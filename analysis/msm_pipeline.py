#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "polars>=1.0",
#   "numpy>=2.0",
#   "scikit-learn>=1.5",
#   "deeptime>=0.4.5",
# ]
# ///
"""Markov state model over a complete partition of embedding space (TASK-76).

The standard MSM pipeline, as the task specifies it: a fine k-means partition
of the unit sphere fitted on a frozen corpus of stationary-regime captions, a
transition matrix at a chosen lag, and PCCA+ to coarse-grain the microstates
into metastable sets from the dynamics rather than from density. Every
timestep gets a label; there is no outlier class and no transit story.

    ./analysis/msm_pipeline.py 019f3645_parquet --network SDXLTurbo_Moondream
    ./analysis/msm_pipeline.py 019f3645_parquet --all --microstates 40

**This is plumbing, not a result.** The only export on hand is
`balanced_panel_5x5` at `max_length` 50, which is about 26 text states per run
against a plateau that does not arrive until 50--75 (see
`long-horizon-design.md`). No trajectory here reaches the stationary regime, so
implied timescales cannot converge and the numbers this prints are not
interpretable as kinetics. Its embeddings also predate TASK-96, which found the
stored vectors were mean-pooled. The script exists so that when TASK-90's panel
lands the analysis is a data swap rather than a build, and so the sample-size
question in `report_power` can be answered before the GPU time is spent.

Covers TASK-76 AC#1 (frozen corpus, subsampling stability), AC#2 (implied
timescales against lag), AC#3 (repetition as a descriptive statistic) and AC#4
(complete assignment). AC#5's kinetic observables are computed but the
seed-resample noise floor they must be tested against needs TASK-90's recorded
seeds; `noise_floor` says so rather than inventing one.

Results -> analysis/msm_pipeline.json, tables to stdout.
"""

import argparse
import json
import pathlib
from itertools import pairwise

import numpy as np
import polars as pl
from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import MaximumLikelihoodMSM
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

OUT = pathlib.Path(__file__).with_suffix(".json")

# Fitted on the frozen corpus only, so the partition cannot drift as the
# database grows -- the failure mode `mix cluster.recompute` has globally.
MICROSTATES = 200
METASTABLE_SETS = 4
LAGS = (1, 2, 3, 5, 8, 12, 20)
N_TIMESCALES = 5
STABILITY_REPS = 5
STABILITY_FRACTION = 0.8
SEED = 0

# Below this many observed residences a dwell distribution has no shape worth
# naming, and below this fraction of the trajectory length an escape time is
# extrapolation rather than measurement. Both guards exist because the pipeline
# will otherwise return confident numbers from data that cannot support them.
MIN_VISITS_FOR_VERDICT = 20
MAX_RESOLVABLE_FRACTION = 0.5


def load_text_trajectories(
    export_dir: pathlib.Path, embedding_model: str = "Qwen3Embed"
) -> dict[tuple[str, str], np.ndarray]:
    """Ordered text-state embeddings per (network, run), newest schema.

    Image states carry no `output_text` and are dropped, as are the synthetic
    `sequence_number == -1` rows that hold each run's initial prompt: they were
    embedded at export time and are not steps of the chain.
    """
    embeddings = pl.read_parquet(export_dir / "embeddings.parquet").filter(
        pl.col("embedding_model") == embedding_model
    )
    invocations = pl.read_parquet(export_dir / "invocations.parquet").select(
        "id", "run_id", "sequence_number", "output_text"
    )
    runs = pl.read_parquet(export_dir / "runs.parquet").select("id", "network")

    frame = (
        embeddings.join(invocations, left_on="invocation_id", right_on="id")
        .join(runs, left_on="run_id", right_on="id")
        .filter(pl.col("output_text").is_not_null() & (pl.col("sequence_number") >= 0))
        .sort("network", "run_id", "sequence_number")
    )

    trajectories: dict[tuple[str, str], np.ndarray] = {}
    for (network, run_id), group in frame.group_by(
        ["network", "run_id"], maintain_order=True
    ):
        trajectories[(network, run_id)] = np.asarray(
            group["vector"].to_list(), dtype=np.float64
        )
    return trajectories


def burn_in_length(trajectories: dict[tuple[str, str], np.ndarray]) -> int:
    """Text states to discard as pre-stationary.

    The 200-step baseline and the 300-step pilot both plateau in step size and
    in drift from t0 by invocation 100--150, which is 50--75 text states. Take
    the conservative end, but never more than a third of the shortest run --
    on a shallow export that would leave nothing at all.
    """
    shortest = min(len(t) for t in trajectories.values())
    return min(75, shortest // 3)


def frozen_corpus(
    trajectories: dict[tuple[str, str], np.ndarray], burn_in: int
) -> tuple[np.ndarray, dict]:
    """Pool post-burn-in states into the corpus the partition is fitted on.

    AC#1: the corpus is defined once and recorded, so it is not the growing
    global pool. Returned provenance is what goes in the methods.
    """
    kept = [t[burn_in:] for t in trajectories.values() if len(t) > burn_in]
    if not kept:
        raise SystemExit(
            f"no trajectory is longer than the {burn_in}-state burn-in; "
            "this export is too shallow to define a stationary corpus"
        )
    corpus = np.vstack(kept)
    provenance = {
        "trajectories_total": len(trajectories),
        "trajectories_contributing": len(kept),
        "burn_in_text_states": burn_in,
        "corpus_states": int(corpus.shape[0]),
        "dimension": int(corpus.shape[1]),
    }
    return corpus, provenance


def fit_partition(corpus: np.ndarray, k: int, seed: int = SEED) -> KMeans:
    """Fine k-means partition. Vectors are already L2-normalised, so Euclidean
    k-means is a monotone surrogate for cosine on the sphere."""
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(corpus)


def partition_stability(
    corpus: np.ndarray,
    k: int,
    reps: int = STABILITY_REPS,
    fraction: float = STABILITY_FRACTION,
    seed: int = SEED,
) -> dict:
    """AC#1: are assignments stable under subsampling?

    Refit on random subsamples and score each refit's labelling of the *full*
    corpus against the reference partition's, by adjusted Rand index. This asks
    whether the partition is a property of the data rather than of the sample.
    """
    reference = fit_partition(corpus, k, seed).predict(corpus)
    rng = np.random.default_rng(seed)
    scores = []
    for rep in range(reps):
        idx = rng.choice(
            corpus.shape[0], size=int(fraction * corpus.shape[0]), replace=False
        )
        refit = fit_partition(corpus[idx], k, seed + rep + 1)
        scores.append(float(adjusted_rand_score(reference, refit.predict(corpus))))
    return {
        "adjusted_rand_mean": float(np.mean(scores)),
        "adjusted_rand_min": float(np.min(scores)),
        "subsample_fraction": fraction,
        "repeats": reps,
    }


def symbolise(
    trajectories: dict[tuple[str, str], np.ndarray], partition: KMeans
) -> dict[tuple[str, str], np.ndarray]:
    """AC#4: every timestep of every run gets a microstate label.

    The partition is fitted on post-burn-in states but applied to the whole
    trajectory, so no frame is unassigned. Burn-in frames are excluded from
    estimation below, not from labelling.
    """
    return {key: partition.predict(traj) for key, traj in trajectories.items()}


def implied_timescales(
    sequences: list[np.ndarray], lags=LAGS, n_timescales: int = N_TIMESCALES
) -> list[dict]:
    """AC#2: implied timescales against lag time, in units of text states.

    Convergence -- a flat plateau in lag -- is what justifies the horizon. A
    timescale still climbing at the largest usable lag is unresolved, and the
    task requires saying so rather than extrapolating.
    """
    usable = [s for s in sequences if len(s) >= 2]
    rows = []
    for lag in lags:
        if min((len(s) for s in usable), default=0) <= lag:
            rows.append({"lag": lag, "status": "lag exceeds trajectory length"})
            continue
        counts = TransitionCountEstimator(lagtime=lag, count_mode="sliding").fit_fetch(
            usable
        )
        connected = counts.submodel_largest()
        msm = MaximumLikelihoodMSM(reversible=True).fit_fetch(connected)
        k = min(n_timescales, msm.n_states - 1)
        rows.append(
            {
                "lag": lag,
                "status": "ok",
                "states_connected": int(msm.n_states),
                "states_total": int(counts.n_states),
                "timescales": [float(t) for t in msm.timescales(k=k)],
            }
        )
    return rows


def coarse_grain(
    sequences: list[np.ndarray], lag: int, n_sets: int
) -> tuple[MaximumLikelihoodMSM, np.ndarray, dict]:
    """PCCA+ coarse-graining: microstates -> metastable sets, from the dynamics.

    Returns the MSM, a microstate->macrostate lookup over the *original*
    microstate ids (-1 where a microstate fell outside the largest connected
    set), and a diagnostic saying how many frames that cost.
    """
    requested = n_sets
    counts = TransitionCountEstimator(lagtime=lag, count_mode="sliding").fit_fetch(
        sequences
    )
    connected = counts.submodel_largest()
    msm = MaximumLikelihoodMSM(reversible=True).fit_fetch(connected)
    n_sets = min(n_sets, msm.n_states)
    pcca = msm.pcca(n_sets)

    n_micro = int(counts.n_states)
    lookup = np.full(n_micro, -1, dtype=int)
    lookup[np.asarray(connected.state_symbols)] = pcca.assignments

    frames = sum(len(s) for s in sequences)
    dropped = sum(int(np.sum(lookup[s] < 0)) for s in sequences)
    diagnostic = {
        "microstates_total": n_micro,
        "microstates_connected": int(msm.n_states),
        "frames_total": frames,
        "frames_unassigned": dropped,
        "frames_unassigned_pct": round(100 * dropped / frames, 3) if frames else 0.0,
        "metastable_sets": n_sets,
        "metastable_sets_requested": requested,
        "lag": lag,
    }
    return msm, lookup, diagnostic


def dwell_times(coarse: list[np.ndarray]) -> dict:
    """AC#5: dwell-time distribution per metastable set.

    Reported as median and coefficient of variation of the residence run
    lengths. A memoryless (exponential) dwell has CV ~ 1; CV well above 1 is
    the heavy-tailed case the task asks to distinguish. Runs censored at the
    trajectory end are excluded, since including them biases dwell downwards.
    """
    per_set: dict[int, list[int]] = {}
    for sequence in coarse:
        valid = sequence[sequence >= 0]
        if valid.size == 0:
            continue
        boundaries = np.flatnonzero(np.diff(valid)) + 1
        segments = np.split(valid, boundaries)
        for segment in segments[:-1]:  # drop the censored final segment
            per_set.setdefault(int(segment[0]), []).append(int(segment.size))

    def verdict(lengths: list[int]) -> str:
        if len(lengths) < MIN_VISITS_FOR_VERDICT:
            return f"unresolved: {len(lengths)} visits, need {MIN_VISITS_FOR_VERDICT}"
        mean = float(np.mean(lengths))
        if not mean:
            return "unresolved: zero mean dwell"
        return (
            "heavy-tailed"
            if np.std(lengths) / mean > 1.5
            else "approximately exponential"
        )

    return {
        str(state): {
            "n_visits": len(lengths),
            "median_dwell": float(np.median(lengths)),
            "mean_dwell": float(np.mean(lengths)),
            "cv": float(np.std(lengths) / np.mean(lengths))
            if np.mean(lengths)
            else None,
            "verdict": verdict(lengths),
        }
        for state, lengths in sorted(per_set.items())
    }


def escape_times(
    msm: MaximumLikelihoodMSM,
    lookup: np.ndarray,
    n_sets: int,
    trajectory_length: float,
) -> dict:
    """AC#5: mean first passage time between metastable sets, in text states.

    These are the escape times RQ1 asks for. Any value approaching the
    trajectory length is unresolved by construction -- an escape longer than a
    run cannot be observed in it, however many runs there are.
    """
    symbols = np.asarray(msm.count_model.state_symbols)
    sets = {s: np.flatnonzero(lookup[symbols] == s) for s in range(n_sets)}
    ceiling = MAX_RESOLVABLE_FRACTION * trajectory_length
    out = {}
    for source in range(n_sets):
        for target in range(n_sets):
            if source == target or not sets[source].size or not sets[target].size:
                continue
            mfpt = float(msm.mfpt(sets[source], sets[target]))
            out[f"{source}->{target}"] = {
                "mfpt_text_states": mfpt,
                "resolved": bool(mfpt <= ceiling),
                "note": (
                    ""
                    if mfpt <= ceiling
                    else f"exceeds {MAX_RESOLVABLE_FRACTION:.0%} of the "
                    f"{trajectory_length:.0f}-state trajectory: extrapolation, not measurement"
                ),
            }
    return out


def repetition_rate(export_dir: pathlib.Path) -> dict:
    """AC#3: exact caption repetition as a descriptive statistic only.

    TASK-75 and the programme settled that repetition is not absorption -- runs
    leave a repeated string immediately -- and that it tracks caption length.
    Reported here so it stays a statistic and never becomes a state definition.
    """
    invocations = (
        pl.read_parquet(export_dir / "invocations.parquet")
        .filter(pl.col("output_text").is_not_null() & (pl.col("sequence_number") >= 0))
        .sort("run_id", "sequence_number")
    )
    repeats = total = 0
    lengths = []
    for (_run,), group in invocations.group_by(["run_id"], maintain_order=True):
        texts = group["output_text"].to_list()
        lengths.extend(len(t.split()) for t in texts)
        for previous, current in pairwise(texts):
            total += 1
            repeats += previous == current
    return {
        "consecutive_pairs": total,
        "exact_repeats": repeats,
        "repeat_rate_pct": round(100 * repeats / total, 3) if total else 0.0,
        "median_caption_words": float(np.median(lengths)) if lengths else None,
    }


def report_power(trajectories: dict[tuple[str, str], np.ndarray], k: int) -> dict:
    """Sample size per microstate, which is the question to settle before the run.

    A transition matrix over k microstates has k^2 entries to populate from
    (states - lag) transitions per trajectory. This is the number that decides
    whether "a few hundred microstates" is estimable at a given horizon, and it
    scales with horizon x trajectories -- so it is an argument about the design,
    not about the analysis.
    """
    per_run = [len(t) for t in trajectories.values()]
    frames = int(sum(per_run))
    return {
        "trajectories": len(per_run),
        "median_text_states_per_trajectory": float(np.median(per_run)),
        "frames_total": frames,
        "microstates": k,
        "frames_per_microstate": round(frames / k, 1),
        "matrix_entries": k * k,
        "transitions_per_matrix_entry_at_lag_1": round(
            (frames - len(per_run)) / (k * k), 3
        ),
    }


def noise_floor() -> dict:
    """AC#5's surrogate, which cannot be built from this export.

    A transition is only real if it exceeds what one diffusion seed draw
    produces with no dynamics. That needs pairs of invocations differing only
    in seed, which requires the per-invocation seeds TASK-93 added and TASK-90
    will be the first run to record. Stated rather than approximated.
    """
    return {
        "status": "unavailable",
        "reason": (
            "needs seed-resampled invocation pairs; recorded seeds arrive with "
            "TASK-90's panel (TASK-93). TASK-89's figures are indicative only "
            "and are measured on a different lineup."
        ),
    }


def analyse(
    trajectories: dict[tuple[str, str], np.ndarray], label: str, k: int, n_sets: int
) -> dict:
    burn_in = burn_in_length(trajectories)
    corpus, provenance = frozen_corpus(trajectories, burn_in)

    k = min(k, corpus.shape[0] // 10)  # keep >=10 corpus states per microstate
    partition = fit_partition(corpus, k)
    symbols = symbolise(trajectories, partition)
    estimation = [s[burn_in:] for s in symbols.values() if len(s) > burn_in]

    result = {
        "label": label,
        "corpus": provenance,
        "power": report_power(trajectories, k),
        "partition_stability": partition_stability(corpus, k),
        "implied_timescales": implied_timescales(estimation),
        "noise_floor": noise_floor(),
    }

    lag = 1
    try:
        msm, lookup, diagnostic = coarse_grain(estimation, lag, n_sets)
        coarse = [lookup[s] for s in estimation]
        result["coarse_graining"] = diagnostic
        result["stationary_distribution"] = [
            float(x) for x in msm.pcca(n_sets).coarse_grained_stationary_probability
        ]
        result["dwell_times"] = dwell_times(coarse)
        result["escape_times"] = escape_times(
            msm, lookup, n_sets, result["power"]["median_text_states_per_trajectory"]
        )
    except (ValueError, RuntimeError) as exc:  # a shallow export cannot support this
        result["coarse_graining"] = {"status": "failed", "reason": str(exc)}

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("export_dir", type=pathlib.Path)
    parser.add_argument("--network", help="analyse one network (default: the largest)")
    parser.add_argument("--all", action="store_true", help="every network separately")
    parser.add_argument("--microstates", type=int, default=MICROSTATES)
    parser.add_argument("--sets", type=int, default=METASTABLE_SETS)
    args = parser.parse_args()

    trajectories = load_text_trajectories(args.export_dir)
    by_network: dict[str, dict] = {}
    for (network, run_id), traj in trajectories.items():
        by_network.setdefault(network, {})[(network, run_id)] = traj

    if args.all:
        targets = sorted(by_network)
    elif args.network:
        targets = [args.network]
    else:
        targets = [max(by_network, key=lambda n: len(by_network[n]))]

    results = {
        "export": str(args.export_dir),
        "repetition": repetition_rate(args.export_dir),
        "networks": {},
    }
    for network in targets:
        if network not in by_network:
            raise SystemExit(f"no such network: {network}")
        print(f"\n=== {network} ===")
        analysis = analyse(by_network[network], network, args.microstates, args.sets)
        results["networks"][network] = analysis

        power = analysis["power"]
        print(
            f"  {power['trajectories']} trajectories, "
            f"{power['median_text_states_per_trajectory']:.0f} text states each, "
            f"{power['frames_per_microstate']} frames per microstate "
            f"({power['microstates']} microstates)"
        )
        print(
            f"  partition stability (adjusted Rand, refit on "
            f"{analysis['partition_stability']['subsample_fraction']:.0%}): "
            f"{analysis['partition_stability']['adjusted_rand_mean']:.3f}"
        )
        print("  implied timescales (text states), slowest first:")
        for row in analysis["implied_timescales"]:
            if row["status"] != "ok":
                print(f"    lag {row['lag']:>3}  {row['status']}")
                continue
            shown = "  ".join(f"{t:8.2f}" for t in row["timescales"][:3])
            print(
                f"    lag {row['lag']:>3}  {shown}"
                f"   [{row['states_connected']}/{row['states_total']} microstates connected]"
            )
        cg = analysis.get("coarse_graining", {})
        if cg.get("status") == "failed":
            print(f"  PCCA+ unavailable: {cg['reason']}")
        else:
            print(
                f"  PCCA+ into {cg['metastable_sets']} sets, "
                f"{cg['frames_unassigned_pct']}% of frames unassigned"
            )
            for pair, esc in analysis["escape_times"].items():
                flag = "" if esc["resolved"] else "  << UNRESOLVED"
                print(
                    f"    escape {pair}: {esc['mfpt_text_states']:.1f} text states{flag}"
                )

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
