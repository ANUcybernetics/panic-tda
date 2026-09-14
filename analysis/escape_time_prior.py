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
"""Can 40 trajectories of ~100 stationary text states resolve escape times?

A pre-launch check on TASK-90's per-cell budget, asked two ways, with the
guards in `analysis/msm_pipeline.py` applied to answers that are known.

1. **Synthetic.** A three-well chain with an exact mean first passage time
   (MFPT) of 100, 400 or 1,000 states, sampled as short trajectories that all
   start in one well (as the panel's runs all start at a prompt). Shows what
   an ensemble of windows shorter than the escape recovers, and what the
   censored dwell-time estimate does to the true mean dwell.

2. **The deepest loop data there is.** `db/length_5000_experiment.sqlite`
   (experiment 067efc98, 2025-04-04) holds 128 runs of 2,500 text states on
   the four SMC networks, already embedded with STSBMpnet. Per network: one
   partition on the pooled stationary frames, a reference MSM on the full
   trajectories, PCCA+ into three sets, then refits on random windows cut from
   the same trajectories at the panel's shape (40 x 100), the panel's
   non-equilibrium start (first 100 stationary states of each run), the
   `balanced_panel_5x5` shape (40 x 25) and a four-times-larger panel
   (160 x 100). Old models, unseeded, truncated captions, two prompts, so a
   prior on the order of magnitude and not a substitute for the panel.

    ./analysis/escape_time_prior.py [db_path]

Results -> analysis/escape_time_prior.json, tables to stdout. Discussion in
`backlog/docs/escape-time-resolvability.md`.
"""

import json
import pathlib
import sqlite3
import sys
import warnings
from itertools import pairwise

import numpy as np
from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import BayesianMSM, MaximumLikelihoodMSM
from deeptime.markov.tools.analysis import mfpt as exact_mfpt
from sklearn.cluster import KMeans

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from msm_pipeline import MIN_CROSSINGS_FOR_ESCAPE, crossings

warnings.filterwarnings("ignore")

DB = pathlib.Path(
    sys.argv[1] if len(sys.argv) > 1 else "db/length_5000_experiment.sqlite"
)
OUT = pathlib.Path(__file__).with_suffix(".json")
SEED = 0
BURN_IN = 75  # text states; the conservative end of the 50--75 plateau
MICROSTATES = 50  # about 1,500 stationary frames each on 32 x 2,425
SETS = 3
LAGS = (1, 2, 5, 10, 20, 50)
DRAWS = 10
# (label, trajectories, text states each, start at the first stationary state?)
DESIGNS = (
    ("40 x 100 random start", 40, 100, False),
    ("32 x 100 first window", 32, 100, True),
    ("40 x 25 random start", 40, 25, False),
    ("160 x 100 random start", 160, 100, False),
)
rng = np.random.default_rng(SEED)


def fit(sequences: list[np.ndarray], lag: int = 1):
    counts = TransitionCountEstimator(lagtime=lag, count_mode="sliding").fit_fetch(
        sequences
    )
    connected = counts.submodel_largest()
    return MaximumLikelihoodMSM(reversible=True).fit_fetch(connected), counts.n_states


def sets_in(msm, lookup: np.ndarray, n_sets: int) -> dict[int, np.ndarray]:
    symbols = np.asarray(msm.count_model.state_symbols)
    return {s: np.flatnonzero(lookup[symbols] == s) for s in range(n_sets)}


def set_mfpts(msm, lookup: np.ndarray, n_sets: int) -> dict[str, float]:
    sets = sets_in(msm, lookup, n_sets)
    return {
        f"{a}->{b}": float(msm.mfpt(sets[a], sets[b]))
        for a in range(n_sets)
        for b in range(n_sets)
        if a != b and sets[a].size and sets[b].size
    }


def bayes_interval(sequences, lookup, n_sets, source, target) -> list[float] | None:
    effective = TransitionCountEstimator(lagtime=1, count_mode="effective").fit_fetch(
        sequences
    )
    posterior = BayesianMSM(n_samples=100, reversible=True).fit_fetch(
        effective.submodel_largest()
    )
    draws = []
    for m in posterior.samples:
        sets = sets_in(m, lookup, n_sets)
        if sets[source].size and sets[target].size:
            draws.append(float(m.mfpt(sets[source], sets[target])))
    return [float(x) for x in np.percentile(draws, [2.5, 97.5])] if draws else None


# --- 1. synthetic -----------------------------------------------------------

WELLS, PER_WELL = 3, 20


def synthetic_chain(p_leave: float) -> np.ndarray:
    """Fast uniform mixing inside each well, rate p_leave out of it."""
    k = WELLS * PER_WELL
    T = np.zeros((k, k))
    for w in range(WELLS):
        lo, hi = w * PER_WELL, (w + 1) * PER_WELL
        T[lo:hi, lo:hi] = (1 - p_leave) / PER_WELL
        for v in range(WELLS):
            if v != w:
                T[lo:hi, v * PER_WELL : (v + 1) * PER_WELL] = p_leave / (
                    (WELLS - 1) * PER_WELL
                )
    return T


def simulate(T: np.ndarray, n: int, length: int, start_well: int = 0) -> list:
    cumulative = np.cumsum(T, axis=1)
    out = []
    for _ in range(n):
        s = int(rng.integers(start_well * PER_WELL, (start_well + 1) * PER_WELL))
        path = [s]
        for _ in range(length - 1):
            s = int(np.searchsorted(cumulative[s], rng.random()))
            path.append(s)
        out.append(np.array(path))
    return out


def censored_mean_dwell(coarse: list[np.ndarray], state: int) -> tuple[float, int]:
    """Mean of the complete residences in `state`, as msm_pipeline reports it."""
    lengths = []
    for sequence in coarse:
        boundaries = np.flatnonzero(np.diff(sequence)) + 1
        for segment in np.split(sequence, boundaries)[1:-1]:
            if segment[0] == state:
                lengths.append(segment.size)
    return (float(np.mean(lengths)) if lengths else float("nan")), len(lengths)


def run_synthetic() -> list[dict]:
    lookup = np.repeat(np.arange(WELLS), PER_WELL)
    well = {w: np.flatnonzero(lookup == w) for w in range(WELLS)}
    rows = []
    print("synthetic three-well chain, escape 0->1 (states)")
    print(
        f"  {'true MFPT':>9} {'n':>4} {'len':>4} {'estimate':>9} {'95% CI':>15} "
        f"{'crossings':>9} {'true dwell':>10} {'censored est':>12} {'n':>4}"
    )
    for p_leave in (1 / 50, 1 / 200, 1 / 500):
        T = synthetic_chain(p_leave)
        truth = float(exact_mfpt(T, target=well[1], origin=well[0]))
        for n, length in ((40, 100), (40, 400), (160, 100)):
            trajectories = simulate(T, n, length)
            coarse = [lookup[t] for t in trajectories]
            n_cross = int(crossings(coarse, 1, WELLS)[0, 1])
            dwell, n_dwell = censored_mean_dwell(coarse, 0)
            row = {
                "true_mfpt": truth,
                "trajectories": n,
                "length": length,
                "crossings_0_to_1": n_cross,
                "true_mean_dwell_0": 1 / p_leave,
                "censored_mean_dwell_0": dwell,
                "complete_residences_0": n_dwell,
            }
            try:
                msm, _ = fit(trajectories)
                sets = sets_in(msm, lookup, WELLS)
                row["estimate"] = float(msm.mfpt(sets[0], sets[1]))
                row["ci95"] = bayes_interval(trajectories, lookup, WELLS, 0, 1)
                row["resolved"] = n_cross >= MIN_CROSSINGS_FOR_ESCAPE
            except (ValueError, RuntimeError, IndexError) as exc:
                row["estimate"] = None
                row["status"] = f"unavailable: {exc}"
            rows.append(row)
            est = f"{row['estimate']:9.0f}" if row["estimate"] else "     none"
            ci = row.get("ci95")
            ci_s = f"[{ci[0]:6.0f},{ci[1]:6.0f}]" if ci else " " * 15
            print(
                f"  {truth:9.0f} {n:4d} {length:4d} {est} {ci_s} {n_cross:9d} "
                f"{1 / p_leave:10.0f} {dwell:12.0f} {n_dwell:4d}"
            )
    return rows


# --- 2. the 5,000-step SMC runs --------------------------------------------


def load_networks(db: pathlib.Path) -> dict[str, dict]:
    con = sqlite3.connect(db)
    rows = con.execute(
        """
        select r.network, i.run_id, i.sequence_number, i.output_text, e.vector
        from invocation i
        join run r on r.id = i.run_id
        join embedding e on e.invocation_id = i.id and e.embedding_model = 'STSBMpnet'
        where i.output_text is not null
        order by r.network, i.run_id, i.sequence_number
        """
    ).fetchall()
    networks: dict[str, dict] = {}
    for network, run, _seq, text, blob in rows:
        entry = networks.setdefault(network, {"vectors": {}, "texts": {}})
        entry["vectors"].setdefault(run, []).append(
            np.frombuffer(blob, dtype=np.float32)
        )
        entry["texts"].setdefault(run, []).append(text)
    return networks


def repetition(texts: dict[str, list[str]]) -> dict:
    pairs = repeats = 0
    words = []
    for captions in texts.values():
        words.extend(len(c.split()) for c in captions)
        pairs += len(captions) - 1
        repeats += sum(a == b for a, b in pairwise(captions))
    return {
        "consecutive_pairs": pairs,
        "exact_repeats": repeats,
        "repeat_rate_pct": round(100 * repeats / pairs, 2),
        "median_caption_words": float(np.median(words)),
    }


def windows(labels: list[np.ndarray], n: int, length: int, first: bool) -> list:
    out = []
    for _ in range(n):
        t = labels[rng.integers(len(labels))]
        start = 0 if first else rng.integers(0, len(t) - length)
        out.append(t[start : start + length])
    return out


def analyse_network(name: str, entry: dict) -> dict:
    trajectories = [
        np.vstack(v)[BURN_IN:].astype(np.float64) for v in entry["vectors"].values()
    ]
    trajectories = [t / np.linalg.norm(t, axis=1, keepdims=True) for t in trajectories]
    corpus = np.vstack(trajectories)
    partition = KMeans(n_clusters=MICROSTATES, n_init=3, random_state=SEED).fit(corpus)
    labels = [partition.predict(t) for t in trajectories]
    n_runs, length = len(labels), len(labels[0])
    print(f"\n=== {name}: {n_runs} runs x {length} stationary text states ===")

    rep = repetition(entry["texts"])
    print(
        f"  exact repeats {rep['repeat_rate_pct']}% of consecutive captions, "
        f"median {rep['median_caption_words']:.0f} words"
    )

    its = []
    print("  implied timescales (text states) vs lag, slowest three:")
    for lag in LAGS:
        msm, _ = fit(labels, lag)
        ts = [float(t) for t in msm.timescales(k=3)]
        its.append({"lag": lag, "timescales": ts})
        print(f"    lag {lag:3d}: " + "  ".join(f"{t:9.1f}" for t in ts))

    reference, n_states = fit(labels)
    pcca = reference.pcca(SETS)
    lookup = np.full(MICROSTATES, -1)
    lookup[np.asarray(reference.count_model.state_symbols)] = pcca.assignments
    coarse = [lookup[s] for s in labels]
    ref_mfpt = set_mfpts(reference, lookup, SETS)
    occupancy = []
    for s in range(SETS):
        frames_by_run = np.array([int(np.sum(c == s)) for c in coarse])
        occupancy.append(
            {
                "set": s,
                "stationary_probability": float(
                    pcca.coarse_grained_stationary_probability[s]
                ),
                "frames": int(frames_by_run.sum()),
                "runs_visiting": int(np.sum(frames_by_run > 0)),
                "top_run_share": float(
                    frames_by_run.max() / max(frames_by_run.sum(), 1)
                ),
            }
        )
    never_leave = int(sum(len(set(c[c >= 0].tolist())) == 1 for c in coarse))
    print(
        f"  reference MSM at lag 1: {reference.n_states}/{n_states} microstates "
        f"connected; {never_leave}/{n_runs} runs never leave their set"
    )
    for o in occupancy:
        print(
            f"    set {o['set']}: pi {o['stationary_probability']:.2f}, "
            f"{o['runs_visiting']}/{n_runs} runs visit, top run holds "
            f"{o['top_run_share']:.0%} of its frames"
        )
    print(
        "  reference escape times: "
        + ", ".join(f"{k} {v:.0f}" for k, v in ref_mfpt.items())
    )

    designs = []
    for label, n, win, first in DESIGNS:
        estimates: dict[str, list[float]] = {k: [] for k in ref_mfpt}
        resolved: dict[str, int] = {k: 0 for k in ref_mfpt}
        connected, n_cross = [], []
        for _ in range(DRAWS):
            sample = windows(labels, n, win, first)
            try:
                msm, _ = fit(sample)
            except (ValueError, RuntimeError):
                continue
            connected.append(msm.n_states / MICROSTATES)
            observed = crossings([lookup[s] for s in sample], 1, SETS)
            n_cross.append(int(observed.sum()))
            for pair, value in set_mfpts(msm, lookup, SETS).items():
                estimates[pair].append(value)
                a, b = (int(x) for x in pair.split("->"))
                resolved[pair] += int(observed[a, b] >= MIN_CROSSINGS_FOR_ESCAPE)
        design = {
            "design": label,
            "trajectories": n,
            "text_states": win,
            "draws_fitted": len(connected),
            "microstates_connected_mean": float(np.mean(connected))
            if connected
            else None,
            "crossings_per_draw_mean": float(np.mean(n_cross)) if n_cross else None,
            "escape_times": {
                pair: {
                    "reference": ref_mfpt[pair],
                    "draws_with_estimate": len(v),
                    "median": float(np.median(v)) if v else None,
                    "min": float(np.min(v)) if v else None,
                    "max": float(np.max(v)) if v else None,
                    "draws_resolved_by_guard": resolved[pair],
                }
                for pair, v in estimates.items()
            },
        }
        designs.append(design)
        print(
            f"  {label}: {design['microstates_connected_mean'] or 0:.0%} of microstates "
            f"connected, {design['crossings_per_draw_mean'] or 0:.0f} set crossings "
            f"per draw"
        )
        for pair, e in design["escape_times"].items():
            if e["median"] is None:
                continue
            print(
                f"    {pair}: reference {e['reference']:6.0f}, median estimate "
                f"{e['median']:6.0f} [{e['min']:.0f}, {e['max']:.0f}] over "
                f"{e['draws_with_estimate']} draws, {e['draws_resolved_by_guard']} "
                f"pass the crossing guard"
            )

    return {
        "runs": n_runs,
        "stationary_text_states_per_run": length,
        "repetition": rep,
        "implied_timescales": its,
        "reference": {
            "microstates_connected": int(reference.n_states),
            "microstates_total": int(n_states),
            "sets": occupancy,
            "runs_never_leaving_set": never_leave,
            "escape_times": ref_mfpt,
        },
        "designs": designs,
    }


def main() -> None:
    results = {
        "db": str(DB),
        "burn_in": BURN_IN,
        "microstates": MICROSTATES,
        "sets": SETS,
        "min_crossings_for_escape": MIN_CROSSINGS_FOR_ESCAPE,
        "synthetic": run_synthetic(),
        "networks": {
            name: analyse_network(name, entry)
            for name, entry in load_networks(DB).items()
        },
    }
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
