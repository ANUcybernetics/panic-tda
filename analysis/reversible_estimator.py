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
"""What does `reversible=True` cost when the chain is not reversible?

`analysis/msm_pipeline.py` fits every transition matrix with
`MaximumLikelihoodMSM(reversible=True)`, and `analysis/escape_time_prior.py`
validated the escape-time guards on a three-well chain whose wells all exchange
at the same rate --- a chain that satisfies detailed balance exactly. So the
assumption has never been off. This asks what it costs when it is wrong.

A three-well chain with the same shape as `escape_time_prior.synthetic_chain`,
but with the leaving probability split unevenly between the two destinations:
a share `f` of departures go forward around the cycle 0->1->2->0 and (1-f) go
back. f=0.5 is the reversible chain already used; f>0.5 is a directed cycle
that breaks detailed balance while leaving the stationary distribution uniform
over the wells, so any departure from a third is estimator error. Exact mean
first passage times come from the true transition matrix, so both estimators
are scored against a known answer.

Sampled at the panel's per-cell shape (40 trajectories x 100 text states, all
starting in well 0, as the panel's runs all start at a prompt). Crossing counts
use `msm_pipeline.crossings`, so "resolved" below means what the pipeline's
guard means.

    ./analysis/reversible_estimator.py

Results -> analysis/reversible_estimator.json, tables to stdout. Discussion in
`backlog/docs/reversible-estimator.md`.
"""

import json
import pathlib
import sys
import warnings

import numpy as np
from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import MaximumLikelihoodMSM
from deeptime.markov.tools.analysis import mfpt as exact_mfpt

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from msm_pipeline import MIN_CROSSINGS_FOR_ESCAPE, crossings

warnings.filterwarnings("ignore")

OUT = pathlib.Path(__file__).with_suffix(".json")

SEED = 20260918
WELLS, PER_WELL = 3, 20
P_LEAVE = 0.01  # exact MFPT between wells is then in the low hundreds of states
N_TRAJ, LENGTH = 40, 100  # the committed per-cell budget, post burn-in
DRAWS = 25
FORWARD = (0.5, 0.7, 0.9, 0.98)  # 0.5 is the reversible chain
PAIRS = ("0->1", "1->0", "0->2")

rng = np.random.default_rng(SEED)


def directed_chain(p_leave: float, forward: float) -> np.ndarray:
    """Three wells, fast uniform mixing inside, a biased cycle between.

    `forward` is the share of departures that go to the next well round the
    cycle; the rest go to the previous one. forward=0.5 reproduces
    `escape_time_prior.synthetic_chain`, which is reversible.
    """
    k = WELLS * PER_WELL
    T = np.zeros((k, k))
    for w in range(WELLS):
        lo, hi = w * PER_WELL, (w + 1) * PER_WELL
        T[lo:hi, lo:hi] = (1 - p_leave) / PER_WELL
        nxt, prv = (w + 1) % WELLS, (w - 1) % WELLS
        T[lo:hi, nxt * PER_WELL : (nxt + 1) * PER_WELL] = p_leave * forward / PER_WELL
        T[lo:hi, prv * PER_WELL : (prv + 1) * PER_WELL] = (
            p_leave * (1 - forward) / PER_WELL
        )
    return T


def stationary(T: np.ndarray) -> np.ndarray:
    evals, evecs = np.linalg.eig(T.T)
    pi = np.real(evecs[:, np.argmin(np.abs(evals - 1))])
    return pi / pi.sum()


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


def true_mfpt(T: np.ndarray, source: int, target: int) -> float:
    """Exact MFPT between wells, averaged over the source well's members."""
    target_states = np.arange(target * PER_WELL, (target + 1) * PER_WELL)
    per_state = exact_mfpt(T, target_states)
    source_states = np.arange(source * PER_WELL, (source + 1) * PER_WELL)
    return float(per_state[source_states].mean())


def flux_asymmetry(T: np.ndarray, pi: np.ndarray) -> float:
    """max |pi_i T_ij - pi_j T_ji| / (pi_i T_ij + pi_j T_ji) over well pairs.

    0 is detailed balance, 1 is one-way flow. Computed on the coarse wells
    rather than the microstates, so it is the asymmetry the metastable-set
    analysis actually sees.
    """
    coarse = np.zeros((WELLS, WELLS))
    for a in range(WELLS):
        lo_a, hi_a = a * PER_WELL, (a + 1) * PER_WELL
        for b in range(WELLS):
            lo_b, hi_b = b * PER_WELL, (b + 1) * PER_WELL
            coarse[a, b] = (pi[lo_a:hi_a, None] * T[lo_a:hi_a, lo_b:hi_b]).sum()
    worst = 0.0
    for a in range(WELLS):
        for b in range(a + 1, WELLS):
            fwd, bwd = coarse[a, b], coarse[b, a]
            if fwd + bwd > 0:
                worst = max(worst, abs(fwd - bwd) / (fwd + bwd))
    return float(worst)


def fit(sequences: list[np.ndarray], reversible: bool) -> dict:
    """MFPTs between wells and the stationary weight of each, as the pipeline
    would report them at the coarse level."""
    counts = TransitionCountEstimator(lagtime=1, count_mode="sliding").fit_fetch(
        sequences
    )
    msm = MaximumLikelihoodMSM(reversible=reversible).fit_fetch(
        counts.submodel_largest()
    )
    symbols = np.asarray(msm.count_model.state_symbols)
    wells = symbols // PER_WELL
    mfpts = {}
    for pair in PAIRS:
        a, b = (int(x) for x in pair.split("->"))
        src, tgt = np.flatnonzero(wells == a), np.flatnonzero(wells == b)
        mfpts[pair] = float(msm.mfpt(src, tgt)) if src.size and tgt.size else None
    pi = np.asarray(msm.stationary_distribution)
    weights = [float(pi[wells == w].sum()) for w in range(WELLS)]
    return {"mfpt": mfpts, "set_weights": weights}


def median_of(draws: list, path: str, key) -> float | None:
    vals = [d[path][key] for d in draws if d[path][key] is not None]
    return float(np.median(vals)) if vals else None


def main() -> None:
    rows = []
    for forward in FORWARD:
        T = directed_chain(P_LEAVE, forward)
        pi = stationary(T)
        exact = {
            pair: true_mfpt(T, *(int(x) for x in pair.split("->"))) for pair in PAIRS
        }

        rev, non, cross = [], [], []
        for _ in range(DRAWS):
            seqs = simulate(T, N_TRAJ, LENGTH)
            rev.append(fit(seqs, reversible=True))
            non.append(fit(seqs, reversible=False))
            coarse = [s // PER_WELL for s in seqs]
            cross.append(crossings(coarse, lag=1, n_sets=WELLS))

        cross_median = {
            pair: float(np.median([c[int(pair[0]), int(pair[-1])] for c in cross]))
            for pair in PAIRS
        }
        row = {
            "forward_share": forward,
            "flux_asymmetry": flux_asymmetry(T, pi),
            "exact_mfpt": exact,
            "reversible_mfpt": {p: median_of(rev, "mfpt", p) for p in PAIRS},
            "nonreversible_mfpt": {p: median_of(non, "mfpt", p) for p in PAIRS},
            "crossings_median": cross_median,
            "resolved_by_guard": {
                p: bool(cross_median[p] >= MIN_CROSSINGS_FOR_ESCAPE) for p in PAIRS
            },
            "exact_set_weights": [
                float(pi[w * PER_WELL : (w + 1) * PER_WELL].sum()) for w in range(WELLS)
            ],
            "reversible_set_weights": [
                median_of(rev, "set_weights", w) for w in range(WELLS)
            ],
            "nonreversible_set_weights": [
                median_of(non, "set_weights", w) for w in range(WELLS)
            ],
        }
        exact_weight = 1.0 / WELLS
        for label in ("reversible", "nonreversible"):
            row[f"{label}_max_weight_error"] = round(
                max(abs(w - exact_weight) for w in row[f"{label}_set_weights"]), 3
            )
        row["ratio_to_exact"] = {
            p: {
                "reversible": round(row["reversible_mfpt"][p] / exact[p], 2)
                if row["reversible_mfpt"][p]
                else None,
                "nonreversible": round(row["nonreversible_mfpt"][p] / exact[p], 2)
                if row["nonreversible_mfpt"][p]
                else None,
            }
            for p in PAIRS
        }
        rows.append(row)

    print(f"escape times, {N_TRAJ} x {LENGTH} states, median of {DRAWS} draws\n")
    print(
        f"{'fwd':>5} {'asym':>6} {'pair':>6} {'exact':>7} {'rev=T':>7} {'rev=F':>7} "
        f"{'xings':>6} {'guard':>10}"
    )
    for row in rows:
        for pair in PAIRS:
            guard = "resolved" if row["resolved_by_guard"][pair] else "UNRESOLVED"
            print(
                f"{row['forward_share']:>5} {row['flux_asymmetry']:>6.2f} {pair:>6} "
                f"{row['exact_mfpt'][pair]:>7.0f} "
                f"{(row['reversible_mfpt'][pair] or float('nan')):>7.0f} "
                f"{(row['nonreversible_mfpt'][pair] or float('nan')):>7.0f} "
                f"{row['crossings_median'][pair]:>6.0f} {guard:>10}"
            )

    print("\nstationary weight per metastable set (exact is 1/3 each)\n")
    print(
        f"{'fwd':>5} {'asym':>6} {'rev=True':>26} {'err':>6} {'rev=False':>26} {'err':>6}"
    )
    for row in rows:
        r = " ".join(f"{x:.3f}" for x in row["reversible_set_weights"])
        n = " ".join(f"{x:.3f}" for x in row["nonreversible_set_weights"])
        print(
            f"{row['forward_share']:>5} {row['flux_asymmetry']:>6.2f} {r:>26} {row['reversible_max_weight_error']:>6.3f} {n:>26} {row['nonreversible_max_weight_error']:>6.3f}"
        )

    OUT.write_text(
        json.dumps(
            {
                "design": {
                    "seed": SEED,
                    "wells": WELLS,
                    "per_well": PER_WELL,
                    "p_leave": P_LEAVE,
                    "trajectories": N_TRAJ,
                    "length": LENGTH,
                    "draws": DRAWS,
                    "min_crossings_for_escape": MIN_CROSSINGS_FOR_ESCAPE,
                },
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
