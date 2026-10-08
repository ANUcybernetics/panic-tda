#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "polars>=1.0",
#   "numpy>=2.0",
#   "scikit-learn>=1.5",
#   "giotto-ph>=0.2.4",
# ]
# ///
"""Does a run come back to where it has been, more than its drift explains? (TASK-77)

A Rips diagram of a run's states cannot see their order (`tda_keep_kill.py`).
Recurrence in time can only show up in measures that use it. Three, per run:

- sliding-window persistence: the run's states taken L at a time (a delay
  embedding, Perea and Harer 2015), and the longest H1 bar of the Rips diagram
  of those windows. A run that cycles through the same sequence of captions
  gives a long bar.
- recurrence rate: the share of pairs of states at least FAR apart in time
  that sit within one step's distance of each other (the run's median distance
  between neighbouring states). A run that returns to an earlier caption, or
  close to it, scores.
- return after excursion: the same, counting only pairs between which the run
  went at least as far from the first state as twice that step distance and
  as its own typical distance at separation FAR. A run that never moves far
  does not score; one that leaves a caption and comes back to it does.
- return depth: for every state, how far the run went away from it and then
  how close it came back afterwards, averaged over states. A run that leaves
  a theme and comes back scores; one that drifts steadily does not.

THE NULL is TASK-103's description and nothing else: a Gaussian process with
the run's own displacement curve (the mean distance between two states against
their separation in time), so its seed noise and slow wander are the run's,
placed around the run's own centre and in the run's main directions, with the
number of directions chosen so its local intrinsic dimension (TwoNN) matches
the run's. Such a process returns only as often as chance and its drift allow.
Nineteen draws per run; a run is recurrent at 5% when it beats all of them on
a measure. Under the null, 5% of runs do so by chance.

The null is not matched on the global spread across directions (participation
ratio): restricting it to few directions to match local dimension leaves it
spread over fewer than the real runs. Real captions are locally lower
dimensional than any Gaussian with their spread, which is itself a property
of the runs the distance description does not cover.

Frozen runs (half or more of their steps repeat the previous caption exactly)
and the three runs that reach a flat image (`panel_audit.json`) are reported apart: an exact
repeat is a return no continuous process makes.

    ./analysis/sliding_window.py long_horizon_panel_01a09e21_parquet

Results -> analysis/sliding_window.json, summary to stdout.
"""

import argparse
import json
import os
import pathlib
from multiprocessing import Pool

# one numerical thread per worker; the pool supplies the parallelism
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
from gph import ripser_parallel
from tda_keep_kill import load, twonn

OUT = pathlib.Path(__file__).with_suffix(".json")
WINDOWS = (5, 10, 20)
FAR = 20
SURROGATES = 19
# The displacement curve is measured out to VARIOGRAM_FIT_TO; beyond it a
# power law fitted from VARIOGRAM_FIT_FROM is used, since few pairs remain.
VARIOGRAM_FIT_FROM, VARIOGRAM_FIT_TO = 30, 110
DIRECTIONS = (2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 64, 149)
FROZEN = 0.5
SEED = 0
WORKERS = 4
AUDIT = pathlib.Path(__file__).parent / "panel_audit.json"


def unit(a: np.ndarray) -> np.ndarray:
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def variogram(x: np.ndarray) -> np.ndarray:
    """Mean cosine distance at every separation 0..T-1, the tail a power law."""
    n = len(x)
    g = 1.0 - x @ x.T
    lag = np.arange(1, n)
    v = np.array([np.diag(g, k).mean() for k in lag])
    fit = (lag >= VARIOGRAM_FIT_FROM) & (lag <= VARIOGRAM_FIT_TO)
    b, a = np.polyfit(np.log(lag[fit]), np.log(np.maximum(v[fit], 1e-8)), 1)
    v[lag > VARIOGRAM_FIT_TO] = np.exp(a) * lag[lag > VARIOGRAM_FIT_TO] ** b
    return np.concatenate([[0.0], v])


class Null:
    """Gaussian process with x's displacement curve, centre and directions.

    For a process with stationary increments and variogram g, the covariance of
    states s and t relative to the first is g(s) + g(t) - g(|s - t|). With
    spatial covariance of unit trace, half the expected squared distance
    between two states, which for unit vectors is the cosine distance, is g.
    """

    def __init__(self, x: np.ndarray, rng: np.random.Generator):
        n = len(x)
        g = variogram(x)
        t = np.arange(n)
        cov = g[t][:, None] + g[t][None, :] - g[np.abs(t[:, None] - t[None, :])]
        w, v = np.linalg.eigh(cov)
        self.time = v * np.sqrt(np.clip(w, 0, None))
        self.centre = x.mean(axis=0)
        _, self.s, self.vt = np.linalg.svd(x - self.centre, full_matrices=False)
        self.rng = rng
        self.scale = 1.0
        target = twonn(x)
        self.k = min(
            (k for k in DIRECTIONS if k <= len(self.s)),
            key=lambda k: abs(
                np.mean([twonn(self.draw(k, np.random.default_rng(j))) for j in range(2)])
                - target
            ),
        )
        # projecting back to the sphere shrinks distances slightly; undo it
        real = np.mean([np.diag(1.0 - x @ x.T, k).mean() for k in range(1, 51)])
        for _ in range(2):
            y = self.draw(self.k, np.random.default_rng(99))
            made = np.mean([np.diag(1.0 - y @ y.T, k).mean() for k in range(1, 51)])
            self.scale *= np.sqrt(real / made)

    def draw(self, k: int | None = None, rng=None) -> np.ndarray:
        k = self.k if k is None else k
        rng = self.rng if rng is None else rng
        space = (self.s[:k, None] * self.vt[:k]) / np.sqrt((self.s[:k] ** 2).sum())
        y = self.time @ rng.standard_normal((len(self.time), k)) @ space * self.scale
        return unit(self.centre + y - y.mean(axis=0))


def sliding_window_h1(x: np.ndarray, window: int) -> float:
    """Longest H1 bar of the Rips diagram of x's delay embedding."""
    d2 = np.maximum(2.0 * (1.0 - x @ x.T), 0.0)
    m = len(x) - window + 1
    sw = sum(d2[k : k + m, k : k + m] for k in range(window))
    dgm = ripser_parallel(np.sqrt(sw), metric="precomputed", maxdim=1, n_threads=1)
    h1 = np.asarray(dgm["dgms"][1]).reshape(-1, 2)
    return float((h1[:, 1] - h1[:, 0]).max()) if len(h1) else 0.0


def recurrence(x: np.ndarray, eps: float, away: float) -> tuple[float, float, float]:
    """Recurrence rate at separations >= FAR, the same after an excursion of at
    least `away`, and mean return depth."""
    n = len(x)
    d = 1.0 - x @ x.T
    i, j = np.triu_indices(n, FAR)
    close = d[i, j] < eps
    rate = float(close.mean())
    # how far the run went from state i before j, less how close j came back
    reach = np.maximum.accumulate(np.triu(d, 1), axis=1)
    returned = float((close & (reach[i, j - 1] >= away)).mean())
    depth = np.full(n, 0.0)
    for a in range(n - FAR):
        later = np.arange(a + FAR, n)
        depth[a] = (reach[a, later - 1] - d[a, later]).max()
    return rate, returned, float(depth[: n - FAR].mean())


def measures(x: np.ndarray, eps: float, away: float) -> dict[str, float]:
    rate, returned, depth = recurrence(x, eps, away)
    out = {"recurrence_rate": rate, "return_after_excursion": returned, "return_depth": depth}
    for w in WINDOWS:
        out[f"sw_h1_max_L{w}"] = sliding_window_h1(x, w)
    return out


def one_run(job: tuple) -> dict:
    index, x, captions, label = job
    rng = np.random.default_rng([SEED, index])
    caps = np.asarray(captions, dtype=object)
    eps = float(np.median(1.0 - (x[1:] * x[:-1]).sum(axis=1)))
    away = max(2 * eps, float(np.median(1.0 - (x[FAR:] * x[:-FAR]).sum(axis=1))))
    real = measures(x, eps, away)
    null = Null(x, rng)
    draws = [measures(null.draw(), eps, away) for _ in range(SURROGATES)]
    out = label | {
        "frozen": bool((caps[1:] == caps[:-1]).mean() >= FROZEN),
        "eps": eps,
        "away": away,
        "null_directions": null.k,
        "twonn_real": twonn(x),
        "twonn_null": float(np.mean([twonn(null.draw()) for _ in range(3)])),
    }
    for name, value in real.items():
        others = np.array([d[name] for d in draws])
        out[name] = value
        out[f"{name}_null_mean"] = float(others.mean())
        # rank p-value: 1/20 when the run beats every draw
        out[f"{name}_p"] = float((1 + (others >= value).sum()) / (SURROGATES + 1))
    return out


def summarise(rows: list[dict], names: list[str]) -> dict:
    out = {"runs": len(rows)}
    for name in names:
        p = np.array([r[f"{name}_p"] for r in rows])
        hits = int((p <= 1 / (SURROGATES + 1)).sum())
        share = hits / len(rows)
        se = np.sqrt(0.05 * 0.95 / len(rows))
        out[name] = {
            "real_mean": float(np.mean([r[name] for r in rows])),
            "null_mean": float(np.mean([r[f"{name}_null_mean"] for r in rows])),
            "runs_beating_every_draw": hits,
            "share_beating_every_draw": share,
            "share_expected_by_chance": 0.05,
            "chance_upper_95": 0.05 + 1.96 * se,
            "mean_p": float(p.mean()),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("export", type=pathlib.Path)
    args = parser.parse_args()

    runs = load(args.export)
    jobs = []
    for i in range(runs.height):
        x = unit(np.asarray(runs["vector"][i].to_list(), dtype=np.float64))
        label = {
            "run_id": runs["id"][i],
            "network": runs["network"][i],
            "generator": runs["generator"][i],
            "captioner": runs["captioner"][i],
            "initial_prompt": runs["initial_prompt"][i],
            "run_number": int(runs["run_number"][i]),
        }
        jobs.append((i, x, runs["output_text"][i].to_list(), label))
    with Pool(WORKERS) as pool:
        rows = pool.map(one_run, jobs, chunksize=4)

    audit = json.loads(AUDIT.read_text())["images"]["first_flat_image_per_run"]
    flat = {f["run_id"] for f in audit}
    for r in rows:
        r["collapsed"] = r["run_id"] in flat

    names = ["recurrence_rate", "return_after_excursion", "return_depth"] + [f"sw_h1_max_L{w}" for w in WINDOWS]
    clean = [r for r in rows if not (r["frozen"] or r["collapsed"])]
    networks = sorted({r["network"] for r in rows})
    result = {
        "export": str(args.export),
        "surrogates_per_run": SURROGATES,
        "far": FAR,
        "windows": list(WINDOWS),
        "null_fit": {
            "twonn_real": float(np.mean([r["twonn_real"] for r in rows])),
            "twonn_null": float(np.mean([r["twonn_null"] for r in rows])),
            "twonn_correlation": float(
                np.corrcoef([r["twonn_real"] for r in rows], [r["twonn_null"] for r in rows])[0, 1]
            ),
        },
        "excluded": {
            "frozen": sum(r["frozen"] for r in rows),
            "collapsed": sum(r["collapsed"] for r in rows),
        },
        "all_runs": summarise(rows, names),
        "without_frozen_or_collapsed": summarise(clean, names),
        "by_captioner": {
            c: summarise([r for r in clean if r["captioner"] == c], names)
            for c in sorted({r["captioner"] for r in rows})
        },
        "by_generator": {
            g: summarise([r for r in clean if r["generator"] == g], names)
            for g in sorted({r["generator"] for r in rows})
        },
        "by_network": {
            n: summarise([r for r in clean if r["network"] == n], names) for n in networks
        },
        "runs": rows,
    }
    OUT.write_text(json.dumps(result, indent=1) + "\n")

    print(f"null local dimension: real {result['null_fit']['twonn_real']:.2f}, "
          f"null {result['null_fit']['twonn_null']:.2f}, "
          f"r = {result['null_fit']['twonn_correlation']:.2f}")
    print(f"excluded: {result['excluded']}")
    for key in ("all_runs", "without_frozen_or_collapsed"):
        s = result[key]
        print(f"\n{key} ({s['runs']} runs); share beating all {SURROGATES} draws, "
              f"chance 0.05 (upper {s[names[0]]['chance_upper_95']:.3f})")
        for name in names:
            v = s[name]
            print(f"  {name:18s} real {v['real_mean']:.4f}  null {v['null_mean']:.4f}"
                  f"  share {v['share_beating_every_draw']:.3f}  mean p {v['mean_p']:.2f}")
    for key in ("by_captioner", "by_generator"):
        print(f"\n{key}: share beating all draws, without frozen or collapsed runs")
        for level, s in result[key].items():
            print(f"  {level:12s} " + "  ".join(
                f"{n.replace('sw_h1_max_', 'SW ')} {s[n]['share_beating_every_draw']:.2f}"
                for n in names))


if __name__ == "__main__":
    main()
