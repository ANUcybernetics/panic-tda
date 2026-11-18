#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "polars>=1.0",
#   "numpy>=2.0",
#   "scipy>=1.14",
#   "scikit-learn>=1.5",
#   "giotto-ph>=0.2.4",
# ]
# ///
"""What is a sliding-window loop made of? (TASK-77, AC#4)

`sliding_window.py` finds that at 1,000 text states a run's delay embedding has
a longer H1 bar than a null built from its own drift, in about 30% of runs.
TASK-77 adopts sliding-window persistence only if that result is also
interpretable in terms of the trajectory: a cycle that can be pointed to in the
captions. This script tries to point to one.

For every run whose longest H1 bar beats all nineteen null draws, it takes that
bar's representative cycle. Ripser names two edges per bar: the edge whose
arrival closes the loop (birth) and the chord that fills it in (death). The
rest of the cycle is recovered as the shortest path between the closing edge's
ends through edges shorter than the birth scale. Vertices are windows, and a
window is a time, so a cycle is a list of times, and each time has a caption.

Three questions, in order:

- WHERE IN TIME does the cycle live: how far apart are the times it joins, how
  far does the whole cycle reach, how many non-adjacent hops does it make?
- WHAT IS IT MADE OF: how far apart are its states compared with the run's own
  spread, and how many distinct captions and distinct caption openings does it
  carry?
- IS IT A CIRCUIT: collapsing each window to its caption's opening, does the
  cycle visit distinct material and come back (A->B->C->A), or shuttle between
  phrasings it has already used (A->B->A->B)? A circuit is what TASK-77 asks
  for; shuttling is not.

The caption opening (first six words, lowercased) is a crude label. It reads
phrasing, not subject, so two labels can be two ways of opening the same
sentence about the same image. Read the sequences printed at the end before
trusting the count of distinct labels.

    ./analysis/loop_anatomy.py 01a10613_parquet \
        --sliding-window analysis/sliding_window_01a10613.json

Results -> analysis/loop_anatomy.json, summary to stdout.
"""

import argparse
import itertools
import json
import os
import pathlib
from multiprocessing import Pool

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import polars as pl  # noqa: F401  (tda_keep_kill.load needs it)
from gph import ripser_parallel
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from tda_keep_kill import load

OUT = pathlib.Path(__file__).with_suffix(".json")
WINDOW = 20
SETTLED = 300  # these runs pick a subject by about here (follow_up_design.py)
WORKERS = 40
EXAMPLES = 3

WHERE_IN_TIME = (
    "closing_gap", "filling_gap", "span", "hops",
    "closing_edge_in_steps", "first_window",
)  # fmt: skip
WHAT_IT_IS_MADE_OF = (
    "windows", "cycle_mean_distance", "run_mean_distance",
    "settled_mean_distance", "distinct_captions", "distinct_openings",
    "share_after_settled",
)  # fmt: skip


def window_distances(x: np.ndarray, length: int) -> np.ndarray:
    """Distances between the run's states taken `length` at a time."""
    d2 = np.maximum(2.0 * (1.0 - x @ x.T), 0.0)
    m = len(x) - length + 1
    return np.sqrt(sum(d2[k : k + m, k : k + m] for k in range(length)))


def opening(caption: str) -> str:
    return " ".join(caption.split()[:6]).lower().rstrip(".,")


def anatomy(job: tuple) -> dict | None:
    """The longest H1 bar of one run's delay embedding, pulled back to times."""
    label, x, captions, window = job
    d = window_distances(x, window)
    out = ripser_parallel(
        d, metric="precomputed", maxdim=1, n_threads=1, return_generators=True
    )
    h1 = np.asarray(out["dgms"][1]).reshape(-1, 2)
    if not len(h1):
        return None
    i = int(np.argmax(h1[:, 1] - h1[:, 0]))
    birth, death = float(h1[i, 0]), float(h1[i, 1])
    g = np.asarray(out["gens"][1][0])[i]
    a, b = sorted((int(g[0]), int(g[1])))
    c, e = sorted((int(g[2]), int(g[3])))

    # the far side of the loop: a path from a to b that avoids the closing edge
    graph = np.where((d < birth) & (d > 0), d, 0.0)
    np.fill_diagonal(graph, 0.0)
    graph[a, b] = graph[b, a] = 0.0
    reach, previous = dijkstra(csr_matrix(graph), indices=a, return_predecessors=True)
    if not np.isfinite(reach[b]):
        return None
    path, v = [b], b
    while v != a:
        v = int(previous[v])
        path.append(v)
    path = path[::-1]

    gram = 1.0 - x @ x.T
    upper = np.triu_indices(len(x), 1)
    step = float(np.median(np.diag(gram, 1)))
    cycle = gram[np.ix_(path, path)]
    cycle_pairs = np.triu_indices(len(path), 1)
    settled = gram[SETTLED:, SETTLED:][np.triu_indices(len(x) - SETTLED, 1)]

    labels = [opening(captions[w]) for w in path]
    seen: dict[str, str] = {}
    letters = [seen.setdefault(lab, chr(65 + len(seen))) for lab in labels]
    visits = [k for k, _ in itertools.groupby(letters)]

    return label | {
        "birth": birth,
        "death": death,
        "life": death - birth,
        "closing_edge": [a, b],
        "closing_gap": b - a,
        "closing_edge_in_steps": float(gram[a, b] / step) if step else None,
        "filling_edge": [c, e],
        "filling_gap": e - c,
        "windows": len(path),
        "span": max(path) - min(path),
        "hops": int(np.sum(np.abs(np.diff(path)) > 1)),
        "first_window": int(min(path)),
        "share_after_settled": float(np.mean([w >= SETTLED for w in path])),
        "cycle_mean_distance": float(cycle[cycle_pairs].mean()),
        "cycle_diameter": float(cycle[cycle_pairs].max()),
        "run_mean_distance": float(gram[upper].mean()),
        "run_diameter": float(gram[upper].max()),
        "settled_mean_distance": float(settled.mean()),
        "distinct_captions": len({captions[w] for w in path}),
        "distinct_openings": len(seen),
        "visit_sequence": "".join(visits),
        "revisits": len(visits) - len(seen),
        "returns_to_start": bool(len(visits) > 1 and visits[0] == visits[-1]),
        "openings": {letter: lab for lab, letter in seen.items()},
        "path": path,
    }


def quartiles(rows: list[dict], key: str) -> dict[str, float]:
    v = np.array([r[key] for r in rows], dtype=float)
    return {
        "median": float(np.median(v)),
        "lower_quartile": float(np.percentile(v, 25)),
        "upper_quartile": float(np.percentile(v, 75)),
        "min": float(v.min()),
        "max": float(v.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("export", type=pathlib.Path)
    parser.add_argument(
        "--sliding-window", type=pathlib.Path, required=True,
        help="sliding_window.py output for this export, for its p-values",
    )  # fmt: skip
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--out", type=pathlib.Path, default=OUT)
    args = parser.parse_args()
    window = args.window

    runs = load(args.export)
    sw = json.loads(args.sliding_window.read_text())
    per_run = {r["run_id"]: r for r in sw["runs"]}
    surrogates = sw["surrogates_per_run"]
    beats_every_draw = 1 / (surrogates + 1)

    jobs, considered = [], 0
    for i in range(runs.height):
        r = per_run[runs["id"][i]]
        if r["frozen"] or r["collapsed"]:
            continue
        considered += 1
        if r[f"sw_h1_max_L{window}_p"] > beats_every_draw:
            continue
        x = np.asarray(runs["vector"][i].to_list(), dtype=np.float64)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        jobs.append((
            {
                "run_id": r["run_id"], "network": r["network"],
                "generator": r["generator"], "captioner": r["captioner"],
                "initial_prompt": r["initial_prompt"], "run_number": r["run_number"],
                "sw_h1_max": r[f"sw_h1_max_L{window}"],
                "sw_h1_max_null_mean": r[f"sw_h1_max_L{window}_null_mean"],
            },
            x,
            runs["output_text"][i].to_list(),
            window,
        ))
    with Pool(args.workers) as pool:
        cycles = [c for c in pool.map(anatomy, jobs, chunksize=1) if c]

    n = len(cycles)
    result = {
        "export": str(args.export),
        "window": window,
        "runs_considered": considered,
        "runs_with_a_significant_bar": n,
        "where_in_time": {
            k: quartiles(cycles, k)
            for k in WHERE_IN_TIME
        },
        "what_it_is_made_of": {
            k: quartiles(cycles, k)
            for k in WHAT_IT_IS_MADE_OF
        },
        "is_it_a_circuit": {
            "visits_three_or_more_openings": sum(c["distinct_openings"] >= 3 for c in cycles),
            "returns_to_the_opening_it_started_from": sum(c["returns_to_start"] for c in cycles),
            "revisits_an_opening_it_had_left": sum(c["revisits"] > 0 for c in cycles),
            "revisits": quartiles(cycles, "revisits"),
        },
        "closing_edge_joins_adjacent_windows": sum(c["closing_gap"] <= 2 for c in cycles),
        "cycles": cycles,
    }
    args.out.write_text(json.dumps(result, indent=1) + "\n")

    print(f"{n} of {considered} runs have a longest H1 bar beating every draw "
          f"at window {window}\n")
    print("where in time the cycle lives (states):")
    for k, v in result["where_in_time"].items():
        print(f"  {k:24s} median {v['median']:8.1f}   quartiles "
              f"{v['lower_quartile']:7.1f} {v['upper_quartile']:7.1f}")
    print(f"\n  closing edge joins adjacent windows (gap <= 2): "
          f"{result['closing_edge_joins_adjacent_windows']}/{n}")
    print("\nwhat the cycle is made of:")
    for k, v in result["what_it_is_made_of"].items():
        print(f"  {k:24s} median {v['median']:8.3f}   quartiles "
              f"{v['lower_quartile']:7.3f} {v['upper_quartile']:7.3f}")
    c = result["is_it_a_circuit"]
    print("\nis it a circuit?")
    print(f"  visits three or more openings:        {c['visits_three_or_more_openings']}/{n}")
    print(f"  returns to the one it started from:   {c['returns_to_the_opening_it_started_from']}/{n}")
    print(f"  revisits one it had already left:     {c['revisits_an_opening_it_had_left']}/{n}"
          f"  (median {c['revisits']['median']:.0f} times)")

    # Examples near the median, not at the extreme: picking by span-per-hop or
    # by bar length selects the degenerate cycles and misreads the population.
    mid_windows = result["what_it_is_made_of"]["windows"]["median"]
    mid_openings = result["what_it_is_made_of"]["distinct_openings"]["median"]
    near = sorted(cycles, key=lambda c: (
        abs(c["windows"] - mid_windows) / max(mid_windows, 1)
        + abs(c["distinct_openings"] - mid_openings) / max(mid_openings, 1)
    ))
    for c in near[:EXAMPLES]:
        print(f"\n{c['network']} run {c['run_number']} | {c['initial_prompt']}")
        print(f"  {c['windows']} windows, {c['distinct_openings']} openings, "
              f"spanning {c['span']} states; bar {c['life']:.3f} against a null "
              f"mean of {c['sw_h1_max_null_mean']:.3f}")
        print(f"  round the cycle: {c['visit_sequence']}")
        for letter, lab in c["openings"].items():
            print(f"    {letter}: {lab}")


if __name__ == "__main__":
    main()
