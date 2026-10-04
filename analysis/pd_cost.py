#!/usr/bin/env python
"""How long does a run's persistence diagram take as the run gets longer? (TASK-104)

The pipeline computes one diagram per run at the end of each cell
(`lib/panic_tda/models/tda.ex`: Vietoris-Rips to dimension 2 over the run's
caption embeddings). The panel's runs have 150 text states and a diagram took
0.03 s. The long follow-up has 700 or more, and the call is made through Snex,
which gives up after five seconds unless told otherwise.

This times the same call on clouds of increasing size, in the pipeline's own
Python environment so the library build is the one the pipeline uses:

- an old run: the first N text states of one 5,000-invocation run from April
  2025, the only real runs this long (STSBMpnet, 768 dimensions)
- the panel, stitched: the first N caption embeddings of one panel cell, its
  runs taken one after another
- a run as described: a prompt's centre, an offset that wanders with the time
  constant `drift-and-memory.md` gives, and fresh noise at every state
- no structure: N points spread evenly over the sphere, which is what the
  diagram costs when nothing ties one caption to the next

    _build/dev/snex/projects/Elixir.PanicTda.Models.PythonInterpreter/venv/bin/python \
        analysis/pd_cost.py

Results -> analysis/pd_cost.json, table to stdout.
"""

import json
import pathlib
import sqlite3
import time

import numpy as np
from gph import ripser_parallel
from persim.persistent_entropy import persistent_entropy

HERE = pathlib.Path(__file__).parent
OUT = pathlib.Path(__file__).with_suffix(".json")
PANEL_DB = (HERE.parent / "priv" / "panic_tda_dev.db").resolve()
OLD_DB = (HERE.parent / "db" / "length_5000_experiment.sqlite").resolve()
PANEL = "01a09e21"
PANEL_CELL = '["SD35Medium","Moondream3"]'

SIZES = (150, 300, 500, 700, 1000, 1400)
# without structure the cost climbs so fast that 150 points is already slow
UNSTRUCTURED_SIZES = (60, 100, 150)
# as `tda.ex` calls it
MAX_DIM = 2
THREADS = 4
# the panel's averages (`drift_memory.json`): a run's own offset at the end,
# the noise in a step, and the time constant its displacement implies
OFFSET, NOISE, RELAXATION = 0.36, 0.034, 145.0
DIMENSION = 256
SEED = 0


def connect(db: pathlib.Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db}?mode=ro", uri=True)


def old_run(n: int) -> np.ndarray:
    con = connect(OLD_DB)
    (run_id,) = con.execute("select id from run order by id limit 1").fetchone()
    rows = con.execute(
        "select e.vector from invocation i join embedding e on e.invocation_id = i.id "
        "and e.embedding_model = 'STSBMpnet' where i.run_id = ? "
        "and i.output_text is not null order by i.sequence_number limit ?",
        (run_id, n),
    )
    return np.stack([np.frombuffer(blob, dtype=np.float32) for (blob,) in rows])


def panel_stitched(n: int) -> np.ndarray:
    con = connect(PANEL_DB)
    rows = con.execute(
        "select e.vector from invocations i join embeddings e on e.invocation_id = i.id "
        "join runs r on r.id = i.run_id where r.experiment_id like ? and r.network = ? "
        "and i.type = 'text' order by r.id, i.sequence_number limit ?",
        (PANEL + "%", PANEL_CELL, n),
    )
    return np.stack([np.frombuffer(blob, dtype=np.float32) for (blob,) in rows])


def unit(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def described_run(n: int) -> np.ndarray:
    """Centre, wandering offset and noise, each with the cosine distance it measures."""
    rng = np.random.default_rng(SEED)
    centre = rng.standard_normal(DIMENSION)
    centre /= np.linalg.norm(centre)
    keep = np.exp(-1.0 / RELAXATION)
    spread = np.sqrt(OFFSET / DIMENSION)
    offset = np.empty((n, DIMENSION))
    offset[0] = spread * rng.standard_normal(DIMENSION)
    for t in range(1, n):
        offset[t] = keep * offset[t - 1] + np.sqrt(
            1 - keep**2
        ) * spread * rng.standard_normal(DIMENSION)
    noise = np.sqrt(NOISE / DIMENSION) * rng.standard_normal((n, DIMENSION))
    return unit(centre + offset + noise)


def unstructured(n: int) -> np.ndarray:
    return unit(np.random.default_rng(SEED).standard_normal((n, DIMENSION)))


def timed(cloud: np.ndarray) -> dict:
    start = time.perf_counter()
    diagram = ripser_parallel(
        cloud, maxdim=MAX_DIM, return_generators=False, n_threads=THREADS
    )
    persistent_entropy(diagram["dgms"], normalize=False)
    return {
        "points": int(cloud.shape[0]),
        "seconds": round(time.perf_counter() - start, 2),
        "pairs_by_dimension": [len(d) for d in diagram["dgms"]],
    }


def main() -> None:
    clouds = {
        "an old run": (old_run, SIZES),
        "the panel, stitched": (panel_stitched, SIZES),
        "a run as described": (described_run, SIZES),
        "no structure": (unstructured, UNSTRUCTURED_SIZES),
    }
    results = {"max_dimension": MAX_DIM, "threads": THREADS, "clouds": {}}
    for name, (source, sizes) in clouds.items():
        rows = []
        for n in sizes:
            rows.append(timed(source(n)))
            print(f"{name:<20}{n:>6} points{rows[-1]['seconds']:>9.2f} s", flush=True)
        results["clouds"][name] = rows
        OUT.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
