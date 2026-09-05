#!/usr/bin/env python
"""Did the 300-step pilot behave? Per-step cost and nothing degrading (TASK-90).

Reads one experiment from the dev database and reports, over its trajectory:

- wall-clock per step for each model, from the invocation timestamps (a batch
  step shares one started_at/completed_at across its runs, so the batch time is
  divided by the batch size to give a per-item figure comparable with CLAUDE.md)
- caption length by step bin, per run, to see whether captions drift in length
- step-to-step cosine distance and distance from the first caption by step bin,
  the two plateau diagnostics from analysis/long_horizon_baseline.py
- exact caption repeats by step bin

    ./analysis/long_horizon_pilot.py <experiment-id-prefix> [db_path]

Results -> analysis/long_horizon_pilot.json, tables to stdout.
"""

import json
import pathlib
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np

PREFIX = sys.argv[1]
DB = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "priv/panic_tda_dev.db")
OUT = pathlib.Path(__file__).with_suffix(".json")
BINS = [(0, 25), (25, 50), (50, 100), (100, 150), (150, 200), (200, 250), (250, 300)]


def ts(s: str) -> datetime:
    return datetime.fromisoformat(s)


def main() -> None:
    con = sqlite3.connect(DB)
    (exp_id, max_length, started, completed) = con.execute(
        "select id, max_length, started_at, completed_at from experiments where id like ?",
        (PREFIX + "%",),
    ).fetchone()
    rows = con.execute(
        """
        select i.run_id, i.sequence_number, i.model, i.type, i.started_at, i.completed_at,
               i.output_text, em.vector
        from invocations i
        join runs r on r.id = i.run_id
        left join embeddings em on em.invocation_id = i.id and em.embedding_model = 'Qwen3Embed'
        where r.experiment_id = ?
        order by i.run_id, i.sequence_number
        """,
        (exp_id,),
    ).fetchall()
    n_runs = len({r[0] for r in rows})

    # per-step cost: one timestamp pair per batch step, shared by every run in it
    step_times: dict[str, dict[tuple, float]] = defaultdict(dict)
    for _run, seq, model, _t, s, c, _txt, _v in rows:
        step_times[model][(seq, s)] = (ts(c) - ts(s)).total_seconds()
    cost = {}
    for model, steps in step_times.items():
        secs = np.array(list(steps.values()))
        cost[model] = {
            "n_steps": len(secs),
            "batch_size": n_runs,
            "median_batch_s": float(np.median(secs)),
            "median_per_item_s": float(np.median(secs) / n_runs),
            "p90_batch_s": float(np.percentile(secs, 90)),
        }

    # per-run trajectories of captions and embeddings
    runs: dict[str, list] = defaultdict(list)
    for run, seq, _m, t, _s, _c, txt, vec in rows:
        if t == "text":
            v = np.frombuffer(vec, dtype=np.float32) if vec is not None else None
            if v is not None:
                v = v / np.linalg.norm(v)
            runs[run].append((seq, txt, v))

    def by_bin(fn) -> list[dict]:
        out = []
        for lo, hi in BINS:
            vals = [x for r in runs.values() for x in fn(r, lo, hi)]
            if vals:
                out.append(
                    {
                        "lo": lo,
                        "hi": hi,
                        "n": len(vals),
                        "median": float(np.median(vals)),
                    }
                )
        return out

    length = by_bin(lambda r, lo, hi: [len(t.split()) for s, t, _ in r if lo <= s < hi])
    repeats = by_bin(
        lambda r, lo, hi: [
            float(r[i][1] == r[i - 1][1])
            for i in range(1, len(r))
            if lo <= r[i][0] < hi
        ]
    )
    have_vectors = all(v is not None for r in runs.values() for _, _, v in r)
    step = drift = None
    if have_vectors:
        step = by_bin(
            lambda r, lo, hi: [
                1 - float(r[i][2] @ r[i - 1][2])
                for i in range(1, len(r))
                if lo <= r[i][0] < hi
            ]
        )
        drift = by_bin(
            lambda r, lo, hi: [1 - float(v @ r[0][2]) for s, _, v in r if lo <= s < hi]
        )

    elapsed_h = (
        (ts(completed) - ts(started)).total_seconds() / 3600 if completed else None
    )
    result = {
        "experiment": exp_id,
        "max_length": max_length,
        "n_runs": n_runs,
        "elapsed_hours": elapsed_h,
        "cost": cost,
        "caption_words_by_bin": length,
        "repeat_rate_by_bin": repeats,
        "step_distance_by_bin": step,
        "drift_from_t0_by_bin": drift,
    }
    OUT.write_text(json.dumps(result, indent=1))

    print(
        f"experiment {exp_id[:8]}: {n_runs} runs x {max_length} steps, elapsed {elapsed_h and round(elapsed_h, 2)} h"
    )
    for model, c in cost.items():
        print(
            f"  {model:12s} median batch {c['median_batch_s']:6.1f}s  per item {c['median_per_item_s']:5.1f}s  p90 batch {c['p90_batch_s']:6.1f}s"
        )
    print("  bin        words  repeat   step    drift")
    for i, b in enumerate(length):
        s = f"{step[i]['median']:.4f}" if step else "   -  "
        d = f"{drift[i]['median']:.4f}" if drift else "   -  "
        print(
            f"  {b['lo']:3d}-{b['hi']:3d}  {b['median']:6.1f}  {repeats[i]['median']:.2f}   {s}  {d}"
        )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
