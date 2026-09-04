#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0"]
# ///
"""What do the existing 200-step trajectories do over time?

Reads the dev database directly (every experiment with max_length 200) and
answers two design questions for the long-horizon run (TASK-90):

1. repetition -- is exact caption repetition an absorbing state? For each run,
   the first step whose caption equals the caption two steps earlier, then what
   fraction of the remaining steps sit at that string and how many distinct
   strings follow. Broken down by network alongside mean caption length.
2. plateau    -- median step-to-step cosine distance and median distance from
   the first caption, by step bin, for the one 200-step experiment with
   Qwen3Embed embeddings (019d2ec7).

    ./analysis/long_horizon_baseline.py [db_path]

These runs predate the v2 lineup and decision-01 (captions were truncated,
Moondream ran in `short` mode), so they inform the design and are not paper
data. Results -> analysis/long_horizon_baseline.json, tables to stdout.
"""

import json
import pathlib
import sqlite3
import sys
from collections import defaultdict

import numpy as np

DB = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "priv/panic_tda_dev.db")
OUT = pathlib.Path(__file__).with_suffix(".json")
PLATEAU_EXPERIMENT = "019d2ec7"
BINS = [(0, 20), (20, 50), (50, 100), (100, 150), (150, 200)]


def load_captions(con: sqlite3.Connection) -> dict[tuple[str, str], list[str]]:
    """Caption sequence per (network, run_id) for every 200-step run."""
    rows = con.execute(
        """
        select r.network, i.run_id, i.sequence_number, i.output_text
        from invocations i
        join runs r on r.id = i.run_id
        join experiments e on e.id = r.experiment_id
        where e.max_length = 200 and i.type = 'text' and i.output_text is not null
        order by i.run_id, i.sequence_number
        """
    ).fetchall()
    runs: dict[tuple[str, str], list[str]] = defaultdict(list)
    for network, run_id, _seq, text in rows:
        runs[(network, run_id)].append(text)
    return runs


def repetition(runs: dict[tuple[str, str], list[str]]) -> dict[str, dict]:
    per_network: dict[str, dict] = defaultdict(
        lambda: {
            "runs": 0,
            "words": [],
            "t_abs": [],
            "frac_after_at_string": [],
            "distinct_after": [],
        }
    )
    for (network, _run_id), captions in runs.items():
        stats = per_network[network]
        stats["runs"] += 1
        stats["words"].extend(len(c.split()) for c in captions)
        # captions are every second invocation, so index k is loop step 2k+1;
        # a consecutive repeat is captions[k] == captions[k-1]
        t_abs = next(
            (k for k in range(1, len(captions)) if captions[k] == captions[k - 1]),
            None,
        )
        if t_abs is None:
            continue
        after = captions[t_abs + 1 :]
        stats["t_abs"].append(2 * t_abs + 1)
        if after:
            stats["frac_after_at_string"].append(
                sum(c == captions[t_abs] for c in after) / len(after)
            )
            stats["distinct_after"].append(len(set(after)))
    out = {}
    for network, s in per_network.items():
        out[network] = {
            "runs": s["runs"],
            "runs_with_consecutive_repeat": len(s["t_abs"]),
            "mean_first_repeat_step": float(np.mean(s["t_abs"]))
            if s["t_abs"]
            else None,
            "mean_frac_after_at_string": float(np.mean(s["frac_after_at_string"]))
            if s["frac_after_at_string"]
            else None,
            "runs_never_leaving": int(sum(f == 1.0 for f in s["frac_after_at_string"])),
            "mean_distinct_strings_after": float(np.mean(s["distinct_after"]))
            if s["distinct_after"]
            else None,
            "mean_caption_words": float(np.mean(s["words"])),
        }
    return out


def plateau(con: sqlite3.Connection) -> dict[str, dict]:
    rows = con.execute(
        """
        select r.network, i.run_id, i.sequence_number, e.vector
        from embeddings e
        join invocations i on i.id = e.invocation_id
        join runs r on r.id = i.run_id
        where r.experiment_id like ? and e.embedding_model = 'Qwen3Embed'
        order by i.run_id, i.sequence_number
        """,
        (f"{PLATEAU_EXPERIMENT}%",),
    ).fetchall()
    runs: dict[tuple[str, str], list[tuple[int, np.ndarray]]] = defaultdict(list)
    for network, run_id, seq, blob in rows:
        v = np.frombuffer(blob, dtype=np.float32)
        runs[(network, run_id)].append((seq, v / np.linalg.norm(v)))
    step = defaultdict(lambda: defaultdict(list))
    drift = defaultdict(lambda: defaultdict(list))
    for (network, _run_id), seq in runs.items():
        seq.sort()
        v0 = seq[0][1]
        for (_s1, a), (s2, b) in zip(seq, seq[1:]):
            for lo, hi in BINS:
                if lo <= s2 < hi:
                    step[network][(lo, hi)].append(1 - float(a @ b))
                    drift[network][(lo, hi)].append(1 - float(v0 @ b))
    return {
        network: {
            "runs": sum(1 for k in runs if k[0] == network),
            "median_step_distance": {
                f"{lo}-{hi}": float(np.median(step[network][(lo, hi)]))
                for lo, hi in BINS
            },
            "median_distance_from_first": {
                f"{lo}-{hi}": float(np.median(drift[network][(lo, hi)]))
                for lo, hi in BINS
            },
        }
        for network in sorted(step)
    }


def fmt(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def main() -> None:
    con = sqlite3.connect(DB)
    rep = repetition(load_captions(con))
    plat = plateau(con)
    con.close()

    print("exact caption repetition in 200-step runs (all experiments)")
    print(
        f"{'network':32s} runs  repeat  t_first  frac_after  stay  distinct_after  words"
    )
    for network, s in sorted(
        rep.items(), key=lambda kv: -kv[1]["runs_with_consecutive_repeat"]
    ):
        print(
            f"{network:32s} {s['runs']:4d}  {s['runs_with_consecutive_repeat']:6d}  "
            f"{fmt(s['mean_first_repeat_step'], 0):>7s}  {fmt(s['mean_frac_after_at_string']):>10s}  "
            f"{s['runs_never_leaving']:4d}  {fmt(s['mean_distinct_strings_after'], 1):>14s}  "
            f"{s['mean_caption_words']:5.0f}"
        )
    print()
    print(
        f"median step-to-step distance | median distance from first caption ({PLATEAU_EXPERIMENT}, Qwen3Embed)"
    )
    header = "".join(f"{lo}-{hi:<6d}" for lo, hi in BINS)
    print(f"{'network':32s}{header} | {header}")
    for network, s in plat.items():
        a = "".join(f"{v:.4f}  " for v in s["median_step_distance"].values())
        b = "".join(f"{v:.3f}   " for v in s["median_distance_from_first"].values())
        print(f"{network:32s}{a} | {b}")

    OUT.write_text(json.dumps({"repetition": rep, "plateau": plat}, indent=2) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
