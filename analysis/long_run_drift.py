#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "polars>=1.0",
#   "numpy>=2.0",
#   "matplotlib>=3.9",
# ]
# ///
"""Does a run ever leave its own region? The 5,000-invocation runs (TASK-103)

`drift_memory.py` leaves one thing open that 150 text states cannot settle:
whether the distance between two states of one run levels off below the
distance to another run of the same prompt, or climbs to it. The only data
deep enough to look is `db/length_5000_experiment.sqlite` (experiment
067efc98, April 2025): 2,500 text states per run, 16 runs of each of two
prompts on the four SMC networks, embedded with STSBMpnet.

For each network this measures the same quantities on the same footing:

- displacement: mean distance between two states of one run against their
  separation, out to 1,250 text states, taken from state 500 on
- the distance between two runs of the same prompt and of different prompts at
  the same text state, in blocks of 250
- the noise in a step, `2 D(1) - D(2)`, and the share of steps that repeat a
  caption exactly

Old models, short captions, no recorded seeds, a different embedding model and
two prompts with no content ("yeah", "nah"), so this says what a loop can do
over a long horizon and nothing about the panel's own numbers.

    ./analysis/long_run_drift.py [db_path]

Results -> analysis/long_run_drift.json, figure in analysis/drift_memory/.
"""

import json
import pathlib
import sqlite3
import sys
from itertools import pairwise

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from drift_memory import AQUA, BLUE, INK, MUTED, SECONDARY, SURFACE, save, styled

DB = pathlib.Path(
    sys.argv[1] if len(sys.argv) > 1 else "db/length_5000_experiment.sqlite"
)
OUT = pathlib.Path(__file__).with_suffix(".json")
LATE = 500
BLOCK = 250
LAGS = (
    1,
    2,
    3,
    5,
    8,
    12,
    20,
    30,
    50,
    75,
    100,
    150,
    200,
    300,
    400,
    600,
    800,
    1000,
    1250,
)
EMBEDDING = "STSBMpnet"


def load(db: pathlib.Path) -> dict[str, dict]:
    """Per network: unit embeddings as (prompt, run, text state, dimension), and captions."""
    con = sqlite3.connect(f"file:{db.resolve()}?mode=ro", uri=True)
    rows = con.execute(
        """
        select r.network, r.initial_prompt, i.run_id, i.output_text, e.vector
        from invocation i
        join run r on r.id = i.run_id
        join embedding e on e.invocation_id = i.id and e.embedding_model = ?
        where i.output_text is not null
        order by r.network, r.initial_prompt, i.run_id, i.sequence_number
        """,
        (EMBEDDING,),
    )
    grouped: dict[str, dict[str, dict[str, tuple[list, list]]]] = {}
    for network, prompt, run_id, text, blob in rows:
        vectors, texts = (
            grouped.setdefault(network, {})
            .setdefault(prompt, {})
            .setdefault(run_id, ([], []))
        )
        vectors.append(np.frombuffer(blob, dtype=np.float32))
        texts.append(text)
    networks = {}
    for network, by_prompt in grouped.items():
        length = min(len(v) for runs in by_prompt.values() for v, _ in runs.values())
        x = np.array(
            [[v[:length] for v, _ in runs.values()] for runs in by_prompt.values()],
            dtype=np.float32,
        )
        x /= np.linalg.norm(x, axis=3, keepdims=True)
        captions = [t[:length] for runs in by_prompt.values() for _, t in runs.values()]
        networks[network] = {"x": x, "captions": captions, "prompts": list(by_prompt)}
    return networks


def displacement(x: np.ndarray, lag: int) -> float:
    late = x[:, :, LATE:]
    return float(
        1.0 - np.einsum("prtd,prtd->prt", late[:, :, :-lag], late[:, :, lag:]).mean()
    )


def between_runs(x: np.ndarray) -> dict[str, list[float]]:
    """Distance between runs at the same text state, in blocks: same prompt, other prompt."""
    n_prompts, n_runs, n_states, _ = x.shape
    same, other = [], []
    for lo in range(0, n_states - BLOCK + 1, BLOCK):
        chunk = x[:, :, lo : lo + BLOCK]
        totals = chunk.sum(axis=1)  # (prompt, t, d): sum over the prompt's runs
        within = (np.einsum("ptd,ptd->pt", totals, totals) - n_runs) / (
            n_runs * (n_runs - 1)
        )
        same.append(float(1.0 - within.mean()))
        if n_prompts > 1:
            cross = np.einsum("ptd,qtd->pqt", totals, totals) / n_runs**2
            other.append(float(1.0 - cross[~np.eye(n_prompts, dtype=bool)].mean()))
    return {"same_prompt": same, "other_prompt": other}


def analyse(entry: dict) -> dict:
    x = entry["x"]
    curve = {lag: displacement(x, lag) for lag in LAGS if lag < x.shape[2] - LATE}
    noise = 2 * curve[1] - curve[2]
    runs = between_runs(x)
    late_blocks = slice(LATE // BLOCK, None)
    twin = float(np.mean(runs["same_prompt"][late_blocks]))
    pairs = repeats = 0
    for captions in entry["captions"]:
        late = captions[LATE:]
        pairs += len(late) - 1
        repeats += sum(a == b for a, b in pairwise(late))
    longest = max(curve)
    return {
        "prompts": entry["prompts"],
        "runs": int(x.shape[0] * x.shape[1]),
        "text_states": int(x.shape[2]),
        "median_caption_words": float(
            np.median([len(c.split()) for run in entry["captions"] for c in run])
        ),
        "repeat_share_late": repeats / pairs,
        "step": curve[1],
        "noise": noise,
        "noise_share": noise / curve[1],
        "displacement": {str(lag): d for lag, d in curve.items()},
        "between_runs": {"block_text_states": BLOCK, **runs},
        "twin_distance_late": twin,
        "stranger_distance_late": float(np.mean(runs["other_prompt"][late_blocks])),
        # how much of the way from the noise floor to another run of the same
        # prompt a single run gets, at separations of 150 (the panel's length)
        # and at the longest separation measured
        "reached_at_150": (curve[150] - noise) / (twin - noise),
        "reached_at_longest": (curve[longest] - noise) / (twin - noise),
        "longest_separation": longest,
    }


def figure(results: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(10.5, 3.4), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (network, r) in zip(axes, results.items(), strict=True):
        lags = [int(k) for k in r["displacement"]]
        styled(ax).plot(lags, list(r["displacement"].values()), color=AQUA, linewidth=2)
        ax.axhline(r["twin_distance_late"], color=BLUE, linewidth=0.8)
        ax.axhline(r["noise"], color=MUTED, linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xlim(1, 1500)
        ax.set_ylim(0, 1.0)
        ax.set_title(" + ".join(json.loads(network)), loc="left", color=INK, fontsize=9)
        ax.set_xlabel("separation in text states", color=SECONDARY, fontsize=8.5)
    axes[0].set_ylabel("cosine distance", color=SECONDARY, fontsize=8.5)
    first = next(iter(results.values()))
    axes[0].text(
        1.3,
        first["twin_distance_late"] + 0.015,
        "distance to another run",
        color=SECONDARY,
        fontsize=7.5,
    )
    axes[0].text(
        40, first["noise"] + 0.015, "seed noise", color=SECONDARY, fontsize=7.5
    )
    fig.suptitle(
        "Distance between two states of one run, in the 5,000-invocation runs of April 2025",
        x=0.01,
        ha="left",
        color=INK,
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "long_run")


def main() -> None:
    results = {network: analyse(entry) for network, entry in load(DB).items()}
    OUT.write_text(
        json.dumps(
            {
                "db": str(DB),
                "embedding": EMBEDDING,
                "late_from_text_state": LATE,
                "networks": results,
            },
            indent=2,
        )
    )
    figure(results)
    print(
        f"{'network':<28}{'words':>6}{'repeat':>8}{'step':>7}{'noise':>7}{'D(150)':>8}"
        f"{'D(1250)':>9}{'twin':>7}{'stranger':>10}{'reached 150':>13}{'longest':>9}"
    )
    for network, r in results.items():
        d = r["displacement"]
        print(
            f"{' + '.join(json.loads(network)):<28}{r['median_caption_words']:>6.0f}"
            f"{r['repeat_share_late']:>8.2f}{r['step']:>7.3f}{r['noise']:>7.3f}{d['150']:>8.3f}"
            f"{d[str(r['longest_separation'])]:>9.3f}{r['twin_distance_late']:>7.3f}"
            f"{r['stranger_distance_late']:>10.3f}{r['reached_at_150']:>13.2f}"
            f"{r['reached_at_longest']:>9.2f}"
        )


if __name__ == "__main__":
    main()
