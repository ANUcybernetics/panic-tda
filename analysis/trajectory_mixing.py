#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "polars>=1.0",
#   "numpy>=2.0",
#   "scikit-learn>=1.5",
# ]
# ///
"""Do a cell's trajectories share a region of embedding space, or keep their own?

A Markov state model estimates escape times from trajectories crossing between
the same states, so it needs the runs of a cell to overlap. This asks that
directly, with no partition in the way, for every cell of an export:

- how far apart two stationary frames are when they come from the same run,
  from the two runs of one prompt, and from runs of different prompts
- how the distance between frames of one run grows with their separation in
  time, against the distance to other runs: a plateau below the between-run
  level is confinement
- whether the two runs of a prompt, and runs of different prompts, converge or
  diverge over the trajectory
- whether a frame's nearest neighbour in the cell is from its own run
- at several k-means resolutions fitted per cell, how many microstates hold
  more than one run or prompt and how often a run changes microstate

Distances are cosine distances between text-state embeddings. The stationary
part of a run is everything after `BURN_IN` text states, the same cut
`msm_pipeline.py` makes.

    ./analysis/trajectory_mixing.py 01a09e21_parquet

Results -> analysis/trajectory_mixing.json, table to stdout.
"""

import argparse
import json
import pathlib

import numpy as np
import polars as pl
from sklearn.cluster import KMeans

OUT = pathlib.Path(__file__).with_suffix(".json")

BURN_IN = 75
SEPARATIONS = (1, 2, 5, 10, 20, 40, 70)
TIME_BINS = ((0, 10), (10, 25), (25, 50), (50, 75), (75, 100), (100, 125), (125, 150))
RESOLUTIONS = (5, 10, 20, 40, 80)
# A microstate counts as shared when a second run (or prompt) holds at least
# this share of its frames, so one stray frame does not make it shared.
SHARED_MIN_SHARE = 0.1
SEED = 0


def load_cells(
    export_dir: pathlib.Path, embedding_model: str = "Qwen3Embed"
) -> dict[str, tuple[np.ndarray, list[str]]]:
    """Per network: embeddings as (runs, text states, dimension), and each run's prompt."""
    embeddings = pl.read_parquet(export_dir / "embeddings.parquet").filter(
        pl.col("embedding_model") == embedding_model
    )
    invocations = pl.read_parquet(export_dir / "invocations.parquet").select(
        "id", "run_id", "sequence_number", "output_text"
    )
    runs = pl.read_parquet(export_dir / "runs.parquet").select(
        "id", "network", "initial_prompt"
    )
    frame = (
        embeddings.join(invocations, left_on="invocation_id", right_on="id")
        .join(runs, left_on="run_id", right_on="id")
        .filter(pl.col("output_text").is_not_null() & (pl.col("sequence_number") >= 0))
        .sort("network", "initial_prompt", "run_id", "sequence_number")
    )
    cells: dict[str, tuple[np.ndarray, list[str]]] = {}
    for (network,), cell in frame.group_by(["network"], maintain_order=True):
        vectors, prompts = [], []
        for (_, prompt), run in cell.group_by(
            ["run_id", "initial_prompt"], maintain_order=True
        ):
            vectors.append(np.asarray(run["vector"].to_list(), dtype=np.float64))
            prompts.append(prompt)
        x = np.stack(vectors)
        x /= np.linalg.norm(x, axis=2, keepdims=True)
        cells[network] = (x, prompts)
    return cells


def pair_masks(prompts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Run-pair masks: the two runs of one prompt, and runs of different prompts."""
    p = np.asarray(prompts)
    same = p[:, None] == p[None, :]
    off_diagonal = ~np.eye(len(p), dtype=bool)
    return same & off_diagonal, ~same


def region_distances(x: np.ndarray, prompts: list[str]) -> dict:
    """Mean distance between stationary frames, by how the two frames are related."""
    s = x[:, BURN_IN:]
    runs, frames, _ = s.shape
    flat = s.reshape(runs * frames, -1)
    # Mean distance between every frame of run a and every frame of run b.
    block = (1.0 - flat @ flat.T).reshape(runs, frames, runs, frames).mean(axis=(1, 3))
    same_prompt, other_prompt = pair_masks(prompts)
    within = np.diag(block) * frames / (frames - 1)  # drop the zero self-distances
    step = 1.0 - np.einsum("rtd,rtd->rt", s[:, :-1], s[:, 1:])
    return {
        "step": float(step.mean()),
        "within_run": float(within.mean()),
        "same_prompt_other_run": float(block[same_prompt].mean()),
        "other_prompt": float(block[other_prompt].mean()),
    }


def separation_curve(x: np.ndarray) -> dict[str, float]:
    """Distance between two stationary frames of one run, by their separation in time."""
    s = x[:, BURN_IN:]
    return {
        str(tau): float(
            (1.0 - np.einsum("rtd,rtd->rt", s[:, :-tau], s[:, tau:])).mean()
        )
        for tau in SEPARATIONS
        if tau < s.shape[1]
    }


def convergence(x: np.ndarray, prompts: list[str]) -> list[dict]:
    """Distance between runs at the same time step, early to late in the trajectory."""
    same_prompt, other_prompt = pair_masks(prompts)
    rows = []
    for start, stop in TIME_BINS:
        if start >= x.shape[1]:
            continue
        chunk = x[:, start:stop]
        # Run-by-run distance at each time step, averaged over the bin.
        d = (1.0 - np.einsum("atd,btd->abt", chunk, chunk)).mean(axis=2)
        rows.append(
            {
                "text_states": f"{start}-{min(stop, x.shape[1]) - 1}",
                "same_prompt_other_run": float(d[same_prompt].mean()),
                "other_prompt": float(d[other_prompt].mean()),
            }
        )
    return rows


def nearest_neighbour(x: np.ndarray, prompts: list[str]) -> dict:
    """Whose frame is nearest to each stationary frame?"""
    s = x[:, BURN_IN:]
    runs, frames, _ = s.shape
    flat = s.reshape(runs * frames, -1)
    run_of = np.repeat(np.arange(runs), frames)
    prompt_of = np.repeat(np.asarray(prompts), frames)
    similarity = flat @ flat.T
    np.fill_diagonal(similarity, -np.inf)
    own_run = run_of[similarity.argmax(axis=1)] == run_of
    # With the frame's own run removed, is the nearest frame from its prompt's other run?
    similarity[run_of[:, None] == run_of[None, :]] = -np.inf
    same_prompt = prompt_of[similarity.argmax(axis=1)] == prompt_of
    return {
        "nearest_is_own_run_pct": float(100 * own_run.mean()),
        "own_run_chance_pct": float(100 * (frames - 1) / (runs * frames - 1)),
        "nearest_other_run_is_same_prompt_pct": float(100 * same_prompt.mean()),
        "same_prompt_chance_pct": float(100 / (runs - 1)),
    }


def sharing(x: np.ndarray, prompts: list[str], k: int) -> dict:
    """At a per-cell partition of k microstates, who shares them and how often runs move."""
    s = x[:, BURN_IN:]
    runs, frames, _ = s.shape
    labels = (
        KMeans(n_clusters=k, n_init=10, random_state=SEED)
        .fit_predict(s.reshape(runs * frames, -1))
        .reshape(runs, frames)
    )
    prompt_index = np.unique(np.asarray(prompts), return_inverse=True)[1]
    by_run = np.stack([np.bincount(row, minlength=k) for row in labels])
    by_prompt = np.zeros((prompt_index.max() + 1, k))
    np.add.at(by_prompt, prompt_index, by_run)

    def shared(counts: np.ndarray) -> np.ndarray:
        second = np.sort(counts, axis=0)[-2]
        return second >= SHARED_MIN_SHARE * counts.sum(axis=0)

    shared_runs, shared_prompts = shared(by_run), shared(by_prompt)
    size = by_run.sum(axis=0)
    return {
        "microstates": k,
        "shared_by_runs": int(shared_runs.sum()),
        "shared_by_prompts": int(shared_prompts.sum()),
        "frames_in_prompt_shared_pct": float(
            100 * size[shared_prompts].sum() / size.sum()
        ),
        "microstates_per_run": float((by_run > 0).sum(axis=1).mean()),
        "modal_microstate_share_pct": float(100 * (by_run.max(axis=1) / frames).mean()),
        "label_changes_per_100_steps": float(
            100 * (labels[:, 1:] != labels[:, :-1]).mean()
        ),
    }


def analyse(x: np.ndarray, prompts: list[str]) -> dict:
    return {
        "runs": int(x.shape[0]),
        "text_states": int(x.shape[1]),
        "distances": region_distances(x, prompts),
        "separation_curve": separation_curve(x),
        "convergence": convergence(x, prompts),
        "nearest_neighbour": nearest_neighbour(x, prompts),
        "sharing": [sharing(x, prompts, k) for k in RESOLUTIONS],
    }


def print_table(results: dict[str, dict]) -> None:
    print(
        f"{'cell':<30}{'step':>7}{'within':>8}{'tau=70':>8}{'twin':>7}{'other':>7}"
        f"{'own-run NN':>12}{'twin NN':>9}{'shared@20':>11}{'moves@20':>10}"
    )
    for network, r in results.items():
        d, nn = r["distances"], r["nearest_neighbour"]
        k20 = next(s for s in r["sharing"] if s["microstates"] == 20)
        print(
            f"{network:<30}{d['step']:>7.3f}{d['within_run']:>8.3f}"
            f"{r['separation_curve']['70']:>8.3f}{d['same_prompt_other_run']:>7.3f}"
            f"{d['other_prompt']:>7.3f}{nn['nearest_is_own_run_pct']:>11.1f}%"
            f"{nn['nearest_other_run_is_same_prompt_pct']:>8.1f}%"
            f"{k20['frames_in_prompt_shared_pct']:>10.1f}%"
            f"{k20['label_changes_per_100_steps']:>10.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("export_dir", type=pathlib.Path)
    args = parser.parse_args()

    results = {
        network: analyse(x, prompts)
        for network, (x, prompts) in load_cells(args.export_dir).items()
    }
    print_table(results)
    OUT.write_text(
        json.dumps(
            {
                "export": str(args.export_dir),
                "burn_in_text_states": BURN_IN,
                "shared_min_share": SHARED_MIN_SHARE,
                "networks": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
