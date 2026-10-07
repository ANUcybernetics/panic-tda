#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0",
# ]
# ///
"""Is an image's compressed size worth keeping as a measure of a run? (TASK-108)

Every image is stored as AVIF at one fixed quality, so its byte length says how
much detail the encoder had to keep, and it says so without going through the
captioner. This reads the byte length of every image in one finished cell (one
network's runs) and asks three things of its logarithm:

- whose it is: how the variance splits between prompts, between the runs of a
  prompt, and within a run, over the whole run and in each quarter of it.
- how it moves within a run: half the mean squared difference between two
  images of a run against their separation (a variogram), which needs no mean
  removed. As in `drift_memory.py`, `2 g(1) - g(2)` is the part a step gives
  back at the next, the seed's noise. The autocorrelation after removing each
  run's mean is reported beside it.
- whether it tells us anything the captions do not: whether a step in size
  comes with a step between caption embeddings, image by image and between
  blocks of images, and how much of the size a linear reading of the caption
  embedding predicts for runs it has not seen.

The image at image index k is `sequence_number` 2k and its caption is the text
state at 2k + 1.

    ./analysis/image_size.py 01a10613 --network Flux2Klein,Gemma4

Results -> analysis/image_size.json, summary to stdout.
"""

import argparse
import json
import pathlib
import sqlite3
import struct

import numpy as np

HERE = pathlib.Path(__file__).parent
OUT = pathlib.Path(__file__).with_suffix(".json")
DB = (HERE.parent / "priv" / "panic_tda_dev.db").resolve()

# What the stored bytes were encoded with. Size compares only across images
# encoded the same way, so this is part of the result. Read off
# lib/panic_tda/models/image_converter.ex, whose encoder call has not changed
# since 219dfe2 (2026-02-07); no caller passes a quality.
ENCODER = {
    "format": "AVIF (AV1 in HEIF)",
    "library": "libvips heifsave_buffer via vix 0.38.0",
    "quality": 50,
    "other_settings": "libvips defaults",
    "source": "lib/panic_tda/models/image_converter.ex",
}

EMBEDDING_MODEL = "Qwen3Embed"
LAGS = [1, 2, 5, 10, 20, 50, 100, 200, 500]
QUARTERS = 4
# A block of images long enough to average the seed's noise out of both size
# and caption, short enough to leave forty of them in a 1000-image run.
BLOCK = 25
# An image this small has little in it. The first look took the figure from
# where this cell's sizes thin out, well under every prompt's typical image.
SMALL_BYTES = 15_000
# The steps counted as jumps when asking whether the two measures jump together.
JUMP_SHARE = 0.05
RIDGE = 1e-2


def connect() -> sqlite3.Connection:
    return sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=60)


def dimensions(header: bytes) -> tuple[int, int]:
    """Width and height from an AVIF header's `ispe` box, without decoding."""
    at = header.index(b"ispe")
    return struct.unpack(">II", header[at + 8 : at + 16])


def load(experiment: str, network: list[str]) -> dict:
    """Byte lengths as [prompt, run, image] and caption embeddings to match."""
    with connect() as db:
        runs = db.execute(
            "select id, initial_prompt, run_number, max_length from runs "
            "where experiment_id like ? and network = ? "
            "order by initial_prompt, run_number",
            (experiment + "%", json.dumps(network, separators=(",", ":"))),
        ).fetchall()
        assert runs, f"no runs of {network} in experiment {experiment}"
        prompts = sorted({prompt for _, prompt, _, _ in runs})
        per_prompt = len(runs) // len(prompts)
        assert sorted({n for _, _, n, _ in runs}) == list(range(per_prompt))
        assert len(runs) == len(prompts) * per_prompt
        (length,) = {max_length for *_, max_length in runs}
        images = length // 2

        size = np.zeros((len(prompts), per_prompt, images))
        captions = np.zeros((len(prompts), per_prompt, images, 256), dtype=np.float32)
        shapes = set()
        for run_id, prompt, r, _ in runs:
            p = prompts.index(prompt)
            rows = db.execute(
                "select sequence_number, length(output_image), "
                "substr(output_image, 1, 4096) from invocations "
                "where run_id = ? and type = 'image' order by sequence_number",
                (run_id,),
            ).fetchall()
            assert [s for s, _, _ in rows] == list(range(0, length, 2)), run_id
            size[p, r] = [n for _, n, _ in rows]
            shapes |= {dimensions(header) for _, _, header in rows}

            vectors = db.execute(
                "select i.sequence_number, e.vector from invocations i "
                "join embeddings e on e.invocation_id = i.id "
                "where i.run_id = ? and e.embedding_model = ? "
                "order by i.sequence_number",
                (run_id, EMBEDDING_MODEL),
            ).fetchall()
            assert [s for s, _ in vectors] == list(range(1, length, 2)), run_id
            captions[p, r] = [np.frombuffer(v, dtype="<f4") for _, v in vectors]

    captions /= np.linalg.norm(captions, axis=3, keepdims=True)
    return {
        "prompts": prompts,
        "size": size,
        "captions": captions.astype(np.float64),
        "image_dimensions": sorted(shapes),
    }


def variance_split(y: np.ndarray) -> dict[str, float]:
    """Shares of the variance of y[prompt, run, image] at each level."""
    prompt_mean = y.mean(axis=(1, 2), keepdims=True)
    run_mean = y.mean(axis=2, keepdims=True)
    total = ((y - y.mean()) ** 2).sum()
    between_prompts = ((prompt_mean - y.mean()) ** 2).sum() * y.shape[1] * y.shape[2]
    between_runs = ((run_mean - prompt_mean) ** 2).sum() * y.shape[2]
    within = ((y - run_mean) ** 2).sum()
    return {
        "variance": float(total / y.size),
        "between_prompts": float(between_prompts / total),
        "between_runs_of_a_prompt": float(between_runs / total),
        "within_a_run": float(within / total),
    }


def variogram(y: np.ndarray, lag: int) -> float:
    """Half the mean squared difference between images `lag` apart in a run."""
    return float(((y[..., lag:] - y[..., :-lag]) ** 2).mean() / 2)


def autocorrelation(y: np.ndarray, lag: int) -> float:
    """Correlation at `lag` within runs, each run's own mean removed."""
    x = y - y.mean(axis=2, keepdims=True)
    return float((x[..., lag:] * x[..., :-lag]).mean() / (x**2).mean())


def rank(values: np.ndarray) -> np.ndarray:
    order = values.argsort()
    ranks = np.empty(len(values))
    ranks[order] = np.arange(len(values))
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def spread(values: list[float]) -> dict[str, float]:
    return {
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def lift(marked_by: np.ndarray, measured: np.ndarray) -> float:
    """Mean of `measured` on the largest steps of `marked_by`, over its mean."""
    cut = np.quantile(marked_by, 1 - JUMP_SHARE)
    return float(measured[marked_by >= cut].mean() / measured.mean())


def steps_together(y: np.ndarray, captions: np.ndarray) -> dict:
    """Does a step in size come with a step between captions, run by run?"""
    correlations, size_on_caption_jumps, caption_on_size_jumps = [], [], []
    block_correlations = []
    blocks = y.shape[2] // BLOCK
    for p in range(y.shape[0]):
        for r in range(y.shape[1]):
            size_step = np.abs(np.diff(y[p, r]))
            caption_step = 1 - (captions[p, r, 1:] * captions[p, r, :-1]).sum(axis=1)
            correlations.append(spearman(size_step, caption_step))
            size_on_caption_jumps.append(lift(caption_step, size_step))
            caption_on_size_jumps.append(lift(size_step, caption_step))

            size_block = y[p, r].reshape(blocks, BLOCK).mean(axis=1)
            caption_block = captions[p, r].reshape(blocks, BLOCK, -1).mean(axis=1)
            caption_block /= np.linalg.norm(caption_block, axis=1, keepdims=True)
            block_correlations.append(
                spearman(
                    np.abs(np.diff(size_block)),
                    1 - (caption_block[1:] * caption_block[:-1]).sum(axis=1),
                )
            )
    return {
        "per_run_rank_correlation": spread(correlations),
        "size_step_on_largest_caption_steps_over_mean": spread(size_on_caption_jumps),
        "caption_step_on_largest_size_steps_over_mean": spread(caption_on_size_jumps),
        "jump_share": JUMP_SHARE,
        "block_images": BLOCK,
        "per_run_rank_correlation_between_blocks": spread(block_correlations),
    }


def predicted_from_captions(y: np.ndarray, captions: np.ndarray) -> dict[str, float]:
    """How much of the size a linear reading of the caption predicts.

    A ridge regression from the caption embedding to the image's log size,
    fitted with one run of every prompt held out in turn and scored on the held
    out runs only.
    """
    predicted = np.zeros_like(y)
    for held in range(y.shape[1]):
        kept = [r for r in range(y.shape[1]) if r != held]
        x = captions[:, kept].reshape(-1, captions.shape[3])
        target = y[:, kept].reshape(-1)
        centre, level = x.mean(axis=0), target.mean()
        x = x - centre
        weights = np.linalg.solve(
            x.T @ x + RIDGE * len(x) * np.eye(x.shape[1]), x.T @ (target - level)
        )
        predicted[:, held] = (captions[:, held] - centre) @ weights + level
    residual = ((y - predicted) ** 2).sum()
    within_y = y - y.mean(axis=2, keepdims=True)
    within_predicted = predicted - predicted.mean(axis=2, keepdims=True)
    return {
        "r_squared_held_out_runs": float(1 - residual / ((y - y.mean()) ** 2).sum()),
        "correlation_of_run_means": float(
            np.corrcoef(y.mean(axis=2).ravel(), predicted.mean(axis=2).ravel())[0, 1]
        ),
        "correlation_within_runs": float(
            np.corrcoef(within_y.ravel(), within_predicted.ravel())[0, 1]
        ),
    }


def run_table(prompts: list[str], size: np.ndarray, y: np.ndarray) -> list[dict]:
    last = y.shape[2] // QUARTERS
    return [
        {
            "prompt": prompt,
            "run_number": r,
            "typical_bytes": round(float(np.exp(y[p, r].mean()))),
            "typical_bytes_last_quarter": round(float(np.exp(y[p, r, -last:].mean()))),
            "sd_log_bytes": round(float(y[p, r].std()), 3),
            "share_small": round(float((size[p, r] < SMALL_BYTES).mean()), 3),
        }
        for p, prompt in enumerate(prompts)
        for r in range(y.shape[1])
    ]


def analyse(data: dict) -> dict:
    size, captions = data["size"], data["captions"]
    y = np.log(size)
    quarter = y.shape[2] // QUARTERS
    g = {lag: variogram(y, lag) for lag in LAGS}
    within_run_variance = float(((y - y.mean(axis=2, keepdims=True)) ** 2).mean())
    noise = 2 * g[1] - g[2]
    return {
        "prompts": data["prompts"],
        "runs_per_prompt": y.shape[1],
        "images_per_run": y.shape[2],
        "image_dimensions": data["image_dimensions"],
        "bytes": {
            "min": int(size.min()),
            "median": int(np.median(size)),
            "max": int(size.max()),
            "share_small": float((size < SMALL_BYTES).mean()),
            "small_bytes": SMALL_BYTES,
        },
        "typical_bytes_by_prompt": {
            prompt: round(float(np.exp(y[p].mean())))
            for p, prompt in enumerate(data["prompts"])
        },
        "variance_split": variance_split(y),
        "variance_split_by_quarter": [
            variance_split(y[..., q * quarter : (q + 1) * quarter])
            for q in range(QUARTERS)
        ],
        "within_run": {
            "variance": within_run_variance,
            "variogram": {str(lag): g[lag] for lag in LAGS},
            "seed_noise": noise,
            "seed_noise_share_of_within_run_variance": noise / within_run_variance,
            "autocorrelation_run_mean_removed": {
                str(lag): autocorrelation(y, lag) for lag in LAGS
            },
        },
        "steps_together": steps_together(y, captions),
        "predicted_from_captions": predicted_from_captions(y, captions),
        "runs": run_table(data["prompts"], size, y),
    }


def print_summary(result: dict) -> None:
    split = result["variance_split"]
    print(
        f"variance of log bytes {split['variance']:.3f}: "
        f"{split['between_prompts']:.0%} between prompts, "
        f"{split['between_runs_of_a_prompt']:.0%} between runs of a prompt, "
        f"{split['within_a_run']:.0%} within a run"
    )
    for q, part in enumerate(result["variance_split_by_quarter"], 1):
        print(
            f"  quarter {q}: {part['between_prompts']:.0%} / "
            f"{part['between_runs_of_a_prompt']:.0%} / {part['within_a_run']:.0%}"
        )
    within = result["within_run"]
    print(
        f"within a run: variance {within['variance']:.3f}, seed noise "
        f"{within['seed_noise']:.3f} "
        f"({within['seed_noise_share_of_within_run_variance']:.0%})"
    )
    print("  lag   variogram  autocorrelation")
    for lag in LAGS:
        print(
            f"  {lag:>3}   {within['variogram'][str(lag)]:.3f}      "
            f"{within['autocorrelation_run_mean_removed'][str(lag)]:.2f}"
        )
    together = result["steps_together"]
    print(
        "size step against caption step, rank correlation per run: "
        f"{together['per_run_rank_correlation']}"
    )
    print(
        f"  between blocks of {BLOCK}: "
        f"{together['per_run_rank_correlation_between_blocks']}"
    )
    print(f"size predicted from captions: {result['predicted_from_captions']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("experiment", help="experiment id prefix")
    parser.add_argument(
        "--network", required=True, help="the cell, as comma-separated model names"
    )
    parser.add_argument("--out", type=pathlib.Path, default=OUT)
    args = parser.parse_args()
    network = args.network.split(",")

    result = {
        "experiment": args.experiment,
        "network": network,
        "encoder": ENCODER,
        "embedding_model": EMBEDDING_MODEL,
        **analyse(load(args.experiment, network)),
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print_summary(result)


if __name__ == "__main__":
    main()
