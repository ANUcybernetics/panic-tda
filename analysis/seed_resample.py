#!/usr/bin/env python
"""What does one seed draw do to a caption? (TASK-103, GPU)

`drift_memory.py` reads the noise in a step off the chain itself: the part of
a step that the next step takes back, `2 D(1) - D(2)`. This measures the same
thing directly. For one late step of every cell, the stored caption is drawn
again with a different seed, the new image captioned, and the new caption
compared with the one the run actually produced from that caption. Two
successors of one caption differ by two independent draws of everything a seed
decides, so if a step is a theme increment `w` plus fresh noise `s`:

    stored step       D(1) = w/2 + s
    two steps on      D(2) = w + s
    resampled twin    R    = w + s

The chain's estimate holds if `R` matches `D(2)`. If part of a step were a
deterministic push that every seed shares, `R` would fall short of it.

Everything goes through the production invoke path. New images are encoded to
AVIF as the pipeline stores them before the captioner sees them, and are
captioned in the stored step's own batch of 40, so the captioner's batch is
the one it had. Flux2Dev costs ~45 s an image, so its cells resample twelve
runs (three of the ten chunks) and keep the stored images in the other slots;
those slots also show what a changed batch does to an unchanged image's
caption.

    _build/dev/snex/projects/Elixir.PanicTda.Models.PythonInterpreter/venv/bin/python \
        analysis/seed_resample.py 01a09e21

Results -> analysis/seed_resample.json. New images are kept under
01a09e21_parquet/seed_resample/ (untracked) so the captioning stages can be
rerun without regenerating.
"""

import base64
import json
import pathlib
import sqlite3
import sys
import time

import numpy as np
import pyvips

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "priv" / "python"))

import panic_models as pm

DB = (HERE.parent / "priv" / "panic_tda_dev.db").resolve()
OUT = pathlib.Path(__file__).with_suffix(".json")

# The image step that is redrawn: its input is the caption at STEP - 1, and the
# stored run went on to the captions at STEP + 1 and STEP + 3.
STEP = 200
GENERATORS = ["SD35Medium", "ZImageTurbo", "Flux2Klein", "Flux2Dev"]
CAPTIONERS = ["Moondream3", "Qwen25VL", "Gemma4", "JoyCaption"]
PARTIAL_CHUNKS = {"Flux2Dev": (0, 3, 6)}
SEED_OFFSET = 0x9E3779B1


def connect() -> sqlite3.Connection:
    return sqlite3.connect(f"file:{DB}?mode=ro", uri=True)


def network(generator: str, captioner: str) -> str:
    return json.dumps([generator, captioner], separators=(",", ":"))


def cell(con: sqlite3.Connection, experiment: str, net: str) -> list[dict]:
    """A cell's runs in batch order, with everything stored around STEP."""
    runs = con.execute(
        "select id, initial_prompt from runs where experiment_id like ? and network = ? "
        "order by id",
        (experiment + "%", net),
    ).fetchall()
    rows = []
    for run_id, prompt in runs:
        steps = {
            sn: (seed, text, image)
            for sn, seed, text, image in con.execute(
                "select sequence_number, seed, output_text, output_image from invocations "
                "where run_id = ? and sequence_number between ? and ?",
                (run_id, STEP - 1, STEP + 3),
            )
        }
        vectors = dict(
            con.execute(
                "select i.sequence_number, e.vector from embeddings e "
                "join invocations i on i.id = e.invocation_id "
                "where i.run_id = ? and i.sequence_number in (?, ?, ?)",
                (run_id, STEP - 1, STEP + 1, STEP + 3),
            )
        )
        rows.append(
            {
                "run_id": run_id,
                "prompt": prompt,
                "caption_in": steps[STEP - 1][1],
                "seed": steps[STEP][0],
                "image": steps[STEP][2],
                "caption_out": steps[STEP + 1][1],
                "vectors": {
                    sn: np.frombuffer(v, dtype=np.float32) for sn, v in vectors.items()
                },
            }
        )
    return rows


def to_avif(png_b64: str) -> bytes:
    """The pipeline's storage encoding (ImageConverter.to_avif!/2)."""
    image = pyvips.Image.new_from_buffer(base64.b64decode(png_b64), "")
    return image.heifsave_buffer(compression="av1", Q=50)


def slots(generator: str, n: int) -> list[int]:
    size = pm._T2I_MAX_BATCH[generator]
    chunks = PARTIAL_CHUNKS.get(generator, range(n // size))
    return [c * size + i for c in chunks for i in range(size)]


def use(name: str) -> None:
    pm.unload_all_models()
    pm.load_model(name)
    pm.swap_to_gpu(name)


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - a @ b)


def main() -> None:
    experiment = sys.argv[1]
    images_dir = HERE.parent / f"{experiment}_parquet" / "seed_resample"
    images_dir.mkdir(parents=True, exist_ok=True)
    pm.setup()
    con = connect()
    cells = {
        (g, c): cell(con, experiment, network(g, c))
        for g in GENERATORS
        for c in CAPTIONERS
    }

    # 1. redraw the step's images at new seeds
    for g in GENERATORS:
        todo = [c for c in CAPTIONERS if not (images_dir / f"{g}_{c}.npz").exists()]
        if not todo:
            continue
        use(g)
        for c in todo:
            rows = cells[g, c]
            chosen = slots(g, len(rows))
            seeds = [(rows[i]["seed"] + SEED_OFFSET) % 2**32 for i in chosen]
            t0 = time.time()
            pngs = pm.invoke_t2i_batch(
                g, [rows[i]["caption_in"] for i in chosen], seeds
            )
            print(
                f"[{g} + {c}] {len(chosen)} images, "
                f"{(time.time() - t0) / len(chosen):.1f} s each",
                flush=True,
            )
            np.savez(
                images_dir / f"{g}_{c}.npz",
                slots=np.asarray(chosen),
                seeds=np.asarray(seeds),
                images=np.asarray([to_avif(p) for p in pngs], dtype=object),
            )

    # 2. caption each redrawn batch in its stored batch of 40
    captions: dict[tuple[str, str], list[str]] = {}
    for c in CAPTIONERS:
        use(c)
        for g in GENERATORS:
            rows = cells[g, c]
            new = np.load(images_dir / f"{g}_{c}.npz", allow_pickle=True)
            batch = [r["image"] for r in rows]
            for slot, image in zip(new["slots"], new["images"], strict=True):
                batch[slot] = image
            t0 = time.time()
            captions[g, c] = pm.invoke_i2t_batch(
                c, [base64.b64encode(b).decode("ascii") for b in batch]
            )
            print(f"[{g} + {c}] captioned in {time.time() - t0:.0f} s", flush=True)

    # 3. embed and compare
    pm.unload_all_models()
    pm.load_model("Qwen3Embed")
    results = {"experiment": experiment, "image_step": STEP, "cells": {}}
    for (g, c), rows in cells.items():
        new = np.load(images_dir / f"{g}_{c}.npz", allow_pickle=True)
        redrawn = set(new["slots"].tolist())
        vectors = [
            np.frombuffer(base64.b64decode(b), dtype=np.float32)
            for b in pm.embed_text("Qwen3Embed", captions[g, c])
        ]
        per_run = []
        for i, (row, caption, vector) in enumerate(
            zip(rows, captions[g, c], vectors, strict=True)
        ):
            stored = row["vectors"]
            per_run.append(
                {
                    "run_id": row["run_id"],
                    "prompt": row["prompt"],
                    "redrawn": i in redrawn,
                    "identical_caption": caption == row["caption_out"],
                    "resampled": distance(vector, stored[STEP + 1]),
                    "parent_to_new": distance(vector, stored[STEP - 1]),
                    "step": distance(stored[STEP - 1], stored[STEP + 1]),
                    "two_steps": distance(stored[STEP - 1], stored[STEP + 3]),
                }
            )
        drawn = [r for r in per_run if r["redrawn"]]
        kept = [r for r in per_run if not r["redrawn"]]

        def mean(rows: list[dict], key: str) -> float | None:
            return float(np.mean([r[key] for r in rows])) if rows else None

        results["cells"][network(g, c)] = {
            "redrawn": len(drawn),
            "resampled": mean(drawn, "resampled"),
            "parent_to_new": mean(drawn, "parent_to_new"),
            "step": mean(drawn, "step"),
            "two_steps": mean(drawn, "two_steps"),
            "identical_captions": sum(r["identical_caption"] for r in drawn),
            "kept": len(kept),
            "kept_identical_captions": sum(r["identical_caption"] for r in kept),
            "kept_distance": mean(kept, "resampled"),
            "runs": per_run,
        }
        r = results["cells"][network(g, c)]
        print(
            f"[{g} + {c}] resampled {r['resampled']:.3f}, step {r['step']:.3f}, "
            f"two steps {r['two_steps']:.3f}, identical {r['identical_captions']}/{r['redrawn']}",
            flush=True,
        )
        OUT.write_text(json.dumps(results, indent=2))
    pm.unload_all_models()


if __name__ == "__main__":
    main()
