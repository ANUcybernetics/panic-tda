#!/usr/bin/env python
"""Does an experiment's stored data reproduce from its own stored inputs? (GPU)

`panel_audit.py` checks that the stored images and captions look right. This
checks that they are what the pipeline says they are, by running stored inputs
back through the production invoke path in a fresh process and comparing:

- a stored caption and seed should regenerate the stored image, and a
  different seed should not, which shows that captions, seeds and images are
  paired as the database says and that the seed is what varies the image
- a stored batch of 40 images should caption to the stored captions
- a run's stored captions should embed to the stored vectors
- every caption is counted under the tokenizer of the generator that read it,
  wrapped as that pipeline wraps it, since a generator reads only its first
  512 tokens and cuts the rest without saying so (TASK-100)

The steps regenerated are chosen to sit either side of the panel's restarts,
on a step that was retried after a CUDA out-of-memory error, and on the step
where a run first went all black.

    _build/dev/snex/projects/Elixir.PanicTda.Models.PythonInterpreter/venv/bin/python \
        analysis/panel_regenerate.py 01a09e21

Results -> analysis/panel_regenerate.json. Sections already present in the
JSON are skipped, so an interrupted run picks up where it stopped.
"""

import base64
import difflib
import io
import json
import pathlib
import sqlite3
import sys
import time

import numpy as np
import torch
from PIL import Image

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "priv" / "python"))
sys.path.insert(0, str(HERE))
from captioner_ceiling_screen import load_encoder_tokenizers

import panic_models as pm

DB = (HERE.parent / "priv" / "panic_tda_dev.db").resolve()
OUT = pathlib.Path(__file__).with_suffix(".json")
CEILING = 512
GEN_CEILING = pm._I2T_MAX_NEW_TOKENS_DEFAULT

# (generator, captioner, image step, chunk of the batch of 40, why)
T2I_TARGETS = [
    ("SD35Medium", "Moondream3", 10, 0, "before the 14 Sep restart"),
    ("SD35Medium", "Qwen25VL", 40, 5, "before the 15 Sep crash"),
    ("SD35Medium", "Qwen25VL", 200, 5, "after the 15 Sep crash"),
    ("ZImageTurbo", "Moondream3", 100, 3, "cell with repeated captions"),
    ("ZImageTurbo", "JoyCaption", 250, 7, ""),
    ("Flux2Klein", "Qwen25VL", 60, 2, "cell with an allocator warning every step"),
    ("Flux2Klein", "Gemma4", 220, 11, ""),
    ("Flux2Klein", "JoyCaption", 150, 15, ""),
    ("Flux2Dev", "Gemma4", 62, 1, "holds the run's first all-black image"),
    ("Flux2Dev", "Gemma4", 118, 0, "step retried after CUDA out of memory"),
    ("Flux2Dev", "Moondream3", 170, 9, "after the 22 Sep OOM kill"),
]
# Flux2Dev costs ~45 s an image, so only one of its chunks gets the control.
T2I_CONTROL_SKIP = {("Flux2Dev", "Gemma4", 118), ("Flux2Dev", "Moondream3", 170)}
# (captioner, generator, caption step)
I2T_TARGETS = [
    ("Moondream3", "ZImageTurbo", 101),
    ("Qwen25VL", "SD35Medium", 41),
    ("Qwen25VL", "SD35Medium", 201),
    ("Gemma4", "Flux2Dev", 119),
    ("JoyCaption", "Flux2Klein", 151),
]
EMBED_CELLS = [
    ("SD35Medium", "Moondream3"),
    ("ZImageTurbo", "Qwen25VL"),
    ("Flux2Klein", "Gemma4"),
    ("Flux2Dev", "JoyCaption"),
]


def connect() -> sqlite3.Connection:
    return sqlite3.connect(f"file:{DB}?mode=ro", uri=True)


def network(generator: str, captioner: str) -> str:
    return json.dumps([generator, captioner], separators=(",", ":"))


def cell_runs(con: sqlite3.Connection, experiment: str, net: str) -> list[tuple]:
    """A cell's runs in the order the engine batches them: creation order."""
    return con.execute(
        "select id, initial_prompt from runs where experiment_id like ? and network = ? "
        "order by id",
        (experiment + "%", net),
    ).fetchall()


def step(con: sqlite3.Connection, run_id: str, sn: int) -> tuple:
    return con.execute(
        "select seed, output_text, output_image from invocations "
        "where run_id = ? and sequence_number = ?",
        (run_id, sn),
    ).fetchone()


def pixels(blob: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"), dtype=np.float32)


def use(name: str) -> None:
    """Load a model the way the pipeline does at the start of a cell."""
    pm.unload_all_models()
    pm.load_model(name)
    pm.swap_to_gpu(name)


def encoder_ceiling(con: sqlite3.Connection, experiment: str) -> dict:
    """Every caption a generator read, counted as that generator counts it."""
    encoders = load_encoder_tokenizers()
    rows = con.execute(
        "select r.network, i.sequence_number, i.output_text from invocations i "
        "join runs r on r.id = i.run_id where r.experiment_id like ? "
        "and i.type = 'text' and i.sequence_number < r.max_length - 1",
        (experiment + "%",),
    ).fetchall()
    by_cell: dict[str, list[int]] = {}
    late: dict[str, list[int]] = {}
    for net, sn, text in rows:
        count = encoders[json.loads(net)[0]]["effective"](text)
        by_cell.setdefault(net, []).append(count)
        if sn >= 150:
            late.setdefault(net, []).append(count)
    cells = {}
    for net, counts in sorted(by_cell.items()):
        c = np.asarray(counts)
        cells[net] = {
            "tokenizer": encoders[json.loads(net)[0]]["tokenizer"],
            "captions_read": int(c.size),
            "median": int(np.median(c)),
            "p99": int(np.percentile(c, 99)),
            "max": int(c.max()),
            "over_512": int((c > CEILING).sum()),
            "over_512_pct": round(float(100 * (c > CEILING).mean()), 2),
            "over_512_pct_second_half": round(
                float(100 * (np.asarray(late[net]) > CEILING).mean()), 2
            ),
            "median_tokens_cut_when_over": int(np.median(c[c > CEILING] - CEILING))
            if (c > CEILING).any()
            else 0,
        }
    total = sum(v["captions_read"] for v in cells.values())
    over = sum(v["over_512"] for v in cells.values())
    return {
        "ceiling": CEILING,
        "captions_read": total,
        "over_512": over,
        "cells": cells,
    }


def generate_pixels(
    generator: str, prompts: list[str], seeds: list[int]
) -> list[np.ndarray]:
    return [
        pixels(base64.b64decode(b))
        for b in pm.invoke_t2i_batch(generator, prompts, seeds)
    ]


def regenerate_images(con: sqlite3.Connection, experiment: str) -> list[dict]:
    results = []
    loaded = None
    for generator, captioner, sn, chunk, why in T2I_TARGETS:
        if generator != loaded:
            use(generator)
            loaded = generator
        size = pm._T2I_MAX_BATCH[generator]
        runs = cell_runs(con, experiment, network(generator, captioner))
        chosen = runs[chunk * size : (chunk + 1) * size]
        prompts, seeds, stored = [], [], []
        for run_id, initial_prompt in chosen:
            seed, _, blob = step(con, run_id, sn)
            prompts.append(initial_prompt if sn == 0 else step(con, run_id, sn - 1)[1])
            seeds.append(seed)
            stored.append(pixels(blob))

        t0 = time.time()
        same = generate_pixels(generator, prompts, seeds)
        seconds = time.time() - t0
        row = {
            "generator": generator,
            "captioner": captioner,
            "image_step": sn,
            "chunk": chunk,
            "why": why,
            "seconds_per_image": round(seconds / len(prompts), 1),
            "stored_luma_mean": [round(float(s.mean()), 1) for s in stored],
            "regenerated_luma_mean": [round(float(s.mean()), 1) for s in same],
            # mean absolute difference per pixel on the 0-255 scale; the stored
            # image has been through AVIF at Q50, the regenerated one has not
            "same_seed_diff": [
                round(float(np.abs(a - b).mean()), 2) for a, b in zip(same, stored)
            ],
        }
        if (generator, captioner, sn) not in T2I_CONTROL_SKIP:
            other = generate_pixels(
                generator, prompts, [(s + 1) % 2**32 for s in seeds]
            )
            row["other_seed_diff"] = [
                round(float(np.abs(a - b).mean()), 2) for a, b in zip(other, stored)
            ]
        print(row, flush=True)
        results.append(row)
    return results


def own_token_counts(name: str, texts: list[str]) -> list[int] | None:
    """Caption lengths under the captioner's own tokenizer, where it has one."""
    model = pm._models[name]
    if isinstance(model, dict):
        tokenizer = model["processor"].tokenizer
        return [len(tokenizer(t, add_special_tokens=False).input_ids) for t in texts]
    # Moondream3 is a bare module carrying a `tokenizers.Tokenizer`
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        return None
    return [len(tokenizer.encode(t).ids) for t in texts]


def regenerate_captions(con: sqlite3.Connection, experiment: str) -> list[dict]:
    results = []
    loaded = None
    for captioner, generator, sn in I2T_TARGETS:
        if captioner != loaded:
            use(captioner)
            loaded = captioner
            every = [
                t
                for (t,) in con.execute(
                    "select i.output_text from invocations i join runs r on r.id = i.run_id "
                    "where r.experiment_id like ? and i.model = ?",
                    (experiment + "%", captioner),
                )
            ]
            counts = own_token_counts(captioner, every)
            ceiling = (
                None
                if counts is None
                else {
                    "captions": len(counts),
                    "max_tokens": max(counts),
                    "at_or_over_generation_ceiling": sum(
                        c >= GEN_CEILING - 1 for c in counts
                    ),
                }
            )
        runs = cell_runs(con, experiment, network(generator, captioner))
        images, stored = [], []
        for run_id, _ in runs:
            images.append(
                base64.b64encode(step(con, run_id, sn - 1)[2]).decode("ascii")
            )
            stored.append(step(con, run_id, sn)[1])
        t0 = time.time()
        again = pm.invoke_i2t_batch(captioner, images)
        seconds = time.time() - t0
        ratios = [
            difflib.SequenceMatcher(None, a, b).ratio() for a, b in zip(again, stored)
        ]
        row = {
            "captioner": captioner,
            "generator": generator,
            "caption_step": sn,
            "batch": len(images),
            "identical": sum(a == b for a, b in zip(again, stored)),
            "min_similarity": round(min(ratios), 4),
            "seconds_per_caption": round(seconds / len(images), 2),
            "generation_ceiling": ceiling,
        }
        print(row, flush=True)
        results.append(row)
    return results


def regenerate_embeddings(con: sqlite3.Connection, experiment: str) -> list[dict]:
    pm.unload_all_models()
    pm.load_model("Qwen3Embed")
    results = []
    for generator, captioner in EMBED_CELLS:
        run_id, _ = cell_runs(con, experiment, network(generator, captioner))[0]
        rows = con.execute(
            "select i.output_text, e.vector from invocations i "
            "join embeddings e on e.invocation_id = i.id "
            "where i.run_id = ? and i.type = 'text' order by i.sequence_number",
            (run_id,),
        ).fetchall()
        stored = np.stack([np.frombuffer(v, dtype=np.float32) for _, v in rows])
        again = np.stack(
            [
                np.frombuffer(base64.b64decode(b), dtype=np.float32)
                for b in pm.embed_text("Qwen3Embed", [t for t, _ in rows])
            ]
        )
        cosine = (stored * again).sum(axis=1)
        row = {
            "generator": generator,
            "captioner": captioner,
            "run_id": run_id,
            "captions": len(rows),
            "min_cosine": round(float(cosine.min()), 6),
            "mean_cosine": round(float(cosine.mean()), 6),
        }
        print(row, flush=True)
        results.append(row)
    return results


def main() -> None:
    experiment = sys.argv[1]
    pm.setup()
    con = connect()
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    results["experiment"] = experiment
    results["versions"] = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }
    sections = {
        "encoder_ceiling": encoder_ceiling,
        "images": regenerate_images,
        "captions": regenerate_captions,
        "embeddings": regenerate_embeddings,
    }
    for name, section in sections.items():
        if name in results:
            continue
        print(f"=== {name} ===", flush=True)
        results[name] = section(con, experiment)
        OUT.write_text(json.dumps(results, indent=2))
    pm.unload_all_models()


if __name__ == "__main__":
    main()
