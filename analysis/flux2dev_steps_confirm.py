#!/usr/bin/env python
"""Confirm the Flux2Dev step-count choice on more prompts.

`analysis/step_sweep.py` found caption cosine flat and non-monotone across
8--35 steps for Flux2Dev on only four prompts, which is thin evidence for the
most expensive model in the panel. This repeats the comparison on twelve
prompts (four short initial prompts and eight natural-length pilot captions) at
the three step counts that matter: the current setting, the cheapest tested
value, and one in between.

Scale for reading the numbers, measured separately: Gemma3n is deterministic,
so the same image gives cosine 1.000, and unrelated images sit near 0.876.

    _build/dev/snex/projects/Elixir.PanicTda.Models.PythonInterpreter/venv/bin/python analysis/flux2dev_steps_confirm.py
"""

import base64
import json
import pathlib
import sqlite3
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/ben/projects/panic_tda/priv/python")
import panic_models as pm  # noqa: E402

HERE = pathlib.Path(__file__).parent
OUT = HERE / "flux2dev_steps_confirm.json"
STEPS = [8, 12, 15]
REF = 25

con = sqlite3.connect("/home/ben/projects/panic_tda/priv/panic_tda_dev.db")
captions = [
    r[0]
    for r in con.execute("""
      select i.output_text from invocations i join runs r on r.id=i.run_id
      join experiments e on e.id=r.experiment_id
      where e.i2t_max_new_tokens=1024 and i.type='text' and i.sequence_number=25
      order by i.id limit 8""")
]
prompts = [
    "a red apple on a wooden table",
    "a storm approaching a fishing village",
    "a machine dreaming of a forest",
    "a library with impossible architecture",
] + captions
print(f"{len(prompts)} prompts ({len(captions)} natural-length captions)", flush=True)

pm.setup()
pm.load_model("Flux2Dev")
pm.swap_to_gpu("Flux2Dev")
cfg = dict(pm._T2I_INVOKE_CONFIGS["Flux2Dev"])
cfg.pop("num_inference_steps")
size = pm._T2I_IMAGE_SIZES["Flux2Dev"]

images, timing = {}, {}
for steps in STEPS + [REF]:
    t0 = time.time()
    for pi, prompt in enumerate(prompts):
        gen = torch.Generator(device="cuda").manual_seed(2000 + pi)
        img = pm._models["Flux2Dev"](
            prompt=prompt,
            height=size,
            width=size,
            generator=gen,
            num_inference_steps=steps,
            **cfg,
        ).images[0]
        images[(steps, pi)] = pm._encode_image_b64(img)
    timing[steps] = (time.time() - t0) / len(prompts)
    print(f"steps={steps}: {timing[steps]:.1f} s/image", flush=True)
pm.unload_model("Flux2Dev")

pm.load_model("Gemma3n")
pm.swap_to_gpu("Gemma3n")
keys = list(images)
caps = dict(zip(keys, pm.invoke_i2t_batch("Gemma3n", [images[k] for k in keys])))
pm.unload_model("Gemma3n")

pm.load_model("Qwen3Embed")
pm.swap_to_gpu("Qwen3Embed")
dec = lambda b: np.frombuffer(base64.b64decode(b), dtype=np.float32)  # noqa: E731
emb = dict(
    zip(keys, (dec(e) for e in pm.embed_text("Qwen3Embed", [caps[k] for k in keys])))
)
pm.unload_model("Qwen3Embed")


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


results = []
for steps in STEPS + [REF]:
    per_prompt = [cos(emb[(steps, pi)], emb[(REF, pi)]) for pi in range(len(prompts))]
    mae = [
        float(
            np.abs(
                np.asarray(
                    pm._decode_image_b64(images[(steps, pi)]).convert("RGB"),
                    dtype=np.float32,
                )
                - np.asarray(
                    pm._decode_image_b64(images[(REF, pi)]).convert("RGB"),
                    dtype=np.float32,
                )
            ).mean()
        )
        for pi in range(len(prompts))
    ]
    results.append(
        {
            "steps": steps,
            "secs_per_image": timing[steps],
            "caption_cos_mean": float(np.mean(per_prompt)),
            "caption_cos_min": float(np.min(per_prompt)),
            "pixel_mae": float(np.mean(mae)),
            "per_prompt": per_prompt,
        }
    )
    r = results[-1]
    print(
        f"steps={steps:3}: {r['secs_per_image']:6.1f} s/img  caption_cos mean "
        f"{r['caption_cos_mean']:.4f} min {r['caption_cos_min']:.4f}  pixel_mae {r['pixel_mae']:.1f}",
        flush=True,
    )

OUT.write_text(json.dumps(results, indent=2))
print("DONE", flush=True)
