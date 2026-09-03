"""Diffusion step-count sweep (backlog task: find the Pareto-optimal step count).

Run with the Snex venv interpreter while no experiment holds the GPU:

    _build/dev/snex/projects/Elixir.PanicTda.Models.PythonInterpreter/venv/bin/python analysis/step_sweep.py

Diffusion step-count sweep for the three T2I models running below their
defaults (SD35Medium, Flux2Dev, GLMImage). Same seed and prompt at each step
count; the max step count is the reference. Metrics per (model, steps):
seconds per image, pixel MAE vs reference, NomicVision image-embedding cosine
vs reference, and the loop-relevant one: cosine (Qwen3Embed) between the
Gemma3n caption of this image and of the reference image.

The NomicVision column is not trustworthy --- that model returns NaN-zeroed or
non-reproducible embeddings depending on the process (TASK-86), so read
caption_cos, which is the metric the step-count decision turns on anyway.
Results -> step_sweep.json; images -> step_sweep/<model>/<prompt>_<steps>.png
"""

import base64, json, sqlite3, sys, time, pathlib

sys.path.insert(0, "/home/ben/projects/panic_tda/priv/python")
import numpy as np, torch
import panic_models as pm

HERE = pathlib.Path(__file__).parent
IMG = HERE / "step_sweep"
IMG.mkdir(exist_ok=True)
OUT = IMG / "results.json"
con = sqlite3.connect("/home/ben/projects/panic_tda/priv/panic_tda_dev.db")
long_caps = [
    r[0]
    for r in con.execute("""
  select i.output_text from invocations i join runs r on r.id=i.run_id
  join experiments e on e.id=r.experiment_id
  where e.i2t_max_new_tokens=1024 and e.num_runs=4 and i.type='text'
  and i.sequence_number=25 order by i.id limit 2""")
]
if len(long_caps) < 2:
    raise SystemExit(
        "need natural-length captions from the caption pilot in priv/panic_tda_dev.db"
    )
prompts = [
    "a bicycle leaning against a brick wall",
    "a bustling Tokyo street at night",
] + long_caps
GRID = {
    "SD35Medium": [10, 15, 20, 28, 40],
    "Flux2Dev": [8, 15, 25, 35, 50],
    "GLMImage": [10, 15, 25, 35, 50],
}


def decode(vec_b64):
    return np.frombuffer(base64.b64decode(vec_b64), dtype=np.float32)


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


pm.setup()
images = {}  # (model, steps, pi) -> b64
timing = {}
for model, grid in GRID.items():
    pm.load_model(model)
    pm.swap_to_gpu(model)
    cfg = dict(pm._T2I_INVOKE_CONFIGS[model])
    cfg.pop("num_inference_steps")
    size = pm._T2I_IMAGE_SIZES[model]
    for steps in grid:
        t0 = time.time()
        for pi, prompt in enumerate(prompts):
            gen = torch.Generator(device="cuda").manual_seed(1000 + pi)
            img = pm._models[model](
                prompt=prompt,
                height=size,
                width=size,
                generator=gen,
                num_inference_steps=steps,
                **cfg,
            ).images[0]
            (IMG / model).mkdir(exist_ok=True)
            img.save(IMG / model / f"p{pi}_{steps}.png")
            images[(model, steps, pi)] = pm._encode_image_b64(img)
        timing[(model, steps)] = (time.time() - t0) / len(prompts)
        print(
            f"{model} steps={steps}: {timing[(model, steps)]:.1f} s/image", flush=True
        )
    pm.unload_model(model)

# metrics
pm.load_model("NomicVision")
pm.swap_to_gpu("NomicVision")
img_emb = {k: decode(pm.embed_images("NomicVision", [v])[0]) for k, v in images.items()}
pm.unload_model("NomicVision")
pm.load_model("Gemma3n")
pm.swap_to_gpu("Gemma3n")
keys = list(images)
caps = dict(zip(keys, pm.invoke_i2t_batch("Gemma3n", [images[k] for k in keys])))
pm.unload_model("Gemma3n")
pm.load_model("Qwen3Embed")
pm.swap_to_gpu("Qwen3Embed")
cap_emb = dict(
    zip(keys, (decode(e) for e in pm.embed_text("Qwen3Embed", [caps[k] for k in keys])))
)
pm.unload_model("Qwen3Embed")

results = []
for model, grid in GRID.items():
    ref = grid[-1]
    for steps in grid:
        rows = []
        for pi in range(len(prompts)):
            a = np.asarray(
                pm._decode_image_b64(images[(model, steps, pi)]).convert("RGB"),
                dtype=np.float32,
            )
            b = np.asarray(
                pm._decode_image_b64(images[(model, ref, pi)]).convert("RGB"),
                dtype=np.float32,
            )
            rows.append(
                {
                    "pixel_mae": float(np.abs(a - b).mean()),
                    "img_cos": cos(
                        img_emb[(model, steps, pi)], img_emb[(model, ref, pi)]
                    ),
                    "caption_cos": cos(
                        cap_emb[(model, steps, pi)], cap_emb[(model, ref, pi)]
                    ),
                }
            )
        agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        results.append(
            {
                "model": model,
                "steps": steps,
                "secs_per_image": timing[(model, steps)],
                **agg,
                "captions": [
                    caps[(model, steps, pi)][:120] for pi in range(len(prompts))
                ],
            }
        )
        print(
            f"{model} steps={steps}: {timing[(model, steps)]:.1f} s/img, pixel_mae {agg['pixel_mae']:.1f}, img_cos {agg['img_cos']:.3f}, caption_cos {agg['caption_cos']:.3f}",
            flush=True,
        )
OUT.write_text(json.dumps(results, indent=2))
print("DONE", flush=True)
