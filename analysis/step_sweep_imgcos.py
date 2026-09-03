#!/usr/bin/env python
"""Recompute the image-embedding column of the step sweep from its saved images.

`analysis/step_sweep.py` recorded img_cos as 0.0 for every row, including
reference-against-itself rows that must be exactly 1.0, so that column of
`step_sweep/results.json` is unusable. The generated images are on disk, so the
metric can be rebuilt without re-running any diffusion.

Run with the Snex venv interpreter while the GPU is free:

    _build/dev/snex/projects/Elixir.PanicTda.Models.PythonInterpreter/venv/bin/python analysis/step_sweep_imgcos.py
"""

import base64
import io
import json
import pathlib
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "/home/ben/projects/panic_tda/priv/python")
import panic_models as pm  # noqa: E402

HERE = pathlib.Path(__file__).parent
IMG = HERE / "step_sweep"
RESULTS = IMG / "results.json"

GRID = {
    "SD35Medium": [10, 15, 20, 28, 40],
    "Flux2Dev": [8, 15, 25, 35, 50],
    "GLMImage": [10, 15, 25, 35, 50],
}
N_PROMPTS = 4


def as_b64(path: pathlib.Path) -> str:
    buf = io.BytesIO()
    Image.open(path).convert("RGB").save(buf, format="WEBP", lossless=True)
    return base64.b64encode(buf.getvalue()).decode()


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


pm.setup()
pm.load_model("NomicVision")
pm.swap_to_gpu("NomicVision")

keys, b64s = [], []
for model, grid in GRID.items():
    for steps in grid:
        for pi in range(N_PROMPTS):
            path = IMG / model / f"p{pi}_{steps}.png"
            if not path.exists():
                raise SystemExit(f"missing {path}")
            keys.append((model, steps, pi))
            b64s.append(as_b64(path))

# embed in one batched call, which is also the difference from the original:
# it called embed_images once per image inside a dict comprehension
vecs = pm.embed_images("NomicVision", b64s)
emb = {
    k: np.frombuffer(base64.b64decode(v), dtype=np.float32) for k, v in zip(keys, vecs)
}
norms = [float(np.linalg.norm(v)) for v in emb.values()]
print(f"{len(emb)} embeddings, norm min {min(norms):.3f} max {max(norms):.3f}")
if min(norms) == 0.0:
    raise SystemExit("still zero vectors --- the embedding call itself is at fault")

results = json.loads(RESULTS.read_text())
for row in results:
    model, steps = row["model"], row["steps"]
    ref = GRID[model][-1]
    row["img_cos"] = float(
        np.mean(
            [
                cos(emb[(model, steps, pi)], emb[(model, ref, pi)])
                for pi in range(N_PROMPTS)
            ]
        )
    )
RESULTS.write_text(json.dumps(results, indent=2))

for row in results:
    print(
        f"{row['model']:12} steps={row['steps']:3} img_cos {row['img_cos']:.4f} "
        f"caption_cos {row['caption_cos']:.4f} pixel_mae {row['pixel_mae']:6.2f} "
        f"{row['secs_per_image']:7.1f} s/img"
    )
