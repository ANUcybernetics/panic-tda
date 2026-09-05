#!/usr/bin/env python
"""Does the tail of a long caption actually condition the image?

The text-to-image encoders read at most 512 tokens by default (SD35Medium
hard-caps there; Flux2's Mistral3 encoder and Z-Image have no such limit).
Before deciding whether to raise that ceiling --- which would mean replacing
SD35Medium to keep the panel uniform --- it is worth knowing whether the text
past 512 tokens changes the image at all.

Method: hold `max_sequence_length` FIXED and vary only the text. For each
caption, generate once from the full text and once from the same text cut to
its first `CUT` T5 tokens, at the same seed. Then caption both images with
Gemma3n and compare in Qwen3Embed space, which is the loop-relevant metric: if
the captioner reads the two images the same way, the discarded text was doing
nothing to the image and a higher ceiling buys nothing.

Padding must be held fixed because `max_sequence_length` is not a neutral
knob: measured 2026-09-04, varying it from 512 to 1024 with identical,
untruncated prompts still moved caption cosine to 0.896-0.988, since it
changes the shape of the conditioning tensor. Comparing 512 against 1024
therefore conflates padding with content, which the first version of this
script did.

Scale for reading the numbers: Gemma3n is deterministic, so identical images
give cosine 1.000. The 0.876 once quoted here for unrelated images came from
mean-pooled embeddings (TASK-96) and understated every distance; on the
corrected scale unrelated captions sit near 0.425. The results in
prompt_tail.json predate the fix and are on the old scale.

    _build/dev/snex/projects/Elixir.PanicTda.Models.PythonInterpreter/venv/bin/python analysis/prompt_tail.py
"""

import base64
import json
import pathlib
import sqlite3
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/ben/projects/panic_tda/priv/python")
import panic_models as pm  # noqa: E402

HERE = pathlib.Path(__file__).parent
OUT = HERE / "prompt_tail.json"
MAX_SEQ = 512  # held fixed for both arms
CUT = 256  # tokens retained in the truncated arm
MODEL = "Flux2Klein"

con = sqlite3.connect("/home/ben/projects/panic_tda/priv/panic_tda_dev.db")
captions = [
    r[0]
    for r in con.execute("""
      select i.output_text from invocations i join runs r on r.id=i.run_id
      join experiments e on e.id=r.experiment_id
      where e.i2t_max_new_tokens=1024 and i.type='text'
      order by length(i.output_text) desc limit 6""")
]

pm.setup()
from transformers import AutoTokenizer  # noqa: E402

t5 = AutoTokenizer.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium", subfolder="tokenizer_3"
)
tok_counts = [len(t5(c, padding=False, truncation=False).input_ids) for c in captions]
print(f"{len(captions)} captions, T5 tokens {tok_counts}", flush=True)
print(
    f"over the {CUT}-token cut: {sum(t > CUT for t in tok_counts)}/{len(captions)}",
    flush=True,
)

pm.load_model(MODEL)
pm.swap_to_gpu(MODEL)
cfg = dict(pm._T2I_INVOKE_CONFIGS[MODEL])
size = pm._T2I_IMAGE_SIZES[MODEL]
# the truncated arm: same text, cut to CUT tokens and decoded back
short = [
    t5.decode(
        t5(c, padding=False, truncation=True, max_length=CUT).input_ids,
        skip_special_tokens=True,
    )
    for c in captions
]
images: dict[tuple[str, int], str] = {}
for arm, texts in (("full", captions), ("cut", short)):
    for i, cap in enumerate(texts):
        gen = torch.Generator(device="cuda").manual_seed(7000 + i)
        img = pm._models[MODEL](
            prompt=cap,
            height=size,
            width=size,
            generator=gen,
            max_sequence_length=MAX_SEQ,
            **cfg,
        ).images[0]
        images[(arm, i)] = pm._encode_image_b64(img)
    print(f"generated arm={arm}", flush=True)
pm.unload_model(MODEL)

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


rows = []
for i, cap in enumerate(captions):
    a = np.asarray(
        pm._decode_image_b64(images[("full", i)]).convert("RGB"), dtype=np.float32
    )
    b = np.asarray(
        pm._decode_image_b64(images[("cut", i)]).convert("RGB"), dtype=np.float32
    )
    rows.append(
        {
            "t5_tokens": tok_counts[i],
            "tokens_dropped": max(0, tok_counts[i] - CUT),
            "caption_cos_full_vs_cut": cos(emb[("full", i)], emb[("cut", i)]),
            "pixel_mae": float(np.abs(a - b).mean()),
        }
    )
    r = rows[-1]
    print(
        f"  {r['t5_tokens']:5} tokens ({r['tokens_dropped']:4} dropped)  "
        f"caption_cos {r['caption_cos_full_vs_cut']:.4f}  pixel_mae {r['pixel_mae']:6.2f}",
        flush=True,
    )

over = [r for r in rows if r["tokens_dropped"] > 0]
summary = {
    "model": MODEL,
    "n": len(rows),
    "n_truncated": len(over),
    "mean_caption_cos_truncated": float(
        np.mean([r["caption_cos_full_vs_cut"] for r in over])
    )
    if over
    else None,
    "mean_pixel_mae_truncated": float(np.mean([r["pixel_mae"] for r in over]))
    if over
    else None,
    "rows": rows,
}
OUT.write_text(json.dumps(summary, indent=2))
print(
    f"\nfor captions that actually lost text: mean caption_cos {summary['mean_caption_cos_truncated']}, "
    f"mean pixel_mae {summary['mean_pixel_mae_truncated']}",
    flush=True,
)
print("DONE", flush=True)
