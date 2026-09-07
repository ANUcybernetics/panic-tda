#!/usr/bin/env python
"""Does a candidate captioner stay under the 512-token encoder ceiling? (TASK-101)

Qwen3VL left the panel because its captions of early-step images ran past the
512 tokens the text-to-image encoders read (TASK-90 smoke: 14 of 80 under
SD35Medium's T5 tokenizer, max 744). This is the measurement that decides
whether a replacement clears the ceiling, run through the production invoke
path so the batch cap, greedy decoding and prompt are the panel's own.

Images: the first three image states (sequence 0, 2, 4) of the twenty panel
prompts from each of the four generators, drawn from the 50-step balanced
panel (an older captioner lineup, but these are captioner inputs, and early
trajectory images are the visually busiest ones the loop produces). Each
(generator, step) gives two batches of 40 composed like a panel cell, 20
prompts x 2, since greedy captions change with batch composition.

Token counts are reported two ways. Raw: the caption alone under each
generator's tokenizer, which is how the TASK-90 smoke counted. Effective: the
caption wrapped the way each pipeline wraps it before truncating at 512 ---
SD35Medium tokenises the bare text with T5, Flux2Klein and ZImageTurbo apply a
Qwen3 chat template, Flux2Dev prepends its Mistral3 system message --- so the
effective count is what the generator actually cuts at. Pass: p99 under 512
and max recorded; any max over 512 fails.

Also checks the TASK-87 screens on the way through: bf16 with no quantised
modules, the rendered prompt (no system prompt or template beyond the model's
own chat format), greedy decoding deterministic on a repeated batch, and
natural termination below the 1024-token generation ceiling.

    _build/dev/snex/projects/Elixir.PanicTda.Models.PythonInterpreter/venv/bin/python \
        analysis/captioner_ceiling_screen.py Qwen3VL2B Qwen3VL4B Qwen3VL

Results -> analysis/captioner_ceiling_screen.json (merged per model, so
candidates can be run one at a time). Qwen3VL, the 8B that failed the smoke,
runs on the same images as the reference that calibrates the image set.
"""

import base64
import json
import pathlib
import random
import re
import sqlite3
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/ben/projects/panic_tda/priv/python")
import panic_models as pm  # noqa: E402

HERE = pathlib.Path(__file__).parent
DB = pathlib.Path("/home/ben/projects/panic_tda/priv/panic_tda_dev.db")
OUT = HERE / "captioner_ceiling_screen.json"
HUB = pathlib.Path("/data/huggingface/hub")
SOURCE_EXPERIMENT = "019f3645"
GENERATORS = ["SD35Medium", "ZImageTurbo", "Flux2Klein", "Flux2Dev"]
IMAGE_STEPS = [0, 2, 4]
BATCH = pm._I2T_MAX_BATCH  # 40: the panel's cell shape, 20 prompts x 2 runs
BATCHES_PER_CELL = 2
CEILING = 512
GEN_CEILING = pm._I2T_MAX_NEW_TOKENS_DEFAULT
ENDS = re.compile(r'[.!?"”)]$')

# The candidates live here, not in panic_models.py: neither passed, and the
# registry carries the panel only. They load the way a panel captioner would
# and caption through the Qwen-VL invoke path, in bf16 with no quantisation
# (9 GB and 4 GB), so unlike the 4-bit 8B they need no device map.
CANDIDATES = {
    "Qwen3VL4B": (
        "Qwen/Qwen3-VL-4B-Instruct",
        "ebb281ec70b05090aa6165b016eac8ec08e71b17",  # HEAD on 2026-09-07
    ),
    "Qwen3VL2B": (
        "Qwen/Qwen3-VL-2B-Instruct",
        "89644892e4d85e24eaac8bacfd4f463576704203",  # HEAD on 2026-09-07
    ),
}


def register_candidates() -> None:
    import functools

    import transformers

    def load(name: str) -> None:
        repo = CANDIDATES[name][0]
        model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
            repo,
            revision=pm._rev(repo),
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        processor = transformers.AutoProcessor.from_pretrained(
            repo, revision=pm._rev(repo)
        )
        pm._models[name] = {"processor": processor, "model": model.to("cuda").eval()}

    for name, (repo, rev) in CANDIDATES.items():
        pm._REVISIONS[repo] = rev
        pm._QWEN_VL_CONFIGS[name] = {
            "repo": repo,
            "cls": "Qwen3VLForConditionalGeneration",
            "quantize": False,
        }
        pm._I2T_LOADERS[name] = functools.partial(load, name)
        pm._I2T_STRATEGIES[name] = pm._invoke_qwen_vl
        pm._I2T_BATCH_STRATEGIES[name] = pm._invoke_qwen_vl_batch


def snapshot(repo: str) -> pathlib.Path:
    return HUB / f"models--{repo.replace('/', '--')}" / "snapshots" / pm._rev(repo)


def load_encoder_tokenizers() -> dict[str, dict]:
    """Each generator's tokenizer and the wrapper its pipeline applies.

    Loaded from the cached snapshot directories of the pinned revisions; hub
    resolution is not needed and has failed offline before.
    """
    from transformers import AutoProcessor, AutoTokenizer

    t5 = AutoTokenizer.from_pretrained(
        snapshot("stabilityai/stable-diffusion-3.5-medium") / "tokenizer_3"
    )
    klein = AutoTokenizer.from_pretrained(
        snapshot("black-forest-labs/FLUX.2-klein-9B") / "tokenizer"
    )
    zimage = AutoTokenizer.from_pretrained(
        snapshot("Tongyi-MAI/Z-Image-Turbo") / "tokenizer"
    )
    dev = AutoProcessor.from_pretrained(
        snapshot("black-forest-labs/FLUX.2-dev") / "tokenizer"
    )
    from diffusers.pipelines.flux2.system_messages import SYSTEM_MESSAGE

    def raw(tok):
        return lambda c: len(tok(c, padding=False, truncation=False).input_ids)

    def qwen_chat(tok, thinking: bool):
        # Flux2KleinPipeline._get_qwen3_prompt_embeds (enable_thinking=False)
        # and ZImagePipeline.encode_prompt (enable_thinking=True)
        def count(c):
            text = tok.apply_chat_template(
                [{"role": "user", "content": c}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=thinking,
            )
            return len(tok(text, padding=False, truncation=False).input_ids)

        return count

    def mistral_chat(proc):
        # Flux2Pipeline._get_mistral_3_small_prompt_embeds via format_input
        def count(c):
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_MESSAGE}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": c.replace("[IMG]", "")}],
                },
            ]
            ids = proc.apply_chat_template(
                [messages],
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=False,
                truncation=False,
            )["input_ids"]
            return int(ids.shape[1])

        return count

    return {
        "SD35Medium": {
            "tokenizer": "T5 (tokenizer_3)",
            "raw": raw(t5),
            "effective": raw(t5),
        },
        "Flux2Klein": {
            "tokenizer": "Qwen2TokenizerFast (Qwen3 chat template)",
            "raw": raw(klein),
            "effective": qwen_chat(klein, thinking=False),
        },
        "ZImageTurbo": {
            "tokenizer": "Qwen tokenizer (Qwen3 chat template, thinking)",
            "raw": raw(zimage),
            "effective": qwen_chat(zimage, thinking=True),
        },
        "Flux2Dev": {
            "tokenizer": "PixtralProcessor (Mistral3 template + system message)",
            "raw": raw(dev.tokenizer),
            "effective": mistral_chat(dev),
        },
    }


def source_batches() -> list[dict]:
    """Batches of 40 early-step images, one cell-shaped batch at a time."""
    con = sqlite3.connect(DB)
    rng = random.Random(101)
    batches = []
    for gen in GENERATORS:
        for step in IMAGE_STEPS:
            rows = con.execute(
                """
                select r.initial_prompt, r.network, r.id, i.output_image
                from invocations i
                join runs r on r.id = i.run_id
                where r.experiment_id like ?
                  and json_extract(r.network, '$[0]') = ?
                  and i.type = 'image' and i.sequence_number = ?
                  and i.output_image is not null
                order by r.initial_prompt, r.network, r.id
                """,
                (f"{SOURCE_EXPERIMENT}%", gen, step),
            ).fetchall()
            by_prompt: dict[str, list] = {}
            for prompt, network, run_id, blob in rows:
                by_prompt.setdefault(prompt, []).append((network, run_id, blob))
            prompts = sorted(by_prompt)
            assert len(prompts) == 20, (gen, step, len(prompts))
            picks = {p: rng.sample(by_prompt[p], 2 * BATCHES_PER_CELL) for p in prompts}
            for b in range(BATCHES_PER_CELL):
                items = []
                for p in prompts:
                    for network, run_id, blob in picks[p][2 * b : 2 * b + 2]:
                        items.append(
                            {
                                "generator": gen,
                                "step": step,
                                "prompt": p,
                                "source_network": json.loads(network),
                                "run_id": run_id,
                                "b64": base64.b64encode(blob).decode("ascii"),
                            }
                        )
                assert len(items) == BATCH
                batches.append({"generator": gen, "step": step, "items": items})
    return batches


def rendered_prompt(name: str) -> str:
    """What the captioner is actually asked, template included."""
    proc = pm._models[name]["processor"]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "<image>"},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    return proc.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def model_facts(name: str) -> dict:
    model = pm._models[name]["model"]
    dtypes = sorted({str(p.dtype) for p in model.parameters()})
    quantised = sorted(
        {
            type(m).__name__
            for m in model.modules()
            if "bnb" in type(m).__module__ or "Linear4bit" in type(m).__name__
        }
    )
    cfg = pm._QWEN_VL_CONFIGS.get(name, {})
    repo = cfg.get("repo")
    return {
        "repo": repo,
        "revision": pm._rev(repo) if repo else None,
        "parameters_bn": round(sum(p.numel() for p in model.parameters()) / 1e9, 2),
        "dtypes": dtypes,
        "quantised_modules": quantised,
        "offload_only": name in pm._models_offload_only,
        "resident_gb_after_load": round(torch.cuda.memory_allocated() / 2**30, 1),
        "rendered_prompt": rendered_prompt(name),
        "generation_config": {
            k: v
            for k, v in model.generation_config.to_dict().items()
            if k in ("do_sample", "temperature", "top_p", "top_k", "repetition_penalty")
        },
        "forced_greedy": pm._I2T_FORCE_GREEDY,
    }


def percentile(xs: list[int], q: float) -> int:
    return int(np.percentile(xs, q, method="higher"))


def summarise(counts: list[int]) -> dict:
    return {
        "median": int(np.median(counts)),
        "p90": percentile(counts, 90),
        "p99": percentile(counts, 99),
        "max": max(counts),
        "over_512": sum(c > CEILING for c in counts),
        "n": len(counts),
    }


def screen(name: str, batches: list[dict], encoders: dict) -> dict:
    pm.unload_all_models()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    pm.load_model(name)
    load_s = time.time() - t0
    facts = model_facts(name)
    facts["load_s"] = round(load_s, 1)
    print(f"[{name}] loaded in {load_s:.0f}s: {facts}", flush=True)
    own_tok = pm._models[name]["processor"].tokenizer

    records = []
    secs = 0.0
    for bi, batch in enumerate(batches):
        b64s = [it["b64"] for it in batch["items"]]
        t0 = time.time()
        caps = pm.invoke_i2t_batch(name, b64s)
        dt = time.time() - t0
        secs += dt
        for it, cap in zip(batch["items"], caps):
            rec = {k: v for k, v in it.items() if k != "b64"}
            rec["caption"] = cap
            rec["words"] = len(cap.split())
            rec["own_tokens"] = len(own_tok(cap, add_special_tokens=False).input_ids)
            rec["terminal_punctuation"] = bool(ENDS.search(cap.strip()))
            # the small Qwen3-VLs answer with bulleted, bold-headed markdown
            rec["markdown"] = "**" in cap or "\n- " in cap
            rec["raw"] = {g: enc["raw"](cap) for g, enc in encoders.items()}
            rec["effective"] = {g: enc["effective"](cap) for g, enc in encoders.items()}
            records.append(rec)
        words = [len(c.split()) for c in caps]
        t5 = [encoders["SD35Medium"]["raw"](c) for c in caps]
        print(
            f"[{name}] batch {bi + 1}/{len(batches)} {batch['generator']} step {batch['step']}: "
            f"{dt / len(caps):.2f} s/caption, words median {int(np.median(words))} "
            f"max {max(words)}, T5 max {max(t5)}, over 512: {sum(t > CEILING for t in t5)}",
            flush=True,
        )
    peak_gb = torch.cuda.max_memory_allocated() / 2**30

    # greedy determinism: the first batch again, must reproduce exactly
    again = pm.invoke_i2t_batch(name, [it["b64"] for it in batches[0]["items"]])
    first = [r["caption"] for r in records[:BATCH]]
    identical = sum(a == b for a, b in zip(first, again))

    by_gen = {}
    for g in GENERATORS:
        recs = [r for r in records if r["generator"] == g]
        by_gen[g] = {
            "tokenizer": encoders[g]["tokenizer"],
            "words": summarise([r["words"] for r in recs]),
            "raw": summarise([r["raw"][g] for r in recs]),
            "effective": summarise([r["effective"][g] for r in recs]),
            "template_overhead": recs[0]["effective"][g] - recs[0]["raw"][g],
        }
    # every caption under every encoder, i.e. what the whole panel would read
    all_raw = {g: summarise([r["raw"][g] for r in records]) for g in GENERATORS}
    all_eff = {g: summarise([r["effective"][g] for r in records]) for g in GENERATORS}
    longest = sorted(records, key=lambda r: -r["effective"]["SD35Medium"])[:3]

    passes = all(s["p99"] < CEILING and s["max"] <= CEILING for s in all_eff.values())
    return {
        "model": facts,
        "images": {
            "source_experiment": SOURCE_EXPERIMENT,
            "generators": GENERATORS,
            "image_steps": IMAGE_STEPS,
            "batch": BATCH,
            "n": len(records),
        },
        "secs_per_caption": round(secs / len(records), 2),
        "peak_vram_gb": round(peak_gb, 1),
        "greedy_identical_on_repeat": f"{identical}/{BATCH}",
        "hit_generation_ceiling": sum(r["own_tokens"] >= GEN_CEILING for r in records),
        "pct_no_terminal_punctuation": round(
            100 * sum(not r["terminal_punctuation"] for r in records) / len(records), 1
        ),
        "words": summarise([r["words"] for r in records]),
        "all_captions_raw": all_raw,
        "all_captions_effective": all_eff,
        "by_generator": by_gen,
        "passes_512_effective": passes,
        "longest": [
            {k: v for k, v in r.items() if k not in ("run_id", "source_network")}
            for r in longest
        ],
        "records": [{k: v for k, v in r.items() if k != "caption"} for r in records],
    }


if __name__ == "__main__":
    names = sys.argv[1:] or ["Qwen3VL2B", "Qwen3VL4B", "Qwen3VL"]
    pm.setup()
    register_candidates()
    encoders = load_encoder_tokenizers()
    batches = source_batches()
    print(f"{len(batches)} batches of {BATCH} images", flush=True)
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name in names:
        results[name] = screen(name, batches, encoders)
        OUT.write_text(json.dumps(results, indent=2))
        r = results[name]
        print(
            f"\n[{name}] {r['secs_per_caption']} s/caption, peak {r['peak_vram_gb']} GB, "
            f"greedy {r['greedy_identical_on_repeat']}, ceiling hits {r['hit_generation_ceiling']}, "
            f"words {r['words']}\n  effective: "
            + json.dumps(r["all_captions_effective"])
            + "\n  raw: "
            + json.dumps(r["all_captions_raw"])
            + f"\n  PASS={r['passes_512_effective']}\n",
            flush=True,
        )
    pm.unload_all_models()
