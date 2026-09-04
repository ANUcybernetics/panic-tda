#!/usr/bin/env python
"""What is one loop step made of? Drift, generator noise, and the ruler (TASK-89).

The loop settles into a stationary regime with a persistent step-to-step
distance of about 0.012-0.016 (analysis/pilot_vs_panel.py,
analysis/long_horizon_baseline.py). This measures what that step is made of,
per text-to-image model.

Method. Take a fixed, length-stratified set of captions (analysis/caption_set.json,
written by captioner_noise.py). For each caption c and each text-to-image model,
generate N images at N recorded seeds, then caption each image with the SAME
captioner that produced c, forced to greedy decoding. Every difference among the
N successors is then one draw of generator noise and nothing else.

Embeddings are unit-normalised, so ||a - b||^2 = 2(1 - cos) and the mean step
decomposes exactly:

    mean_i ||x_i - c||^2  =  ||xbar - c||^2  +  mean_i ||x_i - xbar||^2

Halving gives, in cosine-distance units:  step = drift + noise. That is the
conditional mean and conditional variance of one step (Callaham et al. 2021),
with no continuum correction, since the loop is a discrete-time chain.

WHAT NOT TO COMPARE. Consecutive trajectory states differ by ONE draw of
generator noise; two resamples of one caption differ by TWO, so their distance
is about double with no dynamics at all. `step` here is the one-draw quantity
and is what the trajectory step size should be compared against.

Also reports the ruler: the same distances against Qwen3Embed's resolution for
these captions (successors of unrelated source captions), and the Bland-Altman
minimal detectable change, MDC = sqrt(2) * 1.96 * SEM, above which a drift is
called real. Metrics are computed both at the pipeline's 256 dimensions and at
Qwen3Embed's native 2560, so the truncation is a measured choice.

Resumable: images are written to analysis/step_decomposition/ and skipped if
present, and captions are checkpointed.

    _build/dev/snex/projects/Elixir.PanicTda.Models.PythonInterpreter/venv/bin/python analysis/step_decomposition.py

Results -> analysis/step_decomposition.json
"""

import base64
import io
import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/ben/projects/panic_tda/priv/python")
import panic_models as pm  # noqa: E402

HERE = pathlib.Path(__file__).parent
IMGDIR = HERE / "step_decomposition"
OUT = HERE / "step_decomposition.json"
CAPTIONS_CKPT = IMGDIR / "captions.json"
T2I_MODELS = ["Flux2Klein", "ZImageTurbo", "SD35Medium", "GLMImage", "Flux2Dev"]
N_SEEDS = 16
SEED_BASE = 890_000


def seeds_for(k: int) -> list[int]:
    """Recorded seeds for caption k --- deterministic so the run is reproducible."""
    return [SEED_BASE + 1000 * k + r for r in range(N_SEEDS)]


def img_path(model: str, k: int, seed: int) -> pathlib.Path:
    return IMGDIR / model / f"c{k:02d}_s{seed}.webp"


def token_check(model: str, captions: list[str]) -> dict:
    """How close do these captions run to the model's conditioning ceiling?

    GLMImage is a different question from the other four: its caption goes to a
    vision-language encoder with no ceiling, and its ByT5 tokenizer sees only the
    quoted substrings GLM renders as text in the image. Measuring the whole
    caption through that tokenizer would report a truncation that never happens.
    """
    pipe = pm._models[model]
    if model == "GLMImage":
        frags = [f for c in captions for f in pipe.get_glyph_texts(c)[0]]
        n = [
            len(pipe.tokenizer(f, padding=False, truncation=False).input_ids)
            for f in frags
        ]
        return {
            "branch": "glyph (ByT5); the caption itself is encoded uncapped",
            "captions_with_glyph_text": sum(
                bool(pipe.get_glyph_texts(c)[0]) for c in captions
            ),
            "glyph_tokens_min_max": [min(n), max(n)] if n else None,
            "ceiling": pm._T2I_INVOKE_CONFIGS[model]["max_sequence_length"],
        }
    tok = getattr(pipe, "tokenizer_3", None) or getattr(pipe, "tokenizer", None)
    if tok is None:
        return {"tokenizer": None}
    n = [len(tok(c, padding=False, truncation=False).input_ids) for c in captions]
    return {
        "branch": "caption",
        "tokenizer": type(tok).__name__,
        "tokens_min_max": [min(n), max(n)],
        "over_512": sum(t > 512 for t in n),
    }


def generate(captions: list[str]) -> dict:
    """Phase 1: N seeded images per (model, caption). Skips what already exists."""
    meta = {}
    for model in T2I_MODELS:
        todo = [
            (k, s)
            for k in range(len(captions))
            for s in seeds_for(k)
            if not img_path(model, k, s).exists()
        ]
        if not todo:
            print(f"{model}: all {len(captions) * N_SEEDS} images present", flush=True)
            continue
        (IMGDIR / model).mkdir(parents=True, exist_ok=True)
        pm.load_model(model)
        pm.swap_to_gpu(model)
        meta[model] = token_check(model, captions)
        print(f"{model}: {len(todo)} images to generate, {meta[model]}", flush=True)
        batch = pm._T2I_MAX_BATCH[model]
        t0 = time.time()
        for i in range(0, len(todo), batch):
            chunk = todo[i : i + batch]
            imgs = pm._t2i_generate_seeded(
                model, [captions[k] for k, _ in chunk], [s for _, s in chunk]
            )
            for (k, s), img in zip(chunk, imgs):
                img.save(img_path(model, k, s), format="WEBP", lossless=True)
            done = i + len(chunk)
            el = time.time() - t0
            print(
                f"  {model} {done}/{len(todo)}  {el / done:.1f}s/img  "
                f"eta {(len(todo) - done) * el / done / 60:.0f}m",
                flush=True,
            )
        pm.unload_model(model)
    return meta


def caption_all(entries: list[dict]) -> dict[str, str]:
    """Phase 2: caption every image with the captioner that wrote its source caption."""
    done: dict[str, str] = (
        json.loads(CAPTIONS_CKPT.read_text()) if CAPTIONS_CKPT.exists() else {}
    )
    by_captioner: dict[str, list[tuple[str, pathlib.Path]]] = {}
    for k, e in enumerate(entries):
        for model in T2I_MODELS:
            for s in seeds_for(k):
                key = f"{model}|{k}|{s}"
                if key not in done:
                    by_captioner.setdefault(e["captioner"], []).append(
                        (key, img_path(model, k, s))
                    )
    for captioner, items in by_captioner.items():
        pm.load_model(captioner)
        pm.swap_to_gpu(captioner)
        pm.set_i2t_greedy(True)
        t0 = time.time()
        for i in range(0, len(items), pm._I2T_MAX_BATCH):
            chunk = items[i : i + pm._I2T_MAX_BATCH]
            b64 = [base64.b64encode(p.read_bytes()).decode("ascii") for _, p in chunk]
            for (key, _), text in zip(chunk, pm.invoke_i2t_batch(captioner, b64)):
                done[key] = text
            CAPTIONS_CKPT.write_text(json.dumps(done))
            n = i + len(chunk)
            el = time.time() - t0
            print(
                f"  {captioner} {n}/{len(items)}  {el / n:.1f}s/caption",
                flush=True,
            )
        pm.set_i2t_greedy(False)
        pm.unload_model(captioner)
    return done


def embed(texts: list[str]) -> np.ndarray:
    """Native-dimension unit-normalised embeddings (no 256-d truncation)."""
    pm.load_model("Qwen3Embed")
    pm.swap_to_gpu("Qwen3Embed")
    out = []
    for i in range(0, len(texts), 64):
        out.append(
            pm._models["Qwen3Embed"].encode(
                texts[i : i + 64], convert_to_numpy=True, normalize_embeddings=True
            )
        )
        print(f"  embedded {min(i + 64, len(texts))}/{len(texts)}", flush=True)
    pm.unload_model("Qwen3Embed")
    return np.concatenate(out).astype(np.float32)


def truncate(v: np.ndarray, dim: int) -> np.ndarray:
    """Matryoshka truncation, matching panic_models._encode_embedding."""
    w = v[..., :dim]
    return w / np.linalg.norm(w, axis=-1, keepdims=True)


def decompose(c: np.ndarray, xs: np.ndarray) -> dict:
    """Exact split of the mean step into drift and noise, in cosine-distance units."""
    n = len(xs)
    xbar = xs.mean(axis=0)
    step = float(((xs - c) ** 2).sum(axis=1).mean()) / 2
    drift = float(((xbar - c) ** 2).sum()) / 2
    noise_b = float(((xs - xbar) ** 2).sum()) / (2 * n)
    noise_u = noise_b * n / (n - 1)
    sem = float(np.sqrt(2 * noise_u))  # SEM in Euclidean units on the sphere
    return {
        "step_cosdist": step,
        "drift_cosdist": drift,
        "noise_cosdist_biased": noise_b,
        "noise_cosdist": noise_u,
        "noise_share": noise_b / step if step > 0 else None,
        "drift_cosdist_bias_corrected": step - noise_u,
        "mdc_euclidean": float(np.sqrt(2) * 1.96 * sem),
        "drift_euclidean": float(np.linalg.norm(xbar - c)),
        "drift_exceeds_mdc": bool(
            float(np.linalg.norm(xbar - c)) > np.sqrt(2) * 1.96 * sem
        ),
    }


def main() -> None:
    entries = json.loads((HERE / "caption_set.json").read_text())
    captions = [e["text"] for e in entries]
    IMGDIR.mkdir(exist_ok=True)
    print(
        f"{len(captions)} captions ({[e['words'] for e in entries]} words), "
        f"{N_SEEDS} seeds, {len(T2I_MODELS)} models "
        f"= {len(captions) * N_SEEDS * len(T2I_MODELS)} images",
        flush=True,
    )
    pm.setup()
    token_meta = generate(captions)
    texts = caption_all(entries)

    keys = [
        f"{m}|{k}|{s}"
        for m in T2I_MODELS
        for k in range(len(captions))
        for s in seeds_for(k)
    ]
    vecs = embed(captions + [texts[key] for key in keys])
    src_full, succ_full = vecs[: len(captions)], vecs[len(captions) :]
    succ_full = succ_full.reshape(len(T2I_MODELS), len(captions), N_SEEDS, -1)

    results = {"dims": {}}
    for dim in (256, src_full.shape[1]):
        src = truncate(src_full, dim)
        succ = truncate(succ_full, dim)
        per_model = {}
        for mi, model in enumerate(T2I_MODELS):
            rows = [decompose(src[k], succ[mi, k]) for k in range(len(captions))]
            per_model[model] = {
                "per_caption": rows,
                **{
                    f"mean_{f}": float(np.mean([r[f] for r in rows]))
                    for f in (
                        "step_cosdist",
                        "drift_cosdist",
                        "noise_cosdist",
                        "noise_share",
                        "drift_cosdist_bias_corrected",
                    )
                },
                "captions_drift_exceeds_mdc": int(
                    sum(r["drift_exceeds_mdc"] for r in rows)
                ),
            }
        # the ruler: successors of DIFFERENT source captions, same model
        unrelated = []
        for mi in range(len(T2I_MODELS)):
            for a in range(len(captions)):
                for b in range(a + 1, len(captions)):
                    unrelated.append(1 - float(succ[mi, a, 0] @ succ[mi, b, 0]))
        results["dims"][str(dim)] = {
            "models": per_model,
            "ruler": {
                "unrelated_caption_cosdist_mean": float(np.mean(unrelated)),
                "unrelated_caption_cosdist_p10_p90": [
                    float(np.percentile(unrelated, 10)),
                    float(np.percentile(unrelated, 90)),
                ],
                "seed_resample_cosdist_mean": float(
                    np.mean(
                        [per_model[m]["mean_noise_cosdist"] * 2 for m in T2I_MODELS]
                    )
                ),
            },
        }

    # truncation check (AC#6): do 256 and native dimensions rank distances alike?
    from scipy.stats import spearmanr

    flat = np.concatenate([src_full, succ_full.reshape(-1, src_full.shape[1])])
    idx = np.random.default_rng(0).choice(
        len(flat), size=min(400, len(flat)), replace=False
    )
    a, b = truncate(flat[idx], 256), flat[idx]
    da = 1 - a @ a.T
    db = 1 - b @ b.T
    iu = np.triu_indices(len(idx), 1)
    results["truncation"] = {
        "n_vectors": int(len(idx)),
        "native_dim": int(src_full.shape[1]),
        "spearman_pairwise_distance": float(spearmanr(da[iu], db[iu]).statistic),
        "pearson_pairwise_distance": float(np.corrcoef(da[iu], db[iu])[0, 1]),
    }
    results["config"] = {
        "n_seeds": N_SEEDS,
        "seeds": {str(k): seeds_for(k) for k in range(len(captions))},
        "captions": entries,
        "t2i_token_check": token_meta,
        "t2i_invoke_configs": {m: pm._T2I_INVOKE_CONFIGS[m] for m in T2I_MODELS},
    }
    OUT.write_text(json.dumps(results, indent=2) + "\n")

    print("\nstep decomposition at 256 dimensions (cosine distance)")
    print(
        f"{'model':13s} {'step':>8s} {'drift':>8s} {'noise':>8s} {'noise%':>7s}  drift>MDC"
    )
    for model in T2I_MODELS:
        r = results["dims"]["256"]["models"][model]
        print(
            f"{model:13s} {r['mean_step_cosdist']:8.4f} {r['mean_drift_cosdist']:8.4f} "
            f"{r['mean_noise_cosdist']:8.4f} {100 * r['mean_noise_share']:6.1f}%  "
            f"{r['captions_drift_exceeds_mdc']}/{len(captions)}"
        )
    ruler = results["dims"]["256"]["ruler"]
    print(
        f"\nruler: unrelated captions {ruler['unrelated_caption_cosdist_mean']:.4f}, "
        f"seed resamples {ruler['seed_resample_cosdist_mean']:.4f}"
    )
    print(
        f"truncation 256 vs {results['truncation']['native_dim']}: "
        f"spearman {results['truncation']['spearman_pairwise_distance']:.4f}"
    )
    print(f"wrote {OUT}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
