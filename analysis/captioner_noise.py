#!/usr/bin/env python
"""How much of a loop step is the captioner's own sampling? (TASK-89/TASK-92)

The drift/noise decomposition in TASK-89 attributes all spread among a
caption's successors to the text-to-image seed, which is only true if the
captioner is deterministic. Three of the five panel captioners ship
`do_sample=True` in their generation config (Qwen25VL, Qwen3VL, JoyCaption),
so they are not. This script measures what that costs and whether forcing
greedy decoding degrades the captions.

Method: hold the IMAGE fixed and vary only the captioner's own randomness.
For each captioner and each source image, caption `N_SAMPLES` times under the
shipped config and twice under forced greedy, then embed with Qwen3Embed.
Spread about the centroid is the captioner's contribution to a step, in the
same cosine-distance units as the trajectory step size.

Also emits the length-stratified caption set that `step_decomposition.py`
uses, taken from the greedy captions so it is reproducible.

    _build/dev/snex/projects/Elixir.PanicTda.Models.PythonInterpreter/venv/bin/python analysis/captioner_noise.py

Results -> analysis/captioner_noise.json, caption set -> analysis/caption_set.json
"""

import base64
import json
import pathlib
import sqlite3
import sys

import numpy as np

sys.path.insert(0, "/home/ben/projects/panic_tda/priv/python")
import panic_models as pm  # noqa: E402

HERE = pathlib.Path(__file__).parent
DB = pathlib.Path("/home/ben/projects/panic_tda/priv/panic_tda_dev.db")
OUT = HERE / "captioner_noise.json"
CAPTION_SET = HERE / "caption_set.json"
CAPTIONERS = ["Moondream3", "Qwen25VL", "Qwen3VL", "Gemma4", "JoyCaption"]
N_SAMPLES = 8  # resamples per (captioner, image) under the shipped config
N_IMAGES = 4  # source images, one per prompt
K_CAPTIONS = 8  # size of the length-stratified caption set


def source_images() -> tuple[list[str], list[str]]:
    """One mid-trajectory image per prompt from the caption pilot (Flux2Klein)."""
    con = sqlite3.connect(DB)
    rows = con.execute(
        """
        select r.initial_prompt, i.output_image
        from invocations i
        join runs r on r.id = i.run_id
        join experiments e on e.id = r.experiment_id
        where e.id like '01a060b4%' and i.type = 'image'
          and i.sequence_number = 20 and i.output_image is not null
        group by r.initial_prompt
        order by r.initial_prompt
        """
    ).fetchall()
    con.close()
    if len(rows) < N_IMAGES:
        raise SystemExit(f"need {N_IMAGES} pilot images, found {len(rows)}")
    step = len(rows) // N_IMAGES
    picked = [rows[i * step] for i in range(N_IMAGES)]
    return (
        [p for p, _ in picked],
        [base64.b64encode(b).decode("ascii") for _, b in picked],
    )


def repeat_5gram_frac(text: str) -> float:
    """Share of 5-grams that are repeats --- greedy decoding's failure mode."""
    w = text.split()
    if len(w) < 6:
        return 0.0
    grams = [tuple(w[i : i + 5]) for i in range(len(w) - 4)]
    return 1.0 - len(set(grams)) / len(grams)


def caption_stats(texts: list[str]) -> dict:
    return {
        "distinct": len(set(texts)),
        "words_median": float(np.median([len(t.split()) for t in texts])),
        "words_min_max": [
            min(len(t.split()) for t in texts),
            max(len(t.split()) for t in texts),
        ],
        "repeat_5gram_frac_max": max(repeat_5gram_frac(t) for t in texts),
        "ends_terminal_frac": float(
            np.mean([t.rstrip().endswith((".", "!", "?", '"')) for t in texts])
        ),
    }


def spread(vecs: np.ndarray) -> dict:
    """Spread about the centroid, in cosine-distance units on the unit sphere.

    Vectors are unit-normalised, so ||a - b||^2 = 2(1 - cos). The mean squared
    displacement about the centroid divided by two is therefore the variance
    term of the exact decomposition used in step_decomposition.py, in the same
    units as a mean cosine distance. Unbiased: the centroid is estimated from
    the same sample, so divide by n - 1.
    """
    n = len(vecs)
    centroid = vecs.mean(axis=0)
    ss = float(((vecs - centroid) ** 2).sum())
    return {
        "n": n,
        "noise_cosdist": ss / (2 * (n - 1)) if n > 1 else 0.0,
        "mean_pairwise_cosdist": float(
            np.mean(
                [
                    1 - float(vecs[i] @ vecs[j])
                    for i in range(n)
                    for j in range(i + 1, n)
                ]
            )
        )
        if n > 1
        else 0.0,
    }


def main() -> None:
    prompts, images = source_images()
    print(f"{len(images)} source images from prompts: {prompts}", flush=True)

    pm.setup()
    texts: dict[tuple[str, str, int], list[str]] = {}
    for model in CAPTIONERS:
        pm.load_model(model)
        pm.swap_to_gpu(model)
        for arm, greedy, reps in (("sampled", False, N_SAMPLES), ("greedy", True, 2)):
            pm.set_i2t_greedy(greedy)
            for ii, b64 in enumerate(images):
                texts[(model, arm, ii)] = pm.invoke_i2t_batch(model, [b64] * reps)
        pm.set_i2t_greedy(False)
        pm.unload_model(model)
        s = caption_stats(
            [t for ii in range(len(images)) for t in texts[(model, "sampled", ii)]]
        )
        g = caption_stats(
            [t for ii in range(len(images)) for t in texts[(model, "greedy", ii)]]
        )
        print(
            f"{model:12s} sampled: {s['distinct']}/{N_SAMPLES * len(images)} distinct, "
            f"median {s['words_median']:.0f}w | greedy: {g['distinct']}/{2 * len(images)} distinct, "
            f"median {g['words_median']:.0f}w, repeat5 {g['repeat_5gram_frac_max']:.3f}",
            flush=True,
        )

    keys = sorted(texts)
    flat = [(k, t) for k in keys for t in texts[k]]
    pm.load_model("Qwen3Embed")
    pm.swap_to_gpu("Qwen3Embed")
    raw = pm.embed_text("Qwen3Embed", [t for _, t in flat])
    pm.unload_model("Qwen3Embed")
    vecs: dict[tuple, list[np.ndarray]] = {}
    for (k, _), b in zip(flat, raw):
        vecs.setdefault(k, []).append(
            np.frombuffer(base64.b64decode(b), dtype=np.float32)
        )

    results = {}
    for model in CAPTIONERS:
        per_image = [
            spread(np.stack(vecs[(model, "sampled", ii)])) for ii in range(len(images))
        ]
        results[model] = {
            "sampled": {
                "per_image_noise_cosdist": [p["noise_cosdist"] for p in per_image],
                "mean_noise_cosdist": float(
                    np.mean([p["noise_cosdist"] for p in per_image])
                ),
                "mean_pairwise_cosdist": float(
                    np.mean([p["mean_pairwise_cosdist"] for p in per_image])
                ),
                **caption_stats(
                    [
                        t
                        for ii in range(len(images))
                        for t in texts[(model, "sampled", ii)]
                    ]
                ),
            },
            "greedy": {
                "deterministic": all(
                    len(set(texts[(model, "greedy", ii)])) == 1
                    for ii in range(len(images))
                ),
                **caption_stats(
                    [
                        t
                        for ii in range(len(images))
                        for t in texts[(model, "greedy", ii)]
                    ]
                ),
            },
            "greedy_vs_sampled_cosdist": float(
                np.mean(
                    [
                        1
                        - float(
                            vecs[(model, "greedy", ii)][0]
                            @ np.stack(vecs[(model, "sampled", ii)]).mean(axis=0)
                            / np.linalg.norm(
                                np.stack(vecs[(model, "sampled", ii)]).mean(axis=0)
                            )
                        )
                        for ii in range(len(images))
                    ]
                )
            ),
        }

    # the caption set: greedy captions, ranked by length, evenly spaced
    pool = sorted(
        (
            {
                "captioner": model,
                "source_prompt": prompts[ii],
                "source_image_index": ii,
                "words": len(texts[(model, "greedy", ii)][0].split()),
                "text": texts[(model, "greedy", ii)][0],
            }
            for model in CAPTIONERS
            for ii in range(len(images))
        ),
        key=lambda d: d["words"],
    )
    picked = [
        pool[round(i * (len(pool) - 1) / (K_CAPTIONS - 1))] for i in range(K_CAPTIONS)
    ]
    CAPTION_SET.write_text(json.dumps(picked, indent=2) + "\n")
    print(
        f"\ncaption set: {[d['words'] for d in picked]} words, "
        f"captioners {[d['captioner'] for d in picked]}",
        flush=True,
    )

    OUT.write_text(
        json.dumps(
            {
                "n_samples": N_SAMPLES,
                "source_prompts": prompts,
                "captioners": results,
                "all_captions": {"|".join(map(str, k)): v for k, v in texts.items()},
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {OUT} and {CAPTION_SET}", flush=True)
    print("\ncaptioner sampling noise, cosine distance about the centroid")
    print(
        f"{'captioner':12s} {'noise':>8s} {'pairwise':>9s}  greedy-det  greedy-repeat5"
    )
    for model in CAPTIONERS:
        r = results[model]
        print(
            f"{model:12s} {r['sampled']['mean_noise_cosdist']:8.4f} "
            f"{r['sampled']['mean_pairwise_cosdist']:9.4f}  "
            f"{str(r['greedy']['deterministic']):10s}  "
            f"{r['greedy']['repeat_5gram_frac_max']:.3f}"
        )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
