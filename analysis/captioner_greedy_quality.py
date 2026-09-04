#!/usr/bin/env python
"""Does forcing greedy decoding degrade any panel captioner? (TASK-92 AC#4)

`captioner_noise.py` established the cost of the shipped sampling configs on
four images and showed greedy is deterministic there. Four images is too thin
to clear greedy decoding for a multi-week run: its failure mode is degeneracy
--- a caption that falls into a repeated phrase, collapses to a stub, or runs
to the token cap and is cut mid-sentence --- and whether that happens depends
on the image. This widens the check to 24 images spanning every generator in
the v2 panel, three trajectory depths and two prompts each.

Method: caption each image twice under forced greedy (determinism check) and
once under the shipped config (a like-for-like reference for length and
repetition, not a noise estimate --- `captioner_noise.py` owns that). Then
compare the two arms on the three degeneracy modes:

  repetition -- share of 5-grams that are repeats, and the longest phrase that
                repeats back-to-back, which is what a decoding loop looks like
  length     -- word count, and the share of captions that collapse to a stub
  truncation -- captions that do not end on terminal punctuation, i.e. that
                probably hit the 1024-token cap mid-sentence

Greedy passes if it is deterministic, and if none of the three modes is worse
than the sampled arm by more than noise on any captioner.

    _build/dev/snex/projects/Elixir.PanicTda.Models.PythonInterpreter/venv/bin/python analysis/captioner_greedy_quality.py

Results -> analysis/captioner_greedy_quality.json
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
OUT = HERE / "captioner_greedy_quality.json"
CAPTIONERS = ["Moondream3", "Qwen25VL", "Qwen3VL", "Gemma4", "JoyCaption"]
GENERATORS = ["Flux2Klein", "ZImageTurbo", "SD35Medium", "Flux2Dev"]
DEPTHS = [8, 24, 48]  # early, mid, late in a 50-step trajectory
PROMPTS_PER_CELL = 2  # 4 generators x 3 depths x 2 prompts = 24 images
STUB_WORDS = 20  # below this a caption has collapsed rather than been terse
SOURCE_EXPERIMENT = "019f3645"


def source_images() -> list[dict]:
    """Images spanning every v2 generator, three trajectory depths and two prompts.

    Drawn from the 50-step panel: an older lineup, but these are inputs to the
    captioner, and what matters is that they cover the range of pictures the
    long run will actually produce.
    """
    con = sqlite3.connect(DB)
    picked = []
    cell = 0
    for gen in GENERATORS:
        for depth in DEPTHS:
            rows = con.execute(
                """
                select r.initial_prompt, i.output_image
                from invocations i
                join runs r on r.id = i.run_id
                where r.experiment_id like ?
                  and json_extract(r.network, '$[0]') = ?
                  and i.type = 'image' and i.sequence_number = ?
                  and i.output_image is not null
                group by r.initial_prompt
                order by r.initial_prompt
                """,
                (f"{SOURCE_EXPERIMENT}%", gen, depth),
            ).fetchall()
            if not rows:
                continue
            # advance the prompt offset from cell to cell, so the 24 images
            # cover as many distinct scenes as the panel has prompts rather
            # than the same two subjects photographed twelve ways
            for j in range(PROMPTS_PER_CELL):
                prompt, blob = rows[(cell * PROMPTS_PER_CELL + j) % len(rows)]
                picked.append({
                    "generator": gen,
                    "depth": depth,
                    "prompt": prompt,
                    "b64": base64.b64encode(blob).decode("ascii"),
                })
            cell += 1
    con.close()
    if not picked:
        raise SystemExit("no source images found")
    print(
        f"{len(picked)} images: {len(set(p['generator'] for p in picked))} generators, "
        f"{len(set(p['prompt'] for p in picked))} prompts, "
        f"depths {sorted(set(p['depth'] for p in picked))}",
        flush=True,
    )
    return picked


def repeat_5gram_frac(text: str) -> float:
    """Share of 5-grams that are repeats --- greedy decoding's failure mode."""
    w = text.split()
    if len(w) < 6:
        return 0.0
    grams = [tuple(w[i : i + 5]) for i in range(len(w) - 4)]
    return 1.0 - len(set(grams)) / len(grams)


def longest_immediate_repeat(text: str) -> int:
    """Longest phrase (in words) that repeats back-to-back.

    A decoding loop shows up here as a large number even when the 5-gram share
    stays small, because a loop repeats one phrase many times rather than
    scattering duplicate n-grams through the text.
    """
    w = text.split()
    best = 0
    for n in range(1, len(w) // 2 + 1):
        for i in range(len(w) - 2 * n + 1):
            if w[i : i + n] == w[i + n : i + 2 * n]:
                best = n
                break
    return best


def arm_stats(texts: list[str]) -> dict:
    words = [len(t.split()) for t in texts]
    rep = [repeat_5gram_frac(t) for t in texts]
    loop = [longest_immediate_repeat(t) for t in texts]
    return {
        "n": len(texts),
        "words_median": float(np.median(words)),
        "words_min_max": [int(min(words)), int(max(words))],
        "stub_frac": float(np.mean([w < STUB_WORDS for w in words])),
        "repeat_5gram_frac_mean": float(np.mean(rep)),
        "repeat_5gram_frac_max": float(max(rep)),
        "longest_immediate_repeat_max": int(max(loop)),
        "unterminated_frac": float(
            np.mean([not t.rstrip().endswith((".", "!", "?", '"', "*")) for t in texts])
        ),
    }


def main() -> None:
    images = source_images()
    b64s = [im["b64"] for im in images]

    pm.setup()
    results, captions = {}, {}
    for model in CAPTIONERS:
        pm.load_model(model)
        pm.swap_to_gpu(model)
        pm.set_i2t_greedy(True)
        g1 = pm.invoke_i2t_batch(model, b64s)
        g2 = pm.invoke_i2t_batch(model, b64s)
        pm.set_i2t_greedy(False)
        s1 = pm.invoke_i2t_batch(model, b64s)
        pm.unload_model(model)

        captions[model] = {"greedy": g1, "sampled": s1}
        greedy = arm_stats(g1)
        sampled = arm_stats(s1)
        results[model] = {
            "deterministic": all(a == b for a, b in zip(g1, g2)),
            "n_images_differing_between_greedy_runs": sum(
                a != b for a, b in zip(g1, g2)
            ),
            "greedy": greedy,
            "sampled": sampled,
        }
        print(
            f"{model:12s} deterministic={results[model]['deterministic']!s:5s} "
            f"greedy: {greedy['words_median']:.0f}w "
            f"(min {greedy['words_min_max'][0]}), stub {greedy['stub_frac']:.2f}, "
            f"rep5 {greedy['repeat_5gram_frac_max']:.3f}, "
            f"loop {greedy['longest_immediate_repeat_max']}, "
            f"unterm {greedy['unterminated_frac']:.2f} | "
            f"sampled: {sampled['words_median']:.0f}w, "
            f"rep5 {sampled['repeat_5gram_frac_max']:.3f}, "
            f"loop {sampled['longest_immediate_repeat_max']}, "
            f"unterm {sampled['unterminated_frac']:.2f}",
            flush=True,
        )

    verdict = {
        "all_deterministic": all(r["deterministic"] for r in results.values()),
        "any_stub": any(r["greedy"]["stub_frac"] > 0 for r in results.values()),
        "worst_repeat_5gram": max(
            r["greedy"]["repeat_5gram_frac_max"] for r in results.values()
        ),
        "worst_immediate_repeat_words": max(
            r["greedy"]["longest_immediate_repeat_max"] for r in results.values()
        ),
        "worst_unterminated_frac": max(
            r["greedy"]["unterminated_frac"] for r in results.values()
        ),
    }
    OUT.write_text(
        json.dumps(
            {
                "images": [
                    {k: v for k, v in im.items() if k != "b64"} for im in images
                ],
                "stub_words_threshold": STUB_WORDS,
                "captioners": results,
                "verdict": verdict,
                "captions": captions,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nverdict: {verdict}", flush=True)
    print(f"wrote {OUT}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
