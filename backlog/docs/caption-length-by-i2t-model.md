# Caption length by image-to-text model

Measured 2026-08-13 by Sungyeon Hong, over the `balanced_panel_5x5` export
(`019f3645_parquet/invocations.parquet`): 50,000 text invocations, initial
prompts excluded.

## Result

Words per caption:

| Image-to-text model | n | median | mean | p10–p90 | min–max |
|---|---|---|---|---|---|
| Moondream | 10,000 | 24 | 24.0 | 19–29 | 14–42 |
| Gemma3n | 10,000 | 83 | 82.7 | 78–87 | 49–93 |
| Qwen25VL | 10,000 | 90 | 89.5 | 68–109 | 41–118 |
| Pixtral | 10,000 | 102 | 100.3 | 89–109 | 59–120 |
| LLaMA32Vision | 10,000 | 106 | 103.9 | 94–112 | 31–120 |

**Caption length is almost entirely determined by which image-to-text model
produced it: eta-squared = 0.915.** (Eta-squared is the proportion of variance
in one variable explained by group membership in another; 0.915 means 91.5% of
the variation in caption length is accounted for by model identity alone.)

Overlap of the p10–p90 ranges, where 0 means the ranges are disjoint and 1 means
identical:

| Pair | Overlap |
|---|---|
| Moondream vs each of the other four | **0.00** |
| Gemma3n vs Qwen25VL | 0.22 |
| Gemma3n vs Pixtral / LLaMA32Vision | 0.00 |
| Qwen25VL vs LLaMA32Vision | 0.37 |
| Qwen25VL vs Pixtral | 0.49 |
| Pixtral vs LLaMA32Vision | 0.75 |

**79.9% of captions exceed 50 words.**

## Captions are already being truncated mid-sentence

Follow-up measurement, same data. A caption cut off mid-generation will not end
in terminal punctuation, which makes truncation directly detectable:

| Image-to-text model | median words | p99 | max | % not ending in terminal punctuation |
|---|---|---|---|---|
| Gemma3n | 83 | 90 | 93 | **89.9%** |
| LLaMA32Vision | 106 | 115 | 120 | **78.3%** |
| Pixtral | 102 | 113 | 120 | **64.0%** |
| Qwen25VL | 90 | 114 | 118 | 17.6% |
| Moondream | 24 | 34 | 42 | **0.0%** |

Examples of what the fragments look like:

- Gemma3n: "... casting a subtle shadow beneath the apple. The overall"
- LLaMA32Vision: "... appears to be a wooden surface, providing a subtle and natural contrast to"
- Pixtral: "... has a light brown color and a smooth texture. The lighting in the"

The cause is the hardcoded `max_new_tokens` values in
`priv/python/panic_models.py` — 128 for some captioners, 100 for others, 1024
for Florence2 — with no sentence-boundary handling. Three of the models cluster
against a ceiling around 118-120 words, consistent with a 128-token limit.

**Consequences:**

1. **Naive truncation is already occurring**, at an inconsistent per-model
   threshold that was never chosen deliberately.
2. **The truncated fragments drive the next image generation.** For four of the
   five captioners this affects the majority of steps, so this shapes the actual
   trajectories rather than merely the measurements.
3. **Moondream is the only captioner producing complete sentences.** So the
   Moondream-versus-others difference is not only length but complete sentences
   versus mid-sentence fragments — a more serious confound than length alone,
   and one that matters especially because Moondream is the proposed anchor for
   cross-era comparison.
4. Part of any measured captioner effect may be a truncation artefact rather
   than a difference in descriptive style.

See TASK-82.

## Why this matters

1. **Model identity and caption length cannot be separated statistically.** At
   eta-squared 0.915 they are effectively one variable, so including both in a
   regression gives unstable coefficients and no attributable split between
   them. Controlling for length after the fact is not available as an option;
   it has to be handled in the experiment design. See TASK-80.
2. **Moondream is the extreme case.** Its length range does not overlap any
   other captioner, so every Moondream-versus-other comparison is fully
   confounded with length.
3. **The four verbose captioners partially overlap.** A length-matched analysis
   restricted to Gemma3n, Qwen25VL, Pixtral and LLaMA32Vision within roughly
   78–112 words is feasible on existing data with no new runs.
4. **Comparability with published work.** Hintze et al. (2026, *Patterns*
   7:101451) capped text outputs at 50 words, citing the finding that
   embedding-based similarity is not comparable across different text lengths.
   Since 79.9% of our captions exceed that cap, our captioner effects are not
   directly comparable to theirs without a length-capped arm.

## Related

- TASK-75 acceptance criterion 3 requires checking outlier status against
  caption length and per-model verbosity. The measurement above is the first
  half of that; what remains is testing whether outlier status tracks length.
- TASK-80 covers the design decision this forces.

## Reproducing

```
./analysis/caption_length.py [parquet_dir ...]
```

Defaults to the `balanced_panel_5x5` dump and reproduces every table above.
Pass several dumps to pool them, which is how the truncated panel and the
natural-length runs get compared in one table.
