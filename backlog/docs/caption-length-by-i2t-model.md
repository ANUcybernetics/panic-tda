# Caption length by image-to-text model

Measured 2026-08-13 by Sungyeon Hong, over the `balanced_panel_5x5` export
(`019f3645_parquet/invocations.parquet`): 50,000 text invocations, initial
prompts excluded.

## Result

Words per caption:

| Image-to-text model | n      | median | mean  | p10–p90 | min–max |
| ------------------- | ------ | ------ | ----- | ------- | ------- |
| Moondream           | 10,000 | 24     | 24.0  | 19–29   | 14–42   |
| Gemma3n             | 10,000 | 83     | 82.7  | 78–87   | 49–93   |
| Qwen25VL            | 10,000 | 90     | 89.5  | 68–109  | 41–118  |
| Pixtral             | 10,000 | 102    | 100.3 | 89–109  | 59–120  |
| LLaMA32Vision       | 10,000 | 106    | 103.9 | 94–112  | 31–120  |

**Caption length is almost entirely determined by which image-to-text model
produced it: eta-squared = 0.915.** (Eta-squared is the proportion of variance
in one variable explained by group membership in another; 0.915 means 91.5% of
the variation in caption length is accounted for by model identity alone.)

Overlap of the p10–p90 ranges, where 0 means the ranges are disjoint and 1 means
identical:

| Pair                                | Overlap  |
| ----------------------------------- | -------- |
| Moondream vs each of the other four | **0.00** |
| Gemma3n vs Qwen25VL                 | 0.22     |
| Gemma3n vs Pixtral / LLaMA32Vision  | 0.00     |
| Qwen25VL vs LLaMA32Vision           | 0.37     |
| Qwen25VL vs Pixtral                 | 0.49     |
| Pixtral vs LLaMA32Vision            | 0.75     |

**79.9% of captions exceed 50 words.**

## Captions are already being truncated mid-sentence

Follow-up measurement, same data. A caption cut off mid-generation will not end
in terminal punctuation, which makes truncation directly detectable:

| Image-to-text model | median words | p99 | max | % not ending in terminal punctuation |
| ------------------- | ------------ | --- | --- | ------------------------------------ |
| Gemma3n             | 83           | 90  | 93  | **89.9%**                            |
| LLaMA32Vision       | 106          | 115 | 120 | **78.3%**                            |
| Pixtral             | 102          | 113 | 120 | **64.0%**                            |
| Qwen25VL            | 90           | 114 | 118 | 17.6%                                |
| Moondream           | 24           | 34  | 42  | **0.0%**                             |

Examples of what the fragments look like:

- Gemma3n: "... casting a subtle shadow beneath the apple. The overall"
- LLaMA32Vision: "... appears to be a wooden surface, providing a subtle and
  natural contrast to"
- Pixtral: "... has a light brown color and a smooth texture. The lighting in
  the"

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

## Natural lengths, measured 2026-09-02

Decision-01 raised the ceiling to 1024 tokens. `analysis/natural_lengths.py`
then ran every panel captioner over 16 images from the caption pilot, with
Moondream in both its new default (`normal`) and the old `short` mode.

| Image-to-text model  | median | min–max | % cut off | s/caption | panel median | panel % cut off |
| -------------------- | ------ | ------- | --------- | --------- | ------------ | --------------- |
| Moondream (`short`)  | 24     | 19–33   | 0.0%      | 0.29      | 24           | 0.0%            |
| Qwen25VL             | 80     | 61–105  | 0.0%      | 0.91      | 90           | 17.6%           |
| Moondream (`normal`) | 82     | 50–121  | 0.0%      | 0.68      | —            | —               |
| Pixtral              | 113    | 74–154  | 0.0%      | 2.97      | 102          | 64.0%           |
| LLaMA32Vision        | 124    | 88–198  | 0.0%      | 2.13      | 106          | 78.3%           |
| Gemma3n              | 154    | 97–212  | 0.0%      | 1.21      | 83           | 89.9%           |

Every captioner now terminates on its own. The ceiling is nowhere near binding:
the longest caption in the entire 2,000-caption pilot is 315 words, against a
1024-token limit. The 16 images all came from Flux2Klein, whereas the panel
column mixes five text-to-image models, so the two median columns are not
strictly comparable --- the cut-off share is the column to read across.

Two findings matter beyond the numbers.

Moondream's brevity was its `length="short"` mode and nothing else. Run at
`short` on pilot images it reproduces the panel median of 24 words exactly, so
none of its shortness was ever truncation. At `normal` it is 3.4 times longer
and 2.4 times slower per caption, and it stops being the disjoint outlier that
made every Moondream-versus-other comparison fully confounded with length.

The verbosity ordering is largely an artefact of where each ceiling bit. Under
truncation Gemma3n looked like the second-shortest captioner; uncapped it is the
longest by a clear margin, and Qwen25VL --- which was barely truncated --- is
now the shortest of the verbose four. Any captioner effect measured on
`balanced_panel_5x5` is therefore partly a measure of how hard each model was
being cut, which is what TASK-82 suspected.

The practical consequence for TASK-80 is that the length distributions now
overlap far more than they did. That is the condition TASK-80 actually needed
--- not matched word counts, but enough overlap that caption length and model
identity stop being one variable.

## SD35Medium with the T5 encoder

Loading T5 (decision-01) costs 6.5 s/image at batch 4 and 25.5 GB peak on the 48
GB card, against roughly 6.5 s/item previously --- no measurable slowdown.

The check that matters is whether the caption survives the encoder. Warnings are
useless as evidence here, because `panic_models.setup()` calls
`diffusers.logging.set_verbosity_error()`, so diffusers' truncation warnings
never fire and their absence proves nothing. Token counts are the evidence
instead. The four longest pilot captions (315, 302, 276 and 262 words) come to
423, 430, 391 and 406 T5 tokens.

That lands between the two ceilings, which is why `max_sequence_length` had to
be set explicitly. diffusers defaults it to 256 and caps it at 512, so simply
loading T5 would still have cut roughly 40% off every one of these captions; at
the 512 decision-01 specifies, none are touched. The same captions are 314–373
CLIP tokens against CLIP's fixed 77, so that branch does truncate, as
decision-01 says --- architectural, and documented rather than worked around.

## The v2 lineup, measured 2026-09-03/04

TASK-87 replaced three of the five captioners. Measured the same way as above
(four Flux2Dev images, natural length, the same `Describe this image.`
instruction for every model so captioner and prompt stay unconfounded). The T5
token counts matter because the text-to-image encoders read at most 512 tokens
--- SD35Medium hard-caps there, Flux2Klein/Flux2Dev/ZImageTurbo default to it,
GLMImage takes 2048.

| Image-to-text model | median | min–max | T5 tokens | % cut off | s/caption |
| ------------------- | ------ | ------- | --------- | --------- | --------- |
| Moondream3          | 59     | 36–62   | 52–81     | 0.0%      | 2.4       |
| Qwen25VL (retained) | 80     | 61–105  | —         | 0.0%      | 0.9       |
| JoyCaption          | 178    | 91–178  | 131–250   | 0.0%      | 2.2       |
| Gemma4 (E4B)        | 225    | 180–252 | 262–389   | 0.0%      | 2.6       |
| Qwen3VL             | 292    | 198–317 | 312–466   | 0.0%      | 4.0       |

The spread is roughly fivefold with overlapping distributions, which is the
separability condition TASK-80 actually wanted --- and a marked improvement on
the old lineup, where Moondream sat disjoint from every other captioner. Note
Qwen3VL has the least headroom against the 512-token ceiling (466 of 512 on easy
images), so it is the one to watch on visually complex inputs.

Two models were rejected during integration, both for reasons invisible in their
caption output:

- **CapRL-Qwen3VL-4B** captions at 466 words median and 466--836 T5 tokens,
  exceeding the 512-token ceiling on three of four images. Its captions would
  have been silently truncated by four of the five text-to-image models,
  reintroducing exactly what decision-01 removed.
- **Gemma 4 26B-A4B** captioned well but occupied 48.4 GB of a 50.9 GB card:
  bitsandbytes quantises `nn.Linear` only, and 47.2 GB of that model sits in
  `Gemma4ClippableLinear` layers it skips silently. The dense E4B variant needs
  no quantisation and takes 15.9 GB. See TASK-87.

The general lesson for future lineup changes: 2026 captioners are three to six
times more verbose than their 2025 predecessors, so the 512-token encoder
ceiling --- not generation length --- is now the binding constraint on which
captioners this loop can use at all.

## Is the 512-token encoder ceiling costing us anything?

Measured 2026-09-04 (`analysis/prompt_tail.py`), prompted by the obvious
question: the ceiling is not a decision anyone made, so what is it buying?

What is possible. Flux2Klein and Flux2Dev encode text with
`Mistral3ForConditionalGeneration`, a full LLM with a long context, and Z-Image
has no hard cap either --- for those, 512 is merely the diffusers default.
GLMImage already reads 2048. **SD35Medium is the sole hard constraint**:
`StableDiffusion3Pipeline` raises above 512, and that limit is real, since its
T5 branch was trained to 512.

What it costs. Nothing at present. The v2 lineup's captions run 382--430 T5
tokens, so no caption is being truncated by any encoder. Raising the ceiling
would change the images without adding any text.

That last point is not obvious, and matters on its own: **`max_sequence_length`
is not a neutral knob.** With identical, untruncated prompts at a fixed seed,
moving it from 512 to 1024 still moved caption cosine to 0.896--0.988 (mean
0.943), because it sets the padding length and so changes the shape of the
conditioning tensor. It must therefore be fixed before a run and held.

Whether the discarded tail carries information is harder to answer than it
looks. Holding padding fixed and cutting 126--174 tokens from the end gives mean
caption cosine 0.920 (0.856--0.959). On the measured scale --- 1.000 for
identical images, 0.876 for unrelated ones --- that is a large change, but it is
barely larger than the padding artefact above. Flux2Klein at four steps is
simply very sensitive to any conditioning perturbation, so this metric cannot
cleanly separate "the tail carries content" from "any change reshuffles the
image". Six captions, one model: treat it as inconclusive on the mechanism and
conclusive only on the practical point.

Practical conclusion: keep 512 uniformly and treat it as a constraint on which
captioners are eligible, not as a parameter to tune. Uniformity matters more
than the value --- raising it on the four models that allow it while SD35Medium
stayed at 512 would reintroduce exactly the per-model truncation asymmetry that
made `balanced_panel_5x5` hard to interpret. Revisit only if a wanted captioner
genuinely exceeds it, and price that as replacing SD35Medium.

## Did the truncation change the dynamics?

The caption pilot (`01a060b4`, Flux2Klein+Gemma3n, ceiling 1024) repeats the
same 20 prompts × 4 runs × 50 steps as the corresponding arm of
`balanced_panel_5x5`, so the two differ only in whether captions were cut.
`analysis/pilot_vs_panel.py` compares them.

The captions differ exactly as intended: median 83 words and 9.1% complete
sentences in the panel, against median 160 and 100% complete in the pilot.

Finite-time Lyapunov exponents are statistically indistinguishable --- panel
mean 0.0119, pilot 0.0109, Wilcoxon p = 0.96 over the 20 paired prompts. The
per-prompt ordering does not survive (Spearman 0.20), but the fits are weak in
both conditions (median r² 0.46) and some pilot exponents come out negative, so
that is better read as estimator noise than as a real reshuffle.

Everything that measures step size, however, moves consistently. Complete
captions make the loop markedly less jittery: mean step-to-step cosine distance
falls from 0.0162 to 0.0116, a 28% reduction, holding at 29% over the second
half of the trajectory. Trajectories are correspondingly stickier in cluster
space, with the self-transition rate rising from 79.9% to 81.1% at the finest
clustering layer and from 87.9% to 94.9% at the coarsest, where the pilot visits
1.4 distinct clusters per run against the panel's 1.9. Occupancy and transition
structure shift accordingly (Jensen–Shannon divergence 0.23 and 0.32 bits at
layer 0, falling to 0.04 and 0.08 bits at layer 3).

Distance from the initial prompt tells the same story from the other side. The
pilot starts further out --- 0.220 versus 0.200 at step 1, since a 160-word
caption departs from a short prompt faster than an 83-word fragment --- but
travels less thereafter, ending at 0.244 versus 0.238. The panel gains 0.038
over the trajectory, the pilot 0.023.

So the answer is that truncation changed the dynamics, not merely the captions.
A mid-sentence fragment is a noisier input than a complete description, and the
loop driven by fragments takes visibly larger and less repeatable steps. That
the FTLE is nonetheless unchanged is consistent rather than contradictory:
divergence between paired runs measures the rate at which nearby trajectories
separate, which a roughly uniform change in step size need not affect.

For the paper, `balanced_panel_5x5` has to be described as a truncated dataset
for four of five captioners and for all five SD35Medium networks, with the
measured consequence stated: step-level dynamics are affected, aggregate
divergence rates appear not to be. Comparisons that turn on step size, cluster
dwell time or transition structure should not be pooled across the two regimes.
Note also that the cluster comparison assigns both conditions to the panel's
existing EVoC medoids by nearest cosine, since a global recluster would relabel
both at once; on panel rows that assignment reproduces EVoC's own labels 72–83%
of the time depending on layer, so the absolute occupancy figures are
approximate while the comparison between conditions is not.

## Why this matters

1. **Model identity and caption length cannot be separated statistically.** At
   eta-squared 0.915 they are effectively one variable, so including both in a
   regression gives unstable coefficients and no attributable split between
   them. Controlling for length after the fact is not available as an option; it
   has to be handled in the experiment design. See TASK-80.
2. **Moondream is the extreme case.** Its length range does not overlap any
   other captioner, so every Moondream-versus-other comparison is fully
   confounded with length.
3. **The four verbose captioners partially overlap.** A length-matched analysis
   restricted to Gemma3n, Qwen25VL, Pixtral and LLaMA32Vision within roughly
   78–112 words is feasible on existing data with no new runs.
4. **Comparability with published work.** Hintze et al. (2026, _Patterns_
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
./analysis/caption_length.py [parquet_dir ...]   # length tables above
./analysis/natural_lengths.py                    # natural lengths, SD35+T5 (GPU)
./analysis/pilot_vs_panel.py                     # pilot vs panel dynamics
```

Defaults to the `balanced_panel_5x5` dump and reproduces every table above. Pass
several dumps to pool them, which is how the truncated panel and the
natural-length runs get compared in one table.
