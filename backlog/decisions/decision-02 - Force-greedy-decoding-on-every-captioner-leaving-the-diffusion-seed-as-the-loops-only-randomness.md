---
id: decision-02
title: >-
  Force greedy decoding on every captioner, leaving the diffusion seed as the
  loop's only randomness
date: '2026-09-04 23:06'
status: accepted
---
## Context

The research programme describes the loop as a Markov chain whose only source
of randomness is a fresh diffusion seed per text-to-image invocation. That is
what makes the spread among one caption's successors attributable to the
generator, and it is the premise TASK-89's drift/noise decomposition rests on.

Three of the five v2 captioners were not greedy. Qwen25VL, Qwen3VL and
JoyCaption inherit `do_sample=True` from their shipped
`generation_config.json`, because the invoke paths in `priv/python/panic_models.py`
never passed `do_sample`. Only Moondream3 (temperature 0.0) and Gemma4
(`do_sample=False`, passed explicitly) were ever deterministic. Qwen25VL is the
trap: its temperature of 1e-06 reads as effectively greedy, but with
`repetition_penalty` 1.05 under bfloat16 it still produced eight distinct
captions from one image.

`analysis/captioner_noise.py` measured what that costs. Holding the image fixed
and captioning it eight times, spread about the centroid in Qwen3Embed space at
256 dimensions --- the same cosine-distance units as a trajectory step:

| captioner  | sampling noise | generator noise (TASK-89) |
| ---------- | -------------- | ------------------------- |
| Moondream3 | 0.0000         |                           |
| Gemma4     | 0.0000         |                           |
| Qwen25VL   | 0.0110         |                           |
| Qwen3VL    | 0.0233         |                           |
| JoyCaption | 0.0304         |                           |
| Flux2Klein |                | 0.0311                    |
| ZImageTurbo |               | 0.0306                    |
| Flux2Dev   |                | 0.0341                    |
| SD35Medium |                | 0.0446                    |

The two columns are commensurable: both are half the mean squared displacement
about a centroid on the unit sphere, and both were computed through the
corrected last-token pooling path (TASK-96). Against a whole stationary step of
0.062--0.083, the captioner's own dice were worth as much as the generator's
for JoyCaption and Qwen3VL. For those networks, step-level motion could have
been mostly the captioner resampling its own prose before the generator
contributed anything.

## Decision

Ben, 2026-09-04: force greedy decoding on every captioner.
`_I2T_FORCE_GREEDY` in `priv/python/panic_models.py` now defaults to `True`,
and `set_i2t_greedy/1` remains only so the analysis scripts can measure what
the shipped configs do. Nothing that writes to the database turns it off.

The alternative was to keep the shipped sampling and carry a three-way
decomposition --- drift, generator noise, captioner noise --- through both
research questions. Greedy was chosen because the sampling temperature is a
packaging choice by five different vendors, not a property of the models worth
measuring: under sampling, RQ2's "captioner effect" would confound descriptive
style with whatever temperature each release happened to ship. Greedy makes it
a statement about style alone.

## Consequences

- the loop is a deterministic captioner composed with a stochastic generator,
  so the diffusion seed (TASK-93) accounts for all within-condition variation
  and every step is regenerable from it
- TASK-89's noise term is the generator's alone, as its method assumed; the
  decomposition needs no captioner term
- every dataset written before this is a sampling dataset for three of the five
  captioners. `balanced_panel_5x5` and the caption pilot are affected; the
  paper's methods say which decoding produced which data
- exact caption repetition is not made more likely: a repeat now requires the
  same image, which requires the same seed
- captions get marginally shorter for two captioners (see below), which is a
  change to caption length as an RQ2 covariate, not to the truncation policy of
  decision-01

## Validated on the GPU, 2026-09-05

`analysis/captioner_greedy_quality.py` over 24 images --- all four v2
generators, 20 distinct prompts, trajectory depths 8, 24 and 48 --- captioning
each twice greedily and once under the shipped config. Results in
`analysis/captioner_greedy_quality.json`.

| captioner  | deterministic | median words | shortest | stubs | repeat 5-gram | repeat loop | unterminated |
| ---------- | ------------- | ------------ | -------- | ----- | ------------- | ----------- | ------------ |
| Moondream3 | yes           | 45           | 30       | 0     | 0.045         | 0           | 0            |
| Qwen25VL   | yes           | 83           | 58       | 0     | 0.068         | 0           | 0            |
| Qwen3VL    | yes           | 238          | 121      | 0     | 0.004         | 0           | 0            |
| Gemma4     | yes           | 192          | 115      | 0     | 0.012         | 0           | 0            |
| JoyCaption | yes           | 179          | 118      | 0     | 0.065         | 0           | 0            |

All five are deterministic: two greedy passes over 24 images produced identical
text every time. None of greedy decoding's failure modes appears --- no caption
collapsed to a stub, no phrase repeats back-to-back anywhere in the set, and
every caption ends on terminal punctuation, so none was cut at the 1024-token
ceiling.

JoyCaption is the only captioner whose repeated-5-gram share is higher greedy
(0.065) than sampled (0.022). Reading the caption shows it is content, not
degeneracy: the image is a 3x3 grid of near-identical rose bouquets and the
caption enumerates them row by row.

Caption length barely moves --- Qwen25VL 102 to 88 words in the earlier
four-image measurement, JoyCaption 182 to 168, Qwen3VL 250 to 246, Gemma4 237
to 234 --- so the verbosity ordering that RQ2 uses as a covariate is unchanged.
