---
id: decision-01
title: >-
  Let every model output run to its natural length: no caption or prompt
  truncation anywhere in the loop
date: '2026-09-02 06:16'
status: accepted
---
## Context

Sungyeon's measurement over `balanced_panel_5x5` (`backlog/docs/caption-length-by-i2t-model.md`, TASK-80, TASK-82) showed four of five captioners hitting hardcoded `max_new_tokens` ceilings of 100--128 on most steps (Gemma3n 90% of captions cut mid-sentence). Moondream was short for a different reason: its `length="short"` captioning mode, which the SMC 2025 code also used. None of the ceilings were chosen deliberately; they date from the February 2026 Elixir port.

Truncation is not confined to the captioners. SD35Medium is loaded without its T5 text encoder, so it sees only CLIP's 77 tokens (about 55 words), and every caption in the panel exceeded that. The other text-to-image encoders take 512 tokens (Flux2, Z-Image) or 2048 (GLMImage). Natural-length Gemma3n captions (median 167 words, max 302) fit under 512 for every encoder except CLIP.

## Decision

Ben, 2026-09-02: let every model produce its natural output. No generation ceiling that a captioner can reach, and no text-to-image encoder configuration that drops the caption. Concretely:

- the default captioner ceiling is 1024 tokens (`_I2T_MAX_NEW_TOKENS_DEFAULT` in `priv/python/panic_models.py`), well above any observed natural length; `i2t_max_new_tokens` in an experiment config overrides it for a deliberately capped arm
- SD35Medium loads its T5 encoder with `max_sequence_length` 512 (pipeline maximum); the 77-token CLIP branch is architectural and is documented, not worked around
- caption truncation rate is reported by `mix experiment.status` so it cannot recur silently

The alternative Sungyeon proposed in TASK-80 (matched capped/uncapped arms) remains available through the config override, but a cap is now a manipulation, not the default.

## Consequences

- `balanced_panel_5x5` is a truncated dataset for four captioners and for all five SD35Medium networks; the paper's methods must say so, and the caption-length pilot (`config/caption_pilot_flux2klein_gemma3n.json`) tests whether the truncation changed the dynamics
- captioner verbosity becomes a genuine property of the captioner rather than of the ceiling; RQ2 attribution treats length as a measured covariate
- Moondream now uses its API default `length="normal"` (Ben: don't get hung up on matching the 2025 paper, which used `short`); the cross-era comparison in TASK-81 has to treat that as a changed setting alongside the changed weight revision
- longer captions cost more captioner time (Gemma3n batches took roughly 2--3x longer in the pilot) and a loaded T5 encoder adds memory and time to SD35Medium

## Validated on the GPU, 2026-09-02

Measurements are in `backlog/docs/caption-length-by-i2t-model.md`; the raw
figures are `analysis/natural_lengths.json` and `analysis/pilot_vs_panel.json`.

All five captioners now terminate on their own at the 1024-token ceiling: 0%
cut off, against 89.9% for Gemma3n and 78.3% for LLaMA32Vision before. The
ceiling is nowhere near binding --- the longest caption in the 2,000-caption
pilot is 315 words. Moondream's brevity was confirmed to be its `short` mode
and nothing else, reproducing the panel median of 24 words exactly, while
`normal` gives 82.

The T5 change needed a second fix that this decision had implied but the code
did not carry. diffusers defaults `max_sequence_length` to 256 and caps it at
512, so loading the encoder alone would have cut roughly 40% off the pilot's
longest captions (391--430 T5 tokens); `max_sequence_length` is now set to 512
explicitly in `_T2I_INVOKE_CONFIGS`. Cost is 6.5 s/image at batch 4 and 25.5 GB
peak, no measurable slowdown. Warnings cannot be used to verify this ---
`panic_models.setup()` sets diffusers to verbosity_error --- so the check is a
token count.

The pilot answers the question this decision left open: the truncation changed
the dynamics, not only the captions. Step-to-step cosine distance falls 28%
once captions are complete, trajectories become stickier in cluster space
(coarsest-layer self-transition 87.9% to 94.9%), and distance from the initial
prompt grows more slowly. Finite-time Lyapunov exponents are unchanged
(p = 0.96), which is consistent: a roughly uniform change in step size need not
change the rate at which paired runs separate. So `balanced_panel_5x5` remains
usable for divergence-rate questions and must not be pooled with natural-length
data for anything turning on step size, cluster dwell time or transitions.

