---
id: TASK-82
title: Fix silent mid-sentence truncation of captions in image-to-text models
status: Done
assignee:
  - sungyeon-hong
created_date: '2026-08-13'
updated_date: '2026-09-02 23:16'
labels:
  - analysis
  - paper
  - bug
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Captions from four of the five image-to-text models in the current panel are being cut off mid-sentence, and this was not a deliberate choice. Measured over the 50,000 captions in `balanced_panel_5x5` by checking whether a caption ends in terminal punctuation:

- Gemma3n 89.9% truncated
- LLaMA32Vision 78.3%
- Pixtral 64.0%
- Qwen25VL 17.6%
- Moondream 0.0%

The cause is the hardcoded `max_new_tokens` values in `priv/python/panic_models.py` (128 for several captioners, 100 for others, 1024 for Florence2) with no stopping criterion or sentence-boundary handling. Three models pile up against a ceiling of roughly 118-120 words, consistent with a 128-token limit being hit.

This matters beyond data tidiness. The truncated fragment is what feeds the next text-to-image invocation, so for the majority of steps in four of five networks the trajectory is being driven by an incomplete sentence. Any measured difference between captioners therefore mixes descriptive style with how severely that model is being truncated. Moondream is the only captioner producing complete sentences, which is also the model proposed as the anchor for cross-era comparison, so the asymmetry is concentrated exactly where it does most damage.

Full measurements in `backlog/docs/caption-length-by-i2t-model.md`.

Note this also reframes TASK-80. The question is not whether to introduce truncation but how to make the truncation that already exists deliberate, uniform and documented.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `max_new_tokens` audited for every image-to-text model in the panel, with current values and the resulting word-length ceiling documented per model
- [x] #2 Natural caption length measured per model with the limit raised high enough that generation terminates on its own, establishing what each model produces unconstrained
- [x] #3 Decision recorded on the truncation policy: either allow natural termination, or impose a uniform cap with trimming back to the last complete sentence rather than mid-word
- [x] #4 Policy applied consistently across all image-to-text models, with the chosen limit recorded in the experiment config rather than hardcoded per model
- [x] #5 Assessment written of what this means for `balanced_panel_5x5` and earlier data: whether affected analyses need re-running, and what has to be disclosed in the paper's methods
- [x] #6 Truncation rate added as a routine data-quality check, so this cannot recur silently
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Ben confirmed the limits were never deliberate. Decision recorded in backlog/decisions/decision-01 (natural termination everywhere). Implemented: uniform ceiling default 1024 with experiment-config override (AC #4), truncation rate in mix experiment.status (AC #6), pilot config/caption_pilot_flux2klein_gemma3n.json running to assess balanced_panel_5x5 (AC #5). Natural-length measurement for the other captioners (AC #2) queued after the pilot.

Closed 2026-09-02/03 after the caption pilot (01a060b4). Full results in backlog/docs/caption-length-by-i2t-model.md; decision record in backlog/decisions/decision-01.

AC #2: all five captioners measured unconstrained at the 1024 ceiling (analysis/natural_lengths.py, analysis/natural_lengths.json). Every one terminates on its own --- 0% cut off. Natural medians: Qwen25VL 80, Moondream(normal) 82, Pixtral 113, LLaMA32Vision 124, Gemma3n 154. Moondream at the old length='short' reproduces the panel median of 24 exactly, confirming its brevity was the mode, never truncation. The verbosity ordering is largely an artefact of where each ceiling bit: Gemma3n went from second-shortest under truncation to longest uncapped.

AC #5: the pilot repeats the panel's Flux2Klein+Gemma3n arm (same 20 prompts x 4 runs x 50 steps) with only the ceiling changed, compared by analysis/pilot_vs_panel.py. Truncation changed the dynamics, not only the captions --- step-to-step cosine distance is 28% lower with complete captions, trajectories are stickier in cluster space (coarsest-layer self-transition 87.9%->94.9%, 1.4 vs 1.9 distinct clusters per run), and drift from the initial prompt grows more slowly. FTLE is statistically unchanged (Wilcoxon p=0.96), which is consistent rather than contradictory: divergence rate between paired runs need not move when step size scales roughly uniformly. Paper methods must describe balanced_panel_5x5 as truncated for four of five captioners and all five SD35Medium networks, keep it for divergence-rate questions, and not pool it with natural-length data for anything turning on step size, cluster dwell time or transitions.
<!-- SECTION:NOTES:END -->
