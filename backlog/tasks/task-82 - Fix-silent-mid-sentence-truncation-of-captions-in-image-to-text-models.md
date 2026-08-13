---
id: TASK-82
title: Fix silent mid-sentence truncation of captions in image-to-text models
status: To Do
assignee:
  - sungyeon-hong
created_date: '2026-08-13'
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
- [ ] #1 `max_new_tokens` audited for every image-to-text model in the panel, with current values and the resulting word-length ceiling documented per model
- [ ] #2 Natural caption length measured per model with the limit raised high enough that generation terminates on its own, establishing what each model produces unconstrained
- [ ] #3 Decision recorded on the truncation policy: either allow natural termination, or impose a uniform cap with trimming back to the last complete sentence rather than mid-word
- [ ] #4 Policy applied consistently across all image-to-text models, with the chosen limit recorded in the experiment config rather than hardcoded per model
- [ ] #5 Assessment written of what this means for `balanced_panel_5x5` and earlier data: whether affected analyses need re-running, and what has to be disclosed in the paper's methods
- [ ] #6 Truncation rate added as a routine data-quality check, so this cannot recur silently
<!-- AC:END -->
