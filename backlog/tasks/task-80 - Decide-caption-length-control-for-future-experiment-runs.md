---
id: TASK-80
title: Decide caption-length control for future experiment runs
status: To Do
assignee:
  - sungyeon-hong
created_date: '2026-08-13'
labels:
  - analysis
  - paper
dependencies:
  - TASK-82
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Caption length is almost entirely determined by which image-to-text model produced it (eta-squared = 0.915 over the 50,000 captions in `balanced_panel_5x5`; Moondream median 24 words, LLaMA32Vision 106, and Moondream's p10-p90 range does not overlap any other captioner). Model identity and caption length are therefore not separable statistically, so length cannot be handled as a covariate after the fact — it has to be decided in the experiment design. Measurements are recorded in `backlog/docs/caption-length-by-i2t-model.md`.

This blocks any new GPU run, because a generation-time length cap cannot be retrofitted without regenerating the data.

Preferred approach is to treat length as a manipulated factor rather than a nuisance: matched capped and uncapped arms over the same networks and prompts, which allows the captioner effect to be decomposed into semantic style versus verbosity. A cap must be applied as a generation-time `max_tokens` limit, not post-hoc truncation, because the next image is generated from the full caption — truncating only before embedding would measure something other than what drives the loop. Prompt-level instructions ("describe in under 50 words") are unsuitable, as they introduce instruction-following ability as a further confound and models differ widely in that ability.

Also relevant to comparability with published work: Hintze et al. (2026, Patterns 7:101451) capped text outputs at 50 words, and 79.9% of our captions exceed that, so our captioner effects are not directly comparable to theirs without a capped arm.

Depends on TASK-82. The observed lengths are not natural model behaviour — four of five captioners are already being truncated by hardcoded token limits, so the length distributions measured here are partly an artefact of those limits. Natural unconstrained lengths have to be established before any length-control design can be chosen. Exact word counts are in any case not achievable, and not necessary: the requirement is that length distributions overlap enough across captioners for length and model identity to be statistically separable, which is a far weaker condition than hitting a specific word count.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Measurement script promoted from scratch into `analysis/caption_length.py`, reproducing the per-model length table and the eta-squared figure
- [ ] #2 Length distribution per image-to-text model produced as a publication-quality figure
- [ ] #3 Length-matched comparison across Gemma3n / Qwen25VL / Pixtral / LLaMA32Vision within the ~78-112 word band run on existing data, establishing whether captioner effects survive when length is held approximately fixed
- [ ] #4 Decision recorded on whether future runs use matched capped/uncapped arms, a single capped condition, or no cap, with rationale
- [ ] #5 If a cap is adopted, generation-time `max_tokens` support confirmed or implemented for every image-to-text model in the panel, and the equivalent word count documented per model
<!-- AC:END -->
