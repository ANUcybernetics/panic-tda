---
id: TASK-100
title: Measure encoder-side caption truncation per cell after the long-horizon run
status: To Do
assignee: []
created_date: '2026-09-07 04:23'
labels:
  - analysis
  - paper
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found 2026-09-07 in the TASK-90 pre-launch smoke (one cell, Flux2Klein + Qwen3VL, 20 prompts x 2 runs x 4 steps, at the panel's batch cap of 40). Qwen3VL captions of early-step images exceed the 512-token text-encoder ceiling: under SD35Medium's T5 tokenizer 14 of 80 captions (median 426, p90 563, max 744 tokens); under the Mistral3 (Flux2) and Qwen3 (Z-Image) tokenizers 2 of 80 (median 371, max 642). The other four captioners stay under 512 on every measured image (analysis/captioner_greedy_quality.json: Gemma4 max 410 T5 tokens, JoyCaption 298, Qwen25VL 174, Moondream3 141), and Qwen3VL itself is under on the 24-image quality sample (max 486), so the overshoot is Qwen3VL on visually busy early-trajectory images. The pipelines truncate silently (diffusers verbosity is set to error), so the generator reads a cut caption for those steps. The programme's 512-token rule is an eligibility constraint on captioners; Qwen3VL is marginal against it for SD35Medium and comfortably under for the other three generators.

WHAT TO DO. Captions are stored and all three tokenizers are in the HF cache, so the share of captions over the ceiling is computable post hoc per cell and per step bin with no GPU. Report it in methods as a property of each network (it is part of what the generator reads), add it to RQ2 as a covariate alongside caption length, and check whether it changes in the stationary regime relative to the first steps. Script pattern: the smoke_tokens check in this session (tokenise with tokenizer_3 of SD3.5 for SD35Medium cells, the Flux2 Mistral3 tokenizer for Flux2Klein/Flux2Dev cells, the Z-Image Qwen3 tokenizer for ZImageTurbo cells; load from the cached snapshot directories, since offline hub resolution failed).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Per-cell, per-step-bin share of captions exceeding the encoder ceiling computed from stored text with each generator's own tokenizer, written to analysis/ as JSON with a table in the design doc
- [ ] #2 mix experiment.status reports the share alongside the existing terminal-punctuation truncation check, so the next lineup change cannot miss it
- [ ] #3 Methods text for the paper states the share for each affected cell and RQ2 carries it as a covariate
<!-- AC:END -->
