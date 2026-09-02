---
id: TASK-83
title: >-
  Find the Pareto-optimal diffusion step count for SD35Medium, Flux2Dev and
  GLMImage
status: To Do
assignee: []
created_date: '2026-09-02 06:30'
labels:
  - experiment
  - gpu
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-66 cut num_inference_steps below the pipeline defaults to save time (SD35Medium 20 vs 28, Flux2Dev 15 vs 50, GLMImage 25 vs 50) without measuring what it cost; ZImageTurbo (8) and Flux2Klein (4) are distilled for their current counts and are not in question. Decision-01 says no output should be limited by an undocumented setting, and Ben asked for the knee rather than the defaults. The loop-relevant quality metric is not pixel fidelity but whether the captioner reads the same content: for each model, generate the same four prompts (two short initial prompts, two natural-length captions from the caption pilot) at a fixed seed across five step counts with the highest as reference, caption every image with Gemma3n, and measure caption-embedding cosine (Qwen3Embed) against the reference caption alongside seconds per image, pixel MAE and NomicVision image-embedding cosine. The Pareto choice is the smallest step count at which caption cosine has flattened. Script is ready at analysis/step_sweep.py (~1.5 h of GPU, needs the GPU free and the caption pilot's captions in priv/panic_tda_dev.db); images land in analysis/step_sweep/<model>/ for eyeballing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 analysis/step_sweep.py run to completion with results.json and per-model images committed or summarised in backlog/docs/model-optimisation-log.md
- [ ] #2 Per model, a table of steps vs seconds/image, pixel MAE, image cosine and caption cosine, with the knee identified
- [ ] #3 _T2I_INVOKE_CONFIGS step counts (and guidance if the sweep suggests it) updated to the chosen values, with the choice recorded in decision-01 or a new decision
- [ ] #4 Panel run-time estimate in CLAUDE.md updated for the new counts
<!-- AC:END -->
