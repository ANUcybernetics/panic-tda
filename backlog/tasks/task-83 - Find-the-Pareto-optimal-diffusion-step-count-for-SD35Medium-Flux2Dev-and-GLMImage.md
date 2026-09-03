---
id: TASK-83
title: >-
  Find the Pareto-optimal diffusion step count for SD35Medium, Flux2Dev and
  GLMImage
status: Done
assignee: []
created_date: '2026-09-02 06:30'
updated_date: '2026-09-03 06:22'
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
- [x] #1 analysis/step_sweep.py run to completion with results.json and per-model images committed or summarised in backlog/docs/model-optimisation-log.md
- [x] #2 Per model, a table of steps vs seconds/image, pixel MAE, image cosine and caption cosine, with the knee identified
- [x] #3 _T2I_INVOKE_CONFIGS step counts (and guidance if the sweep suggests it) updated to the chosen values, with the choice recorded in decision-01 or a new decision
- [x] #4 Panel run-time estimate in CLAUDE.md updated for the new counts
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Swept SD35Medium, Flux2Dev and GLMImage over five step counts each (analysis/step_sweep.py); full write-up in backlog/docs/model-optimisation-log.md iteration 2, raw data in analysis/step_sweep/results.json, contact sheets alongside it.

Core finding: pixel MAE falls steadily with steps while caption cosine does not move. Image fidelity keeps improving but what a captioner reads off the image saturates almost immediately, and only the caption propagates to the next invocation.

Chosen: Flux2Dev 15 -> 12; SD35Medium stays 20; GLMImage stays 25. The stated criterion (smallest count at which caption cosine has flattened) pointed at 8 for Flux2Dev, but caption cosine came out non-monotone there, so it was re-run on 12 prompts (analysis/flux2dev_steps_confirm.py): still flat, still scrambled across 8-15, so the metric genuinely cannot separate them and the grid edge would have been noise-fitting. The contact sheet settles it visually --- at 8 steps composition changes (crowd mushy, apple smaller), at 12 it matches 15. Measured with mix gpu.bench at 12 steps: 57.0 s/item at batch=4 vs ~71 at 15, a 19.7% cut on the panel's dearest model; parity 5.65/12.58.

Metric scale, measured because it was needed to interpret the numbers: Gemma3n is deterministic (same image gives byte-identical captions, cosine 1.000) and unrelated images sit near 0.876.

Discarded: the sweep's NomicVision image-embedding column read 0.000 everywhere including self-comparisons. That model silently returns NaN-zeroed or non-reproducible embeddings (two successful runs disagreed by 0.44 on identical input). Column nulled, raised as TASK-86.
<!-- SECTION:NOTES:END -->
