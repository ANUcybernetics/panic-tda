---
id: TASK-78
title: Probe Flux2Dev/GLMImage T2I batching at batch=8/16
status: To Do
assignee: []
created_date: '2026-07-16 08:15'
labels:
  - gpu
  - performance
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Real-schedule validation in balanced_panel_5x5 (see backlog/docs/model-optimisation-log.md) showed both models on projection at batch=4 with tiny VRAM peaks (Flux2Dev 5.6-11 GB, GLMImage 6.85 GB of 48 GB) thanks to sequential CPU offload — the memory-safety rationale for capping at 4 is resolved. The bench speedup curve (Flux2Dev 1.41x@2 -> 1.59x@4; GLMImage 1.37x@2 -> ~1.7x@4) suggests another 10-20% per-item headroom at batch=8 before compute saturates. Panel lockstep steps hand over 80 prompts, which divides evenly by 8 and 16. Must wait until balanced_panel_5x5 finishes (~2026-07-23) — do not touch the GPU before then.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 mix gpu.bench Flux2Dev GLMImage --batch-sizes 4,8,16 run after balanced_panel_5x5 completes, with per-item timings and parity metrics recorded in backlog/docs/model-optimisation-log.md
- [ ] #2 _T2I_MAX_BATCH updated (or explicitly kept at 4) per model based on the results, with rationale logged
- [ ] #3 if bumped: quality gate passes (deterministic parity per the log's threshold) and non-GPU test suite is green
<!-- AC:END -->
