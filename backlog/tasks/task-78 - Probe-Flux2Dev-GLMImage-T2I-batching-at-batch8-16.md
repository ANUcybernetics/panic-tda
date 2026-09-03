---
id: TASK-78
title: Probe Flux2Dev/GLMImage T2I batching at batch=8/16
status: Done
assignee: []
created_date: '2026-07-16 08:15'
updated_date: '2026-09-03 11:50'
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
- [x] #1 mix gpu.bench Flux2Dev GLMImage --batch-sizes 4,8,16 run after balanced_panel_5x5 completes, with per-item timings and parity metrics recorded in backlog/docs/model-optimisation-log.md
- [x] #2 _T2I_MAX_BATCH updated (or explicitly kept at 4) per model based on the results, with rationale logged
- [x] #3 if bumped: quality gate passes (deterministic parity per the log's threshold) and non-GPU test suite is green
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Run 2026-09-03: mix gpu.bench Flux2Dev GLMImage --batch-sizes 4,8,16 --n 16. Full table in backlog/docs/model-optimisation-log.md iteration 3.

Flux2Dev 103.9 (serial) / 57.7 @4 / 58.4 @8 / 57.8 @16 --- flat within noise across the whole range, so the projected 10-20% headroom simply is not there; it is compute-bound at 4 already. Stays at 4.
GLMImage 76.7 (serial) / 45.8 @4 / 42.4 @8 / OOM @16 --- 7.3% gain at 8, and 16 exhausts the card. Bumped to 8.

Quality gate: parity batched-vs-batched is the right comparison, and GLMImage is 70.69 at 8 vs 70.61 at 4, unchanged from the value verified benign in iteration 1. Non-GPU suite green (102 tests). Verified through the production path too, since what actually changed is invoke_t2i_batch's chunking rather than the bench's own call: 10 prompts chunk to 8+2 and return 10 correct 1024x1024 images at 44.4 s/item with a 33.1 GB peak.

Caveat recorded: batch=16 was measured in a process that had just run Flux2Dev, so part of the exhaustion may be fragmentation rather than a hard ceiling. Not worth chasing --- a mid-run OOM would stall a multi-week panel and 8 captures the available gain.

Also fixed a real bug in mix gpu.bench found on the way: @bench_timeout was a flat hour while the work scales with n and the number of batch sizes, so the first --n 16 --batch-sizes 4,8,16 attempt died an hour in having done most of the work. The budget now scales per image.
<!-- SECTION:NOTES:END -->
