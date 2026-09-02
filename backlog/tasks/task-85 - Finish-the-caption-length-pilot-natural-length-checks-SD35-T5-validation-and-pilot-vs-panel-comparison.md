---
id: TASK-85
title: >-
  Finish the caption-length pilot: natural-length checks, SD35 T5 validation,
  and pilot-vs-panel comparison
status: Done
assignee: []
created_date: '2026-09-02 06:37'
updated_date: '2026-09-02 23:16'
labels:
  - experiment
  - gpu
  - analysis
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up to decision-01 (backlog/decisions/decision-01) and TASK-82, parked so a fresh session can pick it up once the caption pilot has finished. The pilot is experiment 01a060b4-579a-7890-8310-8a4a251ddfb3 (config/caption_pilot_flux2klein_gemma3n.json: Flux2Klein+Gemma3n, same 20 prompts x 4 runs x 50 steps as balanced_panel_5x5 019f3645, captioner ceiling 1024). It was launched detached on 2026-09-02 (log: logs/caption_pilot_flux2klein_gemma3n.log) and is resumable with mix experiment.resume if it died. Check with mix experiment.status 01a060b4-579a-7890-8310-8a4a251ddfb3, which now also prints the caption-truncation share. Once the GPU is free: (1) run analysis/natural_lengths.py; it measures natural caption length, terminal punctuation and seconds per caption for all five panel captioners on 16 pilot images, Moondream at its new default (normal) versus the old short mode, and loads SD35Medium with its T5 encoder (commit 'sd35: load the T5 text encoder', not yet GPU-validated) to generate from the four longest pilot captions, recording peak GPU memory, seconds per image and any diffusers truncation warning; results go to analysis/natural_lengths.json. If T5 does not fit or blows the time budget, revert that commit and record why. (2) Compare the pilot against the original Flux2Klein+Gemma3n runs in 019f3645 (parquet dump in 019f3645_parquet/, loader analysis/load_with_polars.py): caption length distributions, FTLE, EVoC cluster occupancy and transition structure, embedding drift per step. The question is whether mid-sentence truncation changed the dynamics or only the captions; the answer decides what the paper has to say about balanced_panel_5x5 (TASK-82 AC #5). (3) Delete the leftover 4-step smoke experiment with mix experiment.delete 01a060b1 --force. (4) Update decision-01, the TASK-82 notes (AC #2 and #5) and backlog/docs/caption-length-by-i2t-model.md with the measured natural lengths, then commit and push.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 analysis/natural_lengths.py run; per-captioner natural length table (median, max, share cut off, seconds per caption) recorded in backlog/docs/caption-length-by-i2t-model.md, including Moondream normal vs short
- [x] #2 SD35Medium with T5 validated on the GPU (loads, no truncation warning on 300-word captions, memory and time recorded) or the commit reverted with the reason
- [x] #3 Pilot vs balanced_panel_5x5 comparison for Flux2Klein+Gemma3n written up (caption length, FTLE, cluster occupancy/transitions, drift) with a stated conclusion on whether truncation changed the dynamics
- [x] #4 TASK-82 AC #2 and #5 ticked with notes; decision-01 consequences updated
- [x] #5 Smoke experiment 01a060b1 deleted
- [x] #6 Everything committed and pushed
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Pilot 01a060b4 completed 2026-09-02 09:13Z after 3h12m (4000 invocations, 2000 embeddings, 80 PDs, 20 Lyapunov results). Results written up in backlog/docs/caption-length-by-i2t-model.md and backlog/decisions/decision-01; raw figures in analysis/natural_lengths.json and analysis/pilot_vs_panel.json.

AC #1 natural lengths measured for all five captioners plus Moondream normal vs short; all 0% cut off.
AC #2 SD35+T5 validated and the commit kept, but it needed a follow-up fix: diffusers defaults max_sequence_length to 256 and caps it at 512, so loading T5 alone would still have cut ~40% off the longest pilot captions (391-430 T5 tokens). max_sequence_length is now 512 in _T2I_INVOKE_CONFIGS. 6.5 s/image at batch 4, 25.5 GB peak, no slowdown. Note the script's original warning-capture check was worthless --- panic_models.setup() sets diffusers to verbosity_error --- so it now measures token counts.
AC #3 comparison done; conclusion is that truncation changed the dynamics, not only the captions (28% lower step-to-step drift, stickier clusters, unchanged FTLE).
AC #4 decision-01 and TASK-82 updated; TASK-82 and TASK-80 closed.
AC #5 01a060b1 deleted --- this exposed a bug in mix experiment.delete, which never removed lyapunov_results and so failed on a foreign key violation after destroying runs and invocations. Fixed in the same session.

Incidental finding for whoever runs the next pilot: the analysis scripts' word counts must split on any whitespace, not on single spaces, or captions containing newlines are undercounted.
<!-- SECTION:NOTES:END -->
