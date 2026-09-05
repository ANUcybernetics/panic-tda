---
id: TASK-97
title: >-
  resume_batch restarts every run at the batch minimum and can violate the
  sequence index
status: Done
assignee:
  - '@claude'
created_date: '2026-09-04 12:56'
updated_date: '2026-09-05 05:25'
labels:
  - engine
  - instrument
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found 2026-09-04 while implementing seed recording (TASK-93).

PanicTda.Engine.RunExecutor.resume_batch/2 reads each run's last invocation, takes min_completed across the batch, and restarts EVERY run at min_completed + 1. Runs that got further than the minimum are then asked to create an invocation at a sequence_number they already have, which the unique index invocations_unique_run_sequence_index rejects, so the resume crashes.

It also silently uses the wrong input for those runs: each state carries the output of its own last invocation, so a run that is ahead would continue from its latest output while being numbered as though it were at the batch minimum.

Runs within a batch step are created in a loop, one Ash create per run, so a crash between the first and last create leaves the batch ragged. That is exactly the state resume is for.

Not hit yet because no batched experiment has crashed mid-write, but TASK-90 is a multi-week run that will be interrupted, and resume is the thing that makes an interruption cheap rather than fatal.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A batch resumed from a ragged state completes without error and without duplicate sequence numbers
- [x] #2 Each run continues from its own last completed step rather than the batch minimum, with inputs matching that step
- [x] #3 Regression test that resumes a batch whose runs are at different sequence numbers
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Each batch state now carries its own next_seq (RunExecutor.execute_batch / resume_batch). The batch loop starts at the lowest next_seq across the group and, at every step, only invokes the runs whose next_seq equals that step; the others wait and rejoin when the loop reaches them. Inputs come from each run's own last output, so a run that is ahead is neither re-numbered nor fed the wrong input.

Regression test 'ragged batch resume' in test/resume_test.exs: three runs at steps 4, 1 and 0 of a 6-step network, resumed as one batch. Asserts sequence numbers 0..5 per run, an unbroken input_invocation_id chain, and that the pre-existing invocations were kept. Against the old code this test crashed in the first resumed step (an image handed to the text-to-image model, the input-mismatch half of the bug), before the unique index could even be hit.
<!-- SECTION:NOTES:END -->
