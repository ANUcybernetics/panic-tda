---
id: TASK-97
title: >-
  resume_batch restarts every run at the batch minimum and can violate the
  sequence index
status: To Do
assignee: []
created_date: '2026-09-04 12:56'
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
- [ ] #1 A batch resumed from a ragged state completes without error and without duplicate sequence numbers
- [ ] #2 Each run continues from its own last completed step rather than the batch minimum, with inputs matching that step
- [ ] #3 Regression test that resumes a batch whose runs are at different sequence numbers
<!-- AC:END -->
