---
id: TASK-105
title: Read memory and displacement off the long follow-up
status: To Do
assignee: []
created_date: '2026-10-04 07:19'
labels:
  - analysis
  - paper
dependencies:
  - TASK-104
references:
  - backlog/docs/drift-and-memory.md
  - analysis/drift_memory.py
  - analysis/long_run_drift.py
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-104 runs the experiment that TASK-103 could not: a few prompts, eight runs each, to 700 text states or more. This task is the reading of it, with the same quantities TASK-103 defined (backlog/docs/drift-and-memory.md), so that the panel and the follow-up can be set side by side.

The questions it has to answer are the ones the panel left open. Does prompt memory level off above zero, or keep falling until the runs of one prompt are as far apart as runs of different prompts? How long does a run take to cover its prompt's region, and does its displacement stop at the distance between twins? With eight runs per prompt, where is each prompt's centre, how wide is the spread around it, and is that spread one region or several?

analysis/drift_memory.py was written for the panel and fixes a 150-state horizon in several constants (the late window, the ageing windows, the two window pairs used for shared movement). analysis/long_run_drift.py has the long-separation version of the displacement curve but was written for runs whose prompts had no content.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Prompt memory, twin distance and stranger distance are reported over the whole horizon for every network with intervals, and for each network it is stated whether memory levels off above zero, keeps falling, or cannot yet be told, with the reason
- [ ] #2 Displacement within a run is reported out to at least half the run length against the distance between twins, with the time a run takes to cover its prompt's region or a lower bound on it
- [ ] #3 Each prompt's centre and the spread of its runs around it are reported from the eight runs, including whether the spread is one region or several
- [ ] #4 The follow-up's numbers at 150 text states are set beside the panel's for the networks and prompts they share, and any disagreement is explained
- [ ] #5 backlog/docs/drift-and-memory.md and backlog/docs/research-programme.md are updated with what the follow-up shows
<!-- AC:END -->
