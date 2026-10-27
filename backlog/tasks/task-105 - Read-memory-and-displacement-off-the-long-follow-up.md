---
id: TASK-105
title: Read memory and displacement off the long follow-up
status: Done
assignee:
  - '@claude'
created_date: 2026-10-04 07:19
updated_date: '2026-10-27T06:37:00Z'
labels:
  - analysis
  - paper
dependencies:
  - TASK-104
references:
  - analysis/long_follow_up.py
  - analysis/long_follow_up.json
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
- [x] #1 Prompt memory, twin distance and stranger distance are reported over the whole horizon for every network with intervals, and for each network it is stated whether memory levels off above zero, keeps falling, or cannot yet be told, with the reason
- [x] #2 Displacement within a run is reported out to at least half the run length against the distance between twins, with the time a run takes to cover its prompt's region or a lower bound on it
- [x] #3 Each prompt's centre and the spread of its runs around it are reported from the eight runs, including whether the spread is one region or several
- [x] #4 The follow-up's numbers at 150 text states are set beside the panel's for the networks and prompts they share, and any disagreement is explained
- [x] #5 backlog/docs/drift-and-memory.md and backlog/docs/research-programme.md are updated with what the follow-up shows
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
READ 2026-10-27 with analysis/long_follow_up.py (results in analysis/long_follow_up.json, figures in analysis/long_follow_up/), written up in backlog/docs/drift-and-memory.md (What the long follow-up shows) and research-programme.md.

Method choices that differ from drift_memory.py: intervals resample runs within each prompt (the five prompts were chosen, not drawn); displacement is taken from state 150 and set against the distance to a twin at the same separation over the same pairs of states, not against the end-of-run twin distance; the memory verdict compares the rate per doubling over states 400-1,000 with the cell's own over 150-400, because a 300-state slope against zero could not tell a level from the logarithmic course; the one-region test is SigClust with the cell's pooled within-prompt covariance as its null (the sample spectrum of eight points gave p near 1 everywhere); the panel comparison is a permutation test shuffling each prompt's ten runs between the two experiments.

Handoff: Ben asked for the follow-up's parquet (01a10613_parquet, exported without --embed-prompts) and an image folder to go to Sungyeon by magic wormhole. mix experiment.export_images 01a10613 writes the full image set (~13 GB) to 01a10613_images. wormhole is not installed on this machine, so the send is left to Ben.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
No prompt is forgotten in 1,000 text states: memory averages 0.39 at state 150 and 0.25 (0.21-0.30) at the end, 0.12-0.44 by network. It is still falling in the two Gemma4 networks whose twins still separate. In the other four, twins have stopped separating and memory falls only as strangers converge (5-8%), and a level cannot be told from a slow fall. A run covers 42-65% of its prompt's region at a separation of 500 and is still climbing (lower bound 500; projected 1,800-11,000). Prompt centres stop moving after about state 400. In 12 of 30 prompt-network pairs the runs are not one region, mostly a core of several runs at one place. At 150 states the follow-up agrees with the panel on memory, offset and displacement. Step size disagrees in Flux2Klein + Moondream3 (p = 0.001), unexplained after ruling out code, config and batch composition.
<!-- SECTION:FINAL_SUMMARY:END -->
