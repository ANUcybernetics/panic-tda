---
id: TASK-77
title: 'TDA keep/kill pilot: does topology add anything beyond symbolic dynamics?'
status: To Do
assignee: []
created_date: '2026-07-10 00:51'
labels:
  - analysis
  - paper
dependencies:
  - TASK-76
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Decide with data whether TDA earns a place in the next paper or is dropped from the headline claims. Static Rips PH on 25-100 points in 2560-dim is unreliable beyond H0 (curse of dimensionality on persistence diagrams, arXiv:2404.18194) and H0 duplicates hierarchical clustering; persistence entropy is confounded by bar count (duplicate captions). The only candidate value-add is detecting geometric recurrence/limit cycles that cluster-label sequences miss (sliding-window persistence, Perea-Harer). Depends on TASK-76 symbol sequences for comparison; blocked on the running experiment.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Recurrence statistics on raw embedding distances (return-time distributions / recurrence plots) computed for a sample of runs and compared against symbol-sequence recurrence, establishing whether geometric recurrence exists that symbols miss
- [ ] #2 Test of whether (normalised) persistence entropy predicts anything not already predicted by duplicate-caption count and dwell statistics (e.g. partial correlation)
- [ ] #3 Documented keep/kill decision: either sliding-window PH is adopted with demonstrated added value, or TDA is dropped from headline analyses with the rationale written up for the paper's methods discussion
- [ ] #4 If killed: PdStage moved out of the per-experiment hot path (or made opt-in) and docs updated
<!-- AC:END -->
