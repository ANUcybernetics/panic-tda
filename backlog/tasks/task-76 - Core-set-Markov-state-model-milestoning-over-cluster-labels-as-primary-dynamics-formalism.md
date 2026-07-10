---
id: TASK-76
title: >-
  Core-set Markov state model (milestoning) over cluster labels as primary
  dynamics formalism
status: To Do
assignee: []
created_date: '2026-07-10 00:50'
labels:
  - analysis
  - paper
dependencies:
  - TASK-75
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Adopt symbolic dynamics as the primary formalism for trajectory analysis, replacing bag-of-points persistent homology as the headline method. Cluster cores (high-confidence dense regions from EVoC) are metastable states; transit points are assigned via milestoning (last core visited) so every timestep has a well-defined state and the jump process is a proper core-set MSM (cf. Schuette/Noe MSM literature). Exact caption string repetition provides a clustering-free ground-truth layer for fixed-point detection. Depends on TASK-75 (density validation) for the core definition; blocked on the running experiment.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Exact-caption fixed-point detection implemented (string equality), yielding per-run absorption status and time-to-absorption without any clustering dependency
- [ ] #2 Core-set state assignment with milestoning produces a total symbol sequence for every run (no unassigned timesteps)
- [ ] #3 Per-network kinetic observables computed: dwell-time distributions (with exponential vs heavy-tailed characterisation), transition graphs, absorption times, and time-in-transit vs time-in-state balance
- [ ] #4 Clustering used for states is frozen on a defined corpus (not the growing global pool) and its assignments shown stable under subsampling
<!-- AC:END -->
