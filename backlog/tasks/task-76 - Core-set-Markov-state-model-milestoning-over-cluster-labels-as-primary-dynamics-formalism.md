---
id: TASK-76
title: >-
  Core-set Markov state model (milestoning) over cluster labels as primary
  dynamics formalism
status: To Do
assignee: []
created_date: '2026-07-10 00:50'
updated_date: '2026-09-05 05:36'
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
- [ ] #1 Core-set state assignment with milestoning produces a total symbol sequence for every run (no unassigned timesteps)
- [ ] #2 Per-network kinetic observables computed: dwell-time distributions (with exponential vs heavy-tailed characterisation), transition graphs, absorption times, and time-in-transit vs time-in-state balance
- [ ] #3 Clustering used for states is frozen on a defined corpus (not the growing global pool) and its assignments shown stable under subsampling
- [ ] #4 Implied timescales computed as a function of lag time per network; convergence used to justify the horizon, and any cell whose slowest timescale does not converge within the trajectory length reported as unresolved
- [ ] #5 Exact caption repetition reported as a descriptive statistic (rate and dependence on caption length), not used as a state or absorption definition
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
TASK-75 RESULT (2026-09-05, backlog/docs/outlier-sparsity.md): the core-plus-transit design in the description is no longer available. EVoC outliers are not sparse and outlier time is not transit: 75% of it is runs settling into the outlier region for good (median 19-step tails, plus 14% of runs never labelled), only 10% is passage between clusters, and the outlier share (26-45%) is a fixed point of the EVoC procedure rather than a data property (a second pass over the outliers leaves 39% unlabelled again). Milestoning would credit a run's destination to a core it left twenty steps earlier for nearly half of outlier time. Start instead from a state definition that assigns every point: PCCA+ on a fine partition or an HMM over embeddings (the fallbacks already named above), or EVoC forced to a complete partition via approx_n_clusters/base_n_clusters, in each case with AC#3's subsampling stability check. Dwell regions, which is what the outlier set is full of, are a candidate trajectory-aware core definition.
<!-- SECTION:NOTES:END -->
