---
id: TASK-76
title: >-
  Markov state model over a complete partition of embedding space as primary
  dynamics formalism
status: To Do
assignee: []
created_date: '2026-07-10 00:50'
updated_date: '2026-09-07 04:06'
labels:
  - analysis
  - paper
dependencies:
  - TASK-75
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Adopt symbolic dynamics as the primary formalism for trajectory analysis, replacing bag-of-points persistent homology as the headline method, using the standard MSM pipeline rather than density cores.

STATE DEFINITION. A fine k-means partition of the embedding space (a few hundred microstates on the unit sphere, fitted on a frozen corpus of stationary-regime captions) assigns every timestep a symbol. A transition matrix at a chosen lag time is estimated from the symbol sequences, and PCCA+ coarse-grains the microstates into metastable sets from the dynamics. The metastable regions are therefore defined kinetically, by slow exchange, not geometrically by density; that is what the scientific question asks for, and it needs no outlier story at all. An HMM over embeddings (Noe & Wu 2013) is the fallback if the fine partition is unstable.

EVoC is kept for naming and illustrating regions (medoid captions, example images) but does not define the states. The core-plus-transit design that preceded this is recorded in the notes and in backlog/docs/outlier-sparsity.md: TASK-75 showed EVoC outliers are dwell regions rather than transit, so milestoning would have credited a run's destination to a core it left twenty steps earlier.

NOISE FLOOR. A step the size of the generator's seed noise (TASK-89) is what one seed draw produces with no dynamics, so microstate flips near a boundary can be pure sampling. The lag time and the PCCA+ level are chosen so that transitions between metastable sets exceed that floor, tested against the seed-resample surrogate; TASK-90 re-measures the floor from its own trajectories.

OBSERVABLES per network: stationary distribution over metastable sets, dwell-time distributions (exponential vs heavy-tailed), transition graph, escape times, and implied timescales as a function of lag, whose convergence justifies the horizon. Depends on TASK-90's data.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Clustering used for states is frozen on a defined corpus (not the growing global pool) and its assignments shown stable under subsampling
- [ ] #2 Implied timescales computed as a function of lag time per network; convergence used to justify the horizon, and any cell whose slowest timescale does not converge within the trajectory length reported as unresolved
- [ ] #3 Exact caption repetition reported as a descriptive statistic (rate and dependence on caption length), not used as a state or absorption definition
- [ ] #4 A fine partition plus PCCA+ (or HMM fallback) assigns every timestep of every run a metastable-set label, with no unassigned timesteps and the lag time and number of sets recorded with the reason for each
- [ ] #5 Per-network kinetic observables computed: stationary distribution over sets, dwell-time distributions (exponential vs heavy-tailed), transition graph and escape times, each reported against the seed-resample surrogate so transitions are shown to exceed the noise floor
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
TASK-75 RESULT (2026-09-05, backlog/docs/outlier-sparsity.md): the core-plus-transit design in the description is no longer available. EVoC outliers are not sparse and outlier time is not transit: 75% of it is runs settling into the outlier region for good (median 19-step tails, plus 14% of runs never labelled), only 10% is passage between clusters, and the outlier share (26-45%) is a fixed point of the EVoC procedure rather than a data property (a second pass over the outliers leaves 39% unlabelled again). Milestoning would credit a run's destination to a core it left twenty steps earlier for nearly half of outlier time. Start instead from a state definition that assigns every point: PCCA+ on a fine partition or an HMM over embeddings (the fallbacks already named above), or EVoC forced to a complete partition via approx_n_clusters/base_n_clusters, in each case with AC#3's subsampling stability check. Dwell regions, which is what the outlier set is full of, are a candidate trajectory-aware core definition.
<!-- SECTION:NOTES:END -->
