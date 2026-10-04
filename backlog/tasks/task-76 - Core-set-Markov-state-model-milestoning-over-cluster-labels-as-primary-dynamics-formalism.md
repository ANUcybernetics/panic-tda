---
id: TASK-76
title: >-
  Markov state model over a complete partition of embedding space as primary
  dynamics formalism
status: To Do
assignee: []
created_date: '2026-07-10 00:50'
updated_date: '2026-10-04 04:07'
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

PIPELINE LANDED 2026-09-14 (Sungyeon's PR #1, reworked): analysis/msm_pipeline.py covers AC#1-#4 on any experiment.export_data parquet dump and computes AC#5's observables; the seed-resample surrogate waits on TASK-90's recorded seeds. Guards: a dwell verdict needs 20 complete residences (both trajectory ends censored), an escape time is resolved only on >= 10 observed crossings and carries a Bayesian 95% interval, every metastable set reports the number of trajectories its frames came from and is marked private below ten or when one run holds over half of them (set_occupancy, the detector for the old data's non-ergodic networks), the coarse-graining lag is --lag and recorded. The committed JSON is a plumbing run over the 25-state balanced_panel_5x5 export (--microstates 12 --sets 3) and is not a kinetic result. The partition is pooled by default -- one k-means over every cell's stationary frames in the export, each cell's transition matrix estimated on it, since a cell's own 3,000-4,000 frames support 60-80 microstates rather than the few hundred in the description; --partition per-cell keeps the old per-cell fit for comparison. Burn-in is one figure for the whole export, so every cell is cut at the same point. See backlog/docs/escape-time-resolvability.md. Prior on the timescales from the 5,000-step SMC runs in analysis/escape_time_prior.json.

FIRST LOOK AT THE TASK-90 PANEL 2026-10-04 (analysis/trajectory_mixing.py, export 01a09e21_parquet). msm_pipeline.py --all at its defaults (200 pooled microstates) is not a data swap: it finished two cells with 21/195 and 6/196 microstates connected, 28% and 90% of frames unassigned and all 24 escape times unresolved, then crashed on Flux2Dev + Moondream3 (deeptime: no strongly connected subset could be fit), which implied_timescales does not handle. Not fixed.

The direct check says why, and that no microstate count repairs it. In all 16 cells every run keeps to its own region for the 75 stationary text states: a frame's nearest neighbour in its cell is from its own run 89-100% of the time (chance 1.9%). Mean cosine distance between stationary frames is 0.07-0.16 within a run, 0.31-0.48 to the other run of the same prompt, 0.53-0.64 to runs of other prompts. With a per-cell partition of 5, 10, 20 or 40 microstates a run spends 95-98% of its frames in one microstate and changes label 0.3-0.6 times per 100 steps, so coarse states are shared but never crossed; at 80 a run splits across 2.2 microstates that only 2% of frames share with another prompt, so fine states are crossed but private. There is no resolution with both, which is what an MSM needs.

The motion is slow drift, not hopping. Within a run, distance between frames grows with their separation and has not levelled at 70 text states (0.042 at 1, 0.079 at 10, 0.175 at 70, mean over cells). The two runs of a prompt start 0.21 apart and reach 0.39 by states 125-149, still rising; runs of different prompts go from 0.61 to 0.59. So the step size plateaus but the chain is not stationary at 150 text states and has not forgotten its prompt. AC#2's outcome for this panel is that every cell is unresolved, and the observables worth computing are the drift ones (displacement against separation, twin divergence, prompt-memory decay) rather than set-to-set kinetics.
<!-- SECTION:NOTES:END -->
