---
id: TASK-90
title: Design and run the uniform 250-300 step factorial that both RQs need
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-04 01:00'
updated_date: '2026-09-05 11:59'
labels:
  - experiment
  - paper
  - gpu
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
RQ1 asks whether the loop reaches a stationary regime, how many metastable regions it has, and what the escape times between them are. Hintze et al. define attractors by k-means on ENDPOINT embeddings at t=100, which assumes convergence rather than demonstrating it. The gap is a trajectory-based definition over a horizon that resolves the slow timescales. See backlog/docs/research-programme.md.

WHAT THE EXISTING DATA SAYS (analysis/long_horizon_baseline.py, four 200-step experiments from Feb/Mar, old lineup, truncated captions, so design evidence only). Step-to-step distance and drift from the initial caption both plateau by step 100-150 to a persistent nonzero level. Exact caption repetition is not absorption: runs leave the repeated string immediately, and repetition tracks caption length (38/40 runs for a 23-word captioner, 0/32 for a 100-word one). So the loop is a Markov chain with a stationary regime, and the horizon question is whether there are metastable regions with escape times longer than the trajectories, not whether motion stops.

THE DESIGN. One uniform factorial (the v2 5x5 panel), 250-300 steps, 20 prompts, random recorded seed per text-to-image invocation, greedy captioner. Many independent trajectories past burn-in are the standard MSM input. The horizon is justified by implied-timescale convergence with lag time, which TASK-76 needs anyway; cells whose slowest timescale does not converge within the trajectory are reported as unresolved rather than extrapolated. No claim of a fixed 1000-iteration horizon.

COST. Measured per-item times (the model predicted 14.9 days for the panel that took ~17): 250 steps at 4 runs/prompt is 74 GPU-days, 300 steps is 89, 300 steps at 2 runs/prompt is 45. Flux2Dev and GLMImage are 86% of text-to-image time. Runs per prompt is the cheaper lever than horizon once past the plateau.

PREREQUISITES. Seed recording needs a seed attribute on Invocation (Ash migration), the seed passed through to the Python invoke path (currently generator=None), and a test that a stored seed regenerates the image. TASK-89 must land first, so the chosen horizon can be checked against the drift/noise decomposition. Already met: v2 lineup pinned and GPU-green (TASK-87), step counts measured (TASK-83), step-level CUDA retry (TASK-79).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Per-invocation text-to-image seed generated, stored on Invocation and passed to the generator, with a test that a stored seed regenerates the image; captioner stays greedy and both policies are recorded in the paper's methods
- [x] #2 TASK-89 landed first, and its noise share used to sanity-check that the stationary step size at the chosen horizon is resolvable
- [x] #3 Horizon (250-300), runs per prompt and prompt count chosen, with the GPU-day cost and the implied-timescale justification written in a form that can go into methods
- [x] #4 Config committed as a versioned file and a short pilot at the chosen horizon on one fast network confirms per-step cost and that nothing degrades over the trajectory
- [ ] #5 Run launched detached with a resumable config, and the expected completion date recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. AC#1: seed recording landed and GPU-verified (TASK-93); greedy captioner enforced (decision-02); both recorded in body.typ methods notes (2026-09-04/05 entries).
2. AC#2: TASK-89 landed; the settled step is at the noise floor at any horizon past the plateau, so the horizon is justified by state-level kinetics, not step size. Written into backlog/docs/long-horizon-design.md.
3. AC#3: 300 steps, 20 prompts, runs per prompt with the cost table in the design doc; config committed as config/long_horizon_panel_4x5_300.json at num_runs 4.
4. AC#4: pilot config/long_horizon_pilot_flux2klein_moondream3.json (4 prompts x 1 run x 300 steps) launched 2026-09-05 as experiment 01a0708e; checked by analysis/long_horizon_pilot.py.
5. AC#5: launch after Ben chooses runs per prompt (2 = 33 GPU-days, 4 = 67).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
PILOT DONE 2026-09-05 (AC#4): experiment 01a0708e, Flux2Klein + Moondream3, 4 prompts x 1 run x 300 steps, 2 h 05 min wall clock. Per-item times match the cost table (Flux2Klein 4.7 s vs 4.1, Moondream3 2.3 s vs 2.4); the rest of the wall clock is the 11 s model swap on every step, which a full panel cell amortises over 40-80 items (3-5%). Nothing degrades over 300 steps: caption length flat at 44-48 words, zero exact repeats in 600 captions, step distance 0.03-0.05 with no trend; drift from t0 keeps growing while step size does not, the stationary-chain-on-a-large-space signature. Details and table in backlog/docs/long-horizon-design.md, numbers in analysis/long_horizon_pilot.json.

AC#3: 300 steps, 20 prompts, 20 cells; cost by runs per prompt (with the 17/14.9 overhead): 1 run 16.7 GPU-days, 2 runs 33.4, 3 runs 50.1, 4 runs 66.8. Runs per prompt settled at 2 on 2026-09-05 (33.4 GPU-days); config/long_horizon_panel_4x5_300.json has num_runs 2. Only the launch (AC#5) remains.
<!-- SECTION:NOTES:END -->
