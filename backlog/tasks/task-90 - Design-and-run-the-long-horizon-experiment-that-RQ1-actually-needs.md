---
id: TASK-90
title: Design and run the long-horizon experiment that RQ1 actually needs
status: To Do
assignee: []
created_date: '2026-09-04 01:00'
labels:
  - experiment
  - paper
  - gpu
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The paper skeleton's first stated contribution is 1000-iteration trajectories, and RQ1 is framed as answering Hintze et al. 'at 10x the horizon' --- their attractors are k-means on ENDPOINT embeddings at t=100, which assumes convergence rather than demonstrating it. The horizon is the gap.

The longest trajectory in the database is 200 steps and every panel config is max_length 50, so the headline dataset does not exist. Nothing in the backlog represented building it until now. See backlog/docs/research-programme.md.

THE CONSTRAINT. This is not a matter of raising max_length. Using measured per-item times (the model predicts 14.9 days for the panel that actually took ~17, so it is about right):

| scenario | GPU-days |
|---|---|
| current panel: 5x5, 20 prompts, 4 runs, 50 steps | 14.9 |
| the same at 1000 steps | 298 |
| 1000 steps, 5 prompts, 2 runs | 37 |
| 250 steps, full 5x5, 20 prompts, 4 runs | 74 |
| 1000 steps, fast T2I only (3x5), 20 prompts, 4 runs | 55 |
| 1000 steps, fast T2I only, 5 prompts, 4 runs | 13.8 |
| 1000 steps, fast T2I only, 10 prompts, 2 runs | 13.8 |

Flux2Dev and GLMImage are 86% of all text-to-image time (57.7 and 42.4 s/item against 4.1-6.5 for SD35Medium, ZImageTurbo and Flux2Klein). Dropping those two buys roughly seven times the horizon for the same budget.

So the full model factorial and the long horizon are mutually exclusive at any sane cost, and that is exactly the RQ1/RQ2 tension: RQ2's attribution wants the factorial, RQ1's kinetics wants the horizon. The design must choose, and the choice should be made deliberately and stated in the paper rather than falling out of what happened to be affordable.

Options worth costing properly, not an exhaustive list: one long-horizon run on the fast three text-to-image models with fewer prompts, plus the existing or a repeated 50-step 5x5 for RQ2; or a single compromise horizon (250-400) across the full factorial; or a staged design where a cheap wide run identifies which cells are still moving at t=50 and only those go long.

Also settle the DECISION markers in the paper skeleton that gate a new run: per-invocation T2I seeding (recorded, so within-condition variation is attributable and runs reproducible) and captioner sampling policy (greedy, which is current behaviour and gives a deterministic captioner composed with a stochastic generator, versus temperature as a design factor for comparability with Hintze et al.).

Prerequisites already met: the v2 lineup is integrated, pinned and GPU-green (TASK-87); step counts are measured (TASK-83); step-level CUDA retry means a stochastic fault no longer kills a multi-week run (TASK-79). Prerequisite NOT met: TASK-89, the sampling noise floor, should land first --- if a single generation step injects variation comparable to the trajectory movement being measured, the horizon needed to see real kinetics changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Horizon, network set, prompt count and runs-per-condition chosen, with the GPU-day cost stated and the RQ1-versus-RQ2 trade-off explicitly justified in a form that can go into the paper's methods
- [ ] #2 T2I seeding policy and captioner sampling policy decided and recorded, resolving those DECISION markers in the paper skeleton
- [ ] #3 Config committed (a versioned file, not an edit to an existing one) and a short pilot run at the chosen horizon on one network to confirm per-step cost and that nothing degrades over a long trajectory
- [ ] #4 Run launched detached with a resumable config, and the expected completion date recorded
- [ ] #5 TASK-89 landed first, and its noise floor used to sanity-check that the chosen horizon can resolve the kinetics being claimed
<!-- AC:END -->
