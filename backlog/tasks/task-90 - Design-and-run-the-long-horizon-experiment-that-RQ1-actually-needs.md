---
id: TASK-90
title: Design and run the uniform 250-300 step factorial that both RQs need
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-04 01:00'
updated_date: '2026-09-05 13:13'
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

PRE-LAUNCH ENGINEERING REVIEW 2026-09-05 (wall clock). Where the time goes: the July panel's database timeline puts 390.7 of 419.3 wall-clock hours inside model calls; swaps and inserts between steps were 1-4 s per step (about 1 h in total; the other 27.6 h was a GLMImage anomaly, no longer in the lineup). So the only levers that matter are inside the invocations, and Flux2Dev is 69% of them.

1. Allocator flag (the headline). setup() sets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, inherited from the Feb 2026 OOM-fix commit (494ab2b) and never measured on its own. With Flux2Dev's sequential CPU offload it costs a third of the model's time. Measured today, batch of 4 at 12 steps, 1024 px, same seeds:
   - standalone script: 43.6 s/item without the flag, 59.5 with it
   - mix gpu.bench Flux2Dev --batch-sizes 4 --n 4: 57.6 s/item with (current code), 43.4 without; batched-vs-serial parity identical (mean 5.78, max 12.58) in both
   Removing the line saves about 14 s per Flux2Dev image, roughly 4.9 GPU-days off the 33-day panel. Effect on the other three generators and the captioners not yet measured (they are resident, so likely smaller). Fragmentation is what the flag guards against; the panel's batch shapes are fixed within a cell, each cell starts from unload_all_models, and Retry restarts the interpreter on a sticky CUDA error, so the safety net already exists. One-line change; validate with mix gpu.bench on all four generators and a GPU smoke subset before launch.

2. Captioner batch cap. _I2T_MAX_BATCH is 8; a panel cell captions 40 per step. Measured on 40 pilot images, s/caption at cap 8 / 20 / 40: Qwen3VL 1.99 / 1.26 / 1.03 (peak 19.0 GiB at 40), Gemma4 1.01 / 0.53 / 0.37 (17.8), JoyCaption 1.14 / 0.62 / 0.46 (13.4), Qwen25VL 0.84 / 0.62 / 0.54 (17.6). Moondream3 captions serially inside its batch path (1.6 s/caption) and is unaffected. Worth about 0.7 GPU-days. Caveat: greedy captions change with batch composition (8-vs-40 identical for 1, 5, 4 and 18 of 40 respectively), so the cap is part of the captioner's effective definition; it already is at 8, but whatever value launches should stay fixed for the whole panel and be recorded in methods.

3. Image transport. Python returns each image as lossless WEBP (335 ms per 1024 px image) and Elixir re-encodes to AVIF serially (156 ms); about 0.5 s per image inside the measured invocation time, 20 s per 40-image step, roughly 0.7 GPU-days over the panel. PNG at compress_level=1 is 36 ms for the hop and the AVIF encodes parallelise cleanly (8 in 363 ms). Lossless, mechanical.

Measured and rejected as not worth the fragility: diffusers group offloading for Flux2Dev (54.7 s/item vs 57.7, with a GPU memory warning); keeping half the transformer resident via a device map (41.5 vs 43.6 s/item at the same allocator setting, at 39.5 GiB peak); Moondream3 compile() (no-op through the HF wrapper, would break under per-step CPU/GPU swapping anyway); keeping both models resident to skip swaps (about 0.5 GPU-days at most, needs a memory heuristic). Scripts and logs in the session scratchpad only; the numbers above are the record.
<!-- SECTION:NOTES:END -->
