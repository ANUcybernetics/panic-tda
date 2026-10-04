---
id: TASK-103
title: 'Describe each network''s dynamics as seed noise, slow wander and prompt memory'
status: In Progress
assignee:
  - '@claude'
created_date: '2026-10-04 05:11'
updated_date: '2026-10-04 06:25'
labels:
  - analysis
  - paper
dependencies:
  - TASK-90
references:
  - analysis/trajectory_mixing.py
  - analysis/panel_audit.py
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-90's panel (experiment 01a09e21: 16 networks, 20 prompts, 2 runs per prompt, 150 text states per run) does not have the structure a Markov state model needs. In every cell each run keeps to its own region of embedding space for the whole trajectory, so no partition has states that are both shared between runs and crossed by them, and no escape time is resolvable (analysis/trajectory_mixing.py; the rationale is in TASK-76, now archived). What the runs do instead is move slowly away from where their prompt put them. Ben's brief for the replacement (2026-10-04): the Markov state model was only there to speak to Hintze et al. (Patterns 2025) and is not required; the purpose is to understand how different text-to-image and image-to-text network topologies behave, with the simplest and clearest mathematical formalism that explains what is seen when the runs are watched.

THE FORMALISM. Treat a caption's embedding as a slowly moving theme plus fresh noise from that step's diffusion seed, and describe each network by quantities read straight off distances between embeddings, with no clustering, no partition and no lag to choose. Cosine distance between unit vectors is half the squared Euclidean distance, so it adds like a variance and every quantity is a mean distance or a difference of two.

- Noise: the part of a step that the next step takes back. One seed draw moves the caption without moving what the run is about.
- Wander: the part that accumulates. Displacement against separation in time, whether it levels off or keeps growing, and whether it slows as the run ages.
- Memory: how much of where a run sits is still set by its prompt. The distance between the two runs of a prompt against the distance between runs of different prompts, over time.

Two things the description leaves out are counted separately: runs that freeze on an exactly repeating caption, and runs that collapse into a flat image (panel_audit.py found three, each a gradual slide the captions track).

WHAT IT HAS TO ANSWER. RQ1 becomes: does a run forget its prompt, how fast, and does it settle. RQ2 becomes: which of the three quantities the generator sets and which the captioner sets. Hintze's convergence claim is tested by whether runs from different prompts approach each other. TASK-89 measured the noise on captions taken from pilot images and said the panel must re-measure it from its own trajectories; this task does.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every cell has its displacement curve (mean distance between two states of one run against their separation in time, at every lag) from a tracked script with tracked output, and how the curve depends on how far into the run the pair starts is reported
- [ ] #2 Each cell's step is split into the share the next step takes back and the share that accumulates, and the split is checked against a direct measurement: a sample of steps regenerated from the stored caption with a different seed
- [x] #3 Each cell's accumulating displacement is classified as levelling off, still growing or undecided at 150 text states by a stated rule, with an interval from resampling prompts
- [x] #4 Prompt memory (distance between the two runs of a prompt against distance between runs of different prompts, noise removed) is reported over time for every cell with an interval, and whether runs from different prompts approach each other is stated as the test of the convergence claim
- [x] #5 The generator's and the captioner's contribution to noise, wander and memory is stated across the 4x4 panel, with what sixteen cells can and cannot support
- [x] #6 Frozen runs (exactly repeating captions) and collapsed runs (flat images) are counted per cell, and each headline number says whether it includes them
- [x] #7 A write-up in backlog/docs gives the result in plain terms with its tables and figures, says which features of the runs the description explains and which it does not, and names what a longer or larger experiment would need to measure
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Explore the panel export to settle which measurements are robust (done in scratch: steps are ~80% taken back by the next step in every cell, wander slows with age, runs within a cell are very unequal).
2. analysis/drift_memory.py: displacement curves, noise split, ageing, levelling-off rule, prompt memory, distance ladder, generator and captioner contributions, frozen runs; JSON plus figures.
3. analysis/seed_resample.py (GPU): one late step per cell regenerated with new seeds through the production path, to check the noise split directly.
4. backlog/docs write-up, then check the acceptance criteria against the outputs.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
DONE SO FAR 2026-10-04. analysis/drift_memory.py reads every network off distances between embeddings; results in its JSON, figures in analysis/drift_memory/, write-up in backlog/docs/drift-and-memory.md. Headlines: seed noise is 0.034 of a 0.042 step (80%, 73-87% by network); a run's own offset is 0.36 by the end and nearly flat; what the prompt fixes is 0.19, so memory falls from 0.71 to 0.35 (0.08-0.48 by network, 0.64-0.12 by prompt); runs of different prompts approach each other by 4%; within-run displacement is still growing in all sixteen networks and is 23-59% of the way to the twin distance; the captioner sets the noise (75% of its variation across the panel) and the generator sets what a step keeps (78%). analysis/long_run_drift.py takes the same measure on the April 2025 5,000-invocation runs: displacement there does reach the distance between independent runs, in 200 to more than 1,250 text states. The data check that came first is analysis/panel_audit.py and panel_regenerate.py, written up in backlog/docs/panel-audit.md.

OUTSTANDING: AC#2's direct check. analysis/seed_resample.py was launched 2026-10-04 16:26 AEDT and was still running when this was written. It redraws image step 200 of every cell at new seeds (all 40 runs for SD35Medium, ZImageTurbo and Flux2Klein, 12 for Flux2Dev) and caches the new images under 01a09e21_parquet/seed_resample/, so a rerun skips generation. When analysis/seed_resample.json exists:
1. Run ./analysis/drift_memory.py 01a09e21_parquet. It reads that JSON and writes a seed_resample section: the pooled distance between a redrawn caption and the stored one, against the stored run's one-step and two-step distances from the same captions, with intervals.
2. The prediction is that the resampled distance matches the two-step distance (the interval on resampled_less_two_steps includes zero), which makes noise_direct equal noise_chain. Replace the paragraph in drift-and-memory.md that begins 'TASK-89 reached the same reading' with the result either way, and adjust the four-fifths wording there and in research-programme.md if it does not match.
3. Commit seed_resample.json, the regenerated drift_memory.json and the doc, check AC#2 and set the task Done.
If the JSON is missing the job died: rerun the command in the script's docstring. Generation resumes from the cache; captioning and embedding take about ten minutes.
<!-- SECTION:NOTES:END -->
