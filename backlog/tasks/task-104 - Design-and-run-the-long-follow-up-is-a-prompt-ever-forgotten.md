---
id: TASK-104
title: 'Design and run the long follow-up: is a prompt ever forgotten?'
status: To Do
assignee: []
created_date: '2026-10-04 07:19'
labels:
  - experiment
  - paper
  - gpu
dependencies:
  - TASK-103
references:
  - backlog/docs/drift-and-memory.md
  - backlog/docs/research-programme.md
  - analysis/panel_audit.py
  - analysis/panel_regenerate.py
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-103 found that the panel (experiment 01a09e21, 150 text states per run) is too short to answer the question RQ1 now turns on. Prompt memory falls from 0.71 to 0.35 and has almost stopped falling, and the two runs of a prompt have nearly stopped separating, well short of the distance between runs of different prompts. At the rate of the last fifty states they would meet after about another 570 text states, and that rate has been falling, so telling a plateau from a slow climb needs runs of at least 700 text states (1,400 invocations). The same runs would show how long a run takes to cover its prompt's region: at a separation of seventy states the panel's runs are 23-59% of the way to the distance between twins. The old 5,000-invocation runs (analysis/long_run_drift.py) say displacement does get there, in 200 to more than 1,250 text states, but their prompts had no content, so they say nothing about memory. See backlog/docs/drift-and-memory.md and the horizon section of backlog/docs/research-programme.md.

WHAT IS ALREADY KNOWN ABOUT THE DESIGN. Fewer prompts and more runs per prompt than the panel: with two runs a prompt's centre is never observed, only inferred, and eight runs give it directly. Five prompts by eight runs keeps a cell at 40 runs in lockstep, which is the panel's batch shape; captioner output depends on batch size, so 40 keeps the captioners defined as they were. The prompts should span the range of memory the panel measured (prompts.by_prompt in analysis/drift_memory.json: 0.64 for a red apple down to 0.12 for a city turning into a forest). Cost per cell of 40 runs to 700 text states, from the panel's own step times: about 2.5 days with SD35Medium or ZImageTurbo, 1.5 with Flux2Klein, 15 with Flux2Dev. The GPU is shared at times, which cost the panel two retried steps and nothing else.

LAUNCH TRAPS LEFT BY THE PANEL. logs/long-run.id still holds 01a09e21, and bin/long-run resumes whatever id is in that file, so a new launch would exit at once reporting the old experiment complete. bin/panic-experiment.service and the installed copy in ~/.config/systemd/user both name the panel's config in ExecStart. TASK-90's notes record what else went wrong at its launch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The prompts, runs per prompt, networks and horizon are chosen and written down with the GPU-day cost and what each choice is for, and Ben has agreed the design before any GPU time is spent
- [ ] #2 The config is committed and the run is launched detached and resumable, with the experiment id and the expected completion date recorded here
- [ ] #3 The run completes and the audit that passed on the panel passes on it: every image, caption, seed and embedding checked, and a sample of steps regenerated from their stored inputs
<!-- AC:END -->
