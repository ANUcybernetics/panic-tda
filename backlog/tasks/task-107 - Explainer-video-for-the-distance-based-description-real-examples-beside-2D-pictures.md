---
id: TASK-107
title: >-
  Explainer video for the distance-based description: real examples beside 2D
  pictures
status: To Do
assignee: []
created_date: '2026-10-04 23:15'
labels:
  - video
dependencies: []
references:
  - backlog/docs/drift-and-memory.md
  - backlog/docs/long-follow-up-design.md
  - analysis/drift_memory.py
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The description of the loop that replaced the Markov state model (backlog/docs/drift-and-memory.md) rests on four distances between captions. Few people have an intuition for what they mean, or for what the long follow-up's result will say. Ben wants an explainer video that builds one: real examples from the experiments, and beside each a picture of the same thing in two dimensions. It would be made with the ben:styled-video skill, where the key idea and the script are Ben's and the craft is Claude's. Nothing has been built.

WHAT IS UNDECIDED, and Ben's to decide before anything is built. The key idea, in one sentence. Two candidates so far: which two captions you compare decides what you are measuring (for the distances), and the long run asks which clouds survive time (for the results). They may be two short videos. Who it is for and how long. Where the look comes from, and whether the video lives in this repo or the paper's.

A REAL EXAMPLE THAT CARRIES IT. The red apple in the panel (experiment 01a09e21), with its twin and a run of the city prompt as the stranger, ends two ways. In SD35Medium + Moondream3 the twins are 0.04 apart at text state 1, 0.15 at 75 and 0.10 at 150: both still apples, both drifted to a bright yellow background, with the city run 0.49 away as orange maple leaves. In ZImageTurbo + Moondream3 they are 0.04, 0.09 and 0.30 apart: one ends as a grid of sixteen red apples and the other as a grid of 36 brown eggs, and the city run, nine projector screens in a grid, is 0.36 away. Moondream3's captions are about 50 words, short enough to read on screen.

A POSSIBLE SHAPE, for Ben to rewrite. Each beat pairs a real moment with its 2D picture. One step: a caption redrawn at a new seed, and a point jittering in a small ball. One run: the ball drifts, and the displacement curve draws itself as the path grows. The twin: two paths from one start. A stranger: the size of the world. Memory: clouds inside clouds, and how much of the spread the prompt accounts for. Outcomes: the same picture run forward four ways (clouds merge, clouds hold and each run fills its own, each run keeps to a corner, everything creeps).

WHAT A 2D PICTURE GETS WRONG. In 256 dimensions independent displacements are nearly at right angles, which is why squared distances add. On a plane they are not, so the 2D versions have to be drawn to keep the relationships and cannot be projections of the data.

CONSTRAINTS. The project has no video style guide; the only one on the machine is LLMs Unplugged's. The analysis figures already give colours a meaning (blue for runs of one prompt, orange for different prompts, aqua for one run against itself, in analysis/drift_memory.py), and type has to come from a named source. The GPU is taken by experiment 01a10613 until about 25 October 2026, so drafts would be rendered without it, which is untested, and a 4K master waits. Eight apple runs in SD35Medium + Moondream3 are due on 11 October and would let the cloud of a prompt be shown from data. The outcome beat cannot be final before TASK-105 has read the follow-up.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Ben has set the key idea, who the video is for, its length, where its look comes from and where it lives, including whether it is one video or two
- [ ] #2 The project has a video style guide that CLAUDE.md points to, with its palette and type taken from named sources
- [ ] #3 The script is Ben's, with a visual direction and approved reads for every stretch of picture
- [ ] #4 Each distance the analysis uses is shown on a real example from the experiments and as a 2D picture, and the script says what a plane cannot show
- [ ] #5 Ben has watched a 1080p draft whose outcome section rests on the follow-up's measured result
- [ ] #6 A 4K master is rendered for the cut that is kept
<!-- AC:END -->
