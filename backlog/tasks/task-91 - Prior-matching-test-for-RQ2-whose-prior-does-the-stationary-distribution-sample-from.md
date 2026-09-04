---
id: TASK-91
title: >-
  Prior-matching test for RQ2: whose prior does the stationary distribution
  sample from?
status: To Do
assignee: []
created_date: '2026-09-04 02:28'
labels:
  - analysis
  - paper
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Iterated learning theory (Griffiths & Kalish 2007, Cognitive Science 31) shows that a chain of Bayesian samplers is a Gibbs sampler whose stationary distribution is the learner's PRIOR, independent of where the chain started. Applied here, the sharpest form of RQ2 is not a variance share but: does the loop's stationary caption distribution look like the captioner's prior, the generator's prior, or something the pair makes together?

THE TEST. Build each captioner's reference distribution by captioning a fixed, broad image set (existing panel images from other networks, or images from a set of generic prompts) and embedding the captions. Build each generator's reference distribution by captioning its images from a null or generic prompt with a fixed captioner. Then compare the stationary regime of each network (steps past the plateau, from TASK-90 data) against the candidates: distance between distributions in embedding space (MMD or energy distance on Qwen3Embed), and whether the metastable-region medoids (TASK-76) are nearest to one captioner's reference set regardless of generator.

PREDICTION IF HINTZE HOLDS. Networks sharing a captioner have stationary distributions closer to each other, and to that captioner's reference set, than networks sharing a generator. If the generator matters at current scale, the reverse or an interaction.

Cheap: needs captioning of an existing image set per captioner, no new trajectories, and the TASK-90 data it compares against. Runs after TASK-76 so the same clustering and plateau definition are used. This is a candidate secondary result for Results II, not a replacement for the Hintze-matched variance decomposition.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Reference caption distributions built per captioner (fixed image set) and per generator (fixed captioner), embedded with Qwen3Embed
- [ ] #2 Stationary-regime distribution of every network compared against captioner and generator reference sets with a stated distribution distance
- [ ] #3 Result stated as which factor the stationary distribution tracks, with the iterated-learning framing written up for Results II
<!-- AC:END -->
