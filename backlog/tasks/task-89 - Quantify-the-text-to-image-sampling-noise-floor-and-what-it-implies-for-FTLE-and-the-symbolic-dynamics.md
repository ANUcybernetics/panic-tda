---
id: TASK-89
title: >-
  Quantify the text-to-image sampling noise floor, and what it implies for FTLE
  and the symbolic dynamics
status: To Do
assignee: []
created_date: '2026-09-03 23:29'
labels:
  - analysis
  - paper
  - dynamics
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found 2026-09-04 while measuring what the 512-token encoder ceiling costs (analysis/prompt_tail.py, backlog/docs/caption-length-by-i2t-model.md).

THE OBSERVATION. With an IDENTICAL, untruncated prompt at a FIXED seed, changing Flux2Klein's max_sequence_length from 512 to 1024 --- which alters only the padding of the conditioning tensor and carries no semantic information whatsoever --- moved the caption-space cosine between the two images to 0.896-0.988, mean 0.943. On the measured scale for that metric (1.000 for identical images, 0.876 for images of unrelated prompts), a zero-information perturbation produced images nearly as different as unrelated ones.

WHY IT MATTERS. The whole project measures how information flows through a recursive loop, and several headline quantities are differences in caption-embedding space. If a single text-to-image sampling step injects variation of that magnitude regardless of caption content, then we need a noise floor before any of those differences can be interpreted:

- FTLE. Runs sharing a network and prompt differ only by generation randomness, so the Lyapunov exponent is measuring divergence driven by exactly this stochasticity. That is arguably the intended measurement, but without a noise floor we cannot say whether an exponent reflects semantic sensitivity or the sampler's variance. The pilot's fits were weak (median r-squared 0.46, some exponents negative), which a large noise floor would explain.
- The pilot's central result. Complete captions gave 28% lower step-to-step distance than truncated ones (TASK-85). If sampling noise dominates the step, that finding is about how much noise the text-to-image step injects as a function of caption length, not about semantic dynamics --- a materially different claim for the paper.
- Symbolic dynamics (TASK-76). A core-set Markov model over cluster labels is modelling transitions; if transitions are largely resampling noise, the model is fitting noise.

THE APPARENT PARADOX WORTH CHASING. Observed step-to-step cosine within real trajectories is about 0.988 (pilot) and 0.984 (panel) --- i.e. CONSECUTIVE states are more similar than two independent samples from the same caption appear to be (0.943). Either the loop has genuine attractor-like structure that keeps it more stable than naive resampling would suggest, which is a positive and publishable finding, or the two measurements are not comparable (different models, six captions, one four-step distilled model). Resolving that is the point of this task.

PROPOSED MEASUREMENT. (1) Per text-to-image model, take a fixed set of captions and generate N images each at N different seeds; caption them all with one captioner and compute the pairwise cosine distribution. That is the per-step sampling noise floor, per model. (2) Compare it against the observed step-to-step distances in the pilot and panel trajectories. (3) Report the ratio: what share of observed trajectory movement is attributable to sampling noise. Note Flux2Klein runs at four distilled steps and may be the noisiest of the five --- the floor is likely model-dependent, and that dependence is itself a result, since network identity is a panel factor.

Cheap to run: no new models, no new experiments, and analysis/step_sweep.py already contains the generate-caption-embed-compare machinery.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Per-model sampling noise floor measured: distribution of pairwise caption-space cosine over N seeds from a fixed caption set, for all five panel text-to-image models
- [ ] #2 Noise floor compared against observed step-to-step distances in the pilot (01a060b4) and panel (019f3645), with the share of trajectory movement attributable to sampling stated
- [ ] #3 The apparent paradox resolved: whether consecutive trajectory states really are more stable than independent resampling, or the two measurements are not comparable
- [ ] #4 Implications written up for FTLE interpretation (including the weak r-squared), for TASK-85's step-size result, and for the core-set MSM in TASK-76
- [ ] #5 max_sequence_length recorded as a fixed experiment parameter that perturbs generation independently of content, so it is never varied mid-programme
<!-- AC:END -->
