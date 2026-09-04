---
id: TASK-89
title: Decompose each loop step into deterministic drift and generator sampling noise
status: To Do
assignee: []
created_date: '2026-09-03 23:29'
updated_date: '2026-09-04 02:29'
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

THE OBSERVATION. With an IDENTICAL, untruncated prompt at a FIXED seed, changing Flux2Klein's max_sequence_length from 512 to 1024 --- which alters only the padding of the conditioning tensor and carries no semantic information --- moved the caption-space cosine between the two images to 0.896-0.988, mean 0.943. On the measured scale for that metric (1.000 for identical images, 0.876 for images of unrelated prompts), a zero-information perturbation produced images nearly as different as unrelated ones.

WHY IT MATTERS. The loop is a Markov chain on captions (random seed per text-to-image step, greedy captioner), and the 200-step baseline (analysis/long_horizon_baseline.py) shows it settles into a stationary regime with a persistent step-to-step distance of about 0.01-0.02. The question is what that step is made of. If it is mostly generator sampling noise, then the core-set Markov state model (TASK-76) must show its transitions exceed the noise, metastable-region identity becomes the whole kinetic result, and TASK-85's 28% step-size reduction is a statement about noise as a function of caption length rather than about semantics. RQ2's variance decomposition uses step-to-step drift as its response and inherits the same caveat. Establishing the decomposition first is what makes those claims defensible.

WHAT NOT TO COMPARE. Consecutive trajectory states differ by ONE draw of generator noise; two resamples of the same caption differ by TWO independent draws, so their distance is expected to be roughly double even with no dynamics at all. The earlier 'paradox' (consecutive cosine ~0.985 versus resample cosine ~0.943) compared these unlike quantities, and the 0.943 came from a padding perturbation rather than a seed change. The measurement below compares like with like.

THE MEASUREMENT. Per text-to-image model, take a fixed set of captions spanning the panel's length range. For each caption c, generate N images at N recorded seeds and caption each with one captioner, giving N next-captions. Embed everything. Then:
- drift: the displacement from c to the centroid of the N next-captions
- noise: the spread of the N next-captions around their centroid
- step: the mean distance from c to each next-caption, which is the quantity the trajectories measure
Report all three per model in the same units, and the noise/step share. Compare the step against the stationary step size in the 200-step baseline and the v2 pilot/panel. Note Flux2Klein runs at four distilled steps and may be the noisiest; the model dependence is itself a result since network identity is a panel factor.

Cheap to run: no new models, no new experiments; analysis/step_sweep.py already contains the generate-caption-embed-compare machinery and the seeded generation helper exists in panic_models.py.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Per text-to-image model: drift, noise and step measured on a fixed caption set over N recorded seeds, all in caption-embedding cosine distance, with the noise share of the step stated
- [ ] #2 Step compared like-for-like against the stationary step size in the 200-step baseline and the v2 pilot/panel, and the share of stationary movement attributable to sampling stated
- [ ] #3 Implications written up for TASK-85's step-size result, for RQ2's response variable, and for what the core-set MSM in TASK-76 must demonstrate about its transitions
- [ ] #4 max_sequence_length recorded as a fixed experiment parameter that perturbs generation independently of content, so it is never varied mid-programme
- [ ] #5 Ruler calibration: the noise share is reported against Qwen3Embed's resolution for these captions (distance between seed-resamples of one caption versus captions of unrelated prompts), so a step below the ruler's resolution is not read as dynamics
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Serves the paper's Null models section (see backlog/docs/research-programme.md): this IS the i.i.d.-resampling surrogate the skeleton pre-specifies. Also bears directly on RQ2 --- if the generator injects large step-level noise, that is evidence against Hintze et al.'s captioner-dominance, which the skeleton already flags as in tension with our SMC results. Should land BEFORE TASK-75/76, and before the long-horizon run (TASK-90), since it decides whether the transitions the Markov model would fit are signal.

Literature (2026-09-04 search). Toker et al., Padding Tone (NAACL 2025, arXiv:2501.06751): padding tokens are not inert in T2I text encoders and can act in the diffusion process itself, which is the mechanism behind the max_sequence_length observation; it is a genuine perturbation, not a bug. Huang et al. 2026 (arXiv:2606.01651): distillation FLATTENS seed sensitivity (seed-identification accuracy 94% for a multi-step teacher, 53-88% for distilled students), so the distilled models (Flux2Klein, ZImageTurbo) may be the LEAST seed-noisy, not the most; the direction is model-specific and must be measured. Decision rule for 'drift exceeds noise': Bland-Altman minimal detectable change, MDC = sqrt(2) x 1.96 x SEM with SEM from the seed-resample spread, so drift is called real only above the MDC (ISO 5725 repeatability/reproducibility is the same one-draw/two-draw accounting). Frank & Afli 2026, HTEB (arXiv:2605.28190): embedding cosine on long texts often scores paraphrase and substantive change alike, hence the ruler-calibration AC. Callaham et al. 2021 (Langevin regression): conditional mean and variance of the step are the drift and diffusion estimators; the Fokker-Planck finite-interval correction does not scale to embedding dimension and the loop is a discrete-time chain anyway, so use the raw conditional moments.
<!-- SECTION:NOTES:END -->
