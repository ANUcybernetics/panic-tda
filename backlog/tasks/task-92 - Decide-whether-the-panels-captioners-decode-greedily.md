---
id: TASK-92
title: Decide whether the panel's captioners decode greedily
status: Done
assignee: []
created_date: '2026-09-04 06:37'
updated_date: '2026-09-04 23:07'
labels:
  - analysis
  - paper
  - dynamics
  - instrument
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found 2026-09-04 while checking TASK-89's premise. The research programme
describes the loop as a Markov chain with a random text-to-image seed and a
GREEDY captioner, which is what makes the spread among a caption's successors
attributable to the generator. Three of the five v2 captioners are not greedy:
Qwen25VL, Qwen3VL and JoyCaption inherit do_sample=True from their shipped
generation_config.json, because _invoke_qwen_vl* and _invoke_chat_template*
never passed do_sample. Only Moondream3 (temperature 0.0) and Gemma4
(do_sample=False, passed explicitly) were ever greedy.

Qwen25VL is the trap: temperature 1e-06 reads as effectively greedy, but with
repetition_penalty 1.05 under bfloat16 it still produced eight distinct captions
from one image.

HOW BIG IT IS (analysis/captioner_noise.py, one fixed image captioned eight
times, four images, Qwen3Embed at 256 dimensions, cosine distance about the
centroid):

Moondream3 0.0000 Gemma4 0.0000 Qwen25VL 0.0110 Qwen3VL 0.0233 JoyCaption 0.0304

Against a stationary step-to-step distance of 0.0116 (natural-length pilot) to
0.0162 (truncated panel), the captioner's own sampling is the same size as the
entire step for Qwen25VL and roughly two to three times it for Qwen3VL and
JoyCaption. For those three networks the loop's step-level motion could be
mostly the captioner rolling dice, before the generator contributes anything.

FORCING GREEDY IS SAFE. The same script captions each image twice under forced
greedy: all five captioners are then deterministic, no captioner degenerates
(max repeated-5-gram share 0.016 for JoyCaption, 0.010 for Gemma4, 0.000 for the
rest), and median length barely moves (Qwen25VL 102 to 88 words, JoyCaption 182
to 168, Qwen3VL 250 to 246, Gemma4 237 to 234).

WHAT IS ALREADY DONE. panic_models.set_i2t_greedy() switches it; the default is
unchanged pending this decision. TASK-89's measurement runs its captioners
greedy regardless, so its noise term is the generator's alone.

THE DECISION. Greedy makes the text-to-image seed the loop's only source of
randomness, which is the design the programme states and the one that makes
RQ2's captioner effect a statement about descriptive style rather than about
each model's shipped temperature. The alternative is to keep sampling and carry
a three-way decomposition (drift, generator noise, captioner noise) through both
research questions. Settle it before TASK-90 commits GPU-weeks, and record which
way in the paper's methods.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Decision recorded --- greedy everywhere, or sampling kept as a measured
      second noise source --- with the reasoning and the numbers above
- [x] #2 panic_models default matches the decision, and set_i2t_greedy is either
      removed or documented as the switch
- [x] #3 Captioner sampling noise stated per captioner alongside TASK-89's
      generator noise, in the same units, so the two are comparable
- [x] #4 If greedy: a quality check over more than four images confirms no
      captioner degenerates (repetition, length, truncation) before TASK-90
- [x] #5 The programme's 'greedy captioner' claim and the paper's methods say
      which decoding the reported data used
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Greedy everywhere; recorded as decision-02.

Sampling noise per captioner, against TASK-89's generator noise, in the same
units (half the mean squared displacement about a centroid, Qwen3Embed at 256
dims, both through the corrected last-token pooling path):

  Moondream3 0.0000  Gemma4 0.0000  Qwen25VL 0.0110  Qwen3VL 0.0233  JoyCaption 0.0304
  Flux2Klein 0.0311  ZImageTurbo 0.0306  Flux2Dev 0.0341  SD35Medium 0.0446

Against a whole stationary step of 0.062-0.083, two captioners were rolling
dice worth as much as the generator. Greedy chosen over carrying a three-way
decomposition because shipped temperature is a packaging choice by five
vendors, not a model property: under sampling, RQ2's captioner effect would
confound descriptive style with whatever each release happened to ship.

_I2T_FORCE_GREEDY now defaults True; set_i2t_greedy/1 kept and documented as
the analysis-only switch. Guarded by a non-GPU test in engine_test.exs.

AC#4, analysis/captioner_greedy_quality.py over 24 images (4 v2 generators, 20
prompts, depths 8/24/48), two greedy passes plus one sampled: all five
captioners deterministic, no stubs, no back-to-back phrase repetition anywhere,
every caption ends on terminal punctuation so none hit the 1024-token ceiling.
JoyCaption's greedy repeat-5-gram share (0.065 vs 0.022 sampled) is content --
a 3x3 grid of near-identical rose bouquets enumerated row by row -- not
degeneracy. Results in analysis/captioner_greedy_quality.json.

Programme and the paper's methods note both say which decoding produced which
data: every dataset before 2026-09-05 samples for three of the five captioners.
<!-- SECTION:NOTES:END -->
