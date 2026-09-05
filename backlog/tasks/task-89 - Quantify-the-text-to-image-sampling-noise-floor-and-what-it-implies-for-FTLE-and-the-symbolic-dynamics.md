---
id: TASK-89
title: Decompose each loop step into deterministic drift and generator sampling noise
status: Done
assignee: []
created_date: '2026-09-03 23:29'
updated_date: '2026-09-05 05:05'
labels:
  - analysis
  - paper
  - dynamics
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found 2026-09-04 while measuring what the 512-token encoder ceiling costs
(analysis/prompt_tail.py, backlog/docs/caption-length-by-i2t-model.md).

THE OBSERVATION. With an IDENTICAL, untruncated prompt at a FIXED seed, changing
Flux2Klein's max_sequence_length from 512 to 1024 --- which alters only the
padding of the conditioning tensor and carries no semantic information --- moved
the caption-space cosine between the two images to 0.896-0.988, mean 0.943. On
the measured scale for that metric (1.000 for identical images, 0.876 for images
of unrelated prompts), a zero-information perturbation produced images nearly as
different as unrelated ones.

WHY IT MATTERS. The loop is a Markov chain on captions (random seed per
text-to-image step, greedy captioner), and the 200-step baseline
(analysis/long_horizon_baseline.py) shows it settles into a stationary regime
with a persistent step-to-step distance of about 0.01-0.02. The question is what
that step is made of. If it is mostly generator sampling noise, then the
core-set Markov state model (TASK-76) must show its transitions exceed the
noise, metastable-region identity becomes the whole kinetic result, and
TASK-85's 28% step-size reduction is a statement about noise as a function of
caption length rather than about semantics. RQ2's variance decomposition uses
step-to-step drift as its response and inherits the same caveat. Establishing
the decomposition first is what makes those claims defensible.

WHAT NOT TO COMPARE. Consecutive trajectory states differ by ONE draw of
generator noise; two resamples of the same caption differ by TWO independent
draws, so their distance is expected to be roughly double even with no dynamics
at all. The earlier 'paradox' (consecutive cosine ~0.985 versus resample cosine
~0.943) compared these unlike quantities, and the 0.943 came from a padding
perturbation rather than a seed change. The measurement below compares like with
like.

THE MEASUREMENT. Per text-to-image model, take a fixed set of captions spanning
the panel's length range. For each caption c, generate N images at N recorded
seeds and caption each with one captioner, giving N next-captions. Embed
everything. Then:

- drift: the displacement from c to the centroid of the N next-captions
- noise: the spread of the N next-captions around their centroid
- step: the mean distance from c to each next-caption, which is the quantity the
  trajectories measure Report all three per model in the same units, and the
  noise/step share. Compare the step against the stationary step size in the
  200-step baseline and the v2 pilot/panel. Note Flux2Klein runs at four
  distilled steps and may be the noisiest; the model dependence is itself a
  result since network identity is a panel factor.

Cheap to run: no new models, no new experiments; analysis/step_sweep.py already
contains the generate-caption-embed-compare machinery and the seeded generation
helper exists in panic_models.py.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Per text-to-image model: drift, noise and step measured on a fixed
      caption set over N recorded seeds, all in caption-embedding cosine
      distance, with the noise share of the step stated
- [x] #2 Step compared like-for-like against the stationary step size in the
      200-step baseline and the v2 pilot/panel, and the share of stationary
      movement attributable to sampling stated
- [x] #3 Implications written up for TASK-85's step-size result, for RQ2's
      response variable, and for what the core-set MSM in TASK-76 must
      demonstrate about its transitions
- [x] #4 max_sequence_length recorded as a fixed experiment parameter that
      perturbs generation independently of content, so it is never varied
      mid-programme
- [x] #5 Ruler calibration: the noise share is reported against Qwen3Embed's
      resolution for these captions (distance between seed-resamples of one
      caption versus captions of unrelated prompts), so a step below the ruler's
      resolution is not read as dynamics
- [x] #6 Truncation validated on our captions: a sample embedded at Qwen3Embed's
      native 2560 and at 256 dimensions, with rank correlation of pairwise
      distances and agreement of the step-size and plateau statistics reported,
      so the 256-d choice is a measured number in methods
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Validate the premise. The decomposition assumes a deterministic captioner;
   check each of the five and measure what sampling contributes
   (analysis/captioner_noise.py) --- DONE, three of five sample, see TASK-92.
2. Pin max_sequence_length explicitly on all five text-to-image pipelines (AC#4)
   --- DONE.
3. Build the caption set: five captioners x four pilot images, greedy, then
   eight captions evenly spaced by length (45-313 words). Written to
   analysis/caption_set.json.
4. The sweep (analysis/step_decomposition.py): per text-to-image model, 8
   captions x 16 recorded seeds = 640 images; caption each with the SAME
   captioner that wrote its source caption, forced greedy, so all spread is
   generator noise. Decompose exactly on the unit sphere: mean_i ||x_i - c||^2 =
   ||xbar - c||^2 + mean_i ||x_i - xbar||^2, i.e. step = drift + noise in
   cosine-distance units.
5. Ruler (AC#5) and Bland-Altman MDC from the same data; truncation check at 256
   vs 2560 dimensions (AC#6), on the sweep's captions and on trajectory captions
   from the pilot.
6. Compare step like-for-like against the 200-step baseline and the pilot/panel
   (AC#2), and write up the implications for TASK-85, RQ2 and TASK-76 (AC#3).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
AC#2 UNBLOCKED AND DONE 2026-09-05, once TASK-96 put the database and the sweep
on one embedding scale. analysis/step_vs_stationary.py joins the three
measurements; results in analysis/step_vs_stationary.json.

The sweep's step of 0.062-0.083 is the same order as the settled step an actual
run shows (200-step late window 0.030-0.050, 50-step arms 0.050-0.059), which is
what the old-scale comparison could not establish --- against the published
0.012-0.016 the sweep looked five times too large, and it was the baseline that
was wrong.

SHARE OF STATIONARY MOVEMENT ATTRIBUTABLE TO SAMPLING, matching each trajectory
to the generator it used:

  200-step Flux2Klein networks   settled 0.030-0.034   noise 0.0311   93-105%
  200-step SD35Medium networks   settled 0.042-0.050   noise 0.0446   89-107%
  50-step panel arm (Flux2Klein) settled 0.0591        noise 0.0311   53%
  50-step pilot     (Flux2Klein) settled 0.0502        noise 0.0311   62%

In the settled 200-step runs the generator's own sampling accounts for
essentially the whole step. In the 50-step arms it is half to two thirds, which
fits: at 50 steps the chain is still drifting, and by 200 the directed component
has gone and what remains is close to pure resampling.

Two mismatches keep this indicative. The noise term travels through a captioner
--- it is the spread of the CAPTIONS of N images --- and the sweep used the v2
lineup greedy while these trajectories used the old lineup at truncating
ceilings, where caption length moves step size on its own (TASK-85). The
sweep's source captions are also not drawn from each network's stationary
regime. So a ratio at or just above 100% means 'the same size', not 'larger'.
TASK-90 must re-measure the noise floor from its own trajectories.

AC#5 RULER: two seed resamples of one caption sit 0.0702 apart; captions of
unrelated prompts sit 0.5754 apart (p10-p90 0.214-0.745). The settled step of
0.030-0.050 is therefore well inside the seed-resample band and about a
fifteenth of the distance between unrelated captions. Step-to-step motion in a
settled run is not resolvable as semantic travel.

AC#3 IMPLICATIONS.

For TASK-85: its step-size result is a statement about noise as much as about
semantics. The 13% reduction (28% before the rescale) is real and holds late in
the trajectory, but since the noise floor is most of the step, 'complete
captions make the loop less jittery' is better read as complete captions making
the generator's own sampling less dispersive, not as the trajectory travelling
less far. Distance from t_0 supports that: total travel is 0.104 against 0.111,
essentially equal.

For RQ2: step-to-step drift is the wrong response variable on its own. Most of
its variance is a noise floor that differs by generator (0.031 to 0.045), so a
decomposition over that response will attribute to the generator factor
something that is sampling dispersion rather than an effect on the dynamics.
Report the decomposition over stationary-regime responses --- occupancy,
metastable-region identity, escape times --- alongside it, and report the noise
share with it.

For TASK-76: this is the sharpest thing the sweep has to say. The core-set MSM
cannot treat every observed step as a transition, because a step the size of the
noise floor is what one seed draw produces with no dynamics at all. Its
transitions have to be shown to exceed that floor, which is the argument for
core sets with a transit region rather than a hard partition: assignments near a
boundary will flip on seed noise alone. Metastable-region identity, not
step-level motion, is where the kinetic result lives.

Drift exceeds the Bland-Altman minimal detectable change for 0 of 8 captions on
all four generators, so no individual step is distinguishable from noise, while
mean drift is about twice 1.96 x SEM over 16 seeds. Both belong in the paper:
the drift is real in aggregate and unattributable step by step.
<!-- SECTION:NOTES:END -->
