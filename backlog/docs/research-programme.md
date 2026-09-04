# What this repo is for

A map from the code and the backlog to the scientific question, written
2026-09-04 after a long run of instrument work had branched in several
directions at once. If a task is not on this map it is probably not worth
doing.

## The question

What happens to meaning when it is passed repeatedly through a closed loop of
generative models? A text-to-image model draws a caption; an image-to-text
model describes the drawing; the description is drawn again. The weights are
frozen, so the loop is a fixed map and this is an *inference-time* dynamical
system --- explicitly not training-time model collapse, which is a different
mechanism with superficially similar phenomenology.

The text-to-image step is stochastic (a fresh diffusion seed per invocation)
and the captioner is greedy, so the loop is a **Markov chain on captions**, not
a deterministic map. That fixes the vocabulary: the long-run objects are a
stationary distribution, metastable regions of it, and the escape times between
them. "Fixed point" is the wrong word for this system and is not used.

Two research questions, from the paper skeleton:

- **RQ1, kinetics.** Does the chain reach a stationary regime, how many
  metastable regions does it have per network, and what are the escape times
  between them? Hintze et al. (Patterns 2025) report convergence, but define
  attractors by k-means on *endpoint* embeddings at t=100 --- which assumes
  convergence rather than demonstrating it. The gap is a trajectory-based
  definition validated over a horizon long enough to resolve the slow
  timescales.
- **RQ2, attribution.** They find the captioner explains 13.6% of drift
  variance against the generator's 0.2%. Does that hold in a current-generation
  panel? With five levels per factor the answer describes this panel rather
  than the model class, and step-to-step drift is the response variable most
  contaminated by generator sampling noise (TASK-89), so the decomposition is
  reported with both caveats and alongside stationary-regime responses.

## Where the science is written

`~/projects/research-papers/typst/semantic-dynamics-2026/` --- `body.typ` is a
structural skeleton with per-section notes, intended claims, and DECISION
markers for open design choices. **That file is the plan.** This repo is the
instrument and the data; the paper follows the repo, not the other way round.

The superseded SMC 2025 version is `typst/semantic-topologies-2025`.

## What the existing data already says

`analysis/long_horizon_baseline.py` reads the four 200-step experiments from
February and March (old lineup, truncated captions, Moondream in `short` mode;
design evidence, not paper data). Two results shape everything below.

**Exact caption repetition is not absorption.** Runs whose caption repeats on
consecutive steps leave that string immediately: afterwards about 2% of steps
sit at it, no run ever stays, and around fifty distinct strings follow.
Repetition tracks caption length and nothing else --- 38 of 40 runs for a
23-word captioner, 0 of 32 for a 100-word one. Under random seeds a repeat is a
coincidence of a low-entropy captioner, and decision-01 makes every captioner
three to seven times more verbose, so the "clustering-free ground-truth layer"
the skeleton once proposed would be empty in new data. Repetition is a
descriptive statistic, not a state definition.

**Step size and drift plateau by step 100--150.** Median step-to-step distance
falls from roughly 0.015--0.03 to 0.009--0.02 and stays there; distance from
the initial caption stops growing in the same window. That is a stationary
stochastic regime with a persistent, nonzero step size --- consistent with
Hintze et al., and the reason the horizon question is about slow timescales,
not about waiting for motion to stop.

## What the literature adds

A 2026-09-04 search (four angles: closed-loop genAI, MSM methodology,
iterated learning and other analogues, drift/noise measurement) changed
three things and confirmed the rest. Citations are in the paper skeleton's
Related work notes.

- **Many short runs are the right input.** The longest resolvable implied
  timescale scales with aggregate sampling time, not single-trajectory
  length (Sinitskiy & Pande 2018), and core-set MSM error does not depend on
  how the transit region is handled (Sarich, Noé & Schütte 2010). Both
  support the uniform 250--300 step factorial and the outliers-as-transit
  design.
- **RQ2 has a sharper form.** In iterated learning a chain of samplers
  converges to the learner's prior regardless of start (Griffiths & Kalish
  2007). "Whose prior does the stationary distribution sample from?" is
  testable by comparing it with each captioner's captions of a reference
  image set. That is TASK-91, a cheap candidate result alongside the
  Hintze-matched decomposition.
- **The divergence claims are a metric confound.** Conde et al. track
  distance from origin, which keeps growing under a stationary chain on a
  large state space. Step-to-step distance is the stationarity diagnostic,
  and Vats, Crandall & Goree (2026) report the same local-before-cumulative
  plateau. Report both curves and say why they differ.

- **TASK-89 has a decision rule and a caveat.** Drift is called real only
  above the Bland--Altman minimal detectable change computed from the
  seed-resample spread, and distilled generators may be the *least*
  seed-noisy (distillation flattens seed sensitivity), so the noise share is
  measured per model with no assumed direction. Padding is a genuine
  perturbation in T2I text encoders (Toker et al. 2025), which is why
  `max_sequence_length` is frozen.

Also worth carrying: an AR(1) fit to the embedding trajectory (Xu &
Griffiths 2010) gives a clustering-free attractor-strength statistic for
interim checks, and compression pressure under a transmission bottleneck
(Kirby et al. 2015) is the mechanism behind short captions repeating and long
ones not.

## The horizon

The paper does not claim a fixed 1000-iteration horizon. The horizon is chosen
so that the Markov state model's implied timescales converge with lag time,
which is the validation the model needs anyway; a cell whose slowest timescale
does not converge within the trajectory length is reported as unresolved. Many
independent trajectories past burn-in are the standard MSM input, so the design
is **one uniform factorial at 250--300 steps**, not a few very long runs.

Measured per-item times (the model predicts 14.9 days for the panel that
actually took ~17):

| scenario                                        | GPU-days |
| ----------------------------------------------- | -------- |
| current panel: 5x5, 20 prompts, 4 runs, 50 steps | 14.9     |
| 250 steps, full 5x5, 20 prompts, 4 runs          | 74       |
| 300 steps, full 5x5, 20 prompts, 4 runs          | 89       |
| 300 steps, full 5x5, 20 prompts, 2 runs          | 45       |

Flux2Dev and GLMImage are 86% of all text-to-image time. Runs per prompt is
the cheaper lever than horizon once past the plateau, and TASK-90 settles the
count.

## What each open task is for

| task                                | kind                  | serves                                                    |
| ----------------------------------- | --------------------- | --------------------------------------------------------- |
| TASK-90                             | **the experiment**    | the dataset both RQs need                                 |
| TASK-89 drift/noise decomposition   | **null model**        | paper's Null models section; bears directly on RQ2        |
| TASK-75 outliers as sparse space    | **gate**              | decides whether "time in transit" is a real observable    |
| TASK-76 core-set MSM                | **primary formalism** | Results I, the headline kinetic result                    |
| TASK-77 TDA keep/kill               | **gate**              | Results III, which exists only if this passes             |
| TASK-91 prior-matching test         | candidate             | a sharper RQ2 (whose prior does the chain sample?); after 76 |
| TASK-88 new model candidates        | instrument            | nothing yet; deferrable until the lineup is in question   |

Dependency order for the analysis tasks is **89 → 75 → 76 → (77)**. TASK-89
comes first because it decides how much of each step is deterministic drift
and how much is generator sampling noise: if the stationary step size is
mostly noise, metastable-region identity is the whole kinetic result and the
transitions the Markov model fits must be shown to exceed the noise.

## What is instrument, and why it took so long

Everything closed in the 2026-09-02/04 stretch was making the instrument
trustworthy rather than answering the question. It is recorded because the
results matter, but none of it is a paper claim:

- caption truncation was silently cutting four of five captioners
  (TASK-80/82/85, decision-01) --- and it changed the dynamics, not just the
  captions: step-to-step distance fell 28% once captions were complete
- diffusion step counts were never measured against a quality metric
  (TASK-83); Flux2Dev went 15 to 12 steps
- models floated to whatever was cached; all are now pinned (TASK-84)
- the captioner lineup was two generations old (TASK-87)
- batching headroom (TASK-74/78), step-level CUDA retry (TASK-79)
- NomicVision silently wrote zero vectors (TASK-86); image embedding is now
  removed entirely, since every second state in an alternating network is
  already text
- FTLE removed outright (TASK-73): bounded distances on the unit sphere mean
  the fits never worked

The instrument is now in a known state, which is the precondition for the
long-horizon run rather than an achievement in itself.

## Standing constraints

- **Three rules for using the literature.** The trajectory-mapping and
  metastability work (MSM, milestoning, iterated learning, serial
  reproduction) is old, stable and the thing to build on. Results about
  generative models themselves date fast: cite them as context, never as a
  foundation, and re-verify any that a design choice would rest on. Hintze
  et al. (Patterns 2025) is the motivating paper: every result should read as
  a direct answer to something they claimed or left open, and the goal is
  clear, interesting results rather than coverage.
- **Seeds are random and recorded.** Every text-to-image invocation draws its
  own seed and stores it, so within-condition variation is attributable and any
  step can be regenerated. Fixing the seed would turn the chain into one seed's
  deterministic map and change what RQ1 means.
- **Recluster once, at the end.** `mix cluster.recompute` is destructive and
  global: it relabels every experiment. Collect all data, cluster once, then
  make every cluster-dependent figure from that clustering. Any interim check
  uses clustering-free observables (step size, drift from origin, AR(1)
  mean reversion, repetition rate).
- **`max_sequence_length` is not a neutral knob.** It sets padding length and
  perturbs generation even with identical text, so fix it before a run.
- **512 tokens is the binding caption constraint**, not generation length ---
  SD35Medium hard-caps there. It is a constraint on which captioners are
  eligible, not a parameter to tune.
- **Validate every new model before committing GPU time.** Four separate
  traps in TASK-87 were invisible in model output and would each have failed
  hours or days into a run.
