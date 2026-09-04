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

Two research questions, from the paper skeleton:

- **RQ1, kinetics.** At long horizons, do these loops reach fixed points,
  metastable states, or cycles? Hintze et al. (Patterns 2025) report
  convergence, but define attractors by k-means on *endpoint* embeddings at
  t=100 --- which assumes convergence rather than demonstrating it. The gap is
  the horizon.
- **RQ2, attribution.** They find the captioner explains 13.6% of drift
  variance against the generator's 0.2%. Does that replicate at current model
  scale? Our own SMC-era results are in tension with it.

## Where the science is written

`~/projects/research-papers/typst/semantic-dynamics-2026/` --- `body.typ` is a
structural skeleton with per-section notes, intended claims, and DECISION
markers for open design choices. **That file is the plan.** This repo is the
instrument and the data; the paper follows the repo, not the other way round.

The superseded SMC 2025 version is `typst/semantic-topologies-2025`.

## The gap at the centre

The paper's first stated contribution is 1000-iteration trajectories, and RQ1
is framed as answering the prior work "at 10x the horizon".

**The longest trajectory in the database is 200 steps. The panel configs are
50.** The headline dataset does not exist yet, and until 2026-09-04 nothing in
the backlog represented building it. That is TASK-90.

It is not a matter of just setting `max_length` higher. Measured per-item
times give:

| scenario | GPU-days |
|---|---|
| current panel: 5x5, 20 prompts, 4 runs, 50 steps | 14.9 |
| the same at 1000 steps | **298** |
| 1000 steps, 5 prompts, 2 runs | 37 |
| 250 steps, full 5x5 | 74 |
| 1000 steps, fast T2I only (3x5), 5 prompts, 4 runs | 13.8 |

(The model predicts 14.9 days for the panel that actually took ~17, so it is
about right.)

**Flux2Dev and GLMImage account for 86% of all text-to-image time.** Dropping
them buys roughly seven times the horizon for the same budget. So the full
model factorial and the long horizon are mutually exclusive at any sane cost
--- which is precisely the RQ1/RQ2 tension, since RQ2 wants the factorial and
RQ1 wants the horizon. Resolving it is a design decision, not an optimisation.

## What each open task is for

| task | kind | serves |
|---|---|---|
| TASK-90 | **the experiment** | the dataset both RQs need |
| TASK-89 sampling noise floor | **null model** | paper's Null models section; bears directly on RQ2 |
| TASK-75 outliers as sparse space | **gate** | decides whether "time in transit" is a real observable |
| TASK-76 core-set MSM | **primary formalism** | Results I, the headline kinetic result |
| TASK-77 TDA keep/kill | **gate** | Results III, which exists only if this passes |
| TASK-88 new model candidates | instrument | nothing yet; deferrable until the lineup is in question again |

Dependency order for the analysis tasks is **89 → 75 → 76 → (77)**. The noise
floor comes first because it decides whether the transitions the Markov model
would fit are signal at all. TASK-89 is not a curiosity: it is the
i.i.d.-resampling surrogate the paper's Null models section already
pre-specifies.

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

- **Recluster once, at the end.** `mix cluster.recompute` is destructive and
  global: it relabels every experiment. Collect all data, cluster once, then
  make every cluster-dependent figure from that clustering.
- **`max_sequence_length` is not a neutral knob.** It sets padding length and
  perturbs generation even with identical text, so fix it before a run.
- **512 tokens is the binding caption constraint**, not generation length ---
  SD35Medium hard-caps there. It is a constraint on which captioners are
  eligible, not a parameter to tune.
- **Validate every new model before committing GPU time.** Four separate
  traps in TASK-87 were invisible in model output and would each have failed
  hours or days into a run.
