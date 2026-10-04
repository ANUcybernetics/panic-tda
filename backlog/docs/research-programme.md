# What this repo is for

A map from the code and the backlog to the scientific question, written
2026-09-04 after a long run of instrument work had branched in several
directions at once, and redrawn 2026-10-04 once TASK-90's panel was in. If a
task is not on this map it is probably not worth doing.

## The question

What happens to meaning when it is passed repeatedly through a closed loop of
generative models? A text-to-image model draws a caption; an image-to-text model
describes the drawing; the description is drawn again. The weights are frozen,
so the loop is a fixed map and this is an _inference-time_ dynamical system ---
explicitly not training-time model collapse, which is a different mechanism with
superficially similar phenomenology.

The text-to-image step is stochastic (a fresh diffusion seed per invocation), so
the loop is a **Markov chain on captions**, not a deterministic map, and "fixed
point" is the wrong word for it. The vocabulary that usually comes with a
Markov chain (a stationary distribution, metastable regions, escape times) is
not the one the data supports. The panel is described in plainer terms. A
caption's embedding is where its prompt sits in this network, plus the run's
own offset from there, plus the noise of the last seed. Each part has a size
and a persistence, and all of them are read off distances between embeddings
(`drift-and-memory.md`).

The captioner adds no randomness of its own: all five decode greedily
(decision-02), so the chain is a deterministic captioner composed with a
stochastic generator. Three of the five shipped a sampling config whose noise
alone was worth as much as the generator's, which would have made step-level
motion partly the captioner resampling its own prose; forcing greedy costs
nothing measurable in caption quality and makes RQ2's captioner effect a
statement about descriptive style rather than about each vendor's shipped
temperature.

Two research questions:

- **RQ1, memory and movement.** Does a run forget its prompt, how fast, and
  does it settle? Hintze et al. (Patterns 2025) report convergence on generic
  motifs, but define attractors by k-means on _endpoint_ embeddings at t=100
  --- which assumes convergence rather than demonstrating it. The
  trajectory-based test is direct: whether runs from different prompts approach
  each other, and whether the two runs of one prompt stay closer than
  strangers.
- **RQ2, attribution.** They find the captioner explains 13.6% of drift variance
  against the generator's 0.2%. Which half of the loop sets what in a
  current-generation panel? There are two answers to give. One is for where a
  caption ends up: how far it moves when the generator, the captioner or the
  prompt is swapped. The other is for how it moves: the noise in a step, what a
  step keeps, and memory. The Hintze-matched decomposition over step-to-step
  drift is reported as one comparability table, since that response is mostly
  seed noise (TASK-89, TASK-103). With four levels per factor either answer
  describes this panel rather than the model class.

## Where the science is written

`~/projects/research-papers/typst/semantic-dynamics-2026/` --- `body.typ` is a
structural skeleton with per-section notes, intended claims, and DECISION
markers for open design choices. **That file is the plan.** This repo is the
instrument and the data; the paper follows the repo, not the other way round.
Its RQ1 still reads as metastable regions and escape times, and has yet to be
brought into line with this map.

The superseded SMC 2025 version is `typst/semantic-topologies-2025`.

## What the data says

### The panel: prompt, offset and noise

TASK-90's panel is 16 networks, 20 prompts, two runs per prompt and 150 text
states per run, and `panel-audit.md` is the check that its data is sound.
TASK-103 reads it off distances between embeddings, with no clustering.
`drift-and-memory.md` has the tables and figures.

- Four-fifths or more of the distance between consecutive captions is seed
  noise that the next step takes back, in every network.
- A run's own offset is the largest part. The two runs of a prompt are 0.36
  apart by the end, noise removed, and have nearly stopped separating.
- The prompt still fixes a third of where a run sits. Memory, the prompt's
  share, falls from 0.71 to 0.35 and has almost stopped falling. It is 0.64 for
  a red apple and 0.12 for a city turning into a forest.
- Runs from different prompts do not converge. The distance between them falls
  4% over a run, and the fall is distinguishable from none only in networks
  with Flux2Dev or JoyCaption.
- Within a run, displacement is still growing at the longest separation the run
  allows, and is 23--59% of the way to the distance between twins.
- The captioner sets the noise and the generator sets how much of a step is
  kept. For where a caption ends up, swapping either moves it 0.07 once the
  captioner's way of writing is removed, against 0.19 for swapping the prompt.
- Seven of the 640 runs spend a third or more of their length on images with
  almost nothing in them, and four end there.

### The Markov state model had nothing to count

A Markov state model estimates escape times from runs crossing between states
they share. In every cell of the panel each run keeps to its own region. A
partition coarse enough for runs to share states is one they almost never
leave, and one fine enough to be crossed gives states private to a single run
(`analysis/trajectory_mixing.py`). TASK-76 carries the rationale and is
archived.

### The step settles before the run does

`analysis/long_horizon_baseline.py` reads the four 200-step experiments from
February and March (old lineup, truncated captions, Moondream in `short` mode;
design evidence, not paper data). Median step-to-step distance falls from
roughly 0.051--0.082 to 0.030--0.050 and stays there, and distance from the
initial caption stops growing in the same window. That was read as a stationary
regime.

The panel agrees about the step and corrects the reading. Its step size changes
little after the first 25 text states (0.046 falling to 0.041). Two states of
one run still get further apart with separation, as far as a 150-state run can
show, and movement keeps slowing as a run ages. A steady step is what seed noise
looks like, and says little about whether the slow part has settled.

Every number from the old runs was recomputed on 2026-09-05 after TASK-96 found
the stored vectors had been mean-pooled; the distances are about three times
larger on the corrected scale, and the plateau keeps its shape.

### Exact caption repetition is not absorption

In the 200-step runs, a run whose caption repeats on consecutive steps leaves
that string immediately: afterwards about 2% of steps sit at it, no run ever
stays, and around fifty distinct strings follow. Repetition tracks caption
length and nothing else --- 38 of 40 runs for a 23-word captioner, 0 of 32 for a
100-word one. Under random seeds a repeat is a coincidence of a low-entropy
captioner. In the panel, where decision-01 has made every captioner three to
seven times more verbose, repeats are 1.5% of late steps or less in thirteen
networks. The other three, at 3--9%, all use Moondream3, the shortest
captioner. Repetition is a descriptive statistic, not a state definition.

### A run covers its region, given long enough

The 5,000-invocation runs of April 2025 are the only data deep enough to see
where displacement goes (`analysis/long_run_drift.py`; old models, short
captions, two prompts with no content). In all four networks a run's
displacement climbs to the distance between independent runs. One network is
there within 200 text states, and two are still short of it at 1,250. One,
SDXLTurbo + BLIP2, also converges: the distance between its runs falls from
0.82 to 0.61 over 2,500 text states.

### EVoC's outliers are not sparse space, and outlier time is not transit time

TASK-75 (`backlog/docs/outlier-sparsity.md`) measured the reading the
programme had assumed. Outliers have the same local density as clustered
points; the 26--45% outlier share holds across every EVoC hyperparameter but
the outlier set changes wholesale under it, and a second EVoC pass over the
outliers alone leaves 39% of them unlabelled again, so the share is the
procedure's, not the data's. Length and captioner explain nothing. Three
quarters of outlier time is runs that end in the outlier region or never leave
it (median 19-step tails); genuine transits between clusters are a tenth. The
outlier region is one connected, ordinarily dense place that thousands of runs
settle into and EVoC declines to partition. Consequence: no analysis rests on
EVoC's labels.

## What the literature adds

A 2026-09-04 search (four angles: closed-loop genAI, MSM methodology, iterated
learning and other analogues, drift/noise measurement) changed three things and
confirmed the rest. Citations are in the paper skeleton's Related work notes.

- **Many short runs were the right design, for a different reason.** The
  longest resolvable implied timescale scales with aggregate sampling time, not
  single-trajectory length (Sinitskiy & Pande 2018), and the uniform factorial
  was built on that. The argument needs runs that visit common states, and the
  panel's do not. What two runs per prompt across a full factorial did buy is
  the twin. With it, a run's own offset and its prompt's share can be told
  apart. A few long runs would have had no twins.
- **Iterated learning says what forgetting would look like.** A chain of
  samplers converges to the learner's prior regardless of where it started
  (Griffiths & Kalish 2007). Prompt memory measures "regardless of where it
  started" directly, and at 150 text states the panel's chains are not there:
  memory is 0.35 and has almost stopped falling. "Whose prior does the chain
  sample from?" (TASK-91) asks about a stationary distribution the panel does
  not reach, and needs restating before it is run.
- **Distance from origin and step size each mislead alone.** Conde et al. track
  distance from origin, which keeps growing under a stationary chain on a large
  state space. Step-to-step distance plateaus early, and Vats, Crandall & Goree
  (2026) report the same local-before-cumulative pattern, but a step is
  four-fifths seed noise and says little about the slow part. The curve to
  report is displacement against separation, with the noise removed.

- **TASK-89 has a decision rule and a caveat.** Drift is called real only above
  the Bland--Altman minimal detectable change computed from the seed-resample
  spread, and distilled generators may be the _least_ seed-noisy (distillation
  flattens seed sensitivity), so the noise share is measured per model with no
  assumed direction. In the panel the noise differs more by captioner than by
  generator. Padding is a genuine perturbation in T2I text encoders (Toker et
  al. 2025), which is why `max_sequence_length` is frozen.

Also worth carrying: an AR(1) fit to the embedding trajectory (Xu &
Griffiths 2010) gives a clustering-free attractor-strength statistic, of which
TASK-103's noise estimate is a two-lag relative, and compression pressure under
a transmission bottleneck (Kirby et al. 2015) is the mechanism behind short
captions repeating and long ones not.

## The horizon

The paper does not claim a fixed 1000-iteration horizon. The panel ran 300
invocations, which is 150 text states, and that settles some things and leaves
others open. It is long enough to see the noise, the early movement that twins
share, the twins separating, and memory falling to a level. It is too short to
see whether memory stays at that level. At the rate of the last fifty states
the twins would reach the stranger distance after about another 570 text
states, and that rate has been falling. It is also too short to see a run cover
its prompt's region, which the old runs say takes from 200 to more than 1,250
text states.

So the next experiment is fewer prompts, more runs per prompt, and 700 text
states or more. The panel's own step times price it. A cell of 40 runs to 700
text states costs about two and a half days with SD35Medium or ZImageTurbo.
With Flux2Klein it is a day and a half, and with Flux2Dev fifteen.

Measured per-item times (the model predicted 14.9 days for the July panel,
which took ~17):

| scenario                                | GPU-days |
| --------------------------------------- | -------- |
| a 50-step 4x5 panel, 20 prompts, 4 runs | 9.8      |
| 250 steps, full 4x5, 20 prompts, 4 runs | 48       |
| 300 steps, full 4x5, 20 prompts, 4 runs | 58       |
| 300 steps, full 4x5, 20 prompts, 2 runs | 29       |

The panel is 4x4, not 5x5. GLMImage was removed (TASK-94): dropping it takes the
300-step four-run design from 89 GPU-days to 58, which is what made the horizon
affordable at all, and leaves Flux2Dev at 78% of all text-to-image time. Qwen3VL
was removed (TASK-101): its captions of early-step images exceed the 512-token
encoder ceiling, and so do its smaller variants'. TASK-90 settled on two runs
per prompt, and the 4x4 panel took 19.5 days of wall clock against the 20
GPU-days estimated (`backlog/docs/long-horizon-design.md`).

## What each task is for

| task                              | kind                    | serves                                                          |
| --------------------------------- | ----------------------- | --------------------------------------------------------------- |
| TASK-90                           | closed                  | the dataset both RQs need: 640 runs of 300 invocations          |
| TASK-103 noise, offset and memory | **primary description** | Results I, and both RQs as now stated                           |
| TASK-104 the long follow-up       | **the next experiment** | whether a prompt is ever forgotten; few prompts, eight runs     |
| TASK-105 reading the follow-up    | after 104               | memory and displacement over 700 text states or more            |
| TASK-76 Markov state model        | abandoned               | the panel has no states that runs both share and cross          |
| TASK-89 drift/noise decomposition | closed                  | the noise floor is most of the step; Null models, and RQ2       |
| TASK-75 outliers as sparse space  | closed                  | failed: outliers are not sparse, transit time is not observable |
| TASK-77 TDA keep/kill             | **gate**                | Results III, which exists only if this passes; against TASK-103 |
| TASK-91 prior-matching test       | candidate               | needs restating: the panel reaches no stationary distribution   |
| TASK-100 encoder truncation       | methods                 | share of captions cut at 512 tokens; measured, reporting open   |
| TASK-88 new model candidates      | instrument              | nothing yet; deferrable until the lineup is in question         |
| TASK-92 captioner decoding        | closed                  | why the captioner contributes no noise (decision-02)            |
| TASK-93 seed recording            | closed                  | attributable within-condition variation; RQ2 rests on it        |
| TASK-94 GLMImage removed          | closed                  | why the text-to-image side is four                              |
| TASK-101 Qwen3VL replacement      | closed                  | why the captioner side is four: the 512-token ceiling           |

Dependency order for the analysis tasks is **89 → 75 → 103 → (77)**. TASK-89
came first because it decides how much of each step is deterministic drift and
how much is generator sampling noise. On the old 200-step runs it found the
generator's own sampling accounts for essentially the whole settled step
(89--107% of it, matched by generator), falling to 53--62% in the 50-step arms
where the chain is still drifting. TASK-103 re-measured that on the panel's own
trajectories: about four-fifths of a step by the chain's own estimate, and
nine-tenths by redrawing 528 steps at new seeds. TASK-75 answered negatively: the
outlier share is an artefact of the clustering procedure and outlier time is
settled time. TASK-76 was to be the standard MSM pipeline, and was abandoned
when the panel turned out to have nothing for it to count. TASK-103 replaced it
with a description that needs no states, and TASK-77 is now measured against
that. TASK-92 and TASK-93 gated TASK-90 rather than the analysis chain: both
change what a recorded step means, and neither can be applied to a run after
the fact.

## What is instrument, and why it took so long

Everything closed in the 2026-09-02/04 stretch was making the instrument
trustworthy rather than answering the question. It is recorded because the
results matter, but none of it is a paper claim:

- caption truncation was silently cutting four of five captioners
  (TASK-80/82/85, decision-01) --- and it changed the dynamics, not just the
  captions: step-to-step distance fell 13% once captions were complete
- diffusion step counts were never measured against a quality metric (TASK-83);
  Flux2Dev went 15 to 12 steps
- models floated to whatever was cached; all are now pinned (TASK-84)
- the captioner lineup was two generations old (TASK-87)
- batching headroom (TASK-74/78), step-level CUDA retry (TASK-79)
- NomicVision silently wrote zero vectors (TASK-86); image embedding is now
  removed entirely, since every second state in an alternating network is
  already text
- FTLE removed outright (TASK-73): bounded distances on the unit sphere mean the
  fits never worked

The instrument was in a known state for the long-horizon run. The audit after
it (`panel-audit.md`) regenerated stored images, captions and embeddings from
their stored inputs. It found nothing in the panel that was not the loop's own
doing.

## Standing constraints

- **Three rules for using the literature.** The trajectory-mapping and
  metastability work (MSM, milestoning, iterated learning, serial reproduction)
  is old, stable and the thing to build on. Results about generative models
  themselves date fast: cite them as context, never as a foundation, and
  re-verify any that a design choice would rest on. Hintze et al.
  (Patterns 2025) is the motivating paper: every result should read as a direct
  answer to something they claimed or left open, and the goal is clear,
  interesting results rather than coverage.
- **Captioners decode greedily.** The diffusion seed is the loop's only source
  of randomness (decision-02). `set_i2t_greedy/1` exists so the analysis
  scripts can measure the shipped sampling configs; nothing that writes to the
  database uses it.
- **Seeds are random and must be recorded.** Fixing the seed would turn the
  chain into one seed's deterministic map and change what RQ1 means, so every
  text-to-image invocation draws its own. Storing it is what makes
  within-condition variation attributable and any step regenerable. A run
  cannot be given seeds afterwards, which is why this landed before TASK-90
  (TASK-93).
- **At least two runs per prompt, in every network.** The twin is what
  separates a run's own offset from where its prompt sits, and everything in
  TASK-103 that is not noise rests on it. More runs per prompt would give each
  prompt's centre directly.
- **Nothing in the primary analysis is clustered.** `mix cluster.recompute` is
  destructive and global: it relabels every experiment. If a figure ever needs
  EVoC labels, collect all the data, cluster once, and make every such figure
  from that clustering.
- **`max_sequence_length` is not a neutral knob.** It sets padding length and
  perturbs generation even with identical text, so fix it before a run.
- **512 tokens is the binding caption constraint**, not generation length ---
  SD35Medium hard-caps there. It is a constraint on which captioners are
  eligible, not a parameter to tune. In the panel 0.36% of captions ran past
  it, nearly all Gemma4's (TASK-100).
- **Validate every new model before committing GPU time.** Four separate traps
  in TASK-87 were invisible in model output and would each have failed hours or
  days into a run.
