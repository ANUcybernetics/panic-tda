# Does topology add anything to the distance description?

Written 2026-10-08 by Sungyeon Hong for TASK-77, on TASK-90's panel (experiment
`01a09e21`: 16 networks, 20 prompts, two runs per prompt, 150 text states per
run). Measured with `analysis/tda_keep_kill.py` and
`analysis/sliding_window.py`; the numbers are in the JSON beside each.

Extended 2026-11-04 with the long follow-up (TASK-104, experiment `01a10613`:
6 networks, 5 prompts, eight runs per prompt, 1,000 text states per run), in
"The follow-up at 1,000 states" below. Its numbers are in the JSON suffixed
`_01a10613`, and `_01a10613_states1-150` for the same runs cut to their first
150 states. **One finding there reverses on the longer runs**: sliding-window
persistence beats its null at 1,000 states, though what it finds there is not a
cycle in the captions.

TASK-77 asks whether topological data analysis (TDA) earns a place in the
paper. Since the Markov state model was abandoned (TASK-76), the baseline TDA
has to beat is TASK-103's description: a caption is where its prompt sits, plus
the run's own offset, plus seed noise, all read off distances between
embeddings (`drift-and-memory.md`).

## In short

- **The pipeline's persistence diagrams say nothing the distances do not.**
  H0 is a statement about spacing between captions and is predicted almost
  exactly by distance numbers. H1 and H2 show no loops: real runs have fewer
  and shorter ones than structureless processes with the same drift. Their
  small help in telling captioners apart is local intrinsic dimension seen
  indirectly.
- **Sliding-window persistence finds no cycles at 150 states.** Fewer runs
  show a long loop than chance would give. At 1,000 states its bars beat the
  null in about 30% of runs against a chance 5%, but they are paraphrase
  geometry, not cycles; see below.
- **There is recurrence, but it is not topological.** Some runs leave a
  caption and later return to a near-verbatim copy of it. A plain recurrence
  count on distances finds this; persistent homology does not.
- **Two properties of the runs fall outside TASK-103's description:** captions
  are locally lower dimensional than any Gaussian with the same spread, and
  runs are clumpier, with groups of near-repeats separated by jumps. Both are
  read off distances directly.

**Recommendation:** drop persistent homology from the headline analyses and
keep it in the methods discussion: Rips persistence as a negative result,
sliding-window persistence as a positive one that does not carry the
interpretation a cycle would (see "What this changes"). Report intrinsic
dimension and recurrence directly, as distance statistics. This is a
recommendation for Sungyeon and Ben to decide on, not a decision taken here.

## What the pipeline's diagrams are

One Vietoris-Rips diagram per run, to dimension 2, over the run's 150 caption
embeddings (Qwen3Embed, 256 dimensions), on Euclidean distance between unit
vectors (`lib/panic_tda/models/tda.ex`). Euclidean distance is a monotone
function of cosine distance, so this is the cosine filtration reparametrised.
Recomputing every fortieth diagram from the stored embeddings reproduces the
stored one to 7e-7.

A Rips diagram is computed on the set of states and cannot see their order:
shuffling a run in time leaves it unchanged. Anything about recurrence in time
needs a different measure (below).

## Test 1: are the diagrams predicted by the distances?

Each run gets fifteen diagram features: for H0, H1 and H2 each, the number of
bars, their total and longest lifetime, and persistence entropy raw and
normalised. Each run also gets twelve distance numbers in TASK-103's
vocabulary: step size, seed noise, spread over the whole run and over its second
half, displacement from start to end, distance to its twin late in the run,
share of distinct captions, share of steps repeating the last caption, mean
distance to the nearest other state over the whole run and late, participation
ratio, and TwoNN intrinsic dimension.

Each diagram feature, predicted from the distance numbers by ridge regression
with whole prompts held out (R², 1 is perfect):

| Feature | H0 | H1 | H2 |
| --- | --- | --- | --- |
| total lifetime | 0.98 | 0.51 | 0.36 |
| entropy | 0.95 | 0.72 | 0.50 |
| bar count | 0.69 | 0.68 | 0.48 |
| longest bar | 0.52 | 0.29 | 0.21 |

H0's bars are the edges of the minimum spanning tree, so its total lifetime is
close to a sum of nearest-neighbour distances. H1 and H2 are only partly
predicted. That leaves room for them to carry something, and the next two
tests ask whether they do.

## Test 2: does the diagram help say which models made a run?

Held-out accuracy at naming a run's generator, and its captioner, from each set
of features (chance is 0.25; whole prompts held out; the gain is resampled
over prompts):

| Features | Generator | Captioner |
| --- | --- | --- |
| distances | 0.43 | 0.67 |
| diagram alone | 0.41 | 0.61 |
| distances and H0 | 0.41 | 0.68 |
| distances, H0, H1 and H2 | 0.43 | 0.69 |
| gain from H1 and H2 | +0.01 (−0.02 to +0.05) | +0.02 (−0.00 to +0.04) |

The diagram adds nothing for the generator. For the captioner the gain from H1
and H2 was +0.045 (+0.017 to +0.075) before intrinsic dimension was among the
distance numbers, and +0.017 with an interval that includes zero after. H1
count correlates with TwoNN dimension at r = 0.58. The captioners differ in how
many directions their runs spread into locally, and H1 counts pick that up
indirectly:

| Captioner | TwoNN dimension | Participation ratio | H1 bars |
| --- | --- | --- | --- |
| Gemma4 | 10.9 | 6.3 | 57 |
| JoyCaption | 9.8 | 7.1 | 60 |
| Qwen25VL | 6.2 | 6.1 | 49 |
| Moondream3 | 4.7 | 5.7 | 41 |

The generators lie between 7.3 and 8.6.

## Test 3: are the loops more than the drift makes?

### The null

A null for this has to reproduce TASK-103's description and add nothing to it.
`sliding_window.Null` is a Gaussian process built for each run from:

- the run's own displacement curve (mean distance between two states against
  their separation in time), measured to a separation of 110 and extended
  beyond by a power law fitted from 30. That curve carries the seed noise and
  the wander.
- the run's centre and main directions, with the number of directions chosen
  so that the null's local intrinsic dimension (TwoNN) matches the run's.

The null reproduces the displacement curve to within a few per cent once
rescaled, and local dimension at 7.5 against the runs' 7.9 (r = 0.93 across
runs). It does not match the global spread across directions. Restricting it
to few directions to match local dimension leaves it spread over fewer than the
runs (participation ratio 4.0 against 6.8 in a sample).

Two simpler nulls failed and were not used. A Gaussian cloud with the run's own
covariance has no drift at all; its numbers are kept in
`tda_keep_kill.json` (`against_null`). Shuffling the increments of a smoothed
path moved too little at short separations and too far at long ones (sample of
80 runs, scratch only).

### The Rips diagrams against it

Nine draws of the null per run. The share of runs whose feature lies above,
or below, every draw is 0.10 each way by chance:

| Feature | Whole run: above | Whole run: below | Second half: above | Second half: below |
| --- | --- | --- | --- | --- |
| H0 longest bar | 0.94 | 0.00 | 0.77 | 0.00 |
| H0 total | 0.81 | 0.05 | 0.77 | 0.05 |
| H0 entropy | 0.00 | 0.90 | 0.01 | 0.73 |
| H1 total | 0.00 | 0.96 | 0.01 | 0.79 |
| H1 longest bar | 0.04 | 0.58 | 0.06 | 0.46 |
| H2 total | 0.01 | 0.79 | 0.02 | 0.46 |

Real runs have **less** H1 and H2 than the drift alone produces, never more.
There are no loops to find.

Real runs do differ from the null in H0. The longest gap is larger, and the
lifetimes are more uneven (lower entropy). Captions come in tight groups of
near-repeats with jumps between them, where the null spreads them evenly. The
second half alone shows the same, so this is not the fast movement early in a
run, which the null, moving at one rate throughout, cannot produce. Nearest-
neighbour distance and TwoNN measure this directly, and so does the share of
repeated captions. H0 adds nothing beyond them (Test 1).

## Test 4: does a run come back to where it has been?

`analysis/sliding_window.py`, nineteen draws of the null per run. A run counts
when it beats every draw, which happens to 5% of runs by chance (upper 95%
bound 6.7% over 637 runs). The three runs that reach a flat image
(`panel_audit.json`) are left out; no run repeats its previous caption at half
or more of its steps.

| Measure | What it counts | Share of runs beating the null |
| --- | --- | --- |
| sliding-window H1, window 5 | longest loop in the run's states taken five at a time | 0.03 |
| window 10 | | 0.02 |
| window 20 | | 0.01 |
| return depth | how far a run went from a state before coming back to it | 0.00 |
| recurrence rate | pairs at least 20 states apart, within one step's distance | 0.28 |
| return after excursion | the same, only after the run went at least its typical 20-state distance away | 0.13 |

By captioner:

| Captioner | Recurrence rate | Return after excursion | Sliding-window H1, window 10 |
| --- | --- | --- | --- |
| Gemma4 | 0.50 | 0.20 | 0.01 |
| JoyCaption | 0.39 | 0.19 | 0.02 |
| Qwen25VL | 0.13 | 0.08 | 0.03 |
| Moondream3 | 0.09 | 0.05 | 0.03 |

The generators lie between 0.09 and 0.18 for return after excursion.

**Sliding-window persistence finds no cycles at this length.** A delay
embedding of a run that cycled through the same sequence of captions would show
a long loop. Fewer runs show one than chance gives, at every window. This is
the finding that does not survive the 1,000-state runs.

**Returns do happen, mostly with Gemma4 and JoyCaption.** For example, in
SD35Medium + Gemma4, run 0 of "a market stall displaying fruit, flowers, and
handmade crafts" describes "a close-up, high-angle photograph filled with a
large quantity of bright yellow lemons" at text state 61. It moves away to a
cosine distance of 0.115, and at state 114 gives an almost identical caption,
0.025 from the first. In SD35Medium + Gemma4, run 1 of "a bustling Tokyo street
at night" returns at state 130 to the "brilliant blue light burst" it described
at state 97, after moving 0.19 away.

These are returns to a **point**, a near-verbatim earlier caption, not
movement around a loop. That is why a recurrence count on distances sees them
and persistent homology does not. One reading is that a captioner has
favourite descriptions of an image type and comes back to them whenever the
image does. That fits the clumpiness in Test 3.

**How far to trust the 13%.** The hits concentrate where the null spreads over
the most directions: Gemma4 and JoyCaption runs need 32 to 64 to match their
local dimension, Moondream3 runs five. A Gaussian in many directions almost
never comes back close to where it has been, so part of the excess may be the
null's Gaussian tails rather than the runs. Return depth points the same way
from the other side. The null goes further out and comes back further (mean
return depth 0.27 against the runs' 0.09), so it overstates excursions. Real runs move more
steadily than a Gaussian with the same drift, and come back to particular
captions more often than one.

## The follow-up at 1,000 states

Run 2026-11-04 on the GPU server, on experiment `01a10613`: 240 runs, six
networks (three generators by two captioners), five prompts, eight runs per
prompt, 1,000 text states each. Ten runs reach a flat image (nine of them
SD35Medium + Moondream3) and are left out of Test 4 as before; no run repeats
its previous caption at half or more of its steps. The pipeline's stored
diagrams are over the same 1,000 states this script reads, agreeing to
1.6e-05 in total persistence.

Three readings are available and worth keeping apart: the panel at 150 states,
these runs cut to their **first 150 states**, and these runs **whole**. The cut
is the control: same runs, same networks, same captioners, only shorter. Where
the cut and the whole disagree, the horizon is doing the work.

### Rips persistence: the verdict holds, and hardens

| | panel 150 | follow-up 1:150 | follow-up 1,000 |
| --- | --- | --- | --- |
| R^2 of h1_count from the distances | 0.68 | 0.58 | **0.85** |
| R^2 of h1_entropy | 0.72 | 0.59 | **0.88** |
| R^2 of h2_count | 0.48 | 0.34 | **0.69** |
| R^2 of h0_total | 0.98 | 0.97 | 0.99 |
| generator, gain from adding the diagram | +0.001 | -0.004 | +0.025 [-0.033, +0.083] |
| captioner, gain from adding the diagram | +0.027 | -0.008 | -0.001 [-0.025, +0.017] |

The diagrams get **more** predictable from the distance numbers as runs get
longer, not less, and neither gain interval clears zero. Against TASK-103's
process null the H1 and H2 bars are still mostly **below** what the drift
alone produces: 90% of runs are below every draw on h1_count, 97% on h1_total,
80% on h2_count.

One number moves. The **longest H1 bar** beats every draw in 24% of runs at
1,000 states, against 3-4% at 150 and a chance 10%. It is the only Rips
feature to rise above chance, and it moves in the same direction as the
sliding-window result below. Read on its own it is modest; read alongside
Test 4 it is the same thing seen weakly.

Unchanged from the panel: the longest H0 bar beats every draw in **every**
run (real 0.52, null 0.30). Runs are clumpier than their drift explains.

### Recurrence: sliding-window bars beat the null at 1,000 states

Nineteen draws per run; a run counts when it beats all of them, which happens
to 5% by chance (upper 95% bound 7.8% over 230 runs).

| Measure | panel 150 | follow-up 1:150 | follow-up 1,000 |
| --- | --- | --- | --- |
| sliding-window H1, window 5 | 0.03 | 0.04 | **0.29** |
| window 10 | 0.02 | 0.03 | **0.31** |
| window 20 | 0.01 | 0.03 | **0.27** |
| recurrence rate | 0.28 | 0.37 | **0.49** |
| return after excursion | 0.13 | 0.19 | **0.40** |
| return depth | 0.00 | 0.00 | 0.00 |

**Sliding-window persistence beats the null at 1,000 states and not at 150.**
The first 150 states of these very runs sit at chance. So this is the horizon,
not the network set: the effect needs more than 150 steps to appear. What the
bars are made of is taken up in "What the loops are made of" below.

The two measures come apart, which is worth noticing. Recurrence is a
**captioner** property and sliding-window H1 is not:

| | Recurrence rate | Return after excursion | Sliding-window H1, window 10 |
| --- | --- | --- | --- |
| Gemma4 | 0.78 | 0.66 | 0.30 |
| Moondream3 | 0.18 | 0.13 | 0.32 |

By network, sliding-window H1 at window 10 runs from 0.19 (SD35Medium +
Moondream3) to 0.53 (ZImageTurbo + Moondream3), and it is above chance for
every one of the five prompts (0.26 to 0.40) and every one of the six
networks. It is a broad effect, not one prompt or one pairing. Recurrence
rate, by contrast, separates cleanly: all three Gemma4 networks at 0.69 to
0.95, all three Moondream3 networks at 0.07 to 0.28.

So a near-verbatim return to an earlier caption is something a captioner does,
while going round a loop is something the longer runs do more generally.

**How far to trust these.** The null's candidate direction counts now run to
255, and 55 of the 230 runs want the top of that ladder: their local dimension
is 14.6 and the null reaches 12.8. The ceiling binds for about a quarter of
runs, a little worse than the panel's 95 of 637. It is conservative in both
directions. A null that is locally lower dimensional packs its points closer
together, so it returns to itself more readily, and a real run has to work
harder to beat it. And the ceiling-bound runs are the **weaker** half on
sliding-window H1 (0.24 against 0.34 for the rest), so the cycles are not
coming from the runs whose null is mismatched.

Return depth stays the odd one out, as in the panel: real runs go out less far
and come back less far than the null (0.15 against 0.41), in every run.

### What the loops are made of

`analysis/loop_anatomy.py`, results in `analysis/loop_anatomy_01a10613.json`.
TASK-77 adopts sliding-window persistence only if the result is also
*interpretable in terms of the trajectory*: a cycle that can be pointed to in
the captions. So the 62 runs whose longest H1 bar beats all nineteen draws at
window 20 were taken apart. Ripser names the edge whose arrival closes each
loop and the chord that fills it; the rest of the cycle is the shortest path
between the closing edge's ends below the birth scale. A vertex is a window, a
window is a time, and a time has a caption.

**The cycles are real and they can be pointed at. They are not circuits.**

Where they live in time:

| | median | quartiles |
| --- | --- | --- |
| closing edge spans | 1 state | 1 to 45 |
| filling chord spans | 64 states | 29 to 185 |
| whole cycle spans | 255 states | 101 to 518 |
| non-adjacent hops in the cycle | 13 | 9 to 19 |
| closing edge, in median steps | 1.7 | 1.3 to 2.4 |

The edge that closes the loop joins **adjacent** windows in 41 of the 62 runs,
and is only about 1.7 times a typical step. So the loop is not shut by a
long-range return. It is shut by one slightly larger than usual step, and the
far side of it is a route through a dozen or so temporally scattered hops.

What they are made of. A cycle carries a median of 34.5 windows, nearly all
with distinct captions (33.5), but only **7 distinct caption openings**. The
states on a cycle sit 0.107 apart on average, against 0.141 across the whole
run and 0.089 across the run after it settles: a cycle covers roughly the
extent of the settled run, not a tour of it.

Whether it is a circuit. 55 of 62 cycles visit three or more openings, which
sounds like a tour. But 60 of 62 **revisit an opening they had already left**,
a median of eight times, and only 28 return to the one they started from. The
sequences shuttle rather than circulate. Three runs picked near the median:

| Run | Sequence round the cycle |
| --- | --- |
| ZImageTurbo + Gemma4, "a man standing in a doorway" | `ABCDEBCFCECBGABD` |
| SD35Medium + Gemma4, "firefighters responding to a building fire" | `ABACACADEABDBFAEABAFAGCG` |
| ZImageTurbo + Gemma4, "a city slowly turning into a forest" | `ABACDACAEBDBFDGDAD` |

And the openings those letters stand for, for the first of them:

- A: "this is a studio photograph featuring"
- B: "this is a studio portrait of"
- C: "a full-length studio portrait features a"
- D: "a full-body shot features a young"
- E: "a full-length studio photograph features a"
- F: "a full-length studio shot features a"
- G: "a young man with dark, curly"

Seven ways of opening the same sentence about the same photograph. The other
two are the same: seven ways of saying "a dramatic, atmospheric, low-light
photograph", seven ways of saying "a vibrant autumnal landscape". The count of
distinct openings reads **phrasing, not subject**, which is why seven looks
like variety and the text does not.

So a loop is a hole in the cloud of paraphrases a captioner produces for one
settled scene. The run does not leave a theme, go somewhere else and come back.
It jitters among ways of wording the same description, and that jitter is not
shaped like a Gaussian with the same drift, which is what makes the bar beat
the null.

**A caution about how this was read.** The first examples pulled out were
chosen by span per hop, which selects the most degenerate cycles: one of them
was seven windows all saying "a large tree engulfed in flames", another
nineteen windows all saying "a grid of 36 red apples". Those are real cycles
but they are the tail, not the middle, and reading them first overstates the
case. The examples above are picked near the median on purpose, and
`loop_anatomy.py` picks them that way.

One claim that does **not** hold: cycles are not simply a post-settling
phenomenon. The share of a cycle's windows after state 300 is bimodal (median
0.90, quartiles 0.06 to 1.00); 33 of 62 lie mostly after it, the rest straddle
or precede it.

### What this changes

Rips persistence on a run's state set can be dropped from the headline
analyses on the evidence of both lengths. **Sliding-window persistence should
be dropped too, but for a different reason.** It is not that it finds nothing:
at 1,000 states its bars beat a drift-matched null in about 30% of runs, which
is a real effect and not one the panel could see. It is that what the bars are
made of is not what TASK-77 asks for. The cycles are geometry in a captioner's
paraphrase cloud, not trajectories going round anything, so they will not carry
the interpretation a Results section would need to put on them.

The distinction that survives is still worth stating: order-blind topology
(Rips on the state set) says nothing the distances do not, at either length,
while order-aware topology says nothing at 150 states and something at 1,000.
The something just is not a cycle in any sense a reader would expect.

## What this does and does not settle

- It settles TASK-77 for **Rips persistence** at both 150 and 1,000 states:
  no loops, and nothing the distances miss. The case is stronger at 1,000,
  where the diagrams are more redundant, not less.
- It now settles sliding-window persistence too, and against it. At 1,000
  states its bars beat the null, but the cycles behind them are paraphrase
  geometry rather than circuits, so TASK-77's second condition fails. See
  "What the loops are made of".
- What has **not** been tested is whether the excess bar length is a property
  worth reporting under another name. Real paraphrase clouds are not shaped
  like Gaussians with the same drift, and the H1 bar is one way of noticing
  that. Local intrinsic dimension and clumpiness, already in this document,
  are others, and they are easier to explain.
- The null is matched on drift and local dimension but not on global spread or
  on how evenly distances fluctuate. A better null would be built from the
  runs' own steps rather than from a Gaussian. That would sharpen the
  recurrence figure and leave the Rips result as it is, since real runs
  are below the null there. It now also bears on the sliding-window result,
  which is a positive finding and so rests on the null more heavily than a
  negative one does.
- The follow-up's six networks are not the panel's sixteen, and its five
  prompts are one per group rather than four. Comparisons between the panel
  and the follow-up carry that confound; comparisons between the follow-up
  whole and the follow-up cut to 150 states do not.
- `ripser_parallel` is not deterministic above one thread: at four threads it
  dropped an H1 bar on about one diagram in 350, enough to make the
  Gaussian-cloud null differ between two runs of `tda_keep_kill.py` on the
  same data. Both scripts now compute every diagram on one thread. The panel's
  committed `null_*` columns were produced at four threads and so are
  reproducible only to that tolerance.
