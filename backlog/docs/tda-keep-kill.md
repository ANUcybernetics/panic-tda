# Does topology add anything to the distance description?

Written 2026-10-08 by Sungyeon Hong for TASK-77, on TASK-90's panel (experiment
`01a09e21`: 16 networks, 20 prompts, two runs per prompt, 150 text states per
run). Measured with `analysis/tda_keep_kill.py` and
`analysis/sliding_window.py`; the numbers are in the JSON beside each.

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
- **Sliding-window persistence finds no cycles.** Fewer runs show a long
  loop than chance would give.
- **There is recurrence, but it is not topological.** Some runs leave a
  caption and later return to a near-verbatim copy of it. A plain recurrence
  count on distances finds this; persistent homology does not.
- **Two properties of the runs fall outside TASK-103's description:** captions
  are locally lower dimensional than any Gaussian with the same spread, and
  runs are clumpier, with groups of near-repeats separated by jumps. Both are
  read off distances directly.

**Recommendation:** drop persistent homology from the headline analyses and
keep it as a negative result in the methods discussion. Report intrinsic
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

**Sliding-window persistence finds no cycles.** A delay embedding of a run that
cycled through the same sequence of captions would show a long loop. Fewer runs
show one than chance gives, at every window.

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

## What this does and does not settle

- It settles TASK-77 for **Rips persistence on 150-state runs**: no loops,
  and nothing the distances miss.
- It does **not** test longer runs. The long follow-up (TASK-104, experiment
  `01a10613`, due about 25 October 2026) has 1,000 text states per run. A
  trajectory has more room to come round at that length. The scripts take any
  export, but its diagrams are already computed by the pipeline at that size and
  the null's Gaussian process is 1,000 by 1,000 per run. Both are affordable.
- The null is matched on drift and local dimension but not on global spread or
  on how evenly distances fluctuate. A better null would be built from the
  runs' own steps rather than from a Gaussian. That would sharpen the
  recurrence figure and leave the topology result as it is, since real runs
  are below the null there.
