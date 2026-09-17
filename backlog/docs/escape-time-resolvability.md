# Can a 300-invocation panel resolve the escape times RQ1 asks for?

Asked 2026-09-14 by Sungyeon Hong before TASK-90's launch; measured with
`analysis/escape_time_prior.py` (numbers in `analysis/escape_time_prior.json`)
and `analysis/msm_pipeline.py`, the TASK-76 pipeline. Cost and horizon
arithmetic from the tables in `long-horizon-design.md`.

The question (TASK-90, TASK-76): the committed panel is 300 invocations, which
is 150 text states, and both step size and drift from t0 plateau by invocation
100--150. After burn-in that leaves 75--100 stationary text states per run and
40 runs per cell. RQ1's headline observable is the escape time between
metastable regions. This asks whether that observable is reachable at that
budget, and how fine a partition the budget supports --- before the GPU time is
spent rather than after.

## Verdict

**Escape times longer than a trajectory are estimable from the ensemble, and
what bounds them is the number of crossings the ensemble contains, not the
length of any one run.** An escape time is a mean first passage time inferred
from the transition matrix at the lag, and every trajectory sees a fraction of
every escape however long it is. On a synthetic three-well chain, 40 windows of
100 states recover a true escape of 100 states exactly (101, interval 74--138,
from 18 observed crossings) and one of 400 states to within a factor of 1.4
(291, interval 201--620, from 11 crossings); a 1,000-state escape leaves the
count matrix disconnected at that budget (4 crossings), and 160 windows of 100
recover it (1,224, interval 678--5,478). So the aggregate-sampling argument in
`long-horizon-design.md` stands, and runs per prompt is the lever it says it
is.

**The per-cell budget resolves escapes up to a few hundred text states and no
further.** Ten crossings between two sets is about the least an estimate can
rest on, and at 4,000 stationary frames per cell that is an escape rate of one
per 400 states. Escapes slower than that are reported as unresolved, with the
crossing count and the interval that says so.

**What the old loop data says the escapes are.** The deepest trajectories the
project has, 128 runs of 2,500 text states on the four SMC networks, split two
ways. Three networks are repetition-stuck (29--52% of consecutive captions
identical, 10--22-word captions), and there the slow processes are individual
runs parked in private regions for thousands of states: implied timescales
climb without converging out to lag 50, and the satellite sets PCCA+ finds are
visited by one to four runs each. No affordable horizon resolves those, and
they are the non-ergodicity finding, not a resolution failure. The per-set
trajectory count in the guards below is what names this shape when it appears;
it is a statement about the ensemble rather than the partition, so repartitioning
does not help and only more runs would. The one network
with the v2 lineup's repetition rate (SDXLTurbo + old Moondream, 1.7%) has its
slowest implied timescale converged at about 150 text states, one dominant set
holding 94% of the stationary distribution, and satellites that runs return
from in 66--90 states but leave for only once in about 5,600. At the panel's
budget that return time is recovered to within a factor of two and the
excursion rate is not; at four times the budget the return time is nailed and
the excursion rate is still a lower bound.

**The pipeline's silent failure is real and is now guarded.** On the 25-state
`balanced_panel_5x5` trajectories it returned escape times of 79--382 text
states, dwell verdicts from single residences, and implied timescales over a
3-of-12-microstate connected set, all with the same confidence as a good fit.
The guards below make it say so.

## What the panel can and cannot deliver per cell

| observable                                   | at 2 runs/prompt (40 x ~100)                          | at 4 runs/prompt (80 x ~100)            |
| -------------------------------------------- | ----------------------------------------------------- | --------------------------------------- |
| implied-timescale convergence                | yes, for timescales under ~150 text states            | same, tighter                           |
| number and identity of metastable sets       | yes                                                   | yes                                     |
| escape time, balanced sets, under ~400 states | order of magnitude, 10--20 crossings                  | within a factor of 1.5                  |
| escape time, rare satellite, thousands       | lower bound only (about one excursion per cell)       | lower bound, two excursions             |
| dwell-time shape (exponential vs heavy)      | residences under ~50 states only; longer are censored | same                                    |

Two runs per prompt is enough to answer whether a cell has metastable
structure and to put its timescales in the right decade; four is what it takes
to put an interval on the escape time worth printing. The design doc's second
batch is the same aggregate as launching at four, so nothing is lost by
launching as committed and adding the batch where the first says it is needed.

The non-equilibrium start matters. Every run starts at a prompt, not from the
stationary distribution, and in the old data fitting on each run's first 100
stationary states rather than a random window pulled the excursion rate three
to four times low (1,453 against 5,609). Nüske et al. 2017 characterise this
bias for short trajectories from non-equilibrium starts and give a reweighting
correction; it is worth a line in methods and a check against the random-window
estimate when TASK-90's data arrives.

## The guards in `msm_pipeline.py`

- **Escape times carry their crossing count and a 95% interval** from a
  Bayesian MSM sampled on effective counts, and are marked resolved only when
  at least ten crossings between the two sets were observed at the lag. On the
  synthetic chain this passes the escapes that were recovered and fails the
  one that was not. The earlier version flagged any escape beyond half the
  trajectory length, which would have flagged the 101-state answer above.
- **Every metastable set reports how many trajectories its frames came from**,
  and is marked private when fewer than ten contributed or when one holds more
  than half of them. A set can be thick in frames and thin in trajectories: a
  run parked in a private region fills it with thousands of within-set counts,
  so the matrix looks well sampled exactly where the one or two entries and
  exits an escape rests on are all it has. The transition matrix cannot show
  this, having discarded which run each count came from, and the connected-set
  restriction may amputate such a set silently --- which reads as a clean fit
  unless `frames_unassigned_pct` is read alongside it.
- **A dwell verdict needs 20 complete residences**, and the first and last
  residence of every trajectory are censored and counted rather than measured.
  Dropping them biases the mean dwell short (a true mean of 50 states came back
  as 22--43 across the synthetic designs, because the long residences are the
  ones cut), so the shape verdict is trustworthy only for residences well
  inside the window. A survival estimate is the fix if that margin ever
  matters.
- **Implied timescales are computed against lag** and a cell whose slowest one
  has not flattened by the largest usable lag is reported as unresolved, which
  is what `long-horizon-design.md` already specifies. The lag used for
  coarse-graining is an argument (`--lag`) and is recorded in the JSON with the
  rest of the invocation.

## The microstate budget

A count matrix is sparse, so the figure that sets the partition's resolution
is frames per microstate (the row total), not entries in the k x k matrix:

| design                  | stationary frames/cell | frames per microstate at k=200 | k for ~50 per microstate |
| ----------------------- | ---------------------- | ------------------------------ | ------------------------ |
| committed 4x4, 300 inv  | 3,000--4,000           | 15--20                         | 60--80                   |
| 1,000 invocations       | 17,000--18,000         | 85--90                         | 340--360                 |

TASK-76's "a few hundred microstates" is above what the committed budget
supports per cell. The mitigation costs no horizon: fit one partition on the
pooled stationary frames of every cell (about 64,000 at the committed design)
and estimate each cell's transition matrix on that shared partition. It is what
"a frozen corpus" in TASK-76 AC#1 already implies, and it makes metastable sets
comparable across cells, which RQ2 needs anyway. Per-cell transition sparsity
is unchanged, so the connected-set fraction per cell is the thing to watch.

## Burn-in

| design                         | text states/run | burn-in | stationary states/run | trajectories/cell | stationary frames/cell |
| ------------------------------ | --------------- | ------- | --------------------- | ----------------- | ---------------------- |
| committed 4x4, 300 invocations | 150             | 50--75  | 75--100               | 40                | 3,000--4,000           |
| 1,000 invocations              | 500             | 50--75  | 425--450              | 40                | 17,000--18,000         |

Burn-in is paid once per trajectory, so the committed design spends roughly 40%
of its frames reaching the stationary regime, against about 12% at 1,000
invocations. That is the one argument for horizon over runs: 3.3 times the
cost buys 4.25 times the stationary frames, and residences up to a few hundred
states become observable whole. It is a ten-week panel rather than three, and
it does not reach the thousands-of-states excursion rates the old data shows
either.

## Also measured

Exact caption repetition over the `balanced_panel_5x5` export, as the
descriptive statistic TASK-76 AC#3 asks for and not as a state definition: 517
repeats in 48,000 consecutive pairs, a rate of 1.08%, at a median caption
length of 87 words. Consistent with the length dependence in
`research-programme.md` and with the 1.7% of the one old network whose
kinetics converged: verbose captioners do not repeat, and loops that do not
repeat mix.

## Caveats on the prior

The 5,000-step runs are unseeded, use truncated captions and the 2025 lineup
(FluxSchnell, SDXLTurbo, BLIP2, the 23-word Moondream), and cover two prompts
("yeah" and "nah") at 16 repeats. Their embeddings are STSBMpnet, not
Qwen3Embed. They give the order of magnitude of the timescales a loop of this
shape has and the behaviour of the estimator at the panel's budget; they say
nothing about the v2 networks' state structure, which is what the panel is
for.
