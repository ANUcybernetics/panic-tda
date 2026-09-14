# Can a 300-invocation panel resolve the escape times RQ1 asks for?

Measured 2026-09-14 by Sungyeon Hong with `analysis/msm_pipeline.py`, an
implementation of TASK-76's pipeline, over `019f3645_parquet` (the
`balanced_panel_5x5` export: 2,000 runs, 25 networks, 25 text states per run,
Qwen3Embed at 256 dimensions). Numbers in `analysis/msm_pipeline.json`. Cost and
horizon arithmetic from the tables in `long-horizon-design.md`.

The question (TASK-90, TASK-76): the committed panel is 300 invocations, which
is 150 text states, and both step size and drift from t0 plateau by invocation
100--150. RQ1's headline observable is the escape time between metastable
regions. This asks whether that observable is reachable at that horizon, and how
fine a partition the horizon can support --- before the GPU time is spent rather
than after.

Nothing here disputes the horizon argument in `long-horizon-design.md`. The
claim that the longest resolvable implied timescale scales with aggregate
sampling time rather than single-trajectory length (Sinitskiy & Pande 2018) is
correct, and it is the right reason to prefer many short trajectories over a few
long ones. It governs the *statistical error* on a timescale. The two findings
below concern what happens either side of that: what is observable at all, and
how fine a partition the frame count supports.

## Verdict

**An escape longer than a trajectory cannot be observed in it, and aggregate
sampling does not change that.** More trajectories reduce the error on a
timescale that is already visible; they do not make a 200-state escape appear in
a 100-state window. After burn-in a 300-invocation run leaves about 75--100
stationary text states, so escape times much beyond that are extrapolation.

**The standard pipeline does not announce this.** On 25-state trajectories it
returned escape times of 169 to 991 text states, dwell-time verdicts from single
observations, and implied timescales that never converged --- all without
comment. The failure is silent and has to be tested for deliberately.

**"A few hundred microstates" is not a budget the committed horizon supports
per cell.** At 20 prompts x 2 runs a cell holds 40 trajectories; at 150 text
states that is 6,000 frames, or 0.15 lag-1 transitions per entry of a 200x200
count matrix.

## What a run leaves after burn-in

| design                       | text states/run | burn-in | stationary states/run | trajectories/cell | stationary frames/cell |
| ---------------------------- | --------------- | ------- | --------------------- | ----------------- | ---------------------- |
| committed 4x4, 300 invocations | 150           | 50--75  | 75--100               | 40                | 3,000--4,000           |
| 1,000 invocations            | 500             | 50--75  | 425--450              | 40                | 17,000--18,000         |

Burn-in is paid once per trajectory, so the committed design spends roughly 40%
of its frames reaching the stationary regime, against about 12% at 1,000
invocations. That is the part of the cost that shortening the horizon does not
save.

## The failure is silent

Run on `balanced_panel_5x5` at 25 text states per trajectory --- a sixth of the
committed panel's horizon --- with 80 trajectories, a 12-microstate partition
and 3 metastable sets. The partition itself was reasonable (adjusted Rand 0.68
against refits on 80% subsamples). Everything downstream was not:

| diagnostic                                   | value                                  |
| -------------------------------------------- | -------------------------------------- |
| microstates in the largest connected set      | 5 of 12 at lag 1, 4 of 12 by lag 5     |
| frames left unassigned by PCCA+               | 41.5%                                  |
| slowest implied timescale, lag 1 -> lag 8     | 181 -> 348 text states, still climbing |
| escape times returned                         | 169 to 991 text states                 |
| observed residences behind each dwell verdict | 1 to 3                                 |

Every escape time is between 7 and 40 times the trajectory length. The implied
timescales never flatten, which is the model telling us it is invalid --- but
that signal lives in a separate diagnostic, and the escape times and dwell
verdicts are emitted with the same confidence either way. One metastable set was
labelled "approximately exponential" on the strength of a single observed
residence.

`msm_pipeline.py` now refuses both: a dwell verdict needs at least 20 observed
residences, and an escape time beyond half the trajectory length is returned
flagged as extrapolation rather than measurement. Whatever the panel's horizon,
those guards should stay on, because this is not a failure the numbers announce.

Caveat on the demonstration: this export is shallow, predates the v2 lineup, and
its embeddings predate TASK-96, so its vectors were mean-pooled. The figures
above characterise the method's behaviour on short trajectories. They are not
statements about the system.

## The microstate budget

A transition matrix over k microstates has k^2 entries to fill from
(frames - trajectories) lag-1 transitions per cell:

| design                  | frames/cell | sqrt(N) | transitions per entry at k=200 | k for ~5 per entry |
| ----------------------- | ----------- | ------- | ------------------------------ | ------------------ |
| committed 4x4, 300 inv  | 6,000       | 77      | 0.15                           | 34                 |
| 1,000 invocations       | 20,000      | 141     | 0.50                           | 63                 |

Count matrices are legitimately sparse and PCCA+ operates on the connected part,
so neither row is fatal. But the microstate count is a per-cell budget set by the
horizon, not a free parameter, and TASK-76's "a few hundred" is above what either
design supports per cell. It should be chosen from the frame count and recorded
in methods.

One mitigation costs no horizon: fit a single frozen partition on the pooled
stationary states of every cell and estimate each cell's transition matrix on
that shared partition. It is what "a frozen corpus" in TASK-76 AC#1 already
implies, and it makes metastable sets comparable across cells, which RQ2 needs
anyway. It does not change per-cell transition sparsity, so the budget question
stands.

## Also measured

Exact caption repetition over the same export, as the descriptive statistic
TASK-76 AC#3 asks for and not as a state definition: 517 repeats in 48,000
consecutive pairs, a rate of 1.08%, at a median caption length of 87 words.
Consistent with the length dependence in `research-programme.md` --- verbose
captioners do not repeat.

## What follows

The horizon question is empirical and can be answered cheaply before launch.
`db/length_5000_experiment.sqlite` (experiment `067efc98`) holds 128 runs at
2,500 text states on the SMC networks. Re-embedding those stored captions with
Qwen3Embed costs a few GPU-hours and no generation, and fitting this pipeline to
2,500-state trajectories would give the order of magnitude of the slowest
timescale in a loop of this shape. Old models, so it is a prior and not a
substitute --- but if that timescale sits well inside 100 text states the
committed horizon is safe, and if it is 400 the panel should be deeper before it
launches.

Failing that, the panel launches as committed with the resolution guards on, and
cells whose implied timescales do not converge are reported as unresolved. That
is what `long-horizon-design.md` already specifies. The risk is only that the
fraction of unresolved cells is discovered after three weeks of GPU rather than
estimated in an afternoon beforehand.
