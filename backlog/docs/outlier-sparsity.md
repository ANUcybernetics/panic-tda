# Are EVoC's outliers sparse semantic space?

Measured 2026-09-05 over every Qwen3Embed embedding in the dev database: 147,193
captions from 3,896 runs across 12 experiments, the same pool
`mix cluster.recompute` clusters. Script `analysis/outlier_sparsity.py`, numbers
in `analysis/outlier_sparsity.json`, figures in `analysis/outlier_sparsity/`.
"Outlier" below means the stored layer-0 label unless a sweep is named.

The question (TASK-75): the research programme read the large unlabelled share
of the global clustering as genuine sparsity --- only dense regions are clusters,
in Hartigan's level-set sense, and an outlier is a state in transit between
them. The planned core-set Markov model (TASK-76) treats time spent as an
outlier as "time in transit", a real kinetic observable. This checks that.

## Verdict

**The sparsity interpretation does not survive, and "time in transit" is not
an observable that EVoC's labels can supply.** Outliers are as dense as
clustered points; the outlier share is stable under every hyperparameter but
the outlier _set_ is not; length and captioner explain nothing; and three
quarters of outlier time is runs that end in the outlier region or never leave
it, not runs passing through.

The 62% quoted when the task was written came from the mean-pooled vectors
(TASK-96). On the corrected embeddings the stored clustering leaves 34--41%
unlabelled depending on layer (40.4% at layer 0). The conclusions below do not
depend on which figure is used.

## Outliers are not locally sparse

Cosine distance to the k-th nearest neighbour, self excluded, over all 147,193
points:

| k   | group     | p5     | p25   | median | p75   | p95   |
| --- | --------- | ------ | ----- | ------ | ----- | ----- |
| 5   | outlier   | 0.0004 | 0.033 | 0.058  | 0.095 | 0.174 |
| 5   | clustered | 0.008  | 0.033 | 0.057  | 0.090 | 0.152 |
| 15  | outlier   | 0.013  | 0.057 | 0.094  | 0.148 | 0.238 |
| 15  | clustered | 0.018  | 0.050 | 0.082  | 0.127 | 0.204 |

The probability that a random outlier is sparser than a random clustered point
is 0.51 at k = 5 and 0.56 at k = 15, against 0.50 for no difference. Half of
all outliers are denser than the median clustered point. The distribution of
log distance is one skewed mode (`density.png`); the only second mode is a
spike at zero from exact caption repeats (1.4% of points at k = 5, 79% of them
Moondream, 56% of them outliers against a 40% base rate). A two-component
mixture beats one on BIC
because of that spike, not because there is a dense-cores-plus-diffuse-sea
split.

## The share is stable, the set is not

EVoC refit over a 300x range of `base_min_cluster_size` and the whole range of
`noise_level`, each fit 2--40 s:

| min size | noise | layers | finest clusters | finest outliers | coarsest outliers | ARI vs stored | stored outliers absorbed |
| -------- | ----- | ------ | --------------- | --------------- | ----------------- | ------------- | ------------------------ |
| 5        | 0.5   | 7      | 6,304           | 32%             | 40%               | -0.06         | 75%                      |
| 15       | 0.5   | 6      | 3,369           | 26%             | 40%               | -0.02         | 77%                      |
| 50       | 0.5   | 5      | 904             | 37%             | 40%               | 0.28          | 40%                      |
| 147      | 0.5   | 5      | 261             | 40%             | 40%               | 1.00          | 0%                       |
| 300      | 0.5   | 4      | 126             | 39%             | 40%               | 0.48          | 26%                      |
| 600      | 0.5   | 3      | 63              | 45%             | 33%               | 0.23          | 36%                      |
| 1,500    | 0.5   | 2      | 30              | 34%             | 40%               | 0.11          | 56%                      |
| 147      | 0.1   | 5      | 259             | 43%             | 41%               | 0.53          | 20%                      |
| 147      | 0.25  | 5      | 255             | 41%             | 13%               | 0.55          | 21%                      |
| 147      | 0.75  | 4      | 248             | 41%             | 36%               | 0.58          | 19%                      |
| 147      | 0.9   | 5      | 239             | 44%             | 28%               | 0.57          | 16%                      |

Outlier share is 26--45% everywhere, which would look like threshold-stability
if only the share were reported. The absorbed column says otherwise: at a
minimum size of 5, three quarters of the stored outliers get a label and a
different third of the data loses one, with an adjusted Rand index of zero
against the stored partition. Which points are outliers is a property of the
fit, not of the data.

The second pass makes this explicit. EVoC refit on the 59,513 stored outliers
alone, production parameters scaled to that size, finds 336 clusters at its
finest layer (median 6 runs each) and leaves 39% of the outliers unlabelled
again. The procedure labels about 60% of whatever it is given.

## Not a caption-length artefact

| captioner     | n      | mean words | outlier rate | by length tercile |
| ------------- | ------ | ---------- | ------------ | ----------------- |
| Moondream     | 36,400 | 23         | 0.38         | 0.36, 0.38, 0.39  |
| Qwen25VL      | 35,000 | 81         | 0.39         | 0.41, 0.37, 0.38  |
| Gemma3n       | 22,800 | 88         | 0.45         | 0.45, 0.45, 0.46  |
| Pixtral       | 32,400 | 96         | 0.42         | 0.45, 0.42, 0.40  |
| LLaMA32Vision | 16,600 | 102        | 0.38         | 0.38, 0.39, 0.37  |
| InstructBLIP  | 3,993  | 125        | 0.48         | 0.55, 0.48, 0.42  |

Outlier rate sits between 0.38 and 0.48 for every captioner with no ordering
by verbosity, and is flat within each captioner across its own length
terciles. A logistic fit of outlier status on standardised log density, log
length and captioner identity has coefficients of +0.05 (density) and -0.05
(length) per standard deviation, and predicts no better than the majority
class (59.6% either way). The rate is also spread across experiments (28--64%)
and networks (14--74% over 45 network cells) rather than concentrated in any.

## Where outlier time goes

The temporal structure is the decisive part. Every maximal outlier stretch
within a run, classified by what surrounds it:

| kind      | meaning                                | stretches | share of outlier steps | median length |
| --------- | -------------------------------------- | --------- | ---------------------- | ------------- |
| transit   | leaves one cluster, arrives at another | 1,441     | 10%                    | 2             |
| excursion | leaves and returns to the same cluster | 2,333     | 9%                     | 1             |
| start     | run begins as outlier, then clusters   | 730       | 6%                     | 2             |
| end       | run clusters, then ends as outlier     | 1,250     | 46%                    | 19            |
| whole     | run is never labelled                  | 530       | 29%                    | 25            |

Genuine transits are a tenth of outlier time and are short. Three quarters of
outlier time is runs that settle into the outlier region and stay (median 19
steps to the end of the run), or runs of which 14% never receive a label at
all. Outlier rate rises along the trajectory, from 0.33 in the first five steps
to 0.45 at steps 50--100, the opposite of transients. Clustered stays, for
comparison, have a median length of 2 steps and 36% of them are a single step.

The outlier region is also one place, not many. In the 15-nearest-neighbour
graph, 94% of an outlier's five nearest neighbours are outliers (clustered
points: 2%), and the outlier set restricted to that graph is essentially one
connected component: 93% of outlier points sit in components visited by five
or more runs, and only 4.5% in single-run components. Outliers do lean towards
their own run more than clustered points (73% of neighbours from the same run
against 56%), which is dwell, not isolation. So the picture is a large,
connected region of ordinary density, visited by thousands of runs, that many
runs settle into, which EVoC declines to partition and the labelled clusters
sit apart from.

## What an EVoC outlier actually is

EVoC builds a 15-nearest-neighbour graph, lays it out in a four-dimensional
node embedding (UMAP-style, initialised by label propagation), and runs
HDBSCAN on that layout with leaf extraction. An outlier is HDBSCAN noise in
the layout: a point that falls off the condensed tree above the leaf level.
That density is the layout's, not the 256-dimensional space's, which is why
local density in the embedding space does not predict the label. `noise_level`
is a repulsion coefficient in the layout optimiser, not an outlier threshold,
which is why it barely moves the share. See `evoc/clustering.py` and
`evoc/node_embedding.py` in the venv.

## What this changes

- **"Time in transit" is off the table** as an observable derived from EVoC
  labels, and with it the sparse-space reading of outliers in the paper.
  Outlier time is mostly settled time somewhere unlabelled.
- **TASK-76 cannot milestone outliers as transit.** Milestoning assigns a run's
  unlabelled tail to the last core it visited, which would credit a run's
  destination to a state it left twenty steps earlier, for 46% of outlier time.
  The core definition has to change. Options, in order of preference: a state
  definition that assigns every point (PCCA+ on a fine partition, or an HMM
  over embeddings, both already named as fallbacks in TASK-76); EVoC with
  `approx_n_clusters` or `base_n_clusters` to force a complete partition, with
  the stability check under subsampling that TASK-76 AC#3 already requires; or
  a trajectory-aware definition of cores as regions where runs dwell, since
  dwell is what the outlier region is full of.
- **The 40% is not a data property.** Any figure that reports an outlier share
  should say it is the procedure's, and no analysis should threshold on it.
- **Exact repeats lean outlier.** The zero-distance spike (mostly Moondream
  repeats, TASK-90's design evidence) is 56% outliers against a 40% base rate,
  so a repetition statistic computed over clustered points only would
  undercount it.
