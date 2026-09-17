# Glossary

Terms, abbreviations and conventions used across this repository, its backlog
and its analysis outputs. Started 2026-08-13 by Sungyeon Hong while getting
oriented; additions welcome.

## Repository conventions

| Term | Meaning |
|---|---|
| `TASK-NN` | A numbered item in `backlog/tasks/`. Each file carries YAML front matter with an id, status, labels, dependencies and priority, followed by a description and acceptance criteria. Completed items move to `backlog/completed/`. |
| Acceptance criteria | The checklist at the foot of a task file, marked `- [ ] #1`, defining what has to be true for the task to count as done. Wrapped in `<!-- AC:BEGIN -->` / `<!-- AC:END -->` markers so tooling can find them. |
| `backlog/docs/` | Findings and records, as distinct from work to be done. For example the model timing log and the dataset inventories. |
| ADR | Architecture Decision Record — a task or note that records a decision and its rationale rather than describing work. TASK-72, which switched the default text embedder to Qwen3 at 256 dimensions, is one. |
| `mise` | The tool-version manager used here. Erlang and Elixir are pinned through it, which is why commands are prefixed `mise exec -- mix ...`. |
| Ash | The Elixir framework providing the data model. Resources live in `lib/panic_tda/resources/`. |
| Snex | The library providing the persistent Python interpreter, so models stay loaded in GPU memory across calls. |

## Domain terms

| Term | Meaning |
|---|---|
| T2I / I2T | Text-to-image and image-to-text. The two model roles that alternate in a network. |
| Network | A list of models that cycle, each output feeding the next input. |
| Run | One trajectory: an initial prompt followed through a network for `max_length` steps. |
| Invocation | A single model inference event, with its input, output and timestamps. |
| Experiment | A batch specification: a network, prompts, embedding models and number of runs per prompt. |
| `max_length` | Number of model invocations per run. Because networks alternate text and image, the number of *text* states is roughly half this. |
| Cell | One combination of models in a factorial experiment — a single text-to-image model paired with a single image-to-text model — together with every run it contributes, across all prompts and all repeats. The long-horizon panel's "16 cells" are its 4 generators crossed with its 4 captioners, each holding 20 prompts × 2 runs = 40 trajectories. A cell is a *network* plus its runs within one experiment, and it is the unit in three separate senses: cells execute sequentially on the GPU, a cell is embedded and given its persistence diagrams as soon as it finishes, and the analysis fits one transition matrix per cell. |
| Panel | A factorial experiment: every text-to-image model crossed with every image-to-text model, run over the same prompts with the same number of repeats, so that model choice can be treated as an experimental factor. Named by its shape and size — `balanced_panel_5x5`, the 4x4 long-horizon panel. |

## Analysis methods

| Term | Meaning |
|---|---|
| EVoC | The clustering library used here, from the Tutte Institute — the same lineage as UMAP and HDBSCAN. Built for clustering large sets of embedding vectors. Produces a *hierarchy* of cluster layers, layer 0 finest and later layers progressively coarser. |
| HDBSCAN | Hierarchical density-based clustering. Used for the SMC 2025 analysis, before EVoC. |
| Medoid | The actual data point nearest a cluster's centre, as opposed to a centroid, which is an average and need not correspond to any real point. Clusters here are labelled by their medoid's caption text, which is how a cluster gets a human-readable name. |
| Outlier / label `-1` | Density-based clustering algorithms may leave a point unassigned, labelled `-1`, when it does not sit in any dense region. This is why an "outlier rate" exists at all — k-means and similar methods assign every point to a cluster by construction, so they cannot represent a point being *between* clusters. |
| PH | Persistent homology. Tracks topological features — connected components, loops, voids — appearing and disappearing as a distance threshold grows, summarised as a persistence diagram. |
| H0 / H1 / H2 | Homology dimensions: connected components, loops, and voids respectively. |
| Vietoris-Rips | The standard way of building a shape from a point cloud for persistent homology: connect any points closer than a threshold, then grow the threshold. |
| Persistence entropy | A single number summarising a persistence diagram, treating each feature's lifetime as a contribution to an entropy score. Low values suggest a few dominant long-lived features; high values suggest many short-lived ones. |
| FTLE | Finite-time Lyapunov exponent. In classical dynamics, the rate at which nearby trajectories diverge exponentially; a positive value indicates chaos. Estimated here by fitting a straight line to the log of mean pairwise distance over time. See TASK-73 for why it fits this data poorly. |
| MSM | Markov state model. Describes dynamics as jumps between discrete states with fixed transition probabilities, yielding dwell times, transition graphs and absorbing states. |
| Milestoning | In a Markov state model, assigning a point that lies between defined states to the last state it visited, so every timestep has a well-defined state. |
| Core set | The high-confidence dense interior of a cluster, used as a state in a Markov state model, as opposed to the cluster's diffuse edge. |
| Stationary distribution | The long-run distribution of where a system spends its time, independent of where it started. |
| Microstate | One cell of the fine partition a Markov state model is built on — here a k-means cluster on the unit sphere, a few dozen to a few hundred per analysis. Microstates are not meant to be interpretable on their own; they are the bookkeeping from which metastable regions are derived. |
| Lag time | The fixed time gap at which transitions are counted. A transition matrix is always "at lag τ", and a model's conclusions must not depend on which τ was chosen — which is what implied timescales test. |
| Implied timescale | A relaxation time read off the transition matrix, computed as −τ / ln λ for each eigenvalue λ at lag τ. Its use is diagnostic: a valid model's implied timescales flatten as τ grows. Timescales still climbing at the largest usable lag mean the model has not resolved the dynamics, and the cell should be reported as unresolved rather than extrapolated. |
| PCCA+ | Perron Cluster Cluster Analysis. Groups the microstates into a handful of metastable regions using the slow eigenvectors of the transition matrix, so regions are defined *kinetically* — by how rarely the system leaves them — rather than geometrically by density. |
| Metastable region | A group of microstates the system stays within for a long time relative to its motion inside them. The stochastic-dynamics replacement for "attractor", which is a deterministic term and is not used for this system. |
| Private region | A metastable region whose frames come from only one or a few trajectories — thick in frames, thin in trajectories. One run parked there fills it with within-region transitions, so it looks well sampled while the entries and exits an escape time rests on number one or two. Frames from a single run are not independent samples, so `msm_pipeline.py` reports each set's trajectory count and marks a set private below ten contributing trajectories, or when one holds over half its frames. Private regions across the ensemble are what non-ergodicity looks like: each run in its own corner, no shared stationary distribution, and escape times between regions undefined rather than merely imprecise. |
| Connected set | The largest strongly connected component of the count matrix, which is all an MSM is fitted on; microstates outside it are dropped. This is the only outlier mechanism a k-means partition has, and a blunt one — an isolated region is not modelled badly, it disappears. `frames_unassigned_pct` in the coarse-graining diagnostic is what it cost. |
| Escape time | How long the system takes to get from one metastable region to another, computed as a mean first passage time from the transition matrix. RQ1's headline observable. It is not bounded by the trajectory length; what bounds it is the number of crossings the ensemble of trajectories contains — see `escape-time-resolvability.md`. |
| Burn-in | The initial portion of a run, before the chain reaches its stationary regime, discarded before fitting. Measured here at roughly 50–75 text states, from the step-size and drift plateau. Burn-in is paid once per trajectory, so short trajectories spend a larger fraction of their frames on it. |
| Adjusted Rand index | A measure of agreement between two partitions of the same data, corrected so that chance agreement scores 0 and identical partitions score 1. Used to check that a clustering is a property of the data rather than of the particular sample it was fitted on. |
| Mixing time | How long a system takes to approach its stationary distribution — a bounded-space alternative to a Lyapunov exponent. |
| Eta-squared | The proportion of variance in one variable explained by group membership in another. Used here to show that caption length is almost entirely determined by which image-to-text model produced it. |
| p10 / p90 | The 10th and 90th percentiles: the values below which 10% and 90% of observations fall. Useful for describing a distribution's spread without being distorted by extremes. |
| L2-normalised | Each vector scaled to unit length, placing all points on a sphere. Euclidean distance then becomes a monotone function of cosine similarity, and distances are bounded in [0, 2]. |
| Matryoshka embeddings | Embeddings trained so that truncating to fewer dimensions still yields a usable vector, allowing dimension to be traded against cost. |
| Hartigan level-set view | The view that clusters are the high-density regions of a distribution, so a point in a sparse region genuinely belongs to no cluster rather than being misassigned. This is the justification for reading a high outlier rate as real sparsity in the space rather than as a clustering failure. Referenced in TASK-75. |
| Spectral gap | The distance between the largest and second-largest eigenvalues of a transition matrix. A large gap means fast mixing; a small one means the system lingers in metastable regions. Used to estimate mixing time. |
| Detailed balance | The property that flow from state A to state B equals flow from B to A. Violations indicate directed, irreversible dynamics rather than equilibrium fluctuation. |
| Difference-in-differences | A comparison design that measures how a quantity changes over time in one group relative to the change over the same period in a control group, so shared background trends cancel out. |

## Still to confirm

- **PKB** — referenced in TASK-73 as "PKB note 741". Meaning not established;
  ask Ben Swift.
