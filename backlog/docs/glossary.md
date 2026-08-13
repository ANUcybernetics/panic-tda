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
