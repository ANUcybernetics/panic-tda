# Penguin-campfire paraphrase-FTLE analysis — design

Status: design/discussion spec (no implementation yet)
Related task: `backlog/tasks/task-71`
Author: Ben Swift
Date: 2026-04-21

## Context

Sungyeon proposes framing this project as "study of AI as a dynamical system",
with sensitivity-to-initial-conditions as the hook (journals: Chaos, Entropy).
The original commitment was to report results to her by April 20; that date
has slipped by one day.

The `penguin_campfire` experiment (id prefix `019d2ec7`, completed 2026-04-16)
already contains the data needed to report the first two of her three
variation-types:

1. **Identical prompt**: 8 runs per prompt share the same initial text,
   differing only in model stochasticity. Within-prompt divergence gives a
   noise floor.
2. **Semantic paraphrase**: 5 near-identical phrasings of
   "a penguin sitting by a campfire". Between-prompt divergence within this
   set gives the paraphrase FTLE.
3. **Controlled semantic perturbation** (Sungyeon's 5 categories): physics
   violation, size reversal, role inversion, diet violation, imaginary
   objects. **Not covered by existing data — design only, implementation
   deferred.**

This spec covers case 1 and 2 analysis (post-hoc, no new GPU runs) plus a
written proposal for case 3 to send to Sungyeon alongside the initial
results.

## Goals

- Produce a short writeup (`report.md` + figures) that lets Sungyeon evaluate
  whether sensitivity-to-initial-conditions is a viable framing, using
  existing `penguin_campfire` data only.
- Include a proposal for the 5-category experiment, with specific asks of
  her, so her reply drives the next task (implementation of case 3).
- Avoid any change to the existing `LyapunovStage` / `LyapunovResult` schema.

## Non-goals

- Running new GPU experiments (case 3 is design-only in this spec).
- Changes to the core pipeline or schemas.
- Choice of publication venue.
- Automating the 5-category sweep (that's the follow-up task once Sungyeon
  has given feedback).

## Data source

- Experiment: `penguin_campfire` (id prefix `019d2ec7`).
- Networks: 9 — {SD35Medium, Flux2Klein, GLMImage} × {Moondream, Qwen25VL,
  Pixtral}.
- Prompts: 5 near-paraphrases of "a penguin sitting by a campfire".
- Runs: 8 per (network, prompt) = 360 runs.
- Max length: 200 invocations.
- Embedding models: Nomic, Qwen3Embed (both present in database despite
  current `config/penguin_campfire.json` listing only `Qwen3Embed`).
- Existing `LyapunovResult` rows: 9 networks × 5 prompts × 2 embedding
  models = 90. These are the identical-prompt baseline.

## FTLE definition

The existing `PanicTda.Models.Lyapunov.compute_ftle/5` uses:

1. Stack trajectories into `(num_trajectories, num_timesteps, dimension)`.
2. For each timestep `t`: compute mean pairwise Euclidean distance
   across all trajectory pairs (`scipy.spatial.distance.pdist`, `mean`).
3. Clamp with `epsilon = 1e-10`, take natural log.
4. Fit a linear slope with `np.polyfit(t, ln_divergence, 1)`; slope is the
   FTLE. Also report `r_squared` and `num_pairs`.

This spec keeps the same definition and clamp, changing only which pairs
contribute at each timestep.

## Cross-prompt FTLE

For a prompt-pair (p₁, p₂) in the same (network, embedding) cell, and
trajectories A = 8 runs of p₁ and B = 8 runs of p₂:

- At each timestep t, compute `scipy.spatial.distance.cdist(A[:, t, :], B[:, t, :])` —
  an 8×8 matrix of Euclidean distances across prompts, no within-prompt
  pairs.
- Mean of that matrix is the cross-prompt divergence at t (64 pairs).
- Clamp, log, polyfit identical to within-prompt case.

Aggregation choice: **per-pair** (10 values per cell), not pooled. Reason:
lets Sungyeon see within-category-spread vs between-category-gap directly.

Yields 9 networks × 10 pairs × 2 embeddings = 180 paraphrase-FTLE values.

## Architecture

Three new files. No edits to existing pipeline code.

### `lib/mix/tasks/analyse.paraphrase_ftle.ex`

Mix task: `analyse.paraphrase_ftle <experiment-id-prefix>`.

Responsibilities:

- Resolve experiment by id prefix (match existing `experiment.status` style).
- Load runs, embeddings via Ash; group by `(network, prompt, embedding_model)`.
- For each `(network, embedding_model)` cell, iterate over the `C(n,2)` prompt
  pairs and call the Python cross-prompt FTLE function through Snex.
- Load existing `LyapunovResult` rows for the same experiment to harvest the
  identical-prompt baseline.
- Emit `ftle_values.csv` with one row per value (see schema below).
- Call the plotting functions in the same Python module to produce
  `ftle_grid.png` and `divergence_curves.png`.
- Write an empty `report.md` stub with headings, figures inlined, and TODO
  markers where narrative prose is needed. Authoring the prose is a manual
  step, not automated.

### `priv/python/penguin_analysis.py`

Pure Python module, invoked via Snex. Follows the `panic_models.py`
convention. Exposes:

```python
def cross_prompt_ftle(trajectories_a_b64, trajectories_b_b64,
                      num_runs_a, num_runs_b, num_timesteps, dimension):
    """
    Returns {exponent, r_squared, divergence_curve, num_pairs, num_timesteps}
    matching the return shape of PanicTda.Models.Lyapunov.compute_ftle.
    """

def plot_ftle_grid(csv_path, out_path):
    """
    3x3 panel (one per network). Each panel: strip plot with two x categories
    (identical, paraphrase), colour-coded by embedding model. y axis: FTLE.
    Saves PNG.
    """

def plot_divergence_curves(experiment_id, network, embedding_model, out_path):
    """
    Single panel, log-y. Two curves: mean within-prompt divergence (averaged
    over the 5 prompts) and mean between-prompt divergence (averaged over the
    10 pairs). x axis: invocation step 0..199.
    """
```

Plotting uses matplotlib only, to avoid adding new Python deps to the Snex
venv spec. If matplotlib isn't already in the Snex venv, the mix task adds
it — cheap.

### `analysis/penguin_campfire/`

Output directory, committed to the repo.

- `ftle_values.csv` — 270 rows total.
  - Columns: `experiment_id`, `network` (pipe-separated), `embedding_model`,
    `category` ∈ {`identical`, `paraphrase`}, `prompt_or_pair` (the prompt
    text for `identical`, or `"p1 || p2"` for `paraphrase`), `ftle`,
    `r_squared`, `num_pairs`, `num_timesteps`.
- `ftle_grid.png` — main comparison figure.
- `divergence_curves.png` — qualitative figure for one representative
  (network, embedding) cell. Pick rule: the cell that maximises
  `(median(paraphrase_ftle) − median(identical_ftle)) / mad(identical_ftle)`
  (i.e., cleanest separation, robust to outliers). The mix task prints the
  chosen cell so it can be overridden with a `--cell network,embedding`
  flag if the automatic pick is uninformative.
- `report.md` — writeup for Sungyeon.

## Data flow

```
Ash: LyapunovResult  ──────► identical-prompt FTLE values ──┐
                                                             ├──► ftle_values.csv ──► plots + report.md
Ash: Embedding       ──► cross-prompt FTLE (new Python) ────┘
```

Identical values come straight from `LyapunovResult.lyapunov_data.exponent`,
no recomputation. Paraphrase values come from the new Python code.

## `report.md` structure

Not implemented in code — authored manually but stubbed by the task. Sections:

1. **Setup** (~150 words) — dynamical-systems framing; config summary;
   FTLE definition.
2. **Identical-prompt baseline** — short characterisation of 90 values
   (median, spread, outliers).
3. **Paraphrase FTLE** — same for 180 values.
4. **Comparison** — `ftle_grid.png`, one sentence per network row.
5. **Qualitative figure** — `divergence_curves.png`, one representative
   cell.
6. **Interpretation** (~200 words) — does within-category spread <
   between-category gap? Honest either way.
7. **Proposed next wave** (~300 words) — the 5-category experiment design,
   with specific asks of her:
   - MVP: physics-violation category only + matched controls, reusing
     existing 9-network grid, 5 distinct violating prompts + 5 matched
     non-violating, 8 runs each, 200 steps. ~2-3 days GPU.
   - Full sweep cost: 5 cat × 5 prompts × 2 controls × 9 networks × 8
     runs × 200 steps.
   - Open questions for her:
     - Who writes the violating and control prompts?
     - Cut the 9-network grid down to 3 to afford more prompts per
       category?
     - Add paraphrases of each violating prompt too (nested design)?

## Testing

- **Unit test for `cross_prompt_ftle`** — Python test using synthetic
  trajectories with two Gaussian clusters separating at a known rate
  `e^(λt)`; assert recovered exponent is within a small tolerance of λ.
  Location to match existing conventions (`test/` with Snex smoke or
  `priv/python/tests/` — check at implementation time).
- **Integration smoke test** — run the mix task against experiment
  `019d2ec7`; assert the CSV has 270 rows and both PNGs are non-empty.
- **Manual verification** — spot-check a handful of FTLE values by
  re-plotting the raw divergence curves for those (network, embedding,
  prompt-pair) triples; ensure `report.md` numbers match the CSV.

## Edge cases

- Embedding model missing for some runs in the experiment → log warning,
  skip, matching existing `LyapunovStage` behaviour.
- Prompt with fewer than 2 runs → skip in paraphrase computation too
  (`LyapunovStage` already skips in identical case).
- Divergence curve with non-positive values → reuse the `epsilon = 1e-10`
  clamp pattern.
- Experiment not matching penguin_campfire shape (different prompt count,
  different network count) → task must be generic. No hard-coding of
  "5 prompts" or "9 networks". Works for any experiment with ≥2 prompts
  and ≥2 runs per prompt.

## Out of scope for this spec (handled by a follow-up task after Sungyeon
replies)

- Implementation of the 5-category experiment config and runs.
- Any schema change to add a `group_tag` or category field to `Run`.
- Modifying `LyapunovStage` to support multi-prompt groupings.
- Choosing a publication venue.
- Rebuilding `penguin_campfire` with different prompts or settings.
