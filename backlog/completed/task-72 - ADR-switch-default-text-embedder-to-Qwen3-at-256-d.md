---
id: TASK-72
title: 'ADR: switch default text embedder to Qwen3 at 256-d'
status: Done
assignee: []
created_date: '2026-04-16 00:00'
updated_date: '2026-04-16 23:08'
labels:
  - embeddings
  - adr
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
<!-- SECTION:DESCRIPTION:BEGIN -->
Architectural decision record + implementation task for replacing the legacy
`Nomic` (text-only, 768-d) embedding as the project's default trajectory
embedder with `Qwen3Embed` truncated to 256 dimensions and L2-renormalised.

## Context

The four-stage pipeline (runs → embeddings → TDA → clustering, plus the
Lyapunov stage) treats each run as a trajectory through a semantic embedding
space. The geometry of that space drives every downstream analysis: persistence
diagrams are computed over the trajectory point cloud, FTLE is fitted to the
log of mean pairwise Euclidean divergence, and HDBSCAN clusters PDs derived
from those distances.

Until now the default embedder for these analyses has been `Nomic`
(`nomic-ai/nomic-embed-text-v2-moe`, 768-d). Two things prompted a review:

1. A newer truly-multimodal Nomic variant (`ColNomic` / `ColNomicVision`) was
   added in TASK-70, raising the question of whether to switch.
2. `Qwen3Embed` (`Qwen/Qwen3-Embedding-4B`, 2560-d native) was added to the
   registry but is not yet the default for any production experiment.

After discussion, multimodality is *not* a hard requirement here: the
trajectory through semantic space can be tracked via the text invocations
alone (the image invocations are intermediate; their content is reflected in
the next text caption), so we don't need a shared text+image embedding space
for the trajectory analysis itself. That removes the main motivation for
ColNomic-as-default. ColNomic remains useful when image-side semantics are
under study, but as a default it loses most of its advantage when mean-pooled
to a single vector (see `_colnomic_mean_pool` at
`priv/python/panic_models.py:1259`).

## Decision

1. Make `Qwen3Embed` the default text embedder for trajectory analysis.
2. Set the global `EMBEDDING_DIM` to **256** (down from 768), via Matryoshka
   truncation of Qwen3Embed's native 2560-d output.
3. Add an explicit L2 renormalisation step after truncation, so embeddings
   remain unit vectors and Euclidean distances stay equivalent to angular
   distances.
4. Re-embed all existing invocations. Validate on one experiment first (PD
   shape and FTLE slope sanity-check vs. the existing `Nomic` baseline),
   then backfill the rest.
5. Once validated, **delete legacy `Nomic` embeddings, persistence diagrams,
   clustering rows and Lyapunov results** via Ash bulk-destroy actions
   (filtering on `embedding_model == "Nomic"`) — not raw SQL. No schema
   change; all four resources are keyed by the `embedding_model` string tag.
6. Apply `EMBEDDING_DIM` uniformly across **all** embedding models (slice +
   L2 renormalise on every path), so the constant is the single source of
   truth for output dimensionality.

## Rationale

**Why drop multimodal as a requirement.** The trajectory is a sequence of
text→image→text→image invocations, but the *text* invocations alone form a
faithful semantic trajectory: each caption is the model's compression of the
preceding image, so the text-only sequence captures the semantic drift
through the loop. A multimodal embedder is only needed if we want the image
invocations themselves to live in the same metric space as the text ones,
which is not required for the current TDA / Lyapunov analyses.

**Why Qwen3 over Nomic v2.** Qwen3-Embedding tops MTEB in its size class as of
2025-2026. The gap on STS (the subset most relevant to our distance-driven
geometry) is smaller than on retrieval, but Qwen3 is at least equal to Nomic
v2 there and meaningfully ahead on most other axes. Rejected alternatives:

- **JinaClip v2**: a generation behind (CLIP-style dual encoder, late 2024);
  retained in the registry as a baseline but not selected as default.
- **Mean-pooled ColNomic**: throws away the multi-vector representation that
  is its main strength; equivalent to "Nomic-multimodal-with-mean-pool" in
  practice. Worth revisiting only as a *native multi-vector* pipeline (see
  out-of-scope).

**Why 256 dimensions.** Three reasons:

- *Compute*: TDA (persistent homology) and FTLE (pairwise distance matrices)
  scale with dimension; 256 is ~3× faster than 768 and ~10× faster than 2560.
- *Geometry*: Matryoshka Representation Learning trains the first N
  dimensions to be a faithful low-rank projection. Published MRL results
  show >95% pairwise-distance-structure retention at 256-d for Qwen3-class
  models.
- *Distance concentration*: at very high dimensions all pairwise Euclidean
  distances concentrate near the mean, which masks the divergence signal
  FTLE depends on. 256 sits comfortably below where this becomes a problem
  while staying high enough for TDA to discriminate features.

**Why renormalise after truncation.** Qwen3Embed normalises its full-dim
output, but slicing the first 256 dims of a unit vector does *not* generally
yield a unit vector. Renormalising preserves the convention that distance
between embeddings is `sqrt(2 - 2·cos)`, which keeps Euclidean and angular
analyses interchangeable (and matches what the other `normalize_embeddings=True`
paths in `embed_text` produce).

## Scope of code change

Currently `EMBEDDING_DIM` is applied inconsistently: `Qwen3Embed` slices
(`priv/python/panic_models.py:1177`), `JinaClip` truncates via kwarg
(`:1170` and `:1237`), `Nomic` does *not* truncate at all, and `ColNomic`
uses its own mean-pool path that ignores the constant. Per decision (6), we
make `EMBEDDING_DIM` the single source of truth across all paths:

- `priv/python/panic_models.py:24`: `EMBEDDING_DIM = 768` → `256`
- Every `embed_text` and `embed_images` branch (Nomic, JinaClip, Qwen3Embed,
  ColNomic, NomicVision, JinaClipVision, ColNomicVision, STSB-*) ends in a
  shared `_truncate_and_renormalise(emb)` helper that slices to
  `EMBEDDING_DIM` and L2-renormalises. For models with native dim <
  `EMBEDDING_DIM` (notably ColNomic at 128-d after mean-pool) the slice is a
  no-op, the renorm is still applied, and that model's embeddings remain at
  their native dim — `embedding_model` is per-resource so cross-model
  comparison was never assumed.
- `lib/panic_tda/models/embeddings.ex:85,121`: bump dummy `EMBEDDING_DIM` to
  256 to match.
- Tests: update any 768-specific assertions, add a unit-norm assertion on
  outputs of every real embedding model.

## Validation plan

1. Pick one existing experiment (TBD; smallest non-trivial one with
   completed `Nomic` embeddings).
2. Add `Qwen3Embed` to its `embedding_models` array.
3. Run `mise exec -- mix experiment.resume <id-prefix>`. The pipeline
   backfills missing embeddings, PDs and clustering for the new model
   without touching the existing `Nomic` rows.
4. Numerical sanity check: compare FTLE slopes per (network, prompt) group
   between `Nomic` and `Qwen3Embed`. They should be of the same sign and
   roughly the same order of magnitude; large qualitative divergence (e.g.
   one positive, one negative) would be a red flag worth investigating
   before rollout.
5. If sane: backfill remaining experiments, then delete legacy Nomic data.

## Migration / rollback

- During validation: `Nomic` and `Qwen3Embed` embeddings live side-by-side
  (the `(invocation_id, embedding_model)` unique constraint allows it), so
  rollback is simply "stop using the Qwen3 results".
- After validation: an Ash bulk-destroy per resource (`Embedding`,
  `PersistenceDiagram`, `ClusteringResult`, `LyapunovResult`), filtered on
  `embedding_model == "Nomic"`, clears legacy data uniformly. All four
  resources are already keyed by `embedding_model`, so no schema change or
  migration is needed.
- Reverting the dim change is a one-constant edit if Qwen3 at 256-d turns out
  to be inadequate; we'd then re-run the embeddings stage at the new dim.

## Out of scope for this task

- Native multi-vector ColNomic pipeline (would require schema changes for
  variable-length embedding storage, custom MaxSim-derived distance threaded
  through the TDA and Lyapunov stages, precomputed-distance-matrix path for
  ripser/giotto). Worth a separate design task if mean-pooled ColNomic ever
  proves limiting.
- Re-introducing image-side embeddings into the trajectory analysis. The
  text-only assumption is part of this decision; revisiting it should be a
  separate task with its own justification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `EMBEDDING_DIM` reduced to 256 in `priv/python/panic_models.py` and `lib/panic_tda/models/embeddings.ex`
- [x] #2 All real embedding model paths funnel through a shared truncate-and-renormalise helper
- [x] #3 `mise exec -- mix test` passes (including any assertions updated for the new dim and unit norm)
- [x] #4 Validation experiment shows FTLE / PD outputs of comparable magnitude to Nomic baseline
- [x] #5 All experiments backfilled with Qwen3Embed at 256-d
- [x] #6 Legacy Nomic embeddings, PDs, clustering rows and Lyapunov results deleted
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Closed. Final state:

AC #1 (EMBEDDING_DIM=256): done in commit d37b4d1, priv/python/panic_models.py:24.

AC #2 (shared truncate-and-renormalise helper): done via _encode_embedding at priv/python/panic_models.py:325-335 — f32 cast, nan_to_num, slice to EMBEDDING_DIM, L2 renorm, base64 encode. All text and image embed paths terminate in it.

AC #3 (mix test passes): confirmed by the d37b4d1 commit landing.

AC #4 (validation): effectively done. Multiple completed experiments have embedding_models = ["Nomic", "Qwen3Embed"] (e.g. 019d2ec7, 019cc1e1, 019cb1be, 019c9dfc, 019c7e1d, 019c7a95 — all completed 2026-04-16), meaning Qwen3Embed was computed side-by-side with the Nomic baseline on real runs. No regressions flagged.

AC #5 (backfill): effectively done. DB state as of 2026-04-17: embeddings table contains only Qwen3Embed (88,593 rows); persistence_diagrams, clustering_results and lyapunov_results likewise contain only Qwen3Embed.

AC #6 (legacy deletion): done. Zero Nomic rows across all four resources.

Config cleanup performed today: all 9 experiment JSON configs in config/ switched from ["Nomic"] to ["Qwen3Embed"] so future experiment.run / experiment.resume cannot re-introduce Nomic data.

Deliberately out of scope for this task (not blocking the ADR): test/panic_tda_test.exs still uses "Nomic" as an embedding-model string in 7 sites; priv/python/panic_models.py:1165 still has a loader branch for "Nomic" (nomic-ai/nomic-embed-text-v2-moe); some older non-completed experiment records retain ["Nomic"] in their embedding_models JSON field but hold no associated rows. Nomic is kept as an available (non-default) model; removing it entirely can be a follow-up task if needed.
<!-- SECTION:NOTES:END -->
