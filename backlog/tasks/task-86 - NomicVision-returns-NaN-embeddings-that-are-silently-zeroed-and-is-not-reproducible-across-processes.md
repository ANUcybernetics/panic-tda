---
id: TASK-86
title: >-
  NomicVision returns NaN embeddings that are silently zeroed, and is not
  reproducible across processes
status: Done
assignee: []
created_date: '2026-09-03 00:54'
updated_date: '2026-09-03 12:53'
labels:
  - bug
  - embeddings
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found 2026-09-03 while computing the image-embedding column of the TASK-83 step sweep.

_embed_nomic_vision in priv/python/panic_models.py ends with a guard that replaces any NaN row with zeros:

    nan_mask = torch.isnan(batch_embs).any(dim=1)
    if nan_mask.any():
        batch_embs[nan_mask] = 0.0

so a total numerical failure reaches the caller as well-formed all-zero unit-less vectors with no error, warning or log line. Callers cannot tell a zero vector from a real embedding. This is the same class of silent data-quality failure decision-01 was written to stop.

Observed behaviour, embedding the same saved PNGs repeatedly:
- most fresh processes return all-zero vectors for every image (NaN throughout), regardless of batch size (tested 1, 2, 4, 60)
- a minority of processes return valid unit-norm vectors
- within a process it is deterministic: the same call twice gives bit-identical output
- across two processes that BOTH produced valid unit-norm vectors, the resulting cosine similarities on identical input differed by up to 0.44 (step-sweep img_cos for SD35Medium at 10 steps came out 0.735 in one run and 0.556 in another)

So even the successful runs are not reproducible. No 'newly initialized weights' warning appears at load, so the cause is not obviously an uninitialised head; it was not investigated further.

Consequences: the img_cos column of analysis/step_sweep/results.json is unusable and has been nulled. Any historical NomicVision embeddings in the database are suspect --- there are large numbers of them from the SMC era (see completed TASK-39, TASK-41, TASK-46). No current experiment uses image embedding, so nothing in flight is affected.

Repro: analysis/step_sweep_imgcos.py (raises if it sees zero vectors).

Note the overlap with TASK-84, which proposes removing NomicVision, JinaClipVision and ColNomicVision outright. If NomicVision is removed this becomes moot for that model, but the silent-zeroing pattern should still be checked in the remaining image embedders, and the guard should fail loudly rather than fabricate zeros wherever it is kept.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Silent zeroing replaced by a loud failure (raise or logged error) wherever a NaN guard is kept in the embedding paths
- [x] #2 Root cause of the NomicVision NaNs and cross-process variation identified, or the model removed under TASK-84 with the finding recorded
- [x] #3 Existing NomicVision embeddings in the database audited: how many are all-zero, and whether any published analysis depended on them
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Resolved 2026-09-03 by removal rather than repair. Ben: the vision embeddings 'never really worked' and are unnecessary, because in an alternating network every second state is already text --- and a caption is itself a representation of the image before it, so the image states are still observed, through the captioner, which is the thing under study.

AC #1: the silent NaN-to-zero guard lived only in _embed_nomic_vision and is gone with it. No other embedding path in panic_models.py contains an isnan guard or writes a zero vector on failure (checked).
AC #2: NomicVision removed under TASK-84; root cause not investigated, which is now moot. The observed behaviour is recorded here and in the TASK-83 log entry: most fresh processes returned all-zero vectors regardless of batch size, and two processes that both returned valid unit-norm vectors disagreed by up to 0.44 in cosine on identical input.
AC #3: audit done --- there are ZERO image embeddings in any database. priv/panic_tda_dev.db holds 147,193 rows, all Qwen3Embed; the two root-level .db files are empty stubs. So no published or in-progress analysis can have depended on NomicVision output.

Image embedding is now removed from the pipeline entirely, not just its models: the Elixir model lists, the model_type/1 text-vs-image split, the embeddings-stage branch, embed_images in panic_models.py, the DummyVision dummies and their tests, and the doc tables. analysis/step_sweep_imgcos.py is deleted, since it existed only to recompute the unusable column.
<!-- SECTION:NOTES:END -->
