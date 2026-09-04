---
id: TASK-93
title: Record the text-to-image seed for every invocation
status: In Progress
assignee: []
created_date: '2026-09-04 06:38'
updated_date: '2026-09-04 12:56'
labels:
  - instrument
  - engine
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The research programme lists as a standing constraint: "Seeds are random and
recorded. Every text-to-image invocation draws its own seed and stores it, so
within-condition variation is attributable and any step can be regenerated."
That is the design, not the code. Checked 2026-09-04:
panic_models._invoke_t2i_single and invoke_t2i_batch both pass generator=None,
so diffusers draws from the ambient torch RNG and nothing is kept; the
invocations table has no seed column, and "seed" appears nowhere in lib/ outside
the benchmark harness (gpu.bench.ex) and the dummy embedding model.

WHY IT MATTERS FOR TASK-90. Without the seed, a step cannot be regenerated, an
anomalous image cannot be re-examined, and within-condition variation cannot be
separated into "the seed did that" and "the caption did that" --- which is
exactly the split TASK-89 measures on a small caption set and which RQ2 needs
across the panel. It also means an interrupted-and-resumed run silently changes
the noise draw for the steps it redoes.

The machinery already exists: panic_models._t2i_generate_seeded takes per-item
seeds and passes a list of generators, so the i-th image depends only on
seeds[i]. It was written for the batching-parity benchmark (TASK-74) and
deliberately kept off the production path. What is missing is drawing a seed per
invocation, threading it through invoke_t2i/invoke_t2i_batch, and persisting it
on the invocation.

Settle before TASK-90 starts, since a run without seeds cannot be given them
afterwards.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every text-to-image invocation draws its own seed, passes it to the
      pipeline, and stores it on the invocation
- [ ] #2 Batched invocations get per-item seeds, so an item's image depends only
      on its own seed
- [ ] #3 Regenerating an invocation from its stored seed and input text
      reproduces the stored image, verified on at least one model per pipeline
      family
- [ ] #4 Resume behaviour under seeds is correct and stated: a resumed run draws fresh seeds for the steps it has yet to do, and completed steps keep the seeds they were run with
      it redoes
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
IMPLEMENTED 2026-09-04, apart from the GPU verification in AC#3, which waits for the embedding recompute to release the card.

Elixir draws the seed and Python consumes it, so recording is by construction rather than by reporting: GenAI.draw_seed/0 draws one per text-to-image invocation, RunExecutor stores it on the invocation and hands it down, and panic_models.invoke_t2i/invoke_t2i_batch now REQUIRE a seed --- there is no unseeded path left to fall back to. Batches pass a list of per-item generators, so image i depends only on seeds[i], and the chunking and retry behaviour is unchanged. The retry path regenerates at the same seed, which is what it should do.

Invocation gains a nullable `seed` integer (migration 20260904124059). Nullable because image-to-text steps have none and every invocation recorded before this does not either.

Covered by a non-GPU test over the dummy models: every text-to-image invocation carries an integer seed, captions carry none, and seeds within a batched step are all distinct --- which is the property that would break if a seed were drawn per step or per run rather than per invocation.

ON AC#4, WHICH WAS WRONG AS WRITTEN. Resume starts at last_invocation.sequence_number + 1, so it only runs steps that were never completed; there is no stored seed to reuse, because the invocation does not exist. Replaced with the property that is actually true and worth asserting.

SEPARATE BUG FOUND WHILE READING RESUME --- see TASK-97. resume_batch/2 takes the MINIMUM completed sequence across the batch and restarts every run there, so any run that got further would be re-run at a sequence number it already has, violating the unique index on (run_id, sequence_number). Reachable whenever a crash lands mid-batch-write, which over a multi-week run is likely.
<!-- SECTION:NOTES:END -->
