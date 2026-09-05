---
id: TASK-93
title: Record the text-to-image seed for every invocation
status: Done
assignee: []
created_date: '2026-09-04 06:38'
updated_date: '2026-09-05 07:52'
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
- [x] #2 Batched invocations get per-item seeds, so an item's image depends only
      on its own seed
- [x] #3 Regenerating an invocation from its stored seed and input text
      reproduces the stored image, verified on at least one model per pipeline
      family
- [x] #4 Resume behaviour under seeds is correct and stated: a resumed run draws fresh seeds for the steps it has yet to do, and completed steps keep the seeds they were run with
      it redoes
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
GPU VERIFICATION DONE 2026-09-05 (AC#2, AC#3), test/real_models_test.exs 'seed regeneration'.

AC#3: a run executed through RunExecutor on SD35Medium, then GenAI.invoke at the invocation's stored seed and initial prompt, reproduces the stored AVIF byte for byte; the same prompt at seed+1 differs by a mean of >10 per pixel. A direct Python probe over Flux2Klein, SD35Medium and ZImageTurbo agreed: same seed is bit-exact on all three, a different seed moves the image by a mean of 46-95 (of 255).

AC#2: a batched item at seed s with different batch-mates lands within numerical noise of the same item generated alone (mean pixel difference 0.4-1.8 across the three models, max under 5 asserted), against 46-95 for a different seed. So an item depends on its own seed only, but a batch is NOT bit-exact against a single call at the same seed; regeneration must use the same path (single or batch) to reproduce bytes.

AC#4: resume runs only the steps that were never completed, drawing a fresh seed for each; completed invocations keep the seed they were run with. Batched resume behaves the same since TASK-97 (each run rejoins at its own next step).

ALSO FIXED: GenAI.invoke/invoke_batch with no seed on a real text-to-image model handed None to Python, which now requires an int, so every existing GPU test that calls invoke without a seed was broken. A seedless caller now gets a seed drawn on the Elixir side; the pipeline path is unchanged.
<!-- SECTION:NOTES:END -->
