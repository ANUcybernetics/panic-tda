---
id: TASK-93
title: Record the text-to-image seed for every invocation
status: To Do
assignee: []
created_date: "2026-09-04 06:38"
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

- [ ] #1 Every text-to-image invocation draws its own seed, passes it to the
      pipeline, and stores it on the invocation
- [ ] #2 Batched invocations get per-item seeds, so an item's image depends only
      on its own seed
- [ ] #3 Regenerating an invocation from its stored seed and input text
      reproduces the stored image, verified on at least one model per pipeline
      family
- [ ] #4 Resuming an interrupted experiment reuses the stored seed for any step
      it redoes

<!-- AC:END -->
