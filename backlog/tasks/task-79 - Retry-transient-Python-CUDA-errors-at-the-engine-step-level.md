---
id: TASK-79
title: Retry transient Python/CUDA errors at the engine step level
status: To Do
assignee: []
created_date: '2026-07-22 02:22'
labels:
  - reliability
  - engine
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The 2026-07-19 balanced_panel_5x5 crash showed a single stochastic CUDA device-side assert (GLM-Image AR prior sampled an out-of-range token id -> gather index OOB, ~0.06% per batched pipeline call) kills the entire multi-week experiment: the Snex error propagates as {:error, %Snex.Error{}} and the mix task match-fails. The GPU then idles until a human notices (26h lost). mix experiment.resume recovers cleanly, but the engine should retry a failed model invocation step (fresh noise makes stochastic failures vanish on retry) before giving up. Bound retries (e.g. 2-3 with short backoff) so deterministic failures still surface promptly. Consider whether the retry belongs in run_executor.ex around the invoke_*_batch calls or in the Python layer; note panic_models.py already has a retry path for some errors ('attempt N failed ... retrying') that did not cover this one.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 a transient (single-shot) Python/CUDA error during a batch invocation step is retried and the experiment continues without operator intervention
- [ ] #2 a persistent (deterministic) failure still aborts with the original error after bounded retries, not an infinite loop
- [ ] #3 retries are logged visibly (model, step, attempt count, error summary) so post-hoc analysis can count them
- [ ] #4 test coverage exercises the retry path (dummy model or injected fault), full non-GPU suite green
<!-- AC:END -->
