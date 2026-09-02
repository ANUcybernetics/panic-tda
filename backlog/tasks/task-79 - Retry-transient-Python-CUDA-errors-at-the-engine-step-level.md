---
id: TASK-79
title: Retry transient Python/CUDA errors at the engine step level
status: Done
assignee: []
created_date: '2026-07-22 02:22'
updated_date: '2026-09-02 06:53'
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
- [x] #1 a transient (single-shot) Python/CUDA error during a batch invocation step is retried and the experiment continues without operator intervention
- [x] #2 a persistent (deterministic) failure still aborts with the original error after bounded retries, not an infinite loop
- [x] #3 retries are logged visibly (model, step, attempt count, error summary) so post-hoc analysis can count them
- [x] #4 test coverage exercises the retry path (dummy model or injected fault), full non-GPU suite green
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Retry lives in PanicTda.Engine.Retry, called from RunExecutor around both GenAI.invoke and GenAI.invoke_batch; three attempts, backoff configurable via :panic_tda, :retry_backoff_ms (0 in test).

Key finding that shaped the design: an in-process retry cannot recover the motivating failure. A CUDA device-side assert poisons the process's CUDA context, so every subsequent CUDA call returns the same sticky error --- which is why the existing panic_models.py retry never helped (it also only covered _invoke_t2i_single, never the batch path that actually crashed). So the first retry reuses the process (enough for a fault that leaves the context intact, e.g. an OOM), and later retries replace the interpreter via the new PanicTda.Models.PythonSession, which holds the interpreter+env as one restartable unit and hands out its current env. The retried invocation reloads the model through the existing swap_model_to_gpu path; the experiment's i2t ceiling is re-applied to each fresh interpreter through the session's on_start hook.

RunExecutor and Engine now pass a session instead of a raw env. test/retry_test.exs covers injected faults (error tuple, raise, exit), the bounded-abort path, log content, and a real interpreter restart. Full non-GPU suite green: 102 tests, 0 failures.
<!-- SECTION:NOTES:END -->
