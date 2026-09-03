---
id: TASK-74
title: >-
  Autoresearch loop: optimise model inference efficiency on the RTX 6000 Ada
  before the balanced_panel_5x5 run
status: Done
assignee: []
created_date: '2026-07-06 05:52'
updated_date: '2026-09-03 06:31'
labels:
  - performance
  - gpu
  - research
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
## Why

We are about to run `config/balanced_panel_5x5.json` — a balanced 5×5 factorial
panel (2,000 runs, ~2 weeks of wall-clock on a single RTX 6000 Ada). Cost is
dominated by the text-to-image stage, and especially by the three slow T2I
models. Any per-item speedup compounds across ~100k T2I invocations, so
squeezing inference efficiency **before** the run is high leverage.

This is an **autoresearch loop**: run it repeatedly with a cleared context. Each
iteration reads the findings log, picks the next untried lever, implements one
change in isolation, benchmarks it, verifies output quality is unchanged, keeps
or reverts, then appends the result to the log. Stop when the ranked levers are
exhausted or the remaining ideas are not worth the risk.

## Hard constraint: no quality regression

Speed must never come at the cost of output quality. Every candidate change is
gated:

- **Deterministic levers** (`torch.compile`, `channels_last`, attention-backend
  swaps, batching, TF32/cudnn flags): at a fixed seed the new outputs must be
  numerically identical or near-identical to baseline. Verify by regenerating a
  fixed prompt/seed sample and diffing images (mean abs pixel delta ≈ 0) and
  text (exact match).
- **Lossy levers** (FP8 quant, step-count or scheduler changes): must pass a
  semantic-parity check — for a fixed seed + held-out prompt sample spanning
  every model pair, the embedding cosine similarity between the new output and
  the baseline output stays above threshold (≥ 0.98 for text; embed images with
  `NomicVision`), plus a human-eyeball spot-check. Revert anything that
  regresses.

When in doubt, revert. A slower-but-correct pipeline beats a faster-but-degraded
one.

## Loop methodology

1. Read the findings log at `backlog/docs/model-optimisation-log.md` (create it
   on first iteration). Never repeat an approach already recorded there.
2. Establish or refresh the baseline: per-item wall-clock and reference outputs
   (fixed seed) for each model on the panel's model set (SD35Medium,
   ZImageTurbo, Flux2Klein, GLMImage, Flux2Dev; Moondream, Qwen25VL, Gemma3n,
   Pixtral, LLaMA32Vision).
3. Pick the highest expected-payoff untried lever (see ranked list).
4. Implement exactly one change, isolated.
5. Benchmark per-item wall-clock; run the quality-parity check for that change
   class.
6. Decision: keep only if it is faster AND quality parity holds AND the GPU
   smoke subset for the affected models passes AND the full non-GPU suite
   passes. Otherwise revert cleanly.
7. Append to the log: approach, files touched, measured before/after per-item
   time, quality verdict with evidence, keep/revert decision.
8. Commit accepted wins (green only), then repeat.

## Ranked optimisation avenues (with pointers)

Prioritise by cost share — Flux2Dev, HunyuanImage and GLMImage dominate, so
target them first.

- **A. Batch the slow T2I models.** `_T2I_BATCH_CAPABLE` in
  `priv/python/panic_models.py` currently lists only the three _fast_ models
  (SD35Medium, ZImageTurbo, Flux2Klein); Flux2Dev, GLMImage and HunyuanImage run
  strictly serially. Since the panel batches runs in lockstep, enabling batching
  on the cost-driving models is likely the single biggest lever. Probe headroom
  on 48 GB with `mix gpu.max_batch` / `probe_max_batch` and wire per-model caps
  into `_T2I_MAX_BATCH`.
- **B. `torch.compile`.** No `torch.compile` anywhere today. Try
  `mode="max-autotune"` (or `"reduce-overhead"`) on the diffusion transformers
  and the I2T decoders. Persist the Inductor cache so the run does not pay
  recompile cost repeatedly. Watch for recompiles triggered by dynamic
  batch/shape.
- **C. `channels_last` memory format** for conv/UNet-based pipelines.
- **D. Attention backend.** Confirm the SDPA flash kernel is actually selected
  everywhere; evaluate adding `flash-attn` to the venv for the transformer
  VLMs/T2I transformers. Some models set `attn_implementation="sdpa"`
  explicitly; others use defaults.
- **E. FP8 (e4m3) on Ada** for the big transformers versus current NF4
  (bitsandbytes) — quality-gated via the lossy check. Consider torchao /
  diffusers quantisation.
- **F. Reduce model-swap overhead.** Swaps are naive `.to("cuda")`/`.to("cpu")`
  per transition (`swap_to_gpu`/`swap_to_cpu`). With 48 GB, keep the small I2T
  model resident while a T2I model runs to cut ping-pong; and/or schedule
  batches to minimise T2I↔I2T transitions (see
  `lib/panic_tda/engine/run_executor.ex` and `genai.ex`). Consider pinned-memory
  / faster offload paths.
- **G. Global perf flags.** `torch.set_float32_matmul_precision("high")` (TF32),
  `torch.backends.cudnn.benchmark = True`, VAE tiling/slicing where relevant.
- **H. Scheduler / step-count re-tuning** (quality-gated, low priority — steps
  are already per-model tuned in `_T2I_SETTINGS`).

## Existing tooling to reuse

- `mix gpu.max_batch` and the `probe_max_batch` Python helper for batch-headroom
  probing.
- GPU smoke tests via `mise exec -- mix test --include gpu` (174 tests, ~4h40m
  for the full set — do NOT run the whole thing each iteration; run a targeted
  subset for the models touched).
- The approximate-run-times table in `CLAUDE.md` is the baseline reference;
  update it when a win lands.
- Venv spec is inline in `lib/panic_tda/models/python_interpreter.ex` — keep it
  in sync if adding a dependency (e.g. flash-attn, torchao).

## Guardrails

- One change per iteration; commit only when green (targeted GPU subset for
  affected models + full non-GPU suite).
- Do not alter the four-stage pipeline semantics — this is purely inference
  performance.
- Update `CLAUDE.md`'s timing table and the findings log with every accepted
  change; note any architecture-level swap-scheduling change in `DESIGN.md`.
- Follow the project rules: `mise exec --` prefix, Ash for data-model work,
  small scoped commits, no red commits.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A findings log at backlog/docs/model-optimisation-log.md records every approach tried, with files touched, measured before/after per-item wall-clock, quality-parity verdict, and keep/revert decision
- [x] #2 A repeatable benchmark harness reports per-item wall-clock and a fixed-seed quality-parity metric for each model on the panel's model set
- [ ] #3 The two highest-payoff levers (batching the slow T2I models; torch.compile) have each been evaluated end-to-end with recorded results
- [x] #4 Every accepted optimisation passes the GPU smoke subset for the affected models and the full non-GPU suite
- [x] #5 No accepted change compromises output quality: each kept change has a logged quality-parity pass (deterministic diff or embedding-cosine threshold)
- [x] #6 Net measured per-item speedup on the panel's model set is recorded and CLAUDE.md's timing table is updated to reflect accepted changes
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Iteration 1 (lever A, batching slow T2I) complete — commit b7bf94a. Flux2Dev 113->71s/item (1.59x, parity 2.6); GLMImage 76->45s/item (1.67x, parity 68.7 but verified benign). Benchmark harness (mix gpu.bench) + findings log landed. AC #3 half-done: batching evaluated, torch.compile (lever B) still pending — that's the next iteration. Negative result logged: ZImageTurbo batching is slower (0.85x).

Closed 2026-09-03 at Ben's direction. AC #3 is deliberately left unticked: lever A (batching the slow T2I models) was evaluated end to end and kept, but lever B (torch.compile) was never attempted. It stays on the ranked untried-levers list in backlog/docs/model-optimisation-log.md alongside channels_last, attention backend, FP8-vs-NF4, swap scheduling and the global TF32/cudnn flags, so nothing is lost by closing the loop here.

Net effect of the work that did land: Flux2Dev 113 -> 71 s/item and GLMImage 76 -> 45 s/item from batching (both validated on the real balanced_panel_5x5 schedule, not just in the bench), and subsequently Flux2Dev 71 -> 57 s/item from the TASK-83 step-count cut. The benchmark harness (mix gpu.bench) and the findings log are the durable outputs.

Negative result worth keeping: batching ZImageTurbo is slower (0.85x), so it stays serial.
<!-- SECTION:NOTES:END -->
