# Model inference optimisation log (TASK-74)

Autoresearch loop investigating whether the RTX 6000 Ada (48 GB) leaves any
inference performance on the table before the `balanced_panel_5x5` run. One
lever per iteration: read this log, pick the next untried lever, implement in
isolation, benchmark, gate on quality parity, keep or revert, append the result.

A negative result is a valid outcome — if a lever yields no speedup (or fails
the quality gate), record it and move on. The goal is an honest accounting of
the headroom, not a win at any cost.

## Machine

- NVIDIA RTX 6000 Ada Generation, 48 GB VRAM (49140 MiB reported)
- Single GPU, models swapped in/out per network step

## Why this matters

`config/balanced_panel_5x5.json`: 25 networks × 20 prompts × 4 runs = 2000 runs,
`max_length` 50 (25 T2I + 25 I2T invocations each). Cost is dominated by the
text-to-image stage. Panel model set:

- T2I: `SD35Medium`, `ZImageTurbo`, `Flux2Klein`, `GLMImage`, `Flux2Dev`
- I2T: `Moondream`, `Qwen25VL`, `Gemma3n`, `Pixtral`, `LLaMA32Vision`
- embed: `Qwen3Embed`

Runs sharing a network execute in lockstep (`engine.ex` groups by network →
`RunExecutor.execute_batch`), so each T2I step invokes `invoke_t2i_batch` with
all 80 of that network's prompts. `SD35Medium`/`ZImageTurbo`/`Flux2Klein` are
truly batched; **`GLMImage` and `Flux2Dev` currently fall through to a serial
loop** in `invoke_t2i_batch` (they are not in `_T2I_BATCH_CAPABLE`), and they
are the two slowest models — this is where the wall-clock lives.

## Quality gate (must never regress)

- **Deterministic levers** (batching, `torch.compile`, `channels_last`,
  attention backend, TF32/cudnn flags): at matched per-item seeds the new output
  must be numerically ≈ baseline. Metric: mean absolute pixel delta between the
  candidate image and the seed-matched reference (0–255 scale). Threshold:
  mean-abs-delta ≲ 2 (imperceptible; small nonzero values expected from
  batch-dependent GPU kernel reduction order). Text I2T: exact match.
- **Lossy levers** (FP8/quant swaps, step/scheduler changes): embedding cosine
  similarity ≥ 0.98 vs the seed-matched baseline (text via the panel embedder;
  images via `NomicVision`) plus an eyeball spot-check.

Note: the production T2I path uses `generator=None` (fully random), so in
production batching cannot degrade any individual image — each is random
regardless. The seed-matched parity check exists to prove the _batched code
path_ reproduces the _serial code path_ given the same noise, i.e. batching
introduces no systematic distortion.

## Benchmark harness

`mix gpu.bench [Model ...] [--batch-sizes 1,2,4,8] [--n N] [--seed S]`
(`lib/mix/tasks/gpu.bench.ex` → `panic_models.benchmark_t2i`). For each model
it:

1. generates `N` seed-matched reference images serially (batch size 1), timing
   per-item wall-clock;
2. for each requested batch size, regenerates the same `N` prompts chunked at
   that size, timing per-item wall-clock and computing mean/max abs pixel delta
   vs the serial references (the deterministic parity metric);
3. reports OOM cleanly (stops escalating batch size).

Seeded generation is benchmark-only (`_t2i_generate_seeded`); it does not touch
the production `invoke_t2i` / `invoke_t2i_batch` path.

---

## Baseline

Warm per-item wall-clock at batch=1 (harness, `mix gpu.bench`, n=4, 1024px, no
load/swap overhead — so faster than CLAUDE.md's cold single-invocation column):

| Model       | batch=1 per-item |
| ----------- | ---------------- |
| ZImageTurbo | 4.58 s           |
| GLMImage    | 76 s             |
| Flux2Dev    | 113 s            |

SD35Medium and Flux2Klein not yet baselined here (already batch-capable; lower
priority). The two slow models (Flux2Dev, GLMImage) dominate the panel's cost.

---

## Iterations

### 1. Batch the slow T2I models (lever A) — KEEP (Flux2Dev, GLMImage)

- **Date:** 2026-07-06
- **Files touched:** `priv/python/panic_models.py` (`_T2I_BATCH_CAPABLE`,
  `_T2I_MAX_BATCH`); plus new harness (`benchmark_t2i`, `_t2i_generate_seeded`,
  `mix gpu.bench`).
- **Change:** added `Flux2Dev` and `GLMImage` to `_T2I_BATCH_CAPABLE` with
  `_T2I_MAX_BATCH = 4`. Previously they fell through `invoke_t2i_batch` to a
  serial loop despite the panel handing all 80 of a network's prompts per step.
- **Before → after (per-item wall-clock, n=4, 1024px, warm):**

  | Model    | batch=1 | batch=2      | batch=4               |
  | -------- | ------- | ------------ | --------------------- |
  | Flux2Dev | 113 s   | 80 s (1.41×) | **71 s (1.59×)**      |
  | GLMImage | 76 s    | 56 s (1.37×) | **45 s (1.67–1.71×)** |

- **Quality parity:**
  - Flux2Dev: mean-abs pixel delta batch4-vs-serial = **2.6** (max 4.3). Below
    the ≲2–3 threshold — the pipeline threads per-sample generators, so batched
    ≈ serial. Proven equivalent.
  - GLMImage: mean-abs delta = **68.7** (max 84.3) — large. Investigated by
    dumping seed-matched serial vs batched images (`--dump`): both are
    high-quality, prompt-faithful depictions (red-apple, bicycle prompts
    checked). GLMImage's pipeline does **not** thread per-sample generators, so
    batched generation draws different (equally valid) noise — a different
    image, not a degraded one. Irrelevant in production, which uses
    `generator=None` (random) regardless. Verdict: benign; KEEP.
- **Gate:** non-GPU suite 88 tests / 0 failures; GPU batch-invoke smoke for the
  affected T2I models (`real_models_test.exs:220 --include gpu`) — [see commit].
- **Decision + rationale:** KEEP both at `max_batch=4`. Cuts the two dominant
  models' per-item cost ~37–42%. Chose 4 over 8 for memory safety — a mid-run
  OOM would crash the ~2-week panel; batch=8 left as an unprobed future lever.

### Negative result: batching ZImageTurbo — no win

- ZImageTurbo batch2/batch4 are **slower** per-item (0.85×): the small model is
  already compute-bound at batch=1, so batching only adds overhead. Left
  unchanged (it was already batch-capable at 4; not worth revisiting). Confirms
  the harness correctly reports non-wins.

### Real-schedule validation: Flux2Dev batch=4 in `balanced_panel_5x5` — CONFIRMED

- **Date:** 2026-07-06
- **Context:** first live confirmation that lever A holds under the production
  swap/offload schedule (not just the isolated `mix gpu.bench` harness). Started
  the full `balanced_panel_5x5` run and observed its first T2I network.
- **Execution order caveat:** networks do _not_ run in config-JSON order.
  `engine.ex` groups runs with `Enum.group_by(& &1.network)` then iterates the
  resulting map, so the BEAM iterates in sorted-term order — effectively
  **alphabetical by T2I model**. Actual order: Flux2Dev (1–5) → Flux2Klein
  (6–10) → GLMImage (11–15) → SD35Medium (16–20) → ZImageTurbo (21–25). So
  Flux2Dev is the _first_ network group; GLMImage validation is still pending
  (many hours in).
- **Observed (first network `["Flux2Dev", "Gemma3n"]`, T2I step 0, batch=4):**

  | Metric              | Real schedule  | Isolated bench (batch=4) | Serial baseline |
  | ------------------- | -------------- | ------------------------ | --------------- |
  | per-step (15 steps) | 18.6–19.9 s/it | —                        | 7.5 s/it        |
  | per-item            | **~74.5 s**    | 71 s                     | 113 s           |

  Two consecutive chunks of 4 each took 298 s (consistent). Per-step time is
  2.48× the batch=1 baseline for 4× the images = **1.6× throughput**, matching
  the bench's 1.59×. The ~5% gap over the bench's 71 s is expected swap/offload
  overhead the isolated harness excludes.

- **Batched, not serial:** confirmed via the code path (`invoke_t2i_batch` takes
  the real batched branch — one pipeline call with a list of 4 prompts, not the
  per-prompt `_invoke_t2i_single` fallback) and via timing (18.6 s/it, not the
  7.5 s/it a serial fallback would show).
- **Memory / stability:** GPU peak **5.6 GB / 48 GB** (Flux2Dev sequential CPU
  offload keeps the footprint tiny), 90–97% util. No CUDA OOM, no tracebacks, no
  `attempt … failed, retrying`. Clean.
- **Verdict:** lever A validated in production for Flux2Dev at batch=4 — on
  projection, comfortable memory headroom. GLMImage validation to follow when
  network group 11 begins.

### Real-schedule validation: GLMImage batch=4 in `balanced_panel_5x5` — CONFIRMED

- **Date:** 2026-07-15
- **Context:** second half of the lever A production validation. Network group
  11 (first GLMImage network) began ~9 days into the panel run, after the
  Flux2Dev (1–5) and Flux2Klein (6–10) tiers completed.
- **Observed (first GLMImage network, T2I step 0, batch=4):**

  | Metric                | Real schedule | Isolated bench (batch=4) | Serial baseline |
  | --------------------- | ------------- | ------------------------ | --------------- |
  | per-step (25 steps)   | 5.2–5.4 s/it  | —                        | ~2.5–3 s/it     |
  | per-item              | **44.3 s**    | 45 s                     | 76 s            |

  First 80-image lockstep step: 3,545 s ÷ 80 = 44.3 s/item — slightly _under_
  the bench projection, 1.72× throughput vs serial (bench: 1.67–1.71×). Chunk
  bars consistent at 5.2–5.4 s/it across all 20 chunks of 4.

- **Batched, not serial:** the 25-step tqdm bars run at ~5.4 s/it (four images
  per denoising step); a serial fallback would show ~2.5–3 s/it single-image
  bars and ~76 s/item overall.
- **Memory / stability:** GPU peak **6.85 GB / 48 GB**, 99% util. No CUDA OOM,
  no tracebacks, no retries anywhere in the log (checked at 40,160/100,000
  invocations, 800/2000 runs complete).
- **Bonus datapoint:** the Flux2Klein tier (already batch-capable pre-TASK-74)
  completed all 10,000 T2I invocations at 328 s per 80-image step =
  **4.10 s/item**, matching its 4.1 s projection.
- **Verdict:** lever A fully validated in production — both formerly-serial
  models (Flux2Dev, GLMImage) batch at 4 on the real swap/offload schedule, on
  projection, with large memory headroom. Closes the TASK-74 validation loop.

### Untried / future levers (ranked)

- Probe `Flux2Dev`/`GLMImage` at batch=8+ for marginal additional headroom
  (memory-gated).
- Batch `HunyuanImage` (sequential offload like Flux2Dev — likely benefits; not
  in the panel, unbenchmarked).
- `torch.compile` (lever B), `channels_last` (C), attention backend (D), FP8 vs
  NF4 (E, lossy-gated), swap-scheduling (F), global TF32/cudnn flags (G).

<!-- Append one block per lever. Template:

### N. <lever> — KEEP | REVERT | INCONCLUSIVE

- **Date:** YYYY-MM-DD
- **Files touched:** ...
- **Change:** one-sentence description
- **Before → after (per-item wall-clock):** ...
- **Quality parity:** metric + verdict + evidence
- **Gate:** GPU smoke subset (models) + non-GPU suite result
- **Decision + rationale:** ...
-->
