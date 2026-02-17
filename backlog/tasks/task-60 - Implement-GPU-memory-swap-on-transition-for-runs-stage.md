---
id: TASK-60
title: Implement GPU memory swap-on-transition for runs stage
status: To Do
assignee: []
created_date: '2026-02-08 03:42'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The runs stage currently unloads the outgoing model entirely (`unload_model/2` removes it from `_models`, calls `remove_all_hooks()`, moves to CPU, deletes, and runs `gc.collect()` + `empty_cache()`) at each network transition. This means models are re-instantiated from disk cache every cycle, which is slow. T2I batch code also generates images one at a time (serial loop) to avoid OOM, since `enable_model_cpu_offload()` doesn't free enough VRAM for true batched generation.

Replace the current unload-on-transition with a swap-on-transition strategy: at each model change in the network cycle, move the outgoing model to CPU RAM (keeping it in `_models`) and move the incoming model back to GPU. This avoids re-instantiation costs. PCIe 4.0 x16 gives ~25GB/s, so even a 20GB model swaps in under a second --- negligible vs inference time. 128GB system RAM can hold multiple models simultaneously.

With only the active model on GPU, there should be enough VRAM headroom to restore true T2I batched generation (passing all prompts at once to the pipeline) and potentially enable I2T batched inference too.

### Current state (after parent change)

- All T2I pipelines use `enable_model_cpu_offload()` (not `.to("cuda")`)
- `unload_model/2` exists and fully removes models from `_models` dict
- `unload_all_models/1` exists with thorough cleanup (hooks, CPU move, gc)
- `RunExecutor.execute_batch_loop` already unloads the outgoing model when the next model differs
- `Engine` calls `unload_all_models` between network groups
- T2I batch code generates one image at a time (serial loop with `torch.cuda.empty_cache()` between each)
- I2T batch code processes images serially in a Python loop

## Implementation plan

### 1. Add swap helpers to PythonBridge (python_bridge.ex)

Add `swap_model_to_cpu/2` and `swap_model_to_gpu/2` to replace the current `unload_model/2` approach. The Python code should be generic, dispatching on type: dict-style models (InstructBLIP, Qwen25VL, Gemma3n with processor/model keys) swap only the "model" key; SentenceTransformers call `.to()`. Call `torch.cuda.empty_cache()` after moving to CPU.

For diffusers pipelines (all T2I models), the swap needs to undo `enable_model_cpu_offload()` by calling `remove_all_hooks()` before `.to("cpu")`, and re-enable it via `enable_model_cpu_offload()` when moving back to GPU. Alternatively, since CPU offload already manages device placement during inference, it may be sufficient to just call `remove_all_hooks()` + `.to("cpu")` + `empty_cache()` when swapping out, and `.to("cuda")` or `enable_model_cpu_offload()` when swapping back in. Test both approaches.

### 2. Modify RunExecutor.execute_batch_loop (run_executor.ex)

Replace the current `unload_model` call with swap calls. Before invoking the current model: swap the previous model to CPU, then swap the current model to GPU (or `ensure_model_loaded` if first use). The model changes every step (cycling through the network), so swaps happen every step. The recursive loop already tracks `model_name` per step; add tracking of the previous model name.

### 3. Verify ensure_model_loaded interaction

After implementing swapping, a model may be in `_models` but on CPU. `ensure_model_loaded` checks `model_name in _models` --- this returns true for CPU-resident models. Need to either: (a) have `swap_model_to_gpu` handle models already in `_models`, or (b) add a separate check. Simplest approach: always use `swap_model_to_gpu` which is a no-op if already on GPU.

### 4. Restore T2I true batching and add I2T batching (genai.ex)

With only the active model on GPU, there should be enough VRAM for true batched generation:
- T2I: restore passing all prompts to the pipeline at once (revert serial loop)
- I2T: implement true batched inference where supported:
  - InstructBLIP: processor and model.generate support batched inputs
  - Qwen25VL: supports batched message lists through its processor
  - Gemma3n: supports batched inputs through apply_chat_template
  - Moondream: check if .caption() API supports batching; serial loop is fine if not

### 5. Testing

- Test with DummyT2I/DummyI2T first to verify swap logic through the full loop
- GPU smoke test with a real two-model network for correctness and swap overhead measurement
- Compare batch throughput before/after with a representative experiment config

### 6. What NOT to do

- No model-specific swap heuristics --- apply uniformly
- No changes to embeddings/TDA/clustering stages (already have unload_all_models boundaries)
- No multi-GPU support (single RTX 6000 Ada)
- No changes to resume logic (swapping is transparent to persistence)
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 PythonBridge has generic swap_model_to_cpu/2 and swap_model_to_gpu/2 functions that handle all model types (pipelines, dict-style, SentenceTransformers) without model-specific code
- [ ] #2 RunExecutor.execute_batch_loop swaps models at each network transition, with only the active model on GPU
- [ ] #3 T2I batch code restored to true batched generation (all prompts passed to pipeline at once)
- [ ] #4 I2T batch code uses true batched inference for models that support it (InstructBLIP, Qwen25VL, Gemma3n)
- [ ] #5 Existing tests pass (mix test) with swap logic in place
- [ ] #6 GPU smoke test confirms correct output and measures swap overhead
- [ ] #7 Batch throughput improved compared to serial one-at-a-time baseline
<!-- AC:END -->
