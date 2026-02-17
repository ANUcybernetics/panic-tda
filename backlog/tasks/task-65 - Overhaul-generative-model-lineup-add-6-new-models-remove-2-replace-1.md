---
id: TASK-65
title: 'Overhaul generative model lineup: add 6 new models, remove 2, replace 1'
status: In Progress
assignee: []
created_date: '2026-02-17 23:54'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Overhaul the T2I and I2T model lineup to broaden architectural diversity and lab coverage. T2I goes from 5 to 7 models (6 labs), I2T goes from 4 to 6 models (6 labs), covering 8 labs total with sizes ranging from 2B to 32B.

## Progress Report (2026-02-18)

### Completed
- **Cleanup:** Fully removed `FluxSchnell` and `InstructBLIP` from `genai.ex`, `python_bridge.ex`, and `real_models_test.exs`.
- **Implementation:** Added invocation logic (single and batch) for all 6 new models in `genai.ex`.
- **Loading:** Added `@model_loaders` entries for all 6 new models in `python_bridge.ex`.
- **Environment:** Updated `python_interpreter.ex` with missing dependencies: `peft`, `bitsandbytes`, `backoff`.
- **Configuration:** Increased `ownership_timeout` to 600s in `config/test.exs` to prevent DB timeouts during heavy GPU model loading.
- **Documentation:** Updated model tables in `README.md` and `CLAUDE.md`.
- **Initial Verification:** Passed `mix test` (no-GPU) and completed first full `:gpu` test pass.

### Identified & Fixed (GPU Verification Phase)
- **Gated Access:** Added `token=True` for `Flux2Dev`, `LLaMA32Vision`, and `Pixtral`.
- **Path Corrections:** Updated `HunyuanImage` to `tencent/HunyuanImage-2.1` and `GLMImage` to `ZhipuAI/GLM-Image`.
- **Naming:** Corrected `GLMImagePipeline` to `GlmImagePipeline`.
- **Architecture:** Switched `Pixtral` to use `AutoModelForCausalLM` instead of `PixtralForConditionalGeneration` for better compatibility with current transformers version.

### Pending
- Final verification of the fixed loaders via `:gpu` test suite.

## T2I changes (5 → 7)

**Remove:** FluxSchnell (FLUX.1 schnell, replaced by Flux2Dev)

**Add 3 models:**
- **Flux2Dev** — FLUX.2 [dev], Black Forest Labs, 32B. Frontier quality, runs at 8-bit on 48GB. HF: black-forest-labs/FLUX.2-dev. Replaces FluxSchnell; keeps Flux2Klein for same-arch different-scale comparison.
- **HunyuanImage** — HunyuanImage 2.1, Tencent, 17B. Runs at FP8 on 48GB. HF: tencent/HunyuanImage-2.1.
- **GLMImage** — GLM-Image, Zhipu AI, ~16B (9B AR + 7B DiT hybrid). Architecturally unusual: autoregressive + diffusion. Requires 48GB with CPU offloading. HF repo path needs research.

**Resulting T2I lineup:** SD35Medium, Flux2Klein, Flux2Dev, ZImageTurbo, QwenImage, HunyuanImage, GLMImage

## I2T changes (4 → 6)

**Remove:** InstructBLIP (older Q-Former architecture, reliability issues per TASK-61)

**Add 3 models:**
- **Pixtral** — Pixtral 12B, Mistral, 12B. Variable-resolution tokenisation, Apache 2.0. HF: mistralai/Pixtral-12B-2409.
- **LLaMA32Vision** — LLaMA 3.2 Vision, Meta, 11B. Strong on document understanding/VQA. HF: meta-llama/Llama-3.2-11B-Vision-Instruct.
- **Phi4Vision** — Phi-4 Multimodal, Microsoft, 3.8B (3.4B LLM + 0.4B vision encoder). Ultra-lightweight. HF: microsoft/Phi-4-multimodal-instruct.

**Resulting I2T lineup:** Moondream, Qwen25VL, Gemma3n, Pixtral, LLaMA32Vision, Phi4Vision

## Per-model implementation pattern (4 files each)

1. **genai.ex** — add to @real_t2i_models or @real_i2t_models word list; add real_t2i_code/1 or real_i2t_code/1 clause (Python invoke code); add corresponding batch code clause. T2I: accept prompt, return base64 WEBP at IMAGE_SIZE. I2T: accept image_b64, return stripped text string.
2. **python_bridge.ex** — add @model_loaders entry. Must load from HuggingFace, store in _models dict. Pipeline models: .enable_model_cpu_offload(). Processor+model pairs: store as dict. Special offload: add to _models_offload_only.
3. **real_models_test.exs** — update model name lists in for comprehensions. Remove FluxSchnell and InstructBLIP from lists.
4. **README.md and CLAUDE.md** — update available models tables.

## Suggested implementation order

1. Remove FluxSchnell and InstructBLIP (clean up references)
2. Flux2Dev (familiar architecture, replaces FluxSchnell)
3. HunyuanImage (standard diffusers pipeline)
4. GLMImage (may need research on loading pattern)
5. Pixtral (standard transformers VLM)
6. LLaMA32Vision (standard transformers VLM)
7. Phi4Vision (smallest, likely simplest)

## Research needed

- Exact HF repo path for GLMImage and whether it needs a custom pipeline (RESOLVED: `ZhipuAI/GLM-Image`)
- Exact HF repo for Phi4Vision (RESOLVED: `microsoft/Phi-4-multimodal-instruct`)
- Whether any new models require gated repo access (HF tokens) (RESOLVED: Flux2Dev, Pixtral, LLaMA32Vision need token)
- FLUX.2 dev licensing terms / HF terms acceptance
- Note TASK-63 tracks existing Flux2Klein gated repo issues — same pattern may apply

## Verification

- mise exec -- mix test must pass (dummy models, no GPU needed)
- mise exec -- mix test --include gpu must pass for all new models
- Each new model should produce sensible output in a short test run
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 FluxSchnell removed from genai.ex, python_bridge.ex, and real_models_test.exs
- [x] #2 InstructBLIP removed from genai.ex, python_bridge.ex, and real_models_test.exs
- [ ] #3 Flux2Dev added with loader, invoke code, batch code, and GPU test passing
- [ ] #4 HunyuanImage added with loader, invoke code, batch code, and GPU test passing
- [ ] #5 GLMImage added with loader, invoke code, batch code, and GPU test passing
- [ ] #6 Pixtral added with loader, invoke code, batch code, and GPU test passing
- [ ] #7 LLaMA32Vision added with loader, invoke code, batch code, and GPU test passing
- [ ] #8 Phi4Vision added with loader, invoke code, batch code, and GPU test passing
- [x] #9 README.md and CLAUDE.md model tables updated to reflect new lineup
- [x] #10 mise exec -- mix test passes (no GPU)
- [ ] #11 mise exec -- mix test --include gpu passes for all new models
<!-- AC:END -->
