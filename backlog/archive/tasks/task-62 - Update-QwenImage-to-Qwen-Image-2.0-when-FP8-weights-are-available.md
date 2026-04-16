---
id: TASK-62
title: Update QwenImage to Qwen-Image-2.0 when FP8 weights are available
status: To Do
assignee: []
created_date: '2026-02-17 03:50'
labels:
  - models
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The current QwenImage model uses Qwen/Qwen-Image-2512 (bf16) via QwenImagePipeline with cpu_offload because the full model (~40GB bf16) exceeds 48GB VRAM. When FP8 weights for Qwen-Image-2.0 are published on HuggingFace, update the model loader in python_bridge.ex to point to the new checkpoint. If the 2.0 FP8 weights are ~20GB they would fit fully in 48GB VRAM without cpu_offload, allowing removal of QwenImage from _models_offload_only and enabling the standard swap-on-transition path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 FP8 weights for Qwen-Image-2.0 are available on HuggingFace
- [ ] #2 Model loader in python_bridge.ex updated to new checkpoint path
- [ ] #3 If FP8 fits in VRAM: remove QwenImage from _models_offload_only and verify swap_model_to_gpu works
- [ ] #4 GPU tests pass with updated model
<!-- AC:END -->
