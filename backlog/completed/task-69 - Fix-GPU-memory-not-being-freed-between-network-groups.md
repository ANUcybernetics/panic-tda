---
id: TASK-69
title: Fix GPU memory not being freed between network groups
status: Done
assignee: []
created_date: '2026-02-20 08:43'
updated_date: '2026-04-16 21:00'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Flux2Klein OOMs even at batch size 8 because previous models (e.g. SD35Medium + LLaMA32Vision) leave ~45 GiB allocated on a 47 GiB GPU. The model unloading code likely needs to properly delete previous models and call torch.cuda.empty_cache() before loading the next network group. This would allow memory-hungry models like Flux2Klein (which uses Qwen3 for prompt encoding) to run at reasonable batch sizes in multi-network experiments.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Previous model references are deleted before loading the next network group
- [x] #2 torch.cuda.empty_cache() is called after model unloading
- [ ] #3 Flux2Klein runs successfully in a multi-network experiment with batch size 8+
- [x] #4 GPU memory usage drops to baseline between network group transitions
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Resolved. engine.ex:29/81 calls PythonBridge.unload_all_models/1 before each network group (and again at 33/40/43/85/92/95 between pipeline stages). unload_all_models/0 in priv/python/panic_models.py:719 pops every entry from _models, calls _force_free_model (which does .to('cpu') + del obj), clears _models_offload_only, runs gc.collect(), torch.cuda.synchronize() and torch.cuda.empty_cache(). AC #3 ('Flux2Klein at batch size 8+') was superseded: the Flux2Klein OOM turned out to be a per-model VRAM ceiling, not cross-group residue, and was fixed in commit 151450a via _T2I_MAX_BATCH caps ({SD35Medium: 4, ZImageTurbo: 4, Flux2Klein: 2}). Multi-network experiments including HunyuanImage + GLMImage (much larger than Flux2Klein) now run successfully end-to-end (e.g. seasons_3x3_b.json).
<!-- SECTION:NOTES:END -->
