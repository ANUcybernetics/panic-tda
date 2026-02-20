---
id: TASK-69
title: Fix GPU memory not being freed between network groups
status: To Do
assignee: []
created_date: '2026-02-20 08:43'
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
- [ ] #1 Previous model references are deleted before loading the next network group
- [ ] #2 torch.cuda.empty_cache() is called after model unloading
- [ ] #3 Flux2Klein runs successfully in a multi-network experiment with batch size 8+
- [ ] #4 GPU memory usage drops to baseline between network group transitions
<!-- AC:END -->
