---
id: TASK-64
title: Increase Ecto sandbox ownership_timeout for QwenImage GPU tests
status: To Do
assignee: []
created_date: '2026-02-17 21:16'
labels:
  - gpu
  - testing
  - ecto
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The QwenImage model combinations (QwenImage + Gemma3n, QwenImage + Qwen25VL) exceed the 120s default Ecto sandbox ownership_timeout during GPU smoke tests.

QwenImage takes ~53s for a single generation, so a full experiment pipeline (multiple generations plus embeddings/TDA/clustering) easily exceeds the 120s limit.

Options:
- Increase ownership_timeout in test setup for GPU-tagged tests
- Restructure the GPU smoke tests to run shorter pipelines
- Set a per-test timeout override using ExUnit tags
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 QwenImage + Gemma3n GPU smoke test completes without DB ownership timeout
- [ ] #2 QwenImage + Qwen25VL GPU smoke test completes without DB ownership timeout
- [ ] #3 Non-GPU tests are unaffected by any timeout changes
<!-- AC:END -->
