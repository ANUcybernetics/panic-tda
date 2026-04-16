---
id: TASK-64
title: Increase Ecto sandbox ownership_timeout for QwenImage GPU tests
status: Done
assignee: []
created_date: '2026-02-17 21:16'
updated_date: '2026-04-16 20:57'
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
- [x] #3 Non-GPU tests are unaffected by any timeout changes
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Superseded. ownership_timeout was raised to 600_000 in config/test.exs (line 6) as part of TASK-65. Acceptance criteria #1 and #2 are no longer applicable: QwenImage was removed from the codebase in commit 15058ba ('Remove QwenImage model due to unusable output quality with NF4 quantisation'), so there are no QwenImage GPU smoke tests left to time out.
<!-- SECTION:NOTES:END -->
