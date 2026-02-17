---
id: TASK-63
title: Fix Flux2Klein gated repo 403 errors in GPU smoke tests
status: To Do
assignee: []
created_date: '2026-02-17 21:16'
labels:
  - gpu
  - testing
  - huggingface
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The GPU smoke tests for Flux2Klein fail with a 403 error because the HuggingFace model `black-forest-labs/FLUX.2-klein-9B` is a gated repo requiring access approval.

Affected tests:
- "Flux2Klein + Moondream"
- "Flux2Klein + InstructBLIP"

Options:
- Skip the test when HF access is not available (detect 403 and tag-skip)
- Configure HF token/access for the gated repo in CI and dev environments
- Add a runtime check that provides a clear error message instead of a raw 403
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Flux2Klein GPU smoke tests either pass or are gracefully skipped when repo access is unavailable
- [ ] #2 No raw 403 errors in test output
<!-- AC:END -->
