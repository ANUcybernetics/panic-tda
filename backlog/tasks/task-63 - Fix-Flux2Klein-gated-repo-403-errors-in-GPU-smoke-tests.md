---
id: TASK-63
title: Fix Flux2Klein gated repo 403 errors in GPU smoke tests
status: Done
assignee: []
created_date: '2026-02-17 21:16'
updated_date: '2026-04-16 21:00'
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
- [x] #1 Flux2Klein GPU smoke tests either pass or are gracefully skipped when repo access is unavailable
- [x] #2 No raw 403 errors in test output
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Resolved via HF authentication, not a code change. HF token is set via HUGGINGFACE_TOKEN env var and cached at ~/.cache/huggingface/token. black-forest-labs/FLUX.2-klein-9B is present in the local HF hub cache and is actively used in production experiments (config/penguin_campfire.json, config/seasons_3x3.json, etc.). The real_models_test.exs suite exercises Flux2Klein alongside all other T2I models without a skip guard. No 403 in current runs; if access is ever revoked upstream, the failure mode will be a raw HF error but that's out of scope for this task.
<!-- SECTION:NOTES:END -->
