---
id: TASK-68
title: Investigate ZImageTurbo blurry output in recursive trajectories
status: To Do
assignee: []
created_date: '2026-02-20 06:35'
labels:
  - gpu
  - models
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ZImageTurbo images become very blurry almost from the start of recursive T2I→I2T trajectories (observed in batch12_diverse experiment). The model is currently configured with num_inference_steps=8 (correct for Turbo mode), but guidance_scale is not set --- the Turbo distillation expects guidance_scale=0.0, and diffusers may be defaulting to a higher value that degrades quality. Additionally, the recursive captioning loop may be producing prompts that don't work well with Turbo's low-step distillation. Investigate whether setting guidance_scale=0.0 fixes the blurriness, and whether there are other ZImageTurbo-specific settings that improve output quality in recursive trajectories.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Confirm current default guidance_scale when not explicitly set
- [ ] #2 Test ZImageTurbo output with guidance_scale=0.0 vs current default
- [ ] #3 Compare image quality across several trajectory steps with the fix applied
- [ ] #4 Update _T2I_INVOKE_CONFIGS if a fix is found
<!-- AC:END -->
