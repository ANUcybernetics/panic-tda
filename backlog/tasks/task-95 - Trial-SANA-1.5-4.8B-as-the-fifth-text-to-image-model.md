---
id: TASK-95
title: Trial CogView4-6B as the fifth text-to-image model
status: To Do
assignee: []
created_date: '2026-09-04 07:59'
updated_date: '2026-09-04 08:15'
labels:
  - models
  - instrument
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ben, 2026-09-04: after GLMImage was removed (TASK-94) the panel is 4x5, and the question was whether to replace it and keep five generator levels. Decision: trial CogView4-6B.

THUDM/CogView4-6B, 29.0 GB, apache-2.0, revision 63a52b7f6dace7033380cd6da14d0915eab3e6b5, first-class CogView4Pipeline in the installed diffusers.

WHY. It is the only candidate that clears all four screens with headroom --- see backlog/docs/text-to-image-model-screen.md for the method and the full table of sixteen. Context 1024 tokens against our 466-token worst case; no instruction, template or rewrite parameter anywhere in its call signature; 29 GB at bfloat16 with no quantisation, which is what GLMImage failed. Its GLM-4 decoder-only text encoder (GlmModel, 18.3 GB of the 29) gives the panel four encoder families: T5+CLIP, Mistral3, Qwen3, GLM-4. Text encoder is the axis worth varying, since it is how the generator reads the caption.

SANA 1.5 was the first choice and was rejected on screen 2: SanaPipeline conditions on 300 tokens, and 2 of the 8 TASK-89 captions already exceed that, which would truncate Gemma4 and Qwen3VL. It also defaults complex_human_instruction to a prompt-rewriting instruction. The screen found that five of sixteen candidates rewrite or augment the prompt by default, so GLM-Image's glyph branch was a genre convention rather than a one-off.

THE OPEN RISK IS SPEED, and it decides whether this is worth doing at all. CogView4 defaults to 50 inference steps on a 6B transformer, unmeasured here. A generator in the 5-10 s/item band adds about 6 GPU-days to the 300-step four-run design (58 to ~64); at 20-25 s/item it adds about 20, which is most of what removing GLMImage bought back. Measure it first and apply TASK-83's step-count method before accepting the default.

Fallback if CogView4 is rejected: lodestones/Chroma1-HD (apache-2.0, ~26 GB, 512 context, clean prompt path, revision 0e0c60ece1e82b17cb7f77342d765ba5024c40c0), accepting that it is FLUX.1-derived and so overlaps the two Flux2 levels.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Deterministic given a seed, and free of the fault class that motivated TASK-79, over a run long enough to see it
- [ ] #2 If adopted: pinned in _REVISIONS, wired into panic_models, genai.ex, gpu.bench, the GPU tests and the panel config, and CLAUDE.md/README updated --- panel back to 5x5
- [ ] #3 Seconds per image and VRAM measured at bfloat16 with no quantisation, at batch 1 and at the panel's batch size, and a step-count sweep run the way TASK-83 did --- with the GPU-day cost of adopting it stated before anything else is decided
- [ ] #4 Confirmed that no prompt text is added or rewritten: the encoded prompt matches the caption, with nothing prepended
- [ ] #5 The 1024-token context confirmed to accept our longest captions without truncation, measured in the pipeline's own tokenizer rather than by word count
- [ ] #6 Caption fidelity checked the way the rest of the lineup was: captions of CogView4 images sit in the same cosine range as captions of the other generators' images, so it is a comparable generator rather than an outlier
<!-- AC:END -->
