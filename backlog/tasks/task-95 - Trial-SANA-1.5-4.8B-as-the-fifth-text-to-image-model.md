---
id: TASK-95
title: Trial SANA 1.5 4.8B as the fifth text-to-image model
status: To Do
assignee: []
created_date: '2026-09-04 07:59'
labels:
  - models
  - instrument
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ben, 2026-09-04: after GLMImage was removed (TASK-94) the panel is 4x5, and the question was whether to replace it and keep five generator levels. Decision: trial SANA 1.5 4.8B.

WHY SANA. The four remaining generators are only three architectures --- Flux2Klein and Flux2Dev share the Flux2 transformer and its Mistral3 text encoder, differing in distillation (4 steps against 12), which is a deliberate contrast worth naming but not two independent levels. SANA is the most mechanism-diverse candidate that fits 48 GB at bfloat16 with no quantisation: linear-attention DiT, a 32x deep-compression autoencoder rather than the usual 8x, and a decoder-only text encoder (Gemma2Model). A different latent geometry is directly on-topic for a paper about trajectories through semantic space. Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers, 14.8 GB, apache-2.0, revision 9468102c3cebb657f8c4b5f1e5a71e989a15f10d, first-class SanaPipeline in the installed diffusers.

Cost is why this is worth doing at all: GLMImage cost 31 GPU-days because it ran at 42 s/item, not because it was a fifth level. A generator in the 5-10 s/item band adds roughly 6, taking the 300-step four-run design from 58 GPU-days to about 64.

TWO THINGS COULD DISQUALIFY IT, both found before any GPU time was spent.

1. CAPTION TRUNCATION, the serious one. SanaPipeline defaults max_sequence_length to 300, which is SANA's trained conditioning length. Measured on the TASK-89 caption set, 2 of 8 captions already exceed it:

     45w -> 57 tok    83w -> 94     92w -> 108    145w -> 173
    187w -> 220      222w -> 288   245w -> 340   313w -> 411

Projected to the panel captioners, Moondream3 (~70), Qwen25VL (~95) and JoyCaption (~210) are safe, Gemma4 (~290-340) is marginal and Qwen3VL (~380-400) would truncate on nearly every step. That is exactly the per-model truncation asymmetry decision-01 and TASK-82 removed, and it is why balanced_panel_5x5 is hard to interpret. Gemma-2's own context is 8k, so the limit is the transformer's cross-attention, trained at 300 --- mechanically raisable, but out of distribution.

2. PROMPT REWRITING. SanaPipeline's complex_human_instruction defaults to a multi-sentence instruction telling the encoder to produce an 'Enhanced prompt' with added detail, prepended to the caption. Left on, the model is not conditioned on our caption but on a rewrite of it, which breaks the loop's premise. It takes None.

ORDER OF WORK: run the disqualifying test first and stop early if it fails. If SANA is disqualified, the fallback is CogView4-6B (29 GB, apache-2.0, GLM-4 text encoder with a long context, plain DiT) --- GLMImage's slot without the glyph branch, the AR prior fault or the quantisation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 max_sequence_length raised to at least 512 and checked for degradation: images from the same caption at 300 and at 512+ compared against the seed-noise floor from TASK-89, and a visual check that the 512 images are not degraded --- if they are, SANA is rejected and the task closes there
- [ ] #2 complex_human_instruction=None confirmed to disable prompt rewriting, verified by comparing the encoded prompt or the resulting images against the default
- [ ] #3 VRAM and seconds per image measured at batch 1 and at the batch size the panel would use, on the RTX 6000 Ada at bfloat16 with no quantisation
- [ ] #4 Caption fidelity checked the way the rest of the lineup was: captions of SANA images sit in the same cosine range as captions of the other generators' images, so it is a comparable generator and not an outlier
- [ ] #5 Deterministic given a seed, and free of the fault class that motivated TASK-79, over a run long enough to see it
- [ ] #6 If adopted: pinned in _REVISIONS, wired into panic_models, genai.ex, gpu.bench, the GPU tests and the panel config, and CLAUDE.md/README updated --- panel back to 5x5
<!-- AC:END -->
