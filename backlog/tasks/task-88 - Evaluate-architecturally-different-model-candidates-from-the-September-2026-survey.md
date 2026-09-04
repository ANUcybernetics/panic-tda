---
id: TASK-88
title: >-
  Evaluate architecturally different model candidates from the September 2026
  survey
status: To Do
assignee: []
created_date: '2026-09-03 09:30'
updated_date: '2026-09-04 07:59'
labels:
  - models
  - research
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ben, 2026-09-03: asked whether there are significant new releases worth considering --- 'comparable in the task they do but perhaps different in their approach'. Survey run 2026-09-03; every repo below was verified directly against the HF API on that date (dates are lastModified), which corrected several claims from the initial search pass.

The framing that matters: diversity of APPROACH is worth more here than leaderboard rank, because the panel treats model identity as an experimental factor. Like-for-like successors are TASK-87; this task is about genuinely different mechanisms.

| candidate | date | licence | gated | arch entry | why it differs |
|---|---|---|---|---|---|
| fancyfeast/llama-joycaption-beta-one-hf-llava | 2025-05-16 | none declared | no | LlavaForConditionalGeneration | captioning-first training objective, uncensored, verbose --- every current captioner is chat/QA-tuned |
| internlm/CapRL-Qwen3VL-4B | 2026-04-16 | apache-2.0 | no | Qwen3VLForConditionalGeneration | trained by RL against a QA-verification reward for caption coverage, not SFT |
| ByteDance-Seed/BAGEL-7B-MoT | 2026-01-09 | apache-2.0 | no | BagelForConditionalGeneration | unified any-to-any: generates AND captions in one checkpoint |
| deepseek-ai/Janus-Pro-7B | 2025-02-01 | mit (cardData) | no | none | any-to-any with decoupled visual encoding, discrete-token AR generation |
| stepfun-ai/NextStep-1.1 | 2025-12-23 | apache-2.0 | no | LlamaForCausalLM | autoregressive T2I with a flow-matching head --- our entire T2I lineup is diffusion/flow-matching |
| BAAI/Emu3.5 | 2025-12-25 | apache-2.0 | no | Emu3ForCausalLM | 34B native next-token any-to-any over interleaved vision-language |
| jinaai/jina-embeddings-v4 | 2026-04-08 | none declared | no | JinaEmbeddingsV4Model | one checkpoint serving BOTH dense and multi-vector/late-interaction, and cross-modal |
| nvidia/nemotron-colembed-vl-4b-v2 | 2026-02-21 | cc-by-nc-4.0 | no | Qwen3VLNemotronEmbedModel | ColBERT-style late interaction over visual documents |

Top three by value-for-effort: (1) JoyCaption --- standard Llava class, drops straight in, and given that decision-01 made caption length a measured covariate, a caption-density-optimised model is a real datapoint; (2) BAGEL --- the only candidate that puts the whole T2I->I2T loop inside ONE network, which speaks directly to the project's core question about information flow; (3) jina-embeddings-v4 --- the dense/late-interaction switch is the one genuine embedding paradigm gap now that Qwen3Embed is the sole embedder.

Corrections to the first-pass survey, from direct verification: BAGEL DOES declare a transformers architecture (BagelForConditionalGeneration), so it may not need the custom code the survey assumed; CapRL is 2026-04-16, not Dec 2025; Janus-Pro's cardData says MIT, though the survey believed a separate DeepSeek model licence applies --- read the repo licence before relying on either. JoyCaption and jina-v4 declare NO licence in cardData, which needs checking before use. nemotron-colembed was originally rejected as redundant with ColNomic; that no longer holds, since TASK-84 removed ColNomic, making it and jina-v4 the only late-interaction options.

Rejected on hardware or staleness grounds, with reasons: zai-org/GLM-5.3-Flash (320B total --- will not fit 48 GB even sparse); tencent/HunyuanImage-3.0 (80B MoE, only viable via an unofficial NF4 quant estimated at 41-49 GB, too marginal for a controlled pipeline); SPLADE-family sparse embedders (architecturally distinct but no meaningful 2026 activity); FoundationVision/Infinity-8B (bitwise VAR, the most novel T2I candidate, but no verifiable diffusers/transformers integration path --- would need a spike).

Adoption signal worth weighing: BAGEL (935 downloads) and Emu3.5 (380) have very low uptake, so expect thin community knowledge and more integration risk than the download-heavy options (jina-v4 431k, JoyCaption 74k).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Licences confirmed from the repo files (not cardData) for any candidate taken forward, in particular JoyCaption, jina-embeddings-v4 and Janus-Pro
- [ ] #2 VRAM and speed measured on the RTX 6000 Ada for each candidate actually trialled, rather than estimated from parameter count
- [ ] #3 Decision recorded per candidate: adopt, spike further, or reject with reason
- [ ] #4 Any adopted model pinned in _REVISIONS, wired into the Elixir model lists and GPU tests, and its caption or generation behaviour measured the way the current lineup was
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Un-deferred 2026-09-04: the lineup came back into question when GLMImage was removed (TASK-94), which is the condition this task named for itself.

The text-to-image half is now TASK-95 (SANA 1.5 4.8B, with CogView4-6B as the fallback). SANA was not in the survey table above --- it was found by enumerating the first-class pipelines in the installed diffusers rather than by searching releases, which is the cheaper filter for this project, since a model without a diffusers pipeline needs a spike before it can even be measured.

Sizes verified against the HF API 2026-09-04, which rules several survey candidates out on the no-quantisation constraint that GLMImage's removal established (48 GB, bfloat16, no 4-bit): NextStep-1.1 55.7 GB, Qwen-Image 53.7 GB, HiDream-I1-Full 43.9 GB. Emu3.5 at 34B is out for the same reason. What remains of this task is the captioner and embedder candidates.
<!-- SECTION:NOTES:END -->
