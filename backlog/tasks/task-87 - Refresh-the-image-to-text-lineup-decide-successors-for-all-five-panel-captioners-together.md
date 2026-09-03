---
id: TASK-87
title: >-
  Refresh the image-to-text lineup: decide successors for all five panel
  captioners together
status: To Do
assignee: []
created_date: '2026-09-03 09:29'
labels:
  - models
  - experiment-design
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ben, 2026-09-03: asked about moving Moondream to v3; the answer is that it should be one lineup decision across all five captioners rather than a piecemeal swap. TASK-84 deliberately scoped successors out ('a separate lineup decision, not this task') --- this is that task.

Why together rather than one at a time. The five captioners are a designed experimental factor. Upgrading only Moondream would make the factor 'four 2024-25 models plus one 2026 model', confounding model era with model identity, and Moondream specifically is the cross-era anchor for TASK-81 (already weakened: its revision moved and its length mode went short -> normal under decision-01, but an architecture change ends the link entirely). Whatever is decided should land BEFORE the next panel run, not during.

Candidates verified on HF 2026-09-03 (dates are lastModified; all checked directly, not from memory):

| current | candidate successor | date | licence | gated | transformers arch |
|---|---|---|---|---|---|
| Moondream (vikhyatk/moondream2, apache-2.0) | moondream/moondream3-preview | 2026-04-09 | BSL 1.1 -> Apache-2.0 after 2 yrs | no | HfMoondream (custom_code) |
| Qwen25VL (Qwen/Qwen2.5-VL-7B-Instruct) | Qwen/Qwen3-VL-8B-Instruct (or 4B; 30B-A3B is MoE) | 2025-10-15 | apache-2.0 | no | Qwen3VLForConditionalGeneration |
| Gemma3n (google/gemma-3n-E2B-it) | google/gemma-4-26B-A4B-it | 2026-07-20 | apache-2.0 | no | Gemma4ForConditionalGeneration |
| Pixtral (mistral-community/pixtral-12b) | mistralai/Mistral-Small-3.2-24B-Instruct-2506 | 2025-12-22 | apache-2.0 | no | Mistral3ForConditionalGeneration |
| LLaMA32Vision (meta-llama/Llama-3.2-11B-Vision-Instruct) | meta-llama/Llama-4-Scout-17B-16E-Instruct | 2025-05-22 | other | manual gate | Llama4ForConditionalGeneration |

Notes. moondream3.1-9B-A2B (2026-07-08) is NEWER than the preview but ships only config.json plus a single 10.5 GB model.safetensors with no modelling code and architectures null --- it is not transformers-loadable as published, so the preview is the integratable one. google/gemma-4-26B-A4B-it is already in the local HF cache, as is an AWQ-4bit community quant. Llama-4-Scout is manually gated and 17Bx16E, so check both access and whether it fits 48 GB at 4-bit before counting on it.

Also worth weighing from the September 2026 survey (see the model-survey notes in TASK-88): fancyfeast/llama-joycaption-beta-one-hf-llava and internlm/CapRL-Qwen3VL-4B are captioners with genuinely different training objectives (caption-density-first, and RL-against-QA-verification respectively) rather than successors --- they may add more to the panel than a like-for-like upgrade does.

Practical constraints. _load_moondream in priv/python/panic_models.py is bespoke (snapshot_download, pull the class out of transformers_modules, load a single model.safetensors with a prefix strip) and will not survive a sharded v3 checkpoint --- it likely simplifies to a standard trust_remote_code load. Moondream is currently the fastest captioner by a wide margin (0.68 s/caption at natural length, measured 2026-09-02); a 9B/2B-active MoE will cost more across ~100k invocations. Every model swapped needs its caption length and terminal-punctuation share re-measured with analysis/natural_lengths.py, since decision-01 makes caption length a measured covariate and the current length landscape was only just characterised.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Decision recorded per captioner: upgrade, replace with a differently-objective model, or keep, with the reason
- [ ] #2 Whatever is adopted is pinned to an explicit revision in _REVISIONS alongside the rest, and loads through as standard a transformers path as the model allows
- [ ] #3 Caption length, terminal-punctuation share and seconds-per-caption re-measured for every changed model via analysis/natural_lengths.py, and backlog/docs/caption-length-by-i2t-model.md updated
- [ ] #4 Licence implications noted for any non-Apache model adopted (Moondream 3 is BSL 1.1; Llama 4 is gated and custom-licensed)
- [ ] #5 Impact on TASK-81's cross-era comparison stated explicitly once the Moondream decision is made
- [ ] #6 mix test --include gpu green for the new lineup, and CLAUDE.md model table plus run-time rows updated
<!-- AC:END -->
