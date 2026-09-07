---
id: TASK-101
title: >-
  Replace Qwen3VL with a captioner that stays under the 512-token encoder
  ceiling
status: To Do
assignee: []
created_date: '2026-09-07 04:39'
labels:
  - experiment
  - gpu
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Qwen3VL (Qwen/Qwen3-VL-8B-Instruct) is out of the panel: the TASK-90 pre-launch smoke found its captions of early-step images exceed the 512-token text-encoder ceiling for 14 of 80 under SD35Medium's T5 tokenizer (median 426, p90 563, max 744) and 2 of 80 under the Flux2 and Z-Image tokenizers, and the pipelines cut silently. TASK-87 had flagged it as marginal (466 of 512 on easy images). The other four captioners are under 512 on every measured image, Gemma4 at most 410.

GOAL. Find a fifth captioner that clears the ceiling with margin, or conclude that the panel launches as 4x4 (config/long_horizon_panel_4x4_300.json, already prepared). Do not touch the 4x4 config or delay the launch for this: the long-horizon run is either already running or about to be, and a fifth captioner can only join it as a later batch of cells with the same prompts, runs and horizon.

CANDIDATES, in order. (1) A smaller Qwen3-VL instruct variant (Qwen/Qwen3-VL-4B-Instruct or the 2B), which would keep the Qwen2.5-VL vs Qwen3-VL matched-family contrast that put Qwen25VL in the panel; its verbosity is unmeasured. (2) Anything else from the TASK-87/88 surveys that loads through a standard transformers class in bfloat16 on the 48 GB card without quantisation (Mistral Small 3.2 fails that: 24B). The screens are TASK-87's: pinned revision, bfloat16 with no quantisation, greedy decoding forced and verified deterministic, no system prompt or template added to 'Describe this image.', natural termination at the 1024 ceiling.

THE MEASUREMENT THAT DECIDES IT. Caption images that stress length: early-step (steps 1-3) images of the twenty panel prompts from all four generators, at the panel's batch cap of 40 (greedy captions change with batch composition, so measure at the cap that will run). Tokenise every caption with SD3.5's tokenizer_3 (T5), the FLUX.2-klein-9B tokenizer (Mistral3) and the Z-Image-Turbo tokenizer (Qwen3), loading from the cached snapshot directories under /data/huggingface/hub since offline hub resolution fails. Pass criterion: p99 under 512 on all three with the max recorded; a candidate whose max exceeds 512 is out, as CapRL and Qwen3VL were. Record words, tokens, seconds per caption at cap 40 and peak VRAM in backlog/docs/caption-length-by-i2t-model.md.

IF ADOPTED. Pin in _REVISIONS, wire into panic_models.py, genai.ex, gpu.bench and the GPU tests the way TASK-87 did, and write config/long_horizon_panel_4x1_300.json for the four extra cells (same 20 prompts, num_runs 2, max_length 300) to run after the 4x4 finishes. Cost is about 5 GPU-days.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Candidate captioner(s) screened with TASK-87's checks: pinned, bfloat16 unquantised, greedy verified deterministic, no prompt template, natural termination
- [ ] #2 Token counts under all three encoder tokenizers measured on early-step images of the 20 panel prompts at batch cap 40, with p99 under 512 and the max recorded, or the candidate rejected
- [ ] #3 Decision recorded: adopt (wired in, config for the extra four cells committed) or panel stays 4x4, with the reason in caption-length-by-i2t-model.md
<!-- AC:END -->
