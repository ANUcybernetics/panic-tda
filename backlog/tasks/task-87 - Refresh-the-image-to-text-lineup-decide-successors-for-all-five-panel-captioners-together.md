---
id: TASK-87
title: >-
  Refresh the image-to-text lineup: decide successors for all five panel
  captioners together
status: Done
assignee: []
created_date: '2026-09-03 09:29'
updated_date: '2026-09-03 15:53'
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
- [x] #1 Decision recorded per captioner: upgrade, replace with a differently-objective model, or keep, with the reason
- [x] #2 Whatever is adopted is pinned to an explicit revision in _REVISIONS alongside the rest, and loads through as standard a transformers path as the model allows
- [x] #3 Caption length, terminal-punctuation share and seconds-per-caption re-measured for every changed model via analysis/natural_lengths.py, and backlog/docs/caption-length-by-i2t-model.md updated
- [x] #4 Licence implications noted for any non-Apache model adopted (Moondream 3 is BSL 1.1; Llama 4 is gated and custom-licensed)
- [x] #5 Impact on TASK-81's cross-era comparison stated explicitly once the Moondream decision is made
- [x] #6 mix test --include gpu green for the new lineup, and CLAUDE.md model table plus run-time rows updated
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Ben, 2026-09-03: happy with the chat-tuned successors (Qwen3-VL, Gemma 4, Mistral Small 3.2, Llama 4 Scout, Moondream 3) --- those are endorsed as candidates, so the task is scoped to picking sizes/variants and validating them rather than re-litigating whether to upgrade at all.

Still open: whether the caption-first models in TASK-88 (fancyfeast/llama-joycaption-beta-one-hf-llava, internlm/CapRL-Qwen3VL-4B) join the same lineup decision, either as additions to the panel or as replacements for one of the chat-tuned slots. They are not successors, so they widen the panel's objective diversity rather than modernising it --- which matters more now that decision-01 makes caption length a measured covariate.

Recommendation (Claude, 2026-09-03, Ben asked for advice): keep the captioner factor at FIVE levels and spend two slots on objective diversity rather than a fifth chat-tuned model.

Proposed: Moondream 3 (terse end, fastest), Qwen3-VL-8B-Instruct (chat-tuned reference), CapRL-Qwen3VL-4B (matched contrast), Gemma 4 26B-A4B (chat-tuned, different family), JoyCaption (caption-first, verbose end). Drop Mistral Small 3.2 and Llama 4 Scout.

The deciding argument is that CapRL-Qwen3VL-4B is built on Qwen3-VL. Running it alongside Qwen3-VL-8B-Instruct gives a matched-backbone contrast --- same architecture family, different training objective (instruct SFT vs RL against a QA-coverage reward). That is the only clean way to separate training objective from architecture within the captioner factor, which is exactly what RQ2's style-versus-verbosity attribution needs; no unrelated model can provide it. Everything measured on 2026-09-02 supports making objective the axis of interest: caption length is almost entirely determined by captioner identity (eta-squared 0.915), and caption completeness measurably changes the loop dynamics (28% step-size difference), so five near-identical chat-tuned models would sample a narrow part of that space.

Why Llama 4 Scout is the clearest drop: manually gated, custom licence, largest of the candidate set (17Bx16E), and its incumbent (Llama-3.2-Vision) was 78.3% truncated, so its natural captioning behaviour is the least characterised of the five. Most friction, least distinctive contribution. Mistral Small 3.2 is the natural sixth if six levels are wanted; it is dropped only because a fourth chat-tuned family adds less than a caption-first model does.

Cost note: the panel is 5 T2I x N I2T, so holding N at 5 keeps it a clean 5x5 and avoids the ~40% run-time increase that seven captioners would bring.

Caveat to state in the paper: swapping two of five captioners means the new panel is not comparable to balanced_panel_5x5 on the captioner axis --- though that comparability is already gone via decision-01 and the successor upgrades.

Constraints and costings established 2026-09-03, to be respected whenever the new lineup is run:

1. RECLUSTER ONCE, AT THE END. mix cluster.recompute is destructive and global: ClusteringStage.delete_existing_clustering destroys the ClusteringResult and every EmbeddingCluster row for that embedding model before rebuilding, so reclustering after a new panel RELABELS EVERY EXISTING EXPERIMENT including 019f3645. The old labels do not survive alongside the new. Every cluster-dependent figure --- including the pilot-vs-panel occupancy and transition numbers in analysis/pilot_vs_panel.json, which were computed against the current medoids --- must be regenerated from the final clustering. Do all data collection first, cluster once, then make figures.

2. VALIDATE BEFORE COMMITTING GPU. Each new model needs its loader, an explicit revision pin, a GPU smoke pass and a caption-length/terminal-punctuation characterisation before a multi-week run starts. This session alone turned up four traps that would each have quietly corrupted or broken a long run: diffusers defaulting max_sequence_length to 256; setup() suppressing the very warnings a check relied on; a PEFT patch mislabelled as ColNomic-specific that actually guards Qwen25VL; and sentence-transformers 6 renaming two APIs our code used. Assume more of the same per new model. Moondream 3 specifically needs its bespoke single-file loader rewritten, since v3 ships a sharded checkpoint.

3. COST. Captioners are not the bottleneck and should not drive the choice. At natural length the current five total roughly 0.9 days against about 13.5 days of T2I over a full panel (50,000 invocations each side). Even a threefold captioner slowdown adds under 3 days. Gemma 4 26B-A4B is the one worth measuring rather than assuming. Note also a correction to an earlier claim in this session: the optimisations cut about 12% off the panel's T2I time relative to the 019f3645 run (15.3 -> 13.5 days), not half --- that run already had TASK-74 batching for most of its duration. The ~2x figure is against the pre-TASK-74 baseline.

4. PROVENANCE is safe either way: experiments.networks is stored per experiment, so 019f3645 retains its real lineup whatever the config file says. config/balanced_panel_5x5_v2.json has been drafted anyway so the file that produced the published dataset stays readable; its captioner names are placeholders pending the decision below.

5. TASK-81 needs rescoping or dropping once Moondream is decided --- changing its architecture removes the last thread linking to the SMC 2025 era.

DECIDED (Ben, 2026-09-03):
- Captioners: Moondream 3, Qwen3-VL-8B-Instruct, CapRL-Qwen3VL-4B, Gemma 4 26B-A4B, JoyCaption. Mistral Small 3.2 and Llama 4 Scout are dropped.
- Panel stays 5x5 (25 networks), so the factorial design and analysis code carry over unchanged.
- Text-to-image side is UNCHANGED. Four of the five are 2026 releases; SD35Medium is dated but is the cheapest and best-characterised, and changing one factor at a time keeps the comparison interpretable.
- TASK-81 is dropped (see that task).

Registry names to use, matching config/balanced_panel_5x5_v2.json: Moondream3, Qwen3VL, CapRL, Gemma4, JoyCaption. Repos and revisions to pin: moondream/moondream3-preview, Qwen/Qwen3-VL-8B-Instruct, internlm/CapRL-Qwen3VL-4B, google/gemma-4-26B-A4B-it, fancyfeast/llama-joycaption-beta-one-hf-llava.

Order of work, cheapest and least risky first: JoyCaption and CapRL both load through standard transformers classes (LlavaForConditionalGeneration and Qwen3VLForConditionalGeneration), so they are near drop-ins. Qwen3-VL is the same family as the incumbent Qwen25VL. Gemma 4 is a 26B MoE and the one whose speed needs measuring rather than assuming. Moondream 3 is the most work: its bespoke single-file loader must be rewritten for a sharded checkpoint, and it may simplify to a standard trust_remote_code load.

Licence note for the paper: Moondream 3 is BSL 1.1 (converting to Apache-2.0 two years after release); the other four are Apache-2.0. JoyCaption declares no licence in its HF metadata, so read the repo before relying on it.

INTEGRATION FINDINGS 2026-09-03 (three of five done: CapRL, Qwen3VL, JoyCaption load and caption; Gemma 4 and Moondream 3 not yet downloaded).

Measured on four Flux2Dev images, natural length, same 'Describe this image.' instruction as the rest of the panel:

| captioner | median words | range | T5 tokens | over 512 | s/caption |
|---|---|---|---|---|---|
| JoyCaption | 178 | 91-178 | 131-250 | 0/4 | 2.2 |
| Qwen3VL | 292 | 198-317 | 312-466 | 0/4 | 4.0 |
| CapRL | 466 | 307-561 | 466-836 | 3/4 | 4.0 |

For reference the CURRENT lineup at natural length is 80-154 words. The 2026 captioners are three to six times more verbose.

THE PROBLEM: this puts the natural-length policy (decision-01) in direct tension with a modern captioner lineup. The text-to-image encoders take 512 tokens --- SD35Medium HARD-CAPS there (StableDiffusion3Pipeline raises above 512), while Flux2Klein, Flux2Dev and ZImageTurbo merely default to 512 with no hard cap, and GLMImage takes 2048. CapRL's natural output exceeds 512 on three of four images, so its captions would be silently truncated by four of the five T2I models --- reintroducing exactly the invisible truncation decision-01 was written to eliminate, across a two-week run. Qwen3VL is under the limit but close (466 of 512 on easy images), so harder images would breach it too.

Options: drop CapRL as incompatible with the setup (its natural output exceeds what the image models can read); or replace SD35Medium, the only hard-capped model, and raise max_sequence_length on the other three; or impose a uniform generation cap sized to 512 tokens (~380 words) as a deliberate, documented manipulation under decision-01; or accept and document SD35Medium truncation. Needs Ben's call --- it changes the experiment design.

Also noted: CapRL prefixes its captions with 'Based on the provided image, here is a description:', boilerplate that would feed into the next image. JoyCaption produces dense, preamble-free captions and was the best-behaved of the three.

DECIDED (Ben, 2026-09-03): CapRL is dropped as incompatible --- its natural output exceeds what four of the five text-to-image models can read, so including it would mean measuring truncation again rather than captioner behaviour.

Fifth slot goes to Qwen25VL, retained rather than replaced by Mistral Small 3.2. Reasons, which make this a better outcome than CapRL would have been: Qwen2.5-VL against Qwen3-VL is a matched-FAMILY contrast that separates model generation from architecture (the same trick CapRL was meant to provide, on the era axis rather than the objective axis); it leaves one captioner in common with balanced_panel_5x5, partially restoring a cross-panel bridge that looked lost; it carries zero integration risk and is already characterised (80 words median, 0.91 s/caption); and it is the shortest of the verbose group, which is useful ballast now the 2026 models run long.

FINAL v2 LINEUP (config/balanced_panel_5x5_v2.json): text-to-image unchanged (SD35Medium, ZImageTurbo, Flux2Klein, GLMImage, Flux2Dev); captioners Moondream3, Qwen25VL, Qwen3VL, Gemma4, JoyCaption; Qwen3Embed; 25 networks, 20 prompts, 50 steps, 4 runs.

Still to integrate: Gemma 4 26B-A4B (51.6 GB download, needs 4-bit, and the one whose speed must be measured rather than assumed) and Moondream 3 (18.5 GB, bespoke single-file loader must be rewritten for a sharded checkpoint). Both must also be checked against the 512-token encoder ceiling, since that is now a known hazard for 2026 captioners --- Qwen3VL already sits at 466 of 512 on easy images.

COMPLETE 2026-09-04. Full GPU suite green on the new lineup: 158 tests, 0 failures in 88 minutes, no warnings.

Final v2 captioners, all measured on four Flux2Dev images at natural length with the same 'Describe this image.' instruction (T5 token counts matter because the text-to-image encoders read at most 512):

| model | repo | median words | T5 tokens | s/caption |
|---|---|---|---|---|
| Moondream3 | moondream/moondream3-preview | 59 | 52-81 | 2.4 |
| Qwen25VL | Qwen/Qwen2.5-VL-7B-Instruct | 80 | --- | 0.9 |
| JoyCaption | fancyfeast/llama-joycaption-beta-one-hf-llava | 178 | 131-250 | 2.2 |
| Gemma4 | google/gemma-4-E4B-it | 225 | 262-389 | 2.6 |
| Qwen3VL | Qwen/Qwen3-VL-8B-Instruct | 292 | 312-466 | 4.0 |

All pinned in _REVISIONS; text-to-image side unchanged. Roughly a fivefold length spread with overlapping distributions.

Three code simplifications fell out. Moondream 3 ships a working auto_map, so the bespoke loader (fish the class out of transformers_modules, hand-build the config, prefix-strip a single safetensors) is gone. One parameterised loader now covers the whole Qwen-VL family, and one invoke path covers both Gemma models.

Four traps caught by measurement, none visible in caption output --- this is the evidence that 'validate before committing GPU' is a real requirement:
1. CapRL captions at 466 words median / up to 836 T5 tokens, exceeding the 512-token ceiling on 3 of 4 images. Dropped as incompatible.
2. Gemma 4 26B-A4B occupied 48.4 GB of a 50.9 GB card. bitsandbytes replaces nn.Linear only and silently skipped the 189 Gemma4ClippableLinear modules holding 47.2 GB. Because a bnb model must live in _models_offload_only, where swap_to_cpu only empty_cache()s, it would have stayed resident and OOM'd every diffusion model mid-panel. Replaced with the dense E4B: 15.9 GB, no quantisation needed, and swap_to_cpu returns allocated to 0.0 GB.
3. Loading Moondream 3 with device_map would have failed mid-panel: swap_to_cpu calls .to('cpu') on anything outside _models_offload_only, which an accelerate-hooked model cannot do --- and a swap_to_gpu-only test does not catch it.
4. JoyCaption's chat template takes plain-string content, not the structured parts list every other model uses.

Carried forward: Qwen3VL has the least headroom against the 512-token ceiling (466 of 512 on easy images), so it is the one to watch on visually complex inputs, and worth a line in the paper's methods. Gemma3n, Pixtral and LLaMA32Vision remain registered but are outside the panel --- pruning them on the TASK-84 principle is Ben's call, weighed against their being what produced balanced_panel_5x5.
<!-- SECTION:NOTES:END -->
