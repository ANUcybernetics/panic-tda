---
id: TASK-84
title: >-
  Pin every supported model to its latest upstream revision and remove legacy
  models
status: To Do
assignee: []
created_date: '2026-09-02 06:30'
labels:
  - models
  - gpu
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ben, 2026-09-02: comparability with earlier runs is already gone after decision-01, so we should be on the latest weights (bug fixes, efficiency tweaks) for every model we actually use, and drop the ones we do not. Only Moondream pins a revision today (2025-06-21; upstream HEAD is 2025-09-23). Everything else floats to whatever was cached at first download, which for reproducibility should become an explicit revision hash in the loader config. A check on 2026-09-02 found every cached snapshot except Moondream already equal to upstream HEAD, so the bump itself is small; the work is pinning, and pruning. The balanced_panel_5x5 lineup is SD35Medium, ZImageTurbo, Flux2Klein, Flux2Dev, GLMImage x Moondream, Qwen25VL, Gemma3n, Pixtral, LLaMA32Vision with Qwen3Embed. Registered but not in that panel: HunyuanImage (T2I, dropped from the panel for cost), Florence2 (I2T, never used in any experiment), text embedders STSBMpnet/STSBRoberta/STSBDistilRoberta/Nomic (SMC era; Nomic last used March 2026), JinaClip, ColNomic, and the image embedders NomicVision/JinaClipVision/ColNomicVision (the embeddings stage supports image embedding but no current experiment uses it). Elixir side: mix cleanup.phi4vision_runs targets a model that no longer exists in the registry. Library versions also matter for the bug-fix argument: the venv has torch 2.10 / transformers 5.5.3 / diffusers 0.37 dev from git; PyPI has torch 2.13 / transformers 5.16 / diffusers 0.40, and the git diffusers dependency should become a released version if 0.40 carries the Flux2/Z-Image/GLM pipelines. Successor models (Qwen3-VL for Qwen25VL, Gemma 4 for Gemma3n, newer Pixtral/Llama vision) are a separate lineup decision, not this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every model in the balanced_panel_5x5 lineup plus Qwen3Embed loads from an explicit pinned revision hash equal to upstream HEAD at the time of the change, recorded in priv/python/panic_models.py
- [ ] #2 Moondream moved from revision 2025-06-21 to the latest upstream revision, with its caption behaviour re-checked (length, terminal punctuation) on a handful of images
- [ ] #3 Decision made and applied for each non-panel model (HunyuanImage, Florence2, STSB*, Nomic, JinaClip*, ColNomic*): keep with a pin, or remove loader, invoke/batch code, Elixir model lists, GPU tests and CLAUDE.md table entries together
- [ ] #4 mix cleanup.phi4vision_runs removed once its cleanup has been run or judged unnecessary
- [ ] #5 torch, transformers, diffusers bumped to current releases in the Snex venv spec (diffusers from a release rather than git if the needed pipelines have shipped), venv rebuilt
- [ ] #6 mise exec -- mix test --include gpu passes for the retained models, and the model-optimisation-log records any per-model timing change
<!-- AC:END -->
