---
id: TASK-84
title: >-
  Pin every supported model to its latest upstream revision and remove legacy
  models
status: To Do
assignee: []
created_date: '2026-09-02 06:30'
updated_date: '2026-09-02 06:57'
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Ben, 2026-09-02 (session decisions before starting):
- config/full_cartesian_36.json and config/seasons_3x3_b.json reference HunyuanImage/Florence2: prune the dead networks from both rather than deleting the files; full_cartesian_36 stops being a full cartesian product, so rename or note it.
- sentence-transformers is also in scope: bump the spec past <6.0 to 6.x alongside torch/transformers/diffusers, and re-validate the embeddings stage on GPU (it backs Qwen3Embed via the NoSortingSentenceTransformer subclass, which exists to keep embedding order stable).

Research done 2026-09-02, no downloads needed: every panel model's local HF cache is already at upstream HEAD. Revisions to pin --- SD35Medium b940f670f0eda2d07fbb75229e779da1ad11eb80; ZImageTurbo f332072aa78be7aecdf3ee76d5c247082da564a6; Flux2Klein 92196c8e11f7b6cf2b7493e037d8c5345c559216; Flux2Dev 26afe3a78bb242c0a8bb181dcc8937bb16e5c66c; GLMImage 2c433cc0cbc293bde2ac8ca9624f279b5d23fcf4; Moondream 6b714b26eea5cbd9f31e4edb2541c170afa935ba (2025-09-23, replacing the 2025-06-21 date tag); Qwen25VL cc594898137f460bfe9f0759e9844b3ce807cfb5; Gemma3n 5e092ebca197cdcd8d8b195040accf22693501bc; Pixtral c2756cbbb9422eba9f6c5c439a214b0392dfc998 (mistral-community/pixtral-12b now redirects to mistral-experimental); LLaMA32Vision 9eb2daaa8597bf192a8b0e73f848f3a102794df5; Qwen3Embed 5cf2132abc99cad020ac570b19d031efec650f2b.

diffusers 0.40.0 (released) ships flux2 (incl. pipeline_flux2_klein), z_image, glm_image and hunyuan_image pipelines, so the git dependency can become a pinned release. Installed now: torch 2.10 / transformers 5.5.3 / diffusers 0.37.0.dev0 / sentence-transformers 5.2.2; PyPI latest: torch 2.13 / transformers 5.16.1 / diffusers 0.40.0 / sentence-transformers 6.0.1.

Do not rebuild the venv until analysis/natural_lengths.py and analysis/step_sweep.py have run --- they use the existing _build/dev venv.
<!-- SECTION:NOTES:END -->
