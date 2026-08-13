---
id: TASK-81
title: Re-embed SMC 2025 captions with Qwen3Embed for cross-era comparison
status: To Do
assignee:
  - sungyeon-hong
created_date: '2026-08-13'
labels:
  - analysis
  - paper
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The surviving SMC 2025 dataset was embedded with Nomic, STSBRoberta and STSBMpnet at 768 dimensions, while current work uses Qwen3Embed at 256 dimensions. Any comparison between the two eras must happen in a single embedding space, since persistence diagrams, cluster assignments and distance statistics are all relative to the embedding model used.

All 72,000 text outputs are stored in the SMC database, so this requires no image generation — only re-embedding stored strings. At roughly 32 ms per item warm, this is well under an hour of GPU time. Dataset inventory and schema notes are in `backlog/docs/smc2025-dataset-inventory.md`.

The purpose is to allow model generation to be treated as an experimental factor: whether newer, more capable models still converge into attractor-like regions, or converge less. If they still converge, the effect is architectural rather than a capability deficit. Note that Moondream is the only model present in both eras and so is the natural anchor leg; whether its weights are unchanged is an open question with Ben Swift.

Not in scope: pooling the two eras into a single clustering, Markov state model or point cloud. Cluster identity would partly encode model generation, confounding any attractor with era. Comparisons should be run per era and then compared.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Text outputs extracted from `panic-tda_SMC2025/sungyeon.sqlite` with run, network, prompt and sequence-number provenance preserved, allowing for the pre-Ash schema (singular table names, uppercase `type` values)
- [ ] #2 All 72,000 captions embedded with Qwen3Embed at 256 dimensions, L2-normalised, and written to parquet alongside their provenance columns
- [ ] #3 Sanity check that the re-embedded trajectories reproduce the qualitative behaviour previously observed under Nomic (for example that the FluxSchnell/Moondream network still shows the most stationary trajectories)
- [ ] #4 Clustering recomputed for the re-embedded data with the same algorithm and hyperparameters used for the current panel, so outlier rates are comparable across eras
- [ ] #5 Written note on whether cross-era comparison is sound, listing the confounds that remain after the embedding space is unified (prompt set, caption length, clustering algorithm)
<!-- AC:END -->
