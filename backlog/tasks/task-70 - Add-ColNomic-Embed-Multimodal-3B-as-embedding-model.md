---
id: TASK-70
title: Add ColNomic Embed Multimodal 3B as embedding model
status: Done
assignee: []
created_date: '2026-04-09 12:01'
labels:
  - embeddings
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add colnomic-embed-multimodal-3b (Nomic AI, April 2025) as a new embedding model. ColNomic is a joint text/image embedding model built on Qwen2.5-VL, current SOTA on ViDoRe-v2 for multimodal retrieval. Unlike the existing single-vector models, ColNomic produces multi-vector embeddings (one per token/patch) with late interaction scoring. For the initial integration, mean-pool the multi-vector output to a single dense vector so it fits the existing TDA pipeline. This keeps the change scoped to adding a new model variant without requiring changes to the embedding storage or downstream stages. A follow-up task can explore using the full multi-vector representation if the pooled version loses important signal.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ColNomic model variant added to embedding model enum in embeddings.ex with both text and image modes
- [x] #2 Python embedding function implemented via Snex that loads colnomic-embed-multimodal-3b and produces mean-pooled single-vector output
- [x] #3 Embeddings can be generated for both text and image invocations using the new model
- [x] #4 Embedding dimensions and normalisation are consistent with existing models (float32 binary storage)
- [x] #5 Existing tests pass and new model is covered by at least one integration test
<!-- AC:END -->
