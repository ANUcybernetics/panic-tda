---
id: task-17
title: add image embeddings back into the workflow
status: To Do
assignee: []
created_date: "2025-07-23"
labels: []
dependencies: []
---

## Description

Currently, only text invocations get embeddings. However, some of the
EmbeddingModel subclasses in @src/panic_tda/embeddings.py can create embeddings
for images. This task requires:

- adding at least one more image EmbeddingModel
- (at least in an ad-hoc way) create embeddings for all existing image
  Invocations in the db
- udpate the clustering code and `add_semantic_drift_cosine` (and the
  `_euclidean` variant) to handle image OR text embeddings (although it still
  doesn't really work to compare image to text embeddings and vice versa)
