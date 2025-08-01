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

Currently, only text invocations get embeddings (the relevant EmbeddingModel
subclasses are in @src/panic_tda/embeddings.py).

Add two new embedding models for image embeddings:

- `NomicVision` ( https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5)
- `JinaClipVision` (uses the same model as the existing `JinaClip` class, but we
  want it to be a separate embedding model for the purposes of this project
  https://huggingface.co/jinaai/jina-clip-v2)

Add tests to @tests/test_embeddings.py (using the appropriate Dummy
infrastructure) and ensure that all the same invariants (embedding dim, etc) are
preserved. Ensure all these tests pass before closing this ticket.

Currently there are several places in the codebase (e.g. in
@src/panic_tda/engine.py) where Invocations are filtered to only type == TEXT
ones before embeddings are computed. Keep this as-is for now---we will just
create the image embeddings in ad-hoc script code for now.
