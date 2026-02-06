---
id: TASK-53
title: Wire up real Python model execution in Elixir port via Snex
status: To Do
assignee: []
created_date: '2026-02-06 02:14'
labels:
  - elixir
  - python-interop
  - models
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Connect the Elixir port to real Python ML models (genAI and embedding) through the existing Snex interop layer. Snex maintains persistent Python state across pyeval calls, so models can be loaded once into the interpreter environment and reused for all subsequent invocations. The existing Python model classes in src/panic_tda/genai_models.py and src/panic_tda/embeddings.py contain all the invoke logic needed --- the bridge module strips Ray decorators and exposes a clean load/invoke interface callable from Elixir.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Snex pyproject deps include torch, diffusers, transformers, sentence-transformers, and accelerate
- [ ] #2 Python bridge module loads real models once into persistent Snex env (stripping @ray.remote from existing classes)
- [ ] #3 GenAI module dispatches to real T2I models (at minimum SDXLTurbo) and I2T models (at minimum Moondream)
- [ ] #4 Embeddings module dispatches to at least one real text embedding model and one real vision embedding model
- [ ] #5 Base64 image serialisation round-trips correctly with real model outputs (not just 256x256 solid-colour dummies)
- [ ] #6 At least one end-to-end experiment completes with real models: runs stage, embeddings stage, and persistence diagram stage
- [ ] #7 Existing dummy model tests continue to pass unchanged
<!-- AC:END -->
