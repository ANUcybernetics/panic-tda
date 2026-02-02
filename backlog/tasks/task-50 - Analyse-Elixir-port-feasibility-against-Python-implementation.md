---
id: task-50
title: Analyse Elixir port feasibility against Python implementation
status: In Progress
assignee: []
created_date: '2026-02-02 07:07'
updated_date: '2026-02-02 09:07'
labels:
  - elixir
  - architecture
  - investigation
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Compare the ELIXIR_PORT_INVESTIGATION.md proposal with the current Python codebase to identify which components are realistically portable and produce revised effort estimates. The investigation document makes optimistic claims about ONNX model availability and Ortex maturity that need verification against actual implementation requirements.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Document current Python model execution layer (diffusers, transformers, sentence-transformers) with specific model architectures used
- [ ] #2 Verify ONNX export feasibility for each model (Z-Image-Turbo, Florence-2, SigLIP, Moondream)
- [x] #3 Compare Python SQLModel schema with proposed Ash resources and identify mapping gaps
- [x] #4 Map current Ray-based orchestration to proposed Reactor patterns with complexity assessment
- [ ] #5 Assess Rustler NIF requirements for giotto-ph TDA operations
- [ ] #6 Produce revised effort estimates with risk factors for each component
- [ ] #7 Recommend hybrid vs pure-Elixir approach with justification
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
## Implementation Progress

### Completed - Ash Resources (AC #3)
- All 7 resources implemented: Experiment, Run, Invocation, Embedding, PersistenceDiagram, ClusteringResult, EmbeddingCluster
- Custom types: Vector (Nx tensor ↔ binary), Image, PersistenceDiagramData, InvocationType enum
- SQLite with AshSqlite, UUIDv7 primary keys
- Direct 1:1 mapping with Python SQLModel schema

### Completed - Orchestration (AC #4)
- Replaced Ray with simple Elixir sequential execution (Enum.each)
- Engine.perform_experiment/1 orchestrates full pipeline
- RunExecutor executes model network trajectories
- EmbeddingsStage computes embeddings per model type
- Reactor deemed overkill for current use case

### Completed - Python Interop via Snex
- GenAI models: DummyT2I, DummyT2I2, DummyI2T, DummyI2T2
- Embedding models: DummyText, DummyText2, DummyVision, DummyVision2
- Base64 encoding for binary data transfer
- Deterministic outputs matching Python implementation

### Test Coverage
All 14 tests passing:
- 4 Python interop tests
- 6 end-to-end pipeline tests
- 4 resource CRUD tests

### Location
elixir_port/panic_tda/
<!-- SECTION:NOTES:END -->
