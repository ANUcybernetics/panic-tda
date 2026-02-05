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
- [x] #5 Assess Rustler NIF requirements for giotto-ph TDA operations
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
- TDA: giotto-ph ripser_parallel + persim persistent_entropy
- Base64 encoding for binary data transfer
- Deterministic outputs matching Python implementation

### Test Coverage
All 15 tests passing:
- 4 Python interop tests
- 7 end-to-end pipeline tests (including TDA/persistence diagrams)
- 4 resource CRUD tests

### Location
elixir_port/panic_tda/

### TDA Assessment (AC #5)

#### Current Python TDA Stack
- **giotto-ph** (`gph.ripser_parallel`): Persistent homology via optimised Ripser algorithm
- **persim** (`persistent_entropy`): Shannon entropy over persistence lengths
- **Custom Wasserstein**: Sliced approximation in pure numpy

#### TDA Operations Breakdown
1. **Persistent homology** (complex): Vietoris-Rips filtration, cohomology computation, clearing/apparent pairs optimisations
2. **Persistent entropy** (simple): `E = -sum(p * log(p))` over normalised bar lengths
3. **Wasserstein distance** (simple): L1 distance between sorted persistence vectors

#### Rustler NIF Options Assessed

| Option | Complexity | Effort | Risk |
|--------|------------|--------|------|
| A: Wrap giotto-ph (C++) via Rust | High | 3-5 weeks | Medium - C++/Rust/BEAM boundary complexity |
| B: Rust `tda` crate | Medium | 2-3 weeks | High - immature crate, unclear correctness |
| C: Python interop via Snex | Low | 1-2 days | Low - already proven pattern |
| D: Pure Elixir/Nx ripser | Very High | 4-8 weeks | Very High - algorithmic complexity |

#### Key Findings
- No Elixir/Erlang TDA packages exist on hex.pm
- Rust TDA ecosystem is immature (only 2 repos tagged `persistent-homology` on GitHub)
- giotto-ph uses pybind11, not a C API --- wrapping requires Rust FFI to C++ (complex)
- Persistent entropy and Wasserstein could be pure Elixir/Nx, but pointless without ripser

#### Recommendation
**Use Python interop (Option C)** for TDA operations. The Snex pattern is already established for model execution. Point clouds are numpy arrays (same as embeddings), so serialisation overhead is acceptable. A Rustler NIF would add significant complexity for minimal benefit given Python is already required for genAI models.

#### Implementation Status: COMPLETE
- `PanicTda.Models.Tda.compute_persistence_diagram/4` calls giotto-ph via Snex
- `PanicTda.Engine.PdStage` orchestrates TDA computation per embedding model
- Python 3.12 pinned in Snex (giotto-ph lacks 3.14 wheels)
- Returns `%{dgms: [[birth, death], ...], entropy: [...], num_edges: int}`
<!-- SECTION:NOTES:END -->
