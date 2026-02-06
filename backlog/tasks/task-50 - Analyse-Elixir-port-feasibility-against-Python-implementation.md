---
id: task-50
title: Analyse Elixir port feasibility against Python implementation
status: Done
assignee: []
created_date: '2026-02-02 07:07'
updated_date: '2026-02-06'
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
- [x] #1 Document current Python model execution layer (diffusers, transformers, sentence-transformers) with specific model architectures used
- [x] #2 Verify ONNX export feasibility for each model (Z-Image-Turbo, Florence-2, SigLIP, Moondream)
- [x] #3 Compare Python SQLModel schema with proposed Ash resources and identify mapping gaps
- [x] #4 Map current Ray-based orchestration to proposed Reactor patterns with complexity assessment
- [x] #5 Assess Rustler NIF requirements for giotto-ph TDA operations
- [x] #6 Produce revised effort estimates with risk factors for each component
- [x] #7 Recommend hybrid vs pure-Elixir approach with justification
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

### Python Model Execution Layer (AC #1)

#### Production GenAI Models

| Model | ID | Type | Library | GPU |
|-------|----|------|---------|-----|
| FluxDev | `black-forest-labs/FLUX.1-dev` | T2I | `diffusers.FluxPipeline` | 0.7 |
| FluxSchnell | `black-forest-labs/FLUX.1-schnell` | T2I | `diffusers.FluxPipeline` | 0.7 |
| SDXLTurbo | `stabilityai/sdxl-turbo` | T2I | `diffusers.AutoPipelineForText2Image` | 0.15 |
| Moondream | `vikhyatk/moondream2` (rev 2025-01-09) | I2T | `transformers.AutoModelForCausalLM` | 0.1 |
| BLIP2 | `Salesforce/blip2-opt-2.7b` | I2T | `transformers.Blip2ForConditionalGeneration` | 0.15 |

- All T2I models output 256×256 images, use `torch.compile` optimisation
- FLUX models use `torch.bfloat16`; SDXL/BLIP2 use `torch.float16`
- Moondream uses `trust_remote_code=True`, custom caption method
- BLIP2 uses beam search (5 beams) + nucleus sampling (top-p 0.9)

#### Production Embedding Models

| Model | ID | Type | Library | GPU |
|-------|----|------|---------|-----|
| Nomic | `nomic-ai/nomic-embed-text-v2-moe` | Text | `sentence_transformers` (custom) | 0.02 |
| JinaClip | `jinaai/jina-clip-v2` | Text | `transformers.AutoModel` | 0.03 |
| STSBMpnet | `sentence-transformers/stsb-mpnet-base-v2` | Text | `sentence_transformers` | 0.01 |
| STSBRoberta | `sentence-transformers/stsb-roberta-base-v2` | Text | `sentence_transformers` | 0.01 |
| STSBDistilRoberta | `sentence-transformers/stsb-distilroberta-base-v2` | Text | `sentence_transformers` | 0.01 |
| NomicVision | `nomic-ai/nomic-embed-vision-v1.5` | Vision | `transformers.AutoModel` | 0.02 |
| JinaClipVision | `jinaai/jina-clip-v2` | Vision | `transformers.AutoModel` | 0.03 |

- All embeddings normalised to 768 dimensions
- Nomic text uses MoE architecture with `trust_remote_code=True`
- JinaClip is a dual-encoder (shared model for text + vision branches)
- 4 dummy models for each category (T2I, I2T, text embed, vision embed) used in tests

#### Execution Infrastructure
- **Ray** for distributed execution with fractional GPU allocation
- Actor pool pattern: models loaded once, reused across runs
- Batch processing for embeddings (batch_size=32)
- CUDA cache management between pipeline stages
- `torch.compile()` applied to UNet/model forward passes

### ONNX Export Feasibility (AC #2)

#### Ortex Baseline
- Ortex v0.1.10 bundles ONNX Runtime 1.19 via `ort` crate v2.0.0-rc.8
- Approximately one year behind current ORT (1.23)
- ORT 1.19 supports opsets 7--21, IR version up to 10
- CUDA, TensorRT, CoreML, DirectML execution providers available

#### Per-Model Assessment

| Model | Pre-exported ONNX | Ortex-feasible | Key blocker |
|-------|-------------------|----------------|-------------|
| FLUX.1-dev | Yes (official, BF16/FP8/FP4) | **No** | IR v11, bfloat16 ops, 4-model orchestration, 12B params |
| FLUX.1-schnell | Yes (official) | **No** | Same as FLUX.1-dev |
| SDXL-Turbo | Yes (onnxruntime team) | **Difficult** | 3-model orchestration, scheduler reimplementation |
| Moondream2 | Yes (Xenova, onnx branch) | **Difficult** | Autoregressive generation, KV cache, custom arch |
| BLIP2-OPT-2.7b | **No** | **No** | No working export path, 3-component arch |
| nomic-embed-text-v2-moe | **No** | **Difficult** | MoE dynamic routing breaks ONNX tracing |
| jina-clip-v2 | Yes (official, >2GB) | **Yes, with caveats** | Dual-input requirement, external data format |
| stsb-mpnet-base-v2 | Yes (multiple variants + quantised) | **Yes** | Tokenisation in Elixir |
| stsb-roberta-base-v2 | Yes (multiple variants + quantised) | **Yes** | Tokenisation in Elixir |
| stsb-distilroberta-base-v2 | Yes (multiple variants) | **Yes** | Tokenisation in Elixir |
| nomic-embed-vision-v1.5 | Yes (FP16, INT8, Q4 variants) | **Yes** | Image preprocessing in Elixir |
| jina-clip-v2 (vision) | Yes (same as text) | **Yes, with caveats** | Dual-input requirement |

#### Key Findings
- **5 of 12 models are strong ONNX/Ortex candidates** --- all are embedding models (3× STSB, nomic-vision, jina-clip)
- **All generative models are problematic** --- multi-model pipelines, iterative inference, autoregressive generation
- BLIP2 is the worst case: no working ONNX export path exists at all
- FLUX official ONNX exports require ORT 1.22+ (IR v11), beyond Ortex's current ORT 1.19
- Diffusion pipeline orchestration (scheduler, denoising loop) proven feasible in C# but would require significant Elixir implementation
- Tokenisation is the main cross-cutting gap --- Ortex has no tokeniser support; would need Rust NIF wrapping HuggingFace `tokenizers` crate

### Revised Effort Estimates (AC #6)

| Component | Approach | Effort | Risk | Notes |
|-----------|----------|--------|------|-------|
| **Ash resources + schema** | Pure Elixir | Done | Low | 1:1 mapping complete, tested |
| **Orchestration (pipeline)** | Pure Elixir | Done | Low | Sequential execution, no Ray needed |
| **TDA (ripser + entropy)** | Python interop (Snex) | Done | Low | Already implemented |
| **T2I models (FLUX, SDXL)** | Python interop | 1--2 weeks | Low | Proven pattern; ONNX not viable |
| **I2T models (Moondream, BLIP2)** | Python interop | 1--2 weeks | Low | Proven pattern; ONNX not viable |
| **Text embeddings (STSB ×3)** | ONNX via Ortex | 2--3 weeks | Medium | Pre-exported ONNX available; need tokeniser NIF |
| **Text embeddings (Nomic MoE)** | Python interop | 3--5 days | Low | MoE breaks ONNX; fallback to proven pattern |
| **Text embeddings (JinaClip)** | ONNX via Ortex | 1--2 weeks | Medium | Pre-exported but >2GB, dual-input quirks |
| **Vision embeddings (Nomic)** | ONNX via Ortex | 1 week | Low | Pre-exported, straightforward preprocessing |
| **Vision embeddings (JinaClip)** | ONNX via Ortex | 1 week | Medium | Same dual-input caveats as text |
| **Tokeniser NIF** | Rustler + HF tokenizers crate | 2--3 weeks | Medium | Required for any ONNX text embedding; well-documented Rust crate |
| **Image preprocessing** | Pure Elixir (Nx/Image) | 3--5 days | Low | Resize + normalise, standard ops |

#### Risk Factors
1. **Ortex version lag**: ORT 1.19 is ~1 year behind; some models may need newer opsets
2. **Tokeniser gap**: No Elixir tokeniser library; Rustler NIF is the critical path for ONNX embeddings
3. **Python version pinning**: giotto-ph lacks Python 3.14 wheels; Snex must pin 3.12
4. **Dual maintenance**: hybrid approach means maintaining both Elixir and Python codepaths
5. **GPU memory**: FLUX models require ~12GB+ even in reduced precision

### Recommendation: Hybrid Approach (AC #7)

#### Recommended architecture: Elixir orchestration with Python model execution

The analysis strongly favours a **hybrid approach** where Elixir owns orchestration, data management, and the web layer, while Python handles all ML model execution. A pure-Elixir approach is not viable.

#### Justification

**Why not pure Elixir:**
- 0 of 5 generative models can practically run via ONNX/Ortex
- BLIP2 has no ONNX export path at all
- FLUX requires ORT 1.22+, Ortex ships ORT 1.19
- Diffusion pipeline orchestration (scheduler, denoising loop) would need full reimplementation
- Autoregressive text generation (Moondream, BLIP2) requires KV cache management --- complex to implement from scratch
- MoE dynamic routing (Nomic text) breaks ONNX tracing

**Why not pure Python (status quo):**
- Elixir provides superior concurrency, fault tolerance, and web serving (Phoenix LiveView)
- Ash resources give declarative data management with less boilerplate than SQLModel
- OTP supervision trees handle long-running model processes more robustly than Ray
- The Snex Python interop pattern is already proven across all model types

**Why hybrid works:**
- Python interop via Snex is already implemented and tested for dummy models + TDA
- The serialisation boundary (Base64 for images, JSON for text/embeddings) has acceptable overhead --- point clouds and embeddings are small relative to model execution time
- Elixir orchestration is simpler than Ray (sequential `Enum.each` vs distributed actors)
- Embedding models *could* migrate to ONNX/Ortex later (5 of 7 are feasible), but this is an optimisation, not a requirement
- Python remains a hard dependency regardless (TDA, generative models), so eliminating it for embeddings only adds complexity without removing the dependency

#### Recommended phased approach

**Phase 1 (current):** Elixir orchestration + Python interop for everything (production dummy models → real models is a config change in Python)

**Phase 2 (optional):** Migrate STSB text embeddings + nomic-vision to ONNX/Ortex if Python interop latency becomes a bottleneck. Requires tokeniser NIF first.

**Phase 3 (speculative):** If Ortex upgrades to ORT 1.22+ and a broader tokeniser ecosystem emerges in Elixir, reassess SDXL-Turbo and JinaClip migration. Generative models (FLUX, Moondream, BLIP2) should remain in Python indefinitely.
<!-- SECTION:NOTES:END -->
