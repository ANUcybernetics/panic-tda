# PANIC-TDA project overview

PANIC-TDA is an Elixir application for computing recursive text-to-image and
image-to-text model trajectories and analysing them using
[topological data analysis](https://en.wikipedia.org/wiki/Topological_data_analysis).
It systematically explores how information flows through networks of generative
AI models by feeding outputs recursively back as inputs, creating "trajectories"
through semantic space.

The project implements a four-stage computational pipeline:

1. **Runs stage**: execute networks of genAI models where outputs become inputs
2. **Embeddings stage**: embed text outputs into high-dimensional semantic space
3. **TDA stage**: compute persistence diagrams via topological data analysis
4. **Clustering stage**: cluster persistence diagrams via HDBSCAN

For detailed design rationale, see @DESIGN.md.

## Development

- use `mise exec --` to prefix all mix/elixir commands (erlang/elixir managed
  by mise)
- follow the Ash usage rules below (synced via `usage_rules` hex package)
- run tests with `mise exec -- mix test`
- run GPU smoke tests (all real model combinations) with
  `mise exec -- mix test --include gpu`
- Python interop is via Snex --- the interpreter maintains persistent state
  across `pyeval` calls. The model registry (loading, invoking, embedding)
  lives in `priv/python/panic_models.py`; Elixir calls into it via short
  inline `pyeval` glue. The Snex venv spec (dependencies, Python version) is
  declared inline in `lib/panic_tda/models/python_interpreter.ex`
- the project uses a separate SQLite database
- tidewave MCP server is available for dev-time BEAM introspection; start it
  with `mise exec -- mix tidewave` (runs on port 4000)

## Running experiments

Experiments are configured via JSON files in `config/` and run with:

```
mise exec -- mix experiment.run config/my_experiment.json
```

The task handles database setup and runs the full four-stage pipeline
(runs → embeddings → TDA → clustering).

### Configuration format

```json
{
  "networks": [["SD35Medium", "Moondream"], ["FluxSchnell", "InstructBLIP"]],
  "prompts": ["a red apple"],
  "embedding_models": ["Nomic"],
  "max_length": 100,
  "num_runs": 1
}
```

- **networks**: list of networks, where each network is a list of models that
  cycle (T2I → I2T → T2I → ...); runs sharing the same network are batched in
  lockstep, different network groups run sequentially
- **prompts**: initial text inputs; each prompt creates `num_runs` runs per
  network
- **embedding_models**: models used in the embeddings stage
- **max_length**: number of model invocations per run
- **num_runs**: number of runs per prompt per network (optional, default 1)

### Available models

| Type | Models |
|---|---|
| text-to-image | `SD35Medium`, `Flux2Klein`, `Flux2Dev`, `ZImageTurbo`, `HunyuanImage`, `GLMImage` |
| image-to-text | `Moondream`, `Qwen25VL`, `Gemma3n`, `Pixtral`, `LLaMA32Vision`, `Florence2` |
| text embedding | `STSBMpnet`, `STSBRoberta`, `STSBDistilRoberta`, `Nomic`, `JinaClip`, `Qwen3Embed`, `ColNomic` |
| image embedding | `NomicVision`, `JinaClipVision`, `ColNomicVision` |
| dummy (testing) | `DummyT2I`, `DummyI2T`, `DummyT2I2`, `DummyI2T2`, `DummyText`, `DummyText2`, `DummyVision`, `DummyVision2` |

### Approximate model run times

Measured on a single NVIDIA RTX 6000 Ada with NF4 quantisation where
applicable. Times include model loading/swapping overhead. Values marked with †
are medians from the `penguin_campfire` experiment (300 batches of 40 per
model); embedding rows are warm-cache timings after cold load; other values
are rough one-off estimates.

| Model | Single invocation | Batch of 3 | Per-item (batch) |
|---|---|---|---|
| **Text-to-image** | | | |
| SD35Medium | ~9s | ~9s | ~6.5s † |
| ZImageTurbo | ~8s | ~18s | ~6s |
| Flux2Klein | ~20s | ~20s | ~4.1s † |
| Flux2Dev | ~100s | ~226s | ~75s |
| HunyuanImage | ~124s | ~326s | ~109s |
| GLMImage | ~44s | ~85s | ~76s † |
| **Image-to-text** | | | |
| Moondream | ~4s | ~10s | ~0.3s † |
| Qwen25VL | ~12s | ~14s | ~0.9s † |
| Gemma3n | ~16s | ~18s | ~6s |
| Pixtral | ~19s | ~24s | ~2.6s † |
| LLaMA32Vision | ~17s | ~23s | ~8s |
| Florence2 | TBD | TBD | TBD |
| **Embedding** | | | |
| ColNomic (text) | ~32ms | ~56ms | ~19ms |
| ColNomicVision (image) | ~200ms | ~650ms | ~220ms |

ColNomic cold load is ~14s (text) / ~8s (image) on first use; the embedding
rows above exclude that one-off cost.

### Other experiment tasks

- `mise exec -- mix experiment.list` --- list all experiments
- `mise exec -- mix experiment.status <id-prefix>` --- show experiment details and progress
- `mise exec -- mix experiment.resume <id-prefix>` --- resume an interrupted
  experiment (picks up where it left off: skips completed runs, computes missing
  embeddings/PDs, reclusters)
- `mise exec -- mix experiment.export <id-prefix> [--output path.mp4] [--fps 10] [--resolution hd|4k]`
  --- export mosaic video of an experiment
- `mise exec -- mix experiment.export --image <invocation-id> [--output image.png]`
  --- export a single invocation's image
- `mise exec -- mix experiment.delete <id-prefix> [--force]` --- delete an
  experiment and all its data

<!-- usage-rules-start -->
<!-- ash-start -->
## ash usage
_A declarative, extensible framework for building Elixir applications._

[ash usage rules](deps/ash/usage-rules.md)
<!-- ash-end -->
<!-- usage-rules-end -->
