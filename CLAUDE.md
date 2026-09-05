# PANIC-TDA project overview

PANIC-TDA is an Elixir application for computing recursive text-to-image and
image-to-text model trajectories and analysing them using
[topological data analysis](https://en.wikipedia.org/wiki/Topological_data_analysis).
It systematically explores how information flows through networks of generative
AI models by feeding outputs recursively back as inputs, creating "trajectories"
through semantic space.

The project implements a three-stage computational pipeline:

1. **Runs stage**: execute networks of genAI models where outputs become inputs
2. **Embeddings stage**: embed text outputs into high-dimensional semantic space
3. **TDA stage**: compute persistence diagrams via topological data analysis

Clustering is a separate manual step, not part of the pipeline:
`mise exec -- mix cluster.recompute` runs global EVoC clustering over all raw
embeddings for each embedding model, pooled across experiments.

For detailed design rationale, see @DESIGN.md.

## Development

- use `mise exec --` to prefix all mix/elixir commands (erlang/elixir managed by
  mise)
- follow the Ash usage rules below (synced via `usage_rules` hex package)
- run tests with `mise exec -- mix test`
- run GPU smoke tests (all real model combinations) with
  `mise exec -- mix test --include gpu` --- takes ~4h 40m on an RTX 6000 Ada
  (174 tests, mostly bottlenecked by T2I model invocations)
- Python interop is via Snex --- the interpreter maintains persistent state
  across `pyeval` calls. The model registry (loading, invoking, embedding) lives
  in `priv/python/panic_models.py`; Elixir calls into it via short inline
  `pyeval` glue. The Snex venv spec (dependencies, Python version) is declared
  inline in `lib/panic_tda/models/python_interpreter.ex`
- the project uses a separate SQLite database
- tidewave MCP server is available for dev-time BEAM introspection; start it
  with `mise exec -- mix tidewave` (runs on port 4000)

## Running experiments

Experiments are configured via JSON files in `config/` and run with:

```
mise exec -- mix experiment.run config/my_experiment.json
```

The task handles database setup and runs the full three-stage pipeline (runs →
embeddings → TDA).

### Configuration format

```json
{
  "networks": [
    ["SD35Medium", "Moondream3"],
    ["Flux2Klein", "Gemma4"]
  ],
  "prompts": ["a red apple"],
  "embedding_models": ["Qwen3Embed"],
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
- **i2t_max_new_tokens**: uniform generation ceiling for every image-to-text
  model (optional; default is each model's own limit in `panic_models.py`)

### Available models

| Type            | Models                                                                      |
| --------------- | --------------------------------------------------------------------------- |
| text-to-image   | `SD35Medium`, `Flux2Klein`, `Flux2Dev`, `ZImageTurbo`                       |
| image-to-text   | `Moondream3`, `Qwen25VL`, `Qwen3VL`, `Gemma4`, `JoyCaption`                 |
| text embedding  | `Qwen3Embed`                                                                |
| dummy (testing) | `DummyT2I`, `DummyI2T`, `DummyT2I2`, `DummyI2T2`, `DummyText`, `DummyText2` |

Every text-to-image invocation draws its own seed, stores it on the invocation
and hands it to the pipeline, so any step can be regenerated and
within-condition variation is attributable. Batched steps use per-item
generators, so an image depends only on its own seed.

Every model is pinned to an explicit upstream revision (`_REVISIONS` in
`priv/python/panic_models.py`). Only text is embedded --- image embedding was
removed outright, since every second state in an alternating network is already
text and a caption represents the image before it.

### Approximate model run times

Measured on a single NVIDIA RTX 6000 Ada with NF4 quantisation where applicable.
Times include model loading/swapping overhead. Values marked with † are medians
from the `penguin_campfire` experiment (300 batches of 40 per model); embedding
rows are warm-cache timings after cold load; other values are rough one-off
estimates.

| Model             | Single invocation | Batch of 3 | Per-item (batch) |
| ----------------- | ----------------- | ---------- | ---------------- |
| **Text-to-image** |                   |            |                  |
| SD35Medium        | ~9s               | ~9s        | ~6.5s †          |
| ZImageTurbo       | ~8s               | ~18s       | ~6s              |
| Flux2Klein        | ~20s              | ~20s       | ~4.1s †          |
| Flux2Dev          | ~104s             | ~181s §    | ~58s ‡           |
| **Image-to-text** |                   |            |                  |
| Moondream3        | —                 | —          | ~2.4s ¶          |
| Qwen25VL          | ~12s              | ~14s       | ~0.9s †          |
| Qwen3VL           | —                 | —          | ~4.0s ¶          |
| Gemma4            | —                 | —          | ~2.6s ¶          |
| JoyCaption        | —                 | —          | ~2.2s ¶          |

Values marked ‡ are warm per-item timings at `batch=4` on the RTX 6000 Ada,
enabled by TASK-74 (Flux2Dev became truly batch-capable —
`_T2I_BATCH_CAPABLE`/`_T2I_MAX_BATCH` in `priv/python/panic_models.py`). Before
that it ran serially even inside `invoke_t2i_batch`. Benchmark and quality gate:
`mix gpu.bench`; see `backlog/docs/model-optimisation-log.md`.

The Flux2Dev row is measured at its current 12 steps (TASK-83); § is the only
cell scaled rather than measured. ¶ marks seconds per caption at natural length
over a batch of four (TASK-87); those models have no single-invocation or
batch-of-three figure yet. Captioner rows predate decision-01 and now
understate: captions run to natural length, so a Gemma3n batch takes roughly two
to three times longer than it did under the old 128-token ceiling. Measured
seconds per caption at natural length are in
`backlog/docs/caption-length-by-i2t-model.md`.

### Other experiment tasks

- `mise exec -- mix experiment.list` --- list all experiments
- `mise exec -- mix experiment.status <id-prefix>` --- show experiment details
  and progress
- `mise exec -- mix experiment.resume <id-prefix>` --- resume an interrupted
  experiment (picks up where it left off: skips completed runs, computes missing
  embeddings/PDs; does not recluster)
- `mise exec -- mix experiment.export <id-prefix> [--output path.mp4] [--fps 10] [--resolution hd|4k]`
  --- export mosaic video of an experiment
- `mise exec -- mix experiment.export --image <invocation-id> [--output image.png]`
  --- export a single invocation's image
- `mise exec -- mix experiment.export_images <id-prefix> [--output dir] [--limit N]`
  --- dump every image invocation as an AVIF file (organised by
  network/prompt/run) with EXIF/XMP metadata (needs `exiftool`)
- `mise exec -- mix experiment.export_data <id-prefix> [<id-prefix> ...] [--output dir] [--embedding-model NAME] [--embed-prompts]`
  --- dump experiment data (everything except image bytes) to parquet, one file
  per table, for analysis in polars/pandas; `--embed-prompts` also embeds each
  run's initial prompt as a synthetic `sequence_number == -1` row (`t_0`)
- `mise exec -- mix experiment.delete <id-prefix> [--force]` --- delete an
  experiment and all its data
- `bin/long-run config/x.json` --- run an experiment to completion across
  crashes and reboots; the header comment covers the systemd unit
  (`bin/panic-experiment.service`) that keeps it alive

Cells (one network's runs, batched in lockstep) execute in the order the config
lists them, and each cell is embedded and given its persistence diagrams as soon
as its runs finish, so put the slow generators last and analyse the fast cells
while they run.

<!-- usage-rules-start -->
<!-- ash-start -->

## ash usage

_A declarative, extensible framework for building Elixir applications._

[ash usage rules](deps/ash/usage-rules.md)

<!-- ash-end -->
<!-- usage-rules-end -->
