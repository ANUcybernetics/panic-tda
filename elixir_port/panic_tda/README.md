# PANIC-TDA Elixir Port

Elixir implementation of PANIC-TDA using Ash Framework for data modelling and
SQLite for persistence. Real ML model execution is handled via Snex (Python
interop), keeping models loaded in a persistent Python environment.

## Current status

The full three-stage pipeline is operational:

1. **Runs stage** --- execute networks of GenAI models (T2I and I2T) where
   outputs feed back as inputs
2. **Embeddings stage** --- embed text and image outputs into 768-dimensional
   semantic space
3. **Persistence diagrams stage** --- apply topological data analysis via
   giotto-ph

### Ash resources

All 7 resources from the Python schema are implemented:

- `Experiment` --- experiment configurations
- `Run` --- individual trajectory runs
- `Invocation` --- model invocations (text or image output)
- `Embedding` --- vector embeddings stored as Nx tensors
- `PersistenceDiagram` --- TDA results
- `ClusteringResult` --- clustering metadata
- `EmbeddingCluster` --- cluster assignments

### Available models

**Text-to-image**: SDXLTurbo, FluxDev, FluxSchnell (plus DummyT2I, DummyT2I2
for testing)

**Image-to-text**: Moondream, BLIP2 (plus DummyI2T, DummyI2T2 for testing)

**Text embeddings**: STSBMpnet, STSBRoberta, STSBDistilRoberta, Nomic, JinaClip
(plus DummyText, DummyText2 for testing)

**Image embeddings**: NomicVision, JinaClipVision (plus DummyVision, DummyVision2
for testing)

### Not yet implemented

- CLI interface
- Clustering stage

## Prerequisites

- Elixir/Erlang (managed via mise --- see `mise.toml`)
- Python 3.12 (managed via mise/Snex)
- NVIDIA GPU with CUDA for real models (dummy models work without GPU)
- `libvips` system library (`sudo apt install libvips-dev` on Debian/Ubuntu)

## Setup

```bash
# Install dependencies
mise exec -- mix deps.get

# Create database and run migrations
mise exec -- mix ecto.create
mise exec -- mix ash_sqlite.migrate

# Run tests (dummy models only, no GPU required)
mise exec -- mix test

# Run all tests including real GPU models
mise exec -- mix test --include gpu
```

## Running an experiment

### With real models (requires GPU)

```elixir
# Start an IEx session
mise exec -- iex -S mix

# Create and run an experiment
{:ok, experiment} =
  PanicTda.Experiment
  |> Ash.Changeset.for_create(:create, %{
    networks: [["SDXLTurbo", "Moondream"]],
    seeds: [42, 123],
    prompts: ["A beautiful sunset over the ocean"],
    embedding_models: ["STSBMpnet"],
    max_length: 10
  })
  |> Ash.create()

# Execute the full pipeline (runs, embeddings, persistence diagrams)
{:ok, completed} = PanicTda.Engine.perform_experiment(experiment.id)

# Inspect the results
completed = Ash.load!(completed, runs: [:invocations])
run = hd(completed.runs)
Enum.each(run.invocations, fn inv ->
  IO.puts("#{inv.model} (#{inv.type}): #{inv.output_text || "image"}")
end)

# Check embeddings
require Ash.Query
embeddings = PanicTda.Embedding |> Ash.Query.filter(embedding_model == "STSBMpnet") |> Ash.read!()
IO.puts("#{length(embeddings)} embeddings computed")

# Check persistence diagrams
pds = Ash.read!(PanicTda.PersistenceDiagram)
pd = hd(pds)
IO.inspect(pd.diagram_data, label: "TDA result")
```

The first run will be slower as HuggingFace model weights are downloaded and
cached. Subsequent runs reuse the cached weights, and models stay loaded in the
Python environment for the duration of the interpreter's lifetime.

### With dummy models (no GPU required)

```elixir
{:ok, experiment} =
  PanicTda.Experiment
  |> Ash.Changeset.for_create(:create, %{
    networks: [["DummyT2I", "DummyI2T"]],
    seeds: [42],
    prompts: ["A test prompt"],
    embedding_models: ["DummyText"],
    max_length: 4
  })
  |> Ash.create()

{:ok, completed} = PanicTda.Engine.perform_experiment(experiment.id)
```

## Architecture

Python interop is handled by [Snex](https://hex.pm/packages/snex), which
maintains a persistent Python interpreter process. The `PythonBridge` module
manages one-time setup (loading torch, defining helper classes) and lazy model
loading --- each model is loaded into GPU memory on first use and cached in a
`_models` dict for subsequent invocations.

## Database

Uses SQLite via AshSqlite. The database file is stored at
`priv/panic_tda_dev.db` by default.

The schema matches the Python SQLModel implementation:

- UUIDv7 primary keys
- JSON arrays for networks, seeds, prompts
- binary storage for vectors and images
- timestamps with microsecond precision
