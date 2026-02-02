# PANIC-TDA Elixir Port

Elixir implementation of PANIC-TDA using Ash Framework for data modelling and SQLite for persistence.

## Current status

This is an early-stage port implementing the data layer:

- **Ash Resources**: All 7 resources from the Python schema are implemented
  - `Experiment` - experiment configurations
  - `Run` - individual trajectory runs
  - `Invocation` - model invocations (text or image output)
  - `Embedding` - vector embeddings stored as Nx tensors
  - `PersistenceDiagram` - TDA results
  - `ClusteringResult` - clustering metadata
  - `EmbeddingCluster` - cluster assignments

- **Custom Types**:
  - `Vector` - stores float32 vectors as binary, compatible with Python numpy format
  - `Image` - stores images as binary (WebP)
  - `PersistenceDiagramData` - stores TDA results as compressed Erlang terms
  - `InvocationType` - enum for :text/:image

## Not yet implemented

- Model execution (GenAI models, embedding models)
- Reactor orchestration for the three-stage pipeline
- TDA computation (would need Rustler NIF or Python interop)
- CLI interface

## Setup

```bash
# Install dependencies
mix deps.get

# Create database and run migrations
mix ecto.create
mix ash_sqlite.migrate

# Run tests
mix test
```

## Usage

```elixir
# Create an experiment
{:ok, experiment} =
  PanicTda.Experiment
  |> Ash.Changeset.for_create(:create, %{
    networks: [["ZImage", "Florence2"]],
    seeds: [42, 123],
    prompts: ["A beautiful sunset"],
    embedding_models: ["Nomic"],
    max_length: 10
  })
  |> Ash.create()

# Create a run
{:ok, run} =
  PanicTda.Run
  |> Ash.Changeset.for_create(:create, %{
    network: ["ZImage", "Florence2"],
    seed: 42,
    max_length: 10,
    initial_prompt: "A beautiful sunset",
    experiment_id: experiment.id
  })
  |> Ash.create()
```

## Database

Uses SQLite via AshSqlite. The database file is stored at `priv/panic_tda_dev.db` by default.

The schema matches the Python SQLModel implementation:
- UUIDv7 primary keys
- JSON arrays for networks, seeds, prompts
- Binary storage for vectors and images
- Timestamps with microsecond precision
