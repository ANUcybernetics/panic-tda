# PANIC-TDA Elixir Port Investigation

## Executive Summary

This document analyzes porting PANIC-TDA from Python to Elixir using:
- **Ash Framework** for data modelling & persistence
- **Ortex (ONNX Runtime)** for model inference
- **SQLite** via `AshSqlite` for storage
- **Reactor** for distributed, reliable computation orchestration

The port is **feasible** but requires significant architectural decisions around model execution strategy.

---

## 1. Architecture Mapping

### 1.1 Current Python Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI (Typer)                              │
├─────────────────────────────────────────────────────────────────┤
│                      Engine (Ray Distributed)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐     │
│  │ Runs Stage   │→ │ Embeddings   │→ │ Persistence Diagram│     │
│  │ (GenAI)      │  │ Stage        │  │ Stage (TDA)        │     │
│  └──────────────┘  └──────────────┘  └────────────────────┘     │
├─────────────────────────────────────────────────────────────────┤
│  Models Layer                                                    │
│  ┌─────────────┐  ┌────────────────┐  ┌──────────────────┐     │
│  │ GenAI Models│  │ Embedding      │  │ TDA (giotto-ph)  │     │
│  │ (diffusers, │  │ Models         │  │                  │     │
│  │ transformers)│  │(sentence-trans)│  │                  │     │
│  └─────────────┘  └────────────────┘  └──────────────────┘     │
├─────────────────────────────────────────────────────────────────┤
│                    SQLModel + SQLite                            │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Proposed Elixir Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLI (escript / Burrito)                       │
├─────────────────────────────────────────────────────────────────┤
│                     Reactor Orchestration                        │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐     │
│  │ Runs Reactor │→ │ Embeddings   │→ │ PD Reactor         │     │
│  │              │  │ Reactor      │  │                    │     │
│  └──────────────┘  └──────────────┘  └────────────────────┘     │
├─────────────────────────────────────────────────────────────────┤
│  Model Execution Layer                                           │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Option A: Ortex (ONNX)    │ Option B: Python Ports/NIFs     ││
│  │ - Local ONNX models       │ - Keep PyTorch via Pythonx      ││
│  │ - Limited model support   │ - Full HuggingFace access       ││
│  └─────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────┤
│  Ash Framework + AshSqlite                                       │
│  ┌─────────────┐  ┌────────────────┐  ┌──────────────────┐     │
│  │ Experiment  │  │ Run            │  │ Invocation       │     │
│  │ Resource    │  │ Resource       │  │ Resource         │     │
│  └─────────────┘  └────────────────┘  └──────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Model Mapping (SQLModel → Ash)

### 2.1 Ash Resources

The current schema maps cleanly to Ash resources:

```elixir
# lib/panic_tda/resources/experiment.ex
defmodule PanicTda.Experiment do
  use Ash.Resource,
    domain: PanicTda,
    data_layer: AshSqlite.DataLayer

  sqlite do
    table "experiments"
    repo PanicTda.Repo
  end

  attributes do
    uuid_v7_primary_key :id

    attribute :networks, {:array, {:array, :string}}, allow_nil?: false
    attribute :seeds, {:array, :integer}, allow_nil?: false
    attribute :prompts, {:array, :string}, allow_nil?: false
    attribute :embedding_models, {:array, :string}, allow_nil?: false
    attribute :max_length, :integer, allow_nil?: false

    attribute :started_at, :utc_datetime_usec
    attribute :completed_at, :utc_datetime_usec

    create_timestamp :inserted_at
    update_timestamp :updated_at
  end

  relationships do
    has_many :runs, PanicTda.Run do
      destination_attribute :experiment_id
    end
  end

  actions do
    defaults [:read, :destroy]

    create :create do
      accept [:networks, :seeds, :prompts, :embedding_models, :max_length]
    end

    update :start do
      change set_attribute(:started_at, &DateTime.utc_now/0)
    end

    update :complete do
      change set_attribute(:completed_at, &DateTime.utc_now/0)
    end
  end

  validations do
    validate {PanicTda.Validations.NetworksStartWithT2I, []}
  end
end
```

```elixir
# lib/panic_tda/resources/run.ex
defmodule PanicTda.Run do
  use Ash.Resource,
    domain: PanicTda,
    data_layer: AshSqlite.DataLayer

  sqlite do
    table "runs"
    repo PanicTda.Repo
  end

  attributes do
    uuid_v7_primary_key :id

    attribute :network, {:array, :string}, allow_nil?: false
    attribute :seed, :integer, allow_nil?: false
    attribute :max_length, :integer, allow_nil?: false
    attribute :initial_prompt, :string, allow_nil?: false
  end

  relationships do
    belongs_to :experiment, PanicTda.Experiment, allow_nil?: false

    has_many :invocations, PanicTda.Invocation do
      destination_attribute :run_id
      sort sequence_number: :asc
    end

    has_many :persistence_diagrams, PanicTda.PersistenceDiagram
  end

  calculations do
    calculate :stop_reason, :atom, PanicTda.Calculations.StopReason
    calculate :is_complete, :boolean, expr(not is_nil(completed_at))
  end
end
```

```elixir
# lib/panic_tda/resources/invocation.ex
defmodule PanicTda.Invocation do
  use Ash.Resource,
    domain: PanicTda,
    data_layer: AshSqlite.DataLayer

  sqlite do
    table "invocations"
    repo PanicTda.Repo
  end

  attributes do
    uuid_v7_primary_key :id

    attribute :model, :string, allow_nil?: false
    attribute :type, PanicTda.Types.InvocationType, allow_nil?: false
    attribute :seed, :integer, allow_nil?: false
    attribute :sequence_number, :integer, allow_nil?: false

    # Polymorphic output - only one will be set
    attribute :output_text, :string
    attribute :output_image_data, :binary  # WEBP format

    attribute :started_at, :utc_datetime_usec, allow_nil?: false
    attribute :completed_at, :utc_datetime_usec, allow_nil?: false
  end

  relationships do
    belongs_to :run, PanicTda.Run, allow_nil?: false
    belongs_to :input_invocation, PanicTda.Invocation, allow_nil?: true
    has_many :embeddings, PanicTda.Embedding
  end

  calculations do
    calculate :duration, :float, expr(
      fragment("(julianday(?) - julianday(?)) * 86400", completed_at, started_at)
    )

    calculate :output, :map, PanicTda.Calculations.InvocationOutput
  end
end
```

```elixir
# lib/panic_tda/resources/embedding.ex
defmodule PanicTda.Embedding do
  use Ash.Resource,
    domain: PanicTda,
    data_layer: AshSqlite.DataLayer

  sqlite do
    table "embeddings"
    repo PanicTda.Repo
  end

  attributes do
    uuid_v7_primary_key :id

    attribute :embedding_model, :string, allow_nil?: false
    attribute :vector, PanicTda.Types.NumpyVector, allow_nil?: false  # Custom type

    attribute :started_at, :utc_datetime_usec, allow_nil?: false
    attribute :completed_at, :utc_datetime_usec, allow_nil?: false
  end

  relationships do
    belongs_to :invocation, PanicTda.Invocation, allow_nil?: false
    has_many :cluster_assignments, PanicTda.EmbeddingCluster
  end

  identities do
    identity :unique_invocation_model, [:invocation_id, :embedding_model]
  end
end
```

### 2.2 Custom Ash Types

```elixir
# lib/panic_tda/types/numpy_vector.ex
defmodule PanicTda.Types.NumpyVector do
  @moduledoc """
  Custom Ash type for storing float32 vectors as binary.
  Compatible with the Python NumpyArrayType format.
  """
  use Ash.Type

  @impl true
  def storage_type(_), do: :binary

  @impl true
  def cast_input(nil, _), do: {:ok, nil}
  def cast_input(%Nx.Tensor{} = tensor, _) do
    # Convert Nx tensor to binary (float32, little-endian)
    {:ok, Nx.to_binary(tensor, type: :f32)}
  end
  def cast_input(list, _) when is_list(list) do
    tensor = Nx.tensor(list, type: :f32)
    {:ok, Nx.to_binary(tensor)}
  end

  @impl true
  def cast_stored(nil, _), do: {:ok, nil}
  def cast_stored(binary, _) when is_binary(binary) do
    # Convert binary back to Nx tensor
    {:ok, Nx.from_binary(binary, :f32)}
  end

  @impl true
  def dump_to_native(nil, _), do: {:ok, nil}
  def dump_to_native(%Nx.Tensor{} = tensor, _) do
    {:ok, Nx.to_binary(tensor, type: :f32)}
  end
  def dump_to_native(binary, _) when is_binary(binary), do: {:ok, binary}
end
```

```elixir
# lib/panic_tda/types/persistence_diagram_result.ex
defmodule PanicTda.Types.PersistenceDiagramResult do
  @moduledoc """
  Custom type for complex persistence diagram data.
  Stores as compressed binary, compatible with Python np.savez_compressed format.
  """
  use Ash.Type

  @impl true
  def storage_type(_), do: :binary

  @impl true
  def cast_input(nil, _), do: {:ok, nil}
  def cast_input(%{dgms: dgms, entropy: entropy} = data, _) do
    # Serialize to compressed format
    binary = serialize_pd_result(data)
    {:ok, binary}
  end

  @impl true
  def cast_stored(nil, _), do: {:ok, nil}
  def cast_stored(binary, _) when is_binary(binary) do
    {:ok, deserialize_pd_result(binary)}
  end

  defp serialize_pd_result(data) do
    # Use :erlang.term_to_binary with compression
    # Or implement NPZ-compatible format for Python interop
    :erlang.term_to_binary(data, [:compressed])
  end

  defp deserialize_pd_result(binary) do
    :erlang.binary_to_term(binary)
  end
end
```

### 2.3 Ash Domain

```elixir
# lib/panic_tda.ex
defmodule PanicTda do
  use Ash.Domain

  resources do
    resource PanicTda.Experiment
    resource PanicTda.Run
    resource PanicTda.Invocation
    resource PanicTda.Embedding
    resource PanicTda.PersistenceDiagram
    resource PanicTda.ClusteringResult
    resource PanicTda.EmbeddingCluster
  end
end
```

---

## 3. Model Execution Strategy

### Critical Decision Point

The current Python implementation uses:
- **FLUX.1-dev/schnell** (diffusers) - Text-to-Image
- **SDXL-Turbo** (diffusers) - Text-to-Image
- **Moondream** (transformers) - Image-to-Text
- **BLIP2** (transformers) - Image-to-Text
- **sentence-transformers** - Text embeddings
- **nomic-embed** - Text/Vision embeddings

### Option A: Ortex (Pure ONNX)

**Pros:**
- Native Elixir, no Python dependency
- Good performance with ONNX Runtime
- Works with Nx ecosystem

**Cons:**
- **Major limitation**: Most generative models (FLUX, SDXL, Moondream, BLIP2) are NOT available in ONNX format
- Would require manual ONNX export from PyTorch (complex, often lossy)
- Dynamic shapes in generative models are problematic for ONNX

**Available ONNX Models:**
- Some embedding models (sentence-transformers has ONNX exports)
- CLIP models (via Hugging Face Optimum)
- Simple classification/regression models

```elixir
# Example: Ortex for embeddings (if ONNX model available)
defmodule PanicTda.Models.Embeddings.SentenceTransformer do
  @behaviour PanicTda.EmbeddingModel

  def load do
    # Load ONNX model via Ortex
    Ortex.load("models/all-mpnet-base-v2.onnx")
  end

  def embed(model, texts) when is_list(texts) do
    # Tokenize (need separate tokenizer)
    tokens = Tokenizers.encode_batch(tokenizer(), texts)

    # Run inference
    {embeddings} = Ortex.run(model, tokens)

    # Mean pooling + normalize
    embeddings
    |> mean_pool()
    |> normalize()
  end
end
```

### Option B: Python Interop via Ports

**Pros:**
- Full access to HuggingFace ecosystem
- Exact same model behavior as current implementation
- Can reuse existing Python model code

**Cons:**
- Adds Python runtime dependency
- Serialization overhead (can be significant for images)
- More complex deployment

```elixir
# lib/panic_tda/python_port.ex
defmodule PanicTda.PythonPort do
  use GenServer

  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def init(_opts) do
    port = Port.open({:spawn, "python3 -u model_server.py"}, [
      :binary,
      :use_stdio,
      {:packet, 4}
    ])
    {:ok, %{port: port}}
  end

  def invoke(model_name, input, seed) do
    GenServer.call(__MODULE__, {:invoke, model_name, input, seed}, :infinity)
  end

  def handle_call({:invoke, model_name, input, seed}, _from, state) do
    # Serialize request
    request = :erlang.term_to_binary({model_name, input, seed})
    Port.command(state.port, request)

    receive do
      {port, {:data, response}} when port == state.port ->
        {:reply, :erlang.binary_to_term(response), state}
    end
  end
end
```

### Option C: Hybrid Approach (Recommended)

Use **Ortex for embeddings** (where ONNX models exist) and **Python ports for generative models** (where ONNX is impractical).

```elixir
# lib/panic_tda/models/model_dispatcher.ex
defmodule PanicTda.Models.Dispatcher do
  @doc """
  Routes model invocations to appropriate backend.
  """

  @onnx_models ~w(STSBMpnet STSBRoberta JinaClip NomicVision)
  @python_models ~w(FluxDev FluxSchnell SDXLTurbo Moondream BLIP2)

  def invoke(model_name, input, seed) when model_name in @onnx_models do
    PanicTda.Models.OrtexBackend.invoke(model_name, input, seed)
  end

  def invoke(model_name, input, seed) when model_name in @python_models do
    PanicTda.Models.PythonBackend.invoke(model_name, input, seed)
  end
end
```

### Option D: Bumblebee for Transformers

Bumblebee provides native Elixir/Axon implementations of some models:

**Available in Bumblebee:**
- CLIP (for embeddings)
- Stable Diffusion (but NOT FLUX or SDXL-Turbo)
- BLIP (image captioning - partial)
- Text embedding models

**Not Available:**
- FLUX.1-dev/schnell (too new, complex architecture)
- Moondream (custom architecture)
- BLIP2 (uses OPT decoder, complex)

```elixir
# Using Bumblebee for CLIP embeddings
defmodule PanicTda.Models.Embeddings.ClipText do
  def load do
    {:ok, model_info} = Bumblebee.load_model({:hf, "openai/clip-vit-base-patch32"})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai/clip-vit-base-patch32"})
    {model_info, tokenizer}
  end

  def embed({model_info, tokenizer}, texts) do
    inputs = Bumblebee.apply_tokenizer(tokenizer, texts)
    Axon.predict(model_info.model, model_info.params, inputs)
  end
end
```

---

## 4. Pipeline Orchestration with Reactor

Reactor is ideal for the three-stage pipeline, providing:
- Declarative step definitions
- Automatic dependency resolution
- Built-in error handling and retries
- Async execution with configurable concurrency

### 4.1 Experiment Reactor

```elixir
# lib/panic_tda/reactors/perform_experiment.ex
defmodule PanicTda.Reactors.PerformExperiment do
  use Reactor

  input :experiment_id

  step :load_experiment do
    argument :experiment_id, input(:experiment_id)

    run fn %{experiment_id: id}, _ ->
      case Ash.get(PanicTda.Experiment, id, load: [:runs]) do
        {:ok, experiment} -> {:ok, experiment}
        {:error, _} -> {:error, :experiment_not_found}
      end
    end
  end

  step :mark_started, wait_for: [:load_experiment] do
    argument :experiment, result(:load_experiment)

    run fn %{experiment: exp}, _ ->
      Ash.update(exp, :start)
    end
  end

  step :init_runs, wait_for: [:mark_started] do
    argument :experiment, result(:mark_started)

    run fn %{experiment: exp}, _ ->
      runs = PanicTda.RunInitializer.create_runs(exp)
      {:ok, runs}
    end
  end

  # Group runs by network and execute each group
  step :execute_run_groups, wait_for: [:init_runs] do
    argument :runs, result(:init_runs)

    run fn %{runs: runs}, _ ->
      runs
      |> Enum.group_by(& &1.network)
      |> Enum.map(fn {_network, group_runs} ->
        # Each group can run in parallel (same network = same actors)
        Reactor.run(PanicTda.Reactors.RunGroup, %{runs: group_runs})
      end)
      |> Enum.reduce({:ok, []}, fn
        {:ok, invocations}, {:ok, acc} -> {:ok, acc ++ invocations}
        {:error, e}, _ -> {:error, e}
        _, {:error, e} -> {:error, e}
      end)
    end
  end

  step :compute_embeddings, wait_for: [:execute_run_groups] do
    argument :invocation_ids, result(:execute_run_groups)
    argument :experiment, result(:mark_started)

    run fn %{invocation_ids: ids, experiment: exp}, _ ->
      Reactor.run(PanicTda.Reactors.EmbeddingsStage, %{
        invocation_ids: ids,
        embedding_models: exp.embedding_models
      })
    end
  end

  step :compute_persistence_diagrams, wait_for: [:compute_embeddings] do
    argument :experiment, result(:mark_started)

    run fn %{experiment: exp}, _ ->
      run_ids = Enum.map(exp.runs, & &1.id)
      Reactor.run(PanicTda.Reactors.PDStage, %{
        run_ids: run_ids,
        embedding_models: exp.embedding_models
      })
    end
  end

  step :mark_completed, wait_for: [:compute_persistence_diagrams] do
    argument :experiment, result(:mark_started)

    run fn %{experiment: exp}, _ ->
      Ash.update(exp, :complete)
    end
  end

  return :mark_completed
end
```

### 4.2 Run Generator Reactor

```elixir
# lib/panic_tda/reactors/run_generator.ex
defmodule PanicTda.Reactors.RunGenerator do
  @moduledoc """
  Executes a single run through the model network.
  Yields invocations one at a time, checking for loops.
  """
  use Reactor

  input :run

  step :execute_run do
    argument :run, input(:run)

    run fn %{run: run}, _ ->
      execute_run_loop(run, MapSet.new(), [])
    end
  end

  return :execute_run

  defp execute_run_loop(run, seen_hashes, invocation_ids) do
    current_seq = length(invocation_ids)

    if current_seq >= run.max_length do
      {:ok, invocation_ids}
    else
      # Get model for this step
      network_len = length(run.network)
      model_name = Enum.at(run.network, rem(current_seq, network_len))

      # Get input (initial prompt or previous output)
      input = get_input(run, current_seq, invocation_ids)

      # Execute model
      started_at = DateTime.utc_now()
      {:ok, output} = PanicTda.Models.Dispatcher.invoke(model_name, input, run.seed)
      completed_at = DateTime.utc_now()

      # Create invocation record
      {:ok, invocation} = create_invocation(run, model_name, current_seq, output, started_at, completed_at)

      # Check for loops (duplicate detection)
      output_hash = hash_output(output)

      if MapSet.member?(seen_hashes, output_hash) and run.seed != -1 do
        # Loop detected, stop run
        {:ok, invocation_ids ++ [invocation.id]}
      else
        # Continue
        execute_run_loop(
          run,
          MapSet.put(seen_hashes, output_hash),
          invocation_ids ++ [invocation.id]
        )
      end
    end
  end
end
```

### 4.3 Embeddings Stage with Parallel Processing

```elixir
# lib/panic_tda/reactors/embeddings_stage.ex
defmodule PanicTda.Reactors.EmbeddingsStage do
  use Reactor

  input :invocation_ids
  input :embedding_models

  step :compute_embeddings do
    argument :invocation_ids, input(:invocation_ids)
    argument :embedding_models, input(:embedding_models)

    async? true
    max_concurrency 8

    run fn %{invocation_ids: ids, embedding_models: models}, _ ->
      # For each embedding model
      tasks = for model <- models do
        Task.async(fn ->
          compute_embeddings_for_model(ids, model)
        end)
      end

      results = Task.await_many(tasks, :infinity)
      {:ok, List.flatten(results)}
    end
  end

  return :compute_embeddings

  defp compute_embeddings_for_model(invocation_ids, model_name) do
    model_type = PanicTda.Models.get_model_type(model_name)

    # Filter invocations by type
    invocations =
      invocation_ids
      |> load_invocations()
      |> Enum.filter(&(&1.type == model_type))

    # Batch process
    invocations
    |> Enum.chunk_every(32)
    |> Enum.flat_map(fn batch ->
      contents = Enum.map(batch, &get_output_content/1)
      vectors = PanicTda.Models.Dispatcher.embed(model_name, contents)

      Enum.zip(batch, vectors)
      |> Enum.map(fn {inv, vector} ->
        create_embedding(inv.id, model_name, vector)
      end)
    end)
  end
end
```

---

## 5. TDA Computation

For the persistence diagram computation, options include:

### 5.1 Rustler NIF wrapping ripser

```rust
// native/ripser_nif/src/lib.rs
use rustler::{Encoder, Env, NifResult, Term};
use ripser::RipserParams;

#[rustler::nif]
fn compute_persistence_diagram(
    point_cloud: Vec<Vec<f64>>,
    max_dim: usize
) -> NifResult<(Vec<Vec<(f64, f64)>>, f64)> {
    let params = RipserParams::default()
        .max_dimension(max_dim)
        .threshold(f64::INFINITY);

    let diagram = ripser::compute_persistence_diagram(&point_cloud, params);

    // Return (dgms, entropy)
    Ok((diagram.pairs, compute_entropy(&diagram)))
}
```

### 5.2 Python Port for TDA

If maintaining Python compatibility is important:

```elixir
defmodule PanicTda.TDA do
  def compute_persistence_diagram(point_cloud, max_dim \\ 2) do
    # Call Python giotto-ph via port
    PanicTda.PythonPort.call(:compute_pd, [point_cloud, max_dim])
  end
end
```

### 5.3 Pure Elixir Implementation

A pure Elixir implementation of Vietoris-Rips persistence is possible but would be significantly slower than optimized C++/Rust implementations.

---

## 6. Export System

### 6.1 FFmpeg Integration (Unchanged)

The export system shells out to ffmpeg and can remain largely the same:

```elixir
defmodule PanicTda.Export.Video do
  def export(run_ids, opts \\ []) do
    fps = Keyword.get(opts, :fps, 24)
    resolution = Keyword.get(opts, :resolution, :hd)

    with {:ok, frames_dir} <- create_frames(run_ids, resolution),
         {:ok, output_path} <- run_ffmpeg(frames_dir, fps, resolution) do
      {:ok, output_path}
    end
  end

  defp run_ffmpeg(frames_dir, fps, resolution) do
    {width, height} = resolution_dims(resolution)

    args = [
      "-framerate", to_string(fps),
      "-pattern_type", "glob",
      "-i", Path.join(frames_dir, "*.jpg"),
      "-c:v", "libx265",
      "-preset", "medium",
      "-crf", "22",
      "-vf", "scale=#{width}:#{height}:force_original_aspect_ratio=decrease,pad=#{width}:#{height}:(ow-iw)/2:(oh-ih)/2",
      "-pix_fmt", "yuv420p",
      "-movflags", "+faststart",
      output_path()
    ]

    case System.cmd("ffmpeg", args, stderr_to_stdout: true) do
      {_, 0} -> {:ok, output_path()}
      {error, _} -> {:error, error}
    end
  end
end
```

### 6.2 Image Processing with Image (Elixir)

Replace Pillow with the Elixir `image` library:

```elixir
defmodule PanicTda.Export.ImageUtils do
  def create_grid(images, columns) do
    images
    |> Enum.chunk_every(columns)
    |> Enum.map(&Image.append(&1, :horizontal))
    |> Image.append(:vertical)
  end

  def add_text_overlay(image, text, opts \\ []) do
    Image.Draw.text!(image, text, opts)
  end
end
```

---

## 7. Project Structure

```
panic_tda_ex/
├── lib/
│   ├── panic_tda.ex                    # Ash Domain
│   ├── panic_tda/
│   │   ├── application.ex              # OTP Application
│   │   ├── repo.ex                     # Ecto Repo for AshSqlite
│   │   │
│   │   ├── resources/                  # Ash Resources
│   │   │   ├── experiment.ex
│   │   │   ├── run.ex
│   │   │   ├── invocation.ex
│   │   │   ├── embedding.ex
│   │   │   ├── persistence_diagram.ex
│   │   │   ├── clustering_result.ex
│   │   │   └── embedding_cluster.ex
│   │   │
│   │   ├── types/                      # Custom Ash Types
│   │   │   ├── invocation_type.ex
│   │   │   ├── numpy_vector.ex
│   │   │   └── persistence_diagram_result.ex
│   │   │
│   │   ├── reactors/                   # Reactor Orchestration
│   │   │   ├── perform_experiment.ex
│   │   │   ├── run_generator.ex
│   │   │   ├── embeddings_stage.ex
│   │   │   └── pd_stage.ex
│   │   │
│   │   ├── models/                     # Model Execution
│   │   │   ├── dispatcher.ex
│   │   │   ├── ortex_backend.ex
│   │   │   ├── python_backend.ex
│   │   │   ├── model_pool.ex           # GenServer pool for models
│   │   │   ├── genai/
│   │   │   │   ├── flux_dev.ex
│   │   │   │   ├── sdxl_turbo.ex
│   │   │   │   ├── moondream.ex
│   │   │   │   └── blip2.ex
│   │   │   └── embeddings/
│   │   │       ├── nomic.ex
│   │   │       ├── jina_clip.ex
│   │   │       └── stsb_mpnet.ex
│   │   │
│   │   ├── tda/                        # Topological Data Analysis
│   │   │   ├── persistence_diagram.ex
│   │   │   └── wasserstein.ex
│   │   │
│   │   ├── export/                     # Export functionality
│   │   │   ├── video.ex
│   │   │   ├── timeline.ex
│   │   │   ├── mosaic.ex
│   │   │   └── image_utils.ex
│   │   │
│   │   ├── clustering/
│   │   │   ├── hdbscan.ex
│   │   │   └── optics.ex
│   │   │
│   │   └── cli.ex                      # CLI interface
│   │
│   └── mix/
│       └── tasks/                      # Mix tasks
│           └── panic_tda.ex
│
├── native/                             # Rustler NIFs
│   └── ripser_nif/
│       ├── src/lib.rs
│       └── Cargo.toml
│
├── python/                             # Python model server (if hybrid)
│   ├── model_server.py
│   └── requirements.txt
│
├── priv/
│   └── repo/
│       └── migrations/                 # AshSqlite migrations
│
├── test/
├── mix.exs
└── README.md
```

---

## 8. Dependencies (mix.exs)

```elixir
defp deps do
  [
    # Ash Framework
    {:ash, "~> 3.0"},
    {:ash_sqlite, "~> 0.1"},

    # Reactor for orchestration
    {:reactor, "~> 0.9"},

    # ML/AI
    {:ortex, "~> 0.1"},           # ONNX Runtime
    {:nx, "~> 0.7"},              # Numerical computing
    {:bumblebee, "~> 0.5"},       # Transformers (optional)
    {:tokenizers, "~> 0.5"},      # HuggingFace tokenizers
    {:exla, "~> 0.7"},            # XLA backend for Nx

    # Image processing
    {:image, "~> 0.40"},          # Image manipulation
    {:vix, "~> 0.26"},            # libvips bindings

    # Database
    {:ecto_sqlite3, "~> 0.15"},

    # CLI
    {:optimus, "~> 0.5"},         # CLI argument parsing
    # or {:burrito, "~> 1.0"}     # For standalone executables

    # Utils
    {:uuid_v7, "~> 0.2"},
    {:jason, "~> 1.4"},

    # Python interop (if using hybrid approach)
    {:pythonx, "~> 0.2"},         # Python integration

    # Dev/Test
    {:ex_doc, "~> 0.31", only: :dev},
    {:credo, "~> 1.7", only: [:dev, :test]}
  ]
end
```

---

## 9. Key Challenges & Recommendations

### 9.1 Model Availability (Critical)

| Model | ONNX Available | Bumblebee | Recommendation |
|-------|----------------|-----------|----------------|
| FLUX.1-dev | No | No | Python port |
| FLUX.1-schnell | No | No | Python port |
| SDXL-Turbo | Partial | SD only | Python port |
| Moondream | No | No | Python port |
| BLIP2 | Partial | Partial | Python port or Bumblebee |
| sentence-transformers | Yes | Yes | Ortex or Bumblebee |
| nomic-embed | Partial | No | Ortex with export |
| CLIP | Yes | Yes | Bumblebee |

**Recommendation:** Start with a **hybrid approach** - use Python ports for generative models (FLUX, SDXL, Moondream, BLIP2) and Ortex/Bumblebee for embeddings.

### 9.2 GPU Management

Python Ray handles GPU allocation automatically. In Elixir:
- Use `Ortex` with CUDA backend for ONNX models
- Python port models handle their own GPU allocation
- Consider `poolboy` for managing model instances

```elixir
# lib/panic_tda/models/model_pool.ex
defmodule PanicTda.Models.Pool do
  use Supervisor

  def start_link(_) do
    Supervisor.start_link(__MODULE__, :ok, name: __MODULE__)
  end

  def init(:ok) do
    children = [
      :poolboy.child_spec(:flux_pool, [
        {:name, {:local, :flux_pool}},
        {:worker_module, PanicTda.Models.FluxWorker},
        {:size, 1},  # One GPU
        {:max_overflow, 0}
      ]),
      # ... more pools
    ]

    Supervisor.init(children, strategy: :one_for_one)
  end
end
```

### 9.3 Memory Management

Elixir/BEAM handles memory differently than Python:
- Binary data (images, vectors) can be large - use binary references carefully
- Consider `:binary.copy/1` for long-lived binaries
- GPU memory cleanup needs explicit calls to Python/Ortex

### 9.4 Persistence Diagram Format Compatibility

If you need to read/write Python-compatible PD data:
- Implement NPZ reader/writer in Elixir
- Or use a shared format (JSON with base64-encoded arrays)

---

## 10. Migration Path

### Phase 1: Core Infrastructure
1. Set up Ash resources and SQLite
2. Implement custom types for vectors
3. Create basic CLI skeleton

### Phase 2: Model Execution
1. Implement Python port for all models
2. Test with existing Python models
3. Optionally migrate embeddings to Ortex

### Phase 3: Pipeline Orchestration
1. Implement Reactor workflows
2. Port run generator logic
3. Add embeddings and PD stages

### Phase 4: Export & Polish
1. Port video/image export
2. Add clustering functionality
3. CLI completion

### Phase 5: Optimization
1. Replace Python ports with native where beneficial
2. Add Rustler NIF for TDA
3. Performance tuning

---

## 11. Estimated Effort

| Component | Complexity | Notes |
|-----------|------------|-------|
| Ash data model | Medium | Straightforward mapping |
| Custom types | Medium | Vector and PD types need care |
| Python port setup | Medium | Boilerplate but reliable |
| Reactor orchestration | High | Core complexity lives here |
| Ortex embeddings | Medium | If ONNX models available |
| Export system | Low | FFmpeg stays, image lib is good |
| TDA (Rustler) | High | Complex algorithms |
| CLI | Low | Standard Elixir |

**Total estimate:** 4-8 weeks for a single experienced developer, depending on desired model execution strategy.

---

## 12. Pure Elixir Model Alternatives (Bumblebee)

Since exact model parity isn't required, we can use Bumblebee-native models for a **pure Elixir** implementation with no Python dependency.

### 12.1 Text-to-Image Models (Bumblebee)

| Model | Quality | Speed | Bumblebee Support | Notes |
|-------|---------|-------|-------------------|-------|
| **Stable Diffusion v2.1** | Good | Medium | ✅ Full | Best Bumblebee option |
| **Stable Diffusion v1.5** | Good | Medium | ✅ Full | Widely used |
| **Stable Diffusion XL Base** | Excellent | Slow | ⚠️ Partial | May need manual setup |

```elixir
# lib/panic_tda/models/genai/stable_diffusion.ex
defmodule PanicTda.Models.GenAI.StableDiffusion do
  @behaviour PanicTda.Models.GenAIModel

  def load do
    {:ok, model_info} = Bumblebee.load_model({:hf, "stabilityai/stable-diffusion-2-1"})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai/clip-vit-large-patch14"})
    {:ok, scheduler} = Bumblebee.load_scheduler({:hf, "stabilityai/stable-diffusion-2-1"})
    {:ok, featurizer} = Bumblebee.load_featurizer({:hf, "stabilityai/stable-diffusion-2-1"})
    {:ok, safety_checker} = Bumblebee.load_model({:hf, "CompVis/stable-diffusion-safety-checker"})

    serving = Bumblebee.Diffusion.StableDiffusion.text_to_image(
      model_info.unet,
      model_info.vae,
      model_info.text_encoder,
      tokenizer,
      scheduler,
      num_steps: 20,
      guidance_scale: 7.5,
      compile: [batch_size: 1, sequence_length: 77],
      defn_options: [compiler: EXLA]
    )

    {:ok, serving}
  end

  def invoke(serving, prompt, seed) do
    prng_key = if seed == -1, do: Nx.Random.key(:rand.uniform(1_000_000)), else: Nx.Random.key(seed)

    output = Nx.Serving.run(serving, %{prompt: prompt, seed: prng_key})

    # Convert Nx tensor to PNG binary
    image_binary = output.results
      |> hd()
      |> StbImage.from_nx()
      |> StbImage.to_binary(:png)

    {:ok, {:image, image_binary}}
  end
end
```

### 12.2 Image-to-Text Models (Bumblebee)

| Model | Quality | Speed | Bumblebee Support | Notes |
|-------|---------|-------|-------------------|-------|
| **BLIP** (base) | Good | Fast | ✅ Full | Best Bumblebee option |
| **BLIP-2** | Excellent | Medium | ⚠️ Partial | Complex architecture |
| **GIT** (base) | Good | Fast | ✅ Full | Alternative to BLIP |

```elixir
# lib/panic_tda/models/genai/blip.ex
defmodule PanicTda.Models.GenAI.Blip do
  @behaviour PanicTda.Models.GenAIModel

  def load do
    {:ok, model_info} = Bumblebee.load_model({:hf, "Salesforce/blip-image-captioning-base"})
    {:ok, featurizer} = Bumblebee.load_featurizer({:hf, "Salesforce/blip-image-captioning-base"})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "Salesforce/blip-image-captioning-base"})
    {:ok, generation_config} = Bumblebee.load_generation_config({:hf, "Salesforce/blip-image-captioning-base"})

    serving = Bumblebee.Vision.image_to_text(
      model_info,
      featurizer,
      tokenizer,
      generation_config,
      compile: [batch_size: 1],
      defn_options: [compiler: EXLA]
    )

    {:ok, serving}
  end

  def invoke(serving, image_binary, _seed) do
    # Convert binary to Nx tensor
    image = StbImage.read_binary!(image_binary) |> StbImage.to_nx()

    output = Nx.Serving.run(serving, image)
    caption = output.results |> hd() |> Map.get(:text)

    {:ok, {:text, caption}}
  end
end
```

### 12.3 Embedding Models (Bumblebee + Ortex)

| Model | Type | Quality | Support | Notes |
|-------|------|---------|---------|-------|
| **all-MiniLM-L6-v2** | Text | Good | ✅ Bumblebee | Fast, 384-dim |
| **CLIP ViT-B/32** | Text+Image | Good | ✅ Bumblebee | Shared embedding space |
| **BGE-base-en-v1.5** | Text | Excellent | ✅ ONNX | SOTA for retrieval |
| **nomic-embed-text-v1.5** | Text | Excellent | ⚠️ ONNX | Needs export |

```elixir
# lib/panic_tda/models/embeddings/clip.ex
defmodule PanicTda.Models.Embeddings.Clip do
  @moduledoc """
  CLIP provides both text and image embeddings in a shared space.
  This is ideal for PANIC-TDA since we embed both modalities.
  """
  @behaviour PanicTda.Models.EmbeddingModel

  def model_type, do: :multimodal  # Can embed both text and images

  def load do
    {:ok, model_info} = Bumblebee.load_model({:hf, "openai/clip-vit-base-patch32"})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai/clip-vit-base-patch32"})
    {:ok, featurizer} = Bumblebee.load_featurizer({:hf, "openai/clip-vit-base-patch32"})

    {:ok, {model_info, tokenizer, featurizer}}
  end

  def embed_text({model_info, tokenizer, _featurizer}, texts) do
    inputs = Bumblebee.apply_tokenizer(tokenizer, texts, length: 77)
    %{text_embedding: embeddings} = Axon.predict(model_info.model, model_info.params, inputs)

    # Normalize embeddings
    embeddings
    |> Nx.divide(Nx.LinAlg.norm(embeddings, axes: [-1], keep_axes: true))
    |> Nx.to_list()
  end

  def embed_image({model_info, _tokenizer, featurizer}, images) do
    inputs = Bumblebee.apply_featurizer(featurizer, images)
    %{image_embedding: embeddings} = Axon.predict(model_info.model, model_info.params, inputs)

    embeddings
    |> Nx.divide(Nx.LinAlg.norm(embeddings, axes: [-1], keep_axes: true))
    |> Nx.to_list()
  end
end
```

```elixir
# lib/panic_tda/models/embeddings/sentence_transformer.ex
defmodule PanicTda.Models.Embeddings.SentenceTransformer do
  @behaviour PanicTda.Models.EmbeddingModel

  def model_type, do: :text

  def load do
    {:ok, model_info} = Bumblebee.load_model({:hf, "sentence-transformers/all-MiniLM-L6-v2"})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "sentence-transformers/all-MiniLM-L6-v2"})

    serving = Bumblebee.Text.TextEmbedding.text_embedding(
      model_info,
      tokenizer,
      compile: [batch_size: 32, sequence_length: 128],
      defn_options: [compiler: EXLA]
    )

    {:ok, serving}
  end

  def embed(serving, texts) when is_list(texts) do
    output = Nx.Serving.run(serving, texts)

    output.embedding
    |> Nx.to_list()
  end
end
```

### 12.4 Recommended Model Set

For a **pure Elixir** implementation:

| Category | Primary Model | Fallback | Embedding Dim |
|----------|--------------|----------|---------------|
| Text-to-Image | Stable Diffusion 2.1 | SD 1.5 | N/A |
| Image-to-Text | BLIP (base) | GIT (base) | N/A |
| Text Embedding | all-MiniLM-L6-v2 | CLIP text | 384 / 512 |
| Image Embedding | CLIP vision | N/A | 512 |
| Multimodal | CLIP (shared space) | N/A | 512 |

**Note:** Using CLIP for both text and image embeddings ensures they're in the same embedding space, which could be advantageous for the TDA analysis.

---

## 13. Database Migration (Python → Elixir)

### 13.1 Schema Comparison

| Python Table | Elixir Table | Changes |
|--------------|--------------|---------|
| `experiment_configs` | `experiments` | Rename, same structure |
| `runs` | `runs` | Same |
| `invocations` | `invocations` | Same |
| `embeddings` | `embeddings` | Vector format may differ |
| `persistence_diagrams` | `persistence_diagrams` | Complex type format differs |
| `clustering_results` | `clustering_results` | Same |
| `embedding_clusters` | `embedding_clusters` | Same |

### 13.2 Migration Script

```elixir
# lib/mix/tasks/panic_tda.migrate_from_python.ex
defmodule Mix.Tasks.PanicTda.MigrateFromPython do
  @moduledoc """
  Migrates data from a Python PANIC-TDA SQLite database to the Elixir schema.

  Usage:
      mix panic_tda.migrate_from_python /path/to/old/panic_tda.db

  Options:
      --dry-run    Preview migration without writing
      --skip-pds   Skip persistence diagrams (complex format)
  """
  use Mix.Task

  require Logger

  @impl Mix.Task
  def run(args) do
    {opts, [source_db], _} = OptionParser.parse(args,
      switches: [dry_run: :boolean, skip_pds: :boolean]
    )

    Mix.Task.run("app.start")

    Logger.info("Migrating from #{source_db}")

    {:ok, source_conn} = Exqlite.Sqlite3.open(source_db)

    try do
      migrate_experiments(source_conn, opts)
      migrate_runs(source_conn, opts)
      migrate_invocations(source_conn, opts)
      migrate_embeddings(source_conn, opts)

      unless opts[:skip_pds] do
        migrate_persistence_diagrams(source_conn, opts)
      end

      migrate_clustering_results(source_conn, opts)
      migrate_embedding_clusters(source_conn, opts)

      Logger.info("Migration complete!")
    after
      Exqlite.Sqlite3.close(source_conn)
    end
  end

  defp migrate_experiments(conn, opts) do
    {:ok, stmt} = Exqlite.Sqlite3.prepare(conn,
      "SELECT id, networks, seeds, prompts, embedding_models, max_length, started_at, completed_at FROM experiment_configs"
    )

    count = stream_rows(conn, stmt)
    |> Enum.reduce(0, fn row, acc ->
      [id, networks, seeds, prompts, embedding_models, max_length, started_at, completed_at] = row

      attrs = %{
        id: parse_uuid(id),
        networks: Jason.decode!(networks),
        seeds: Jason.decode!(seeds),
        prompts: Jason.decode!(prompts),
        embedding_models: Jason.decode!(embedding_models),
        max_length: max_length,
        started_at: parse_datetime(started_at),
        completed_at: parse_datetime(completed_at)
      }

      unless opts[:dry_run] do
        PanicTda.Experiment
        |> Ash.Changeset.for_create(:create, attrs)
        |> Ash.create!()
      end

      acc + 1
    end)

    Logger.info("Migrated #{count} experiments")
  end

  defp migrate_runs(conn, opts) do
    {:ok, stmt} = Exqlite.Sqlite3.prepare(conn,
      "SELECT id, network, seed, max_length, initial_prompt, experiment_id FROM runs"
    )

    count = stream_rows(conn, stmt)
    |> Enum.reduce(0, fn row, acc ->
      [id, network, seed, max_length, initial_prompt, experiment_id] = row

      attrs = %{
        id: parse_uuid(id),
        network: Jason.decode!(network),
        seed: seed,
        max_length: max_length,
        initial_prompt: initial_prompt,
        experiment_id: parse_uuid(experiment_id)
      }

      unless opts[:dry_run] do
        PanicTda.Run
        |> Ash.Changeset.for_create(:create, attrs)
        |> Ash.create!()
      end

      acc + 1
    end)

    Logger.info("Migrated #{count} runs")
  end

  defp migrate_invocations(conn, opts) do
    {:ok, stmt} = Exqlite.Sqlite3.prepare(conn, """
      SELECT id, model, type, seed, sequence_number, run_id,
             input_invocation_id, output_text, output_image_data,
             started_at, completed_at
      FROM invocations
    """)

    count = stream_rows(conn, stmt)
    |> Enum.reduce(0, fn row, acc ->
      [id, model, type, seed, seq_num, run_id, input_inv_id, output_text, output_image, started, completed] = row

      attrs = %{
        id: parse_uuid(id),
        model: model,
        type: String.to_existing_atom(String.downcase(type)),
        seed: seed,
        sequence_number: seq_num,
        run_id: parse_uuid(run_id),
        input_invocation_id: parse_uuid(input_inv_id),
        output_text: output_text,
        output_image_data: output_image,  # Binary, same format
        started_at: parse_datetime(started),
        completed_at: parse_datetime(completed)
      }

      unless opts[:dry_run] do
        PanicTda.Invocation
        |> Ash.Changeset.for_create(:create, attrs)
        |> Ash.create!()
      end

      acc + 1
    end)

    Logger.info("Migrated #{count} invocations")
  end

  defp migrate_embeddings(conn, opts) do
    {:ok, stmt} = Exqlite.Sqlite3.prepare(conn, """
      SELECT id, invocation_id, embedding_model, vector, started_at, completed_at
      FROM embeddings
    """)

    count = stream_rows(conn, stmt)
    |> Enum.reduce(0, fn row, acc ->
      [id, inv_id, model, vector_binary, started, completed] = row

      # Python stores as float32 little-endian, which is what we expect
      # The vector binary should be compatible
      attrs = %{
        id: parse_uuid(id),
        invocation_id: parse_uuid(inv_id),
        embedding_model: model,
        vector: vector_binary,  # Same binary format
        started_at: parse_datetime(started),
        completed_at: parse_datetime(completed)
      }

      unless opts[:dry_run] do
        PanicTda.Embedding
        |> Ash.Changeset.for_create(:create, attrs)
        |> Ash.create!()
      end

      acc + 1
    end)

    Logger.info("Migrated #{count} embeddings")
  end

  defp migrate_persistence_diagrams(conn, opts) do
    {:ok, stmt} = Exqlite.Sqlite3.prepare(conn, """
      SELECT id, run_id, embedding_model, diagram_data, started_at, completed_at
      FROM persistence_diagrams
    """)

    count = stream_rows(conn, stmt)
    |> Enum.reduce(0, fn row, acc ->
      [id, run_id, model, diagram_data, started, completed] = row

      # diagram_data is stored as np.savez_compressed format
      # We need to either:
      # 1. Convert to Elixir-native format
      # 2. Keep as-is and handle in the custom type
      # Option 2 is simpler for migration

      converted_data = convert_numpy_savez(diagram_data)

      attrs = %{
        id: parse_uuid(id),
        run_id: parse_uuid(run_id),
        embedding_model: model,
        diagram_data: converted_data,
        started_at: parse_datetime(started),
        completed_at: parse_datetime(completed)
      }

      unless opts[:dry_run] do
        PanicTda.PersistenceDiagram
        |> Ash.Changeset.for_create(:create, attrs)
        |> Ash.create!()
      end

      acc + 1
    end)

    Logger.info("Migrated #{count} persistence diagrams")
  end

  defp migrate_clustering_results(conn, opts) do
    {:ok, stmt} = Exqlite.Sqlite3.prepare(conn, """
      SELECT id, embedding_model, algorithm, parameters, created_at, started_at, completed_at
      FROM clustering_results
    """)

    count = stream_rows(conn, stmt)
    |> Enum.reduce(0, fn row, acc ->
      [id, model, algorithm, params, created, started, completed] = row

      attrs = %{
        id: parse_uuid(id),
        embedding_model: model,
        algorithm: algorithm,
        parameters: Jason.decode!(params),
        created_at: parse_datetime(created),
        started_at: parse_datetime(started),
        completed_at: parse_datetime(completed)
      }

      unless opts[:dry_run] do
        PanicTda.ClusteringResult
        |> Ash.Changeset.for_create(:create, attrs)
        |> Ash.create!()
      end

      acc + 1
    end)

    Logger.info("Migrated #{count} clustering results")
  end

  defp migrate_embedding_clusters(conn, opts) do
    {:ok, stmt} = Exqlite.Sqlite3.prepare(conn, """
      SELECT id, embedding_id, clustering_result_id, medoid_embedding_id
      FROM embedding_clusters
    """)

    count = stream_rows(conn, stmt)
    |> Enum.reduce(0, fn row, acc ->
      [id, emb_id, cluster_id, medoid_id] = row

      attrs = %{
        id: parse_uuid(id),
        embedding_id: parse_uuid(emb_id),
        clustering_result_id: parse_uuid(cluster_id),
        medoid_embedding_id: parse_uuid(medoid_id)
      }

      unless opts[:dry_run] do
        PanicTda.EmbeddingCluster
        |> Ash.Changeset.for_create(:create, attrs)
        |> Ash.create!()
      end

      acc + 1
    end)

    Logger.info("Migrated #{count} embedding clusters")
  end

  # Helper functions

  defp stream_rows(conn, stmt) do
    Stream.unfold(:continue, fn
      :done -> nil
      :continue ->
        case Exqlite.Sqlite3.step(conn, stmt) do
          {:row, row} -> {row, :continue}
          :done -> nil
        end
    end)
  end

  defp parse_uuid(nil), do: nil
  defp parse_uuid(uuid_string) when is_binary(uuid_string) do
    # Python stores UUIDs as hex strings
    uuid_string
  end

  defp parse_datetime(nil), do: nil
  defp parse_datetime(dt_string) when is_binary(dt_string) do
    # Python stores as ISO8601
    {:ok, dt, _} = DateTime.from_iso8601(dt_string)
    dt
  end

  defp convert_numpy_savez(nil), do: nil
  defp convert_numpy_savez(binary) when is_binary(binary) do
    # NPZ is a ZIP file containing .npy files
    # For now, store as-is and handle in custom type
    # A full implementation would:
    # 1. Unzip the archive
    # 2. Parse each .npy file (has a header + raw data)
    # 3. Convert to Elixir data structures

    # Simple approach: call Python to convert, or keep binary
    # For stretch goal, this is acceptable:
    binary
  end
end
```

### 13.3 NPZ Format Parser (Optional Enhancement)

For full NPZ compatibility without Python:

```elixir
# lib/panic_tda/formats/numpy.ex
defmodule PanicTda.Formats.Numpy do
  @moduledoc """
  Parse NumPy .npy and .npz (zipped .npy) formats.
  """

  @npy_magic <<0x93, "NUMPY">>

  def parse_npy(binary) do
    <<@npy_magic, version::binary-size(2), header_len::little-16, rest::binary>> = binary

    header = String.slice(rest, 0, header_len)
    data = binary_part(rest, header_len, byte_size(rest) - header_len)

    # Parse header (Python dict format)
    {dtype, shape, fortran_order} = parse_header(header)

    # Convert binary to Nx tensor
    tensor = Nx.from_binary(data, dtype_to_nx(dtype))
    Nx.reshape(tensor, List.to_tuple(shape))
  end

  def parse_npz(binary) do
    # NPZ is a ZIP file
    {:ok, files} = :zip.unzip(binary, [:memory])

    files
    |> Enum.map(fn {name, data} ->
      {String.trim_trailing(to_string(name), ".npy"), parse_npy(data)}
    end)
    |> Map.new()
  end

  defp parse_header(header) do
    # Header is Python dict: {'descr': '<f4', 'fortran_order': False, 'shape': (10, 768)}
    # Simplified parsing - production would use proper parser
    descr = Regex.run(~r/'descr':\s*'([^']+)'/, header) |> List.last()
    shape = Regex.run(~r/'shape':\s*\(([^)]+)\)/, header)
      |> List.last()
      |> String.split(",")
      |> Enum.map(&String.trim/1)
      |> Enum.reject(&(&1 == ""))
      |> Enum.map(&String.to_integer/1)

    fortran = String.contains?(header, "'fortran_order': True")

    {descr, shape, fortran}
  end

  defp dtype_to_nx("<f4"), do: :f32
  defp dtype_to_nx("<f8"), do: :f64
  defp dtype_to_nx("<i4"), do: :s32
  defp dtype_to_nx("<i8"), do: :s64
  defp dtype_to_nx(">f4"), do: {:f, 32, :big}
  defp dtype_to_nx(">f8"), do: {:f, 64, :big}
end
```

### 13.4 Migration Considerations

1. **Vector Format**: Python stores float32 little-endian arrays. Elixir's Nx uses the same format, so vectors should migrate directly.

2. **Persistence Diagrams**: The NPZ format is more complex. Options:
   - Keep as binary blob and parse on read (simplest)
   - Convert to Elixir term format during migration (cleaner)
   - Use Python helper script for this specific conversion

3. **UUIDs**: Python uses UUIDv7 hex strings, which Elixir handles natively.

4. **Timestamps**: Python uses ISO8601 strings, which DateTime.from_iso8601/1 handles.

5. **JSON fields**: Python stores arrays/dicts as JSON strings, which Jason handles.

---

## 14. Conclusion

Porting PANIC-TDA to Elixir is **highly feasible** with a pure Elixir approach using Bumblebee models:

### Recommended Stack

| Layer | Technology |
|-------|------------|
| Data Model | Ash 3.x + AshSqlite |
| Database | SQLite (same as Python) |
| Orchestration | Reactor |
| T2I Model | Stable Diffusion 2.1 (Bumblebee) |
| I2T Model | BLIP base (Bumblebee) |
| Embeddings | CLIP or all-MiniLM-L6-v2 (Bumblebee) |
| TDA | Rustler NIF wrapping ripser |
| Export | FFmpeg (unchanged) + Image library |

### Benefits of Elixir Port

- **No Python dependency** with Bumblebee models
- **Better concurrency** via OTP supervision trees
- **Cleaner orchestration** via Reactor's declarative steps
- **Type safety** with Ash validations and constraints
- **Distributed execution** across BEAM nodes (future)
- **Hot code reloading** for development

### Migration Path

1. Set up Ash resources (map 1:1 from SQLModel)
2. Implement Bumblebee model wrappers (SD, BLIP, CLIP)
3. Build Reactor pipelines (direct translation of Ray workflow)
4. Add Rustler NIF for TDA (or Python port initially)
5. Port export functionality (straightforward)
6. Run migration script for existing data

### Estimated Effort (Pure Elixir)

| Component | Effort |
|-----------|--------|
| Ash data model | 1 week |
| Bumblebee models | 1-2 weeks |
| Reactor pipelines | 1-2 weeks |
| TDA (Rustler) | 1 week |
| Export | 3-4 days |
| CLI | 2-3 days |
| Migration script | 2-3 days |
| **Total** | **5-7 weeks**|
