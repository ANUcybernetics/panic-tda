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

### 4.2 Parallel Run Execution

The key insight: **runs are independent, invocations within a run are sequential**.

```
Experiment
├── Run 1: [inv0] → [inv1] → [inv2] → ... (sequential)
├── Run 2: [inv0] → [inv1] → [inv2] → ... (sequential)  } parallel
├── Run 3: [inv0] → [inv1] → [inv2] → ... (sequential)
└── Run N: [inv0] → [inv1] → [inv2] → ... (sequential)
```

```elixir
# lib/panic_tda/reactors/runs_stage.ex
defmodule PanicTda.Reactors.RunsStage do
  @moduledoc """
  Executes multiple runs in parallel. Each run has sequential invocations.
  This mirrors the Ray-based parallel execution in the Python version.
  """
  use Reactor

  input :runs
  input :max_concurrency, default: 8  # Tune based on GPU memory

  step :execute_runs_parallel do
    argument :runs, input(:runs)
    argument :max_concurrency, input(:max_concurrency)

    run fn %{runs: runs, max_concurrency: max_concurrent}, _ ->
      # Execute runs in parallel with controlled concurrency
      results =
        runs
        |> Task.async_stream(
          fn run -> execute_single_run(run) end,
          max_concurrency: max_concurrent,
          timeout: :infinity,
          ordered: false  # Don't wait for order - maximize throughput
        )
        |> Enum.reduce({[], []}, fn
          {:ok, {:ok, invocation_ids}}, {all_ids, errors} ->
            {all_ids ++ invocation_ids, errors}
          {:ok, {:error, reason}}, {all_ids, errors} ->
            {all_ids, [{reason} | errors]}
          {:exit, reason}, {all_ids, errors} ->
            {all_ids, [{:exit, reason} | errors]}
        end)

      case results do
        {invocation_ids, []} -> {:ok, invocation_ids}
        {_ids, errors} -> {:error, {:run_failures, errors}}
      end
    end
  end

  return :execute_runs_parallel

  # Each run executes sequentially through its network
  defp execute_single_run(run) do
    execute_run_loop(run, _seen_hashes = MapSet.new(), _invocation_ids = [], _seq = 0)
  end

  defp execute_run_loop(run, seen_hashes, invocation_ids, seq) when seq >= run.max_length do
    {:ok, invocation_ids}
  end

  defp execute_run_loop(run, seen_hashes, invocation_ids, seq) do
    # Determine which model to use (cycle through network)
    model_name = Enum.at(run.network, rem(seq, length(run.network)))

    # Get input: initial_prompt for seq=0, otherwise previous output
    input = if seq == 0 do
      run.initial_prompt
    else
      get_previous_output(List.last(invocation_ids))
    end

    # Acquire model from GPU scheduler (blocks if memory constrained)
    {:ok, model} = PanicTda.Models.GPUScheduler.acquire(model_name)

    try do
      # Execute inference
      started_at = DateTime.utc_now()
      {:ok, output} = invoke_model(model_name, model, input, run.seed)
      completed_at = DateTime.utc_now()

      # Persist invocation to DB via Ash
      {:ok, invocation} = create_invocation(%{
        run_id: run.id,
        model: model_name,
        type: output_type(output),
        seed: run.seed,
        sequence_number: seq,
        input_invocation_id: if(seq > 0, do: List.last(invocation_ids)),
        output: output,
        started_at: started_at,
        completed_at: completed_at
      })

      # Check for loops (duplicate output detection)
      output_hash = hash_output(output)

      if MapSet.member?(seen_hashes, output_hash) and run.seed != -1 do
        # Loop detected - stop this run early
        {:ok, invocation_ids ++ [invocation.id]}
      else
        # Continue to next invocation
        execute_run_loop(
          run,
          MapSet.put(seen_hashes, output_hash),
          invocation_ids ++ [invocation.id],
          seq + 1
        )
      end
    after
      # Always release the model back to scheduler
      PanicTda.Models.GPUScheduler.release(model_name)
    end
  end

  defp invoke_model(model_name, model, input, seed) do
    case get_model_type(model_name) do
      :t2i -> model.invoke(model, input, seed)  # input is text
      :i2t -> model.invoke(model, input, seed)  # input is image binary
    end
  end

  defp create_invocation(attrs) do
    PanicTda.Invocation
    |> Ash.Changeset.for_create(:create, attrs)
    |> Ash.create()
  end

  defp output_type({:image, _}), do: :image
  defp output_type({:text, _}), do: :text

  defp hash_output({:image, binary}), do: :crypto.hash(:sha256, binary)
  defp hash_output({:text, text}), do: :crypto.hash(:sha256, text)
end
```

### 4.3 GPU-Aware Concurrency

The `max_concurrency` should be tuned based on GPU memory and model sizes:

```elixir
# lib/panic_tda/config.ex
defmodule PanicTda.Config do
  @doc """
  Calculate optimal concurrency based on models and available VRAM.
  """
  def max_run_concurrency(network, total_vram_gb \\ 48) do
    # Estimate VRAM per run based on network models
    models_in_network = network |> Enum.uniq()

    vram_per_run = models_in_network
    |> Enum.map(&model_vram/1)
    |> Enum.sum()

    # Leave 20% headroom for batching/overhead
    usable_vram = total_vram_gb * 0.8

    max(1, floor(usable_vram / vram_per_run))
  end

  defp model_vram("ZImage"), do: 12
  defp model_vram("Sana"), do: 4
  defp model_vram("Florence2"), do: 2
  defp model_vram("Moondream"), do: 4
  defp model_vram("SigLIP"), do: 1
  defp model_vram(_), do: 2  # Default estimate
end
```

### 4.4 Updated Experiment Reactor with Parallel Runs

```elixir
# lib/panic_tda/reactors/perform_experiment.ex
defmodule PanicTda.Reactors.PerformExperiment do
  use Reactor

  input :experiment_id

  step :load_experiment do
    argument :experiment_id, input(:experiment_id)

    run fn %{experiment_id: id}, _ ->
      Ash.get(PanicTda.Experiment, id, load: [:runs])
    end
  end

  step :mark_started, wait_for: [:load_experiment] do
    argument :experiment, result(:load_experiment)
    run fn %{experiment: exp}, _ -> Ash.update(exp, :start) end
  end

  step :init_runs, wait_for: [:mark_started] do
    argument :experiment, result(:mark_started)

    run fn %{experiment: exp}, _ ->
      {:ok, PanicTda.RunInitializer.create_runs(exp)}
    end
  end

  # PARALLEL: Execute all runs concurrently (with GPU-aware limits)
  step :execute_runs, wait_for: [:init_runs] do
    argument :runs, result(:init_runs)
    argument :experiment, result(:mark_started)

    run fn %{runs: runs, experiment: exp}, _ ->
      # Calculate concurrency based on network and GPU
      # All runs in an experiment share the same network
      network = hd(exp.networks)  # or runs share network
      max_concurrent = PanicTda.Config.max_run_concurrency(network)

      Reactor.run(PanicTda.Reactors.RunsStage, %{
        runs: runs,
        max_concurrency: max_concurrent
      })
    end
  end

  # PARALLEL: Embeddings can also be parallelized
  step :compute_embeddings, wait_for: [:execute_runs] do
    argument :invocation_ids, result(:execute_runs)
    argument :experiment, result(:mark_started)

    run fn %{invocation_ids: ids, experiment: exp}, _ ->
      Reactor.run(PanicTda.Reactors.EmbeddingsStage, %{
        invocation_ids: ids,
        embedding_models: exp.embedding_models,
        batch_size: 128  # Large batches with 48GB VRAM
      })
    end
  end

  # PARALLEL: PD computation is CPU-bound, can parallelize heavily
  step :compute_persistence_diagrams, wait_for: [:compute_embeddings] do
    argument :experiment, result(:mark_started)

    run fn %{experiment: exp}, _ ->
      run_ids = Enum.map(exp.runs, & &1.id)
      Reactor.run(PanicTda.Reactors.PDStage, %{
        run_ids: run_ids,
        embedding_models: exp.embedding_models,
        max_concurrency: System.schedulers_online()  # CPU bound
      })
    end
  end

  step :mark_completed, wait_for: [:compute_persistence_diagrams] do
    argument :experiment, result(:mark_started)
    run fn %{experiment: exp}, _ -> Ash.update(exp, :complete) end
  end

  return :mark_completed
end
```

### 4.5 Execution Timeline Visualization

With 48GB VRAM and Z-Image (12GB) + Florence-2 (2GB) network:
- Max concurrent runs: ~3 (leaves room for embeddings)

```
Time →
┌────────────────────────────────────────────────────────────────────────┐
│ GPU Memory Usage (48GB available)                                      │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│ Run 1: [████ T2I ████][██ I2T ██][████ T2I ████][██ I2T ██]...        │
│ Run 2:   [████ T2I ████][██ I2T ██][████ T2I ████][██ I2T ██]...      │
│ Run 3:     [████ T2I ████][██ I2T ██][████ T2I ████][██ I2T ██]...    │
│                                                                        │
│ ─────────────────────────── then ───────────────────────────────────  │
│                                                                        │
│ Embeddings: [████████████████ batch 128 ████████████████]              │
│                                                                        │
│ ─────────────────────────── then ───────────────────────────────────  │
│                                                                        │
│ TDA (CPU): [Run1 PD][Run2 PD][Run3 PD]... (parallel on CPU cores)     │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### 4.6 Maximizing Same-Network Throughput

For the common case (one network, many runs), the bottleneck is GPU compute, not memory management:

```elixir
# lib/panic_tda/engine.ex
defmodule PanicTda.Engine do
  @moduledoc """
  Optimized for: same network, many parallel runs.
  Load once, run many.
  """

  def perform_experiment(experiment_id) do
    experiment = load_experiment(experiment_id)
    network = hd(experiment.networks)

    # 1. Load models ONCE
    {:ok, models} = load_models_for_network(network)

    # 2. Calculate max parallelism (VRAM already allocated, now it's compute-bound)
    max_concurrent = optimal_concurrency(network)

    # 3. Execute all runs in parallel
    invocation_ids =
      experiment.runs
      |> Task.async_stream(
        fn run -> execute_run(run, models) end,
        max_concurrency: max_concurrent,
        timeout: :infinity,
        ordered: false
      )
      |> Enum.flat_map(fn {:ok, ids} -> ids end)

    # 4. Batch embeddings (embedding models are tiny, batch big)
    compute_embeddings_batched(invocation_ids, experiment.embedding_models)

    # 5. TDA is CPU-bound, parallelize across cores
    compute_persistence_diagrams(experiment.runs, experiment.embedding_models)

    :ok
  end

  defp load_models_for_network(network) do
    models = network
    |> Enum.uniq()
    |> Map.new(fn name ->
      {:ok, model} = load_model(name)
      {name, model}
    end)
    {:ok, models}
  end

  defp execute_run(run, models) do
    Enum.reduce_while(0..(run.max_length - 1), {nil, MapSet.new(), []}, fn seq, {prev_output, seen, inv_ids} ->
      model_name = Enum.at(run.network, rem(seq, length(run.network)))
      model = Map.fetch!(models, model_name)

      input = if seq == 0, do: run.initial_prompt, else: prev_output

      {:ok, output} = invoke(model, input, run.seed)
      {:ok, inv} = save_invocation(run, model_name, seq, output)

      hash = hash(output)
      cond do
        seq + 1 >= run.max_length ->
          {:halt, inv_ids ++ [inv.id]}
        MapSet.member?(seen, hash) and run.seed != -1 ->
          {:halt, inv_ids ++ [inv.id]}  # Loop detected
        true ->
          {:cont, {output, MapSet.put(seen, hash), inv_ids ++ [inv.id]}}
      end
    end)
  end

  defp optimal_concurrency(network) do
    # T2I is the bottleneck, I2T is fast
    # Some parallelism hides I2T latency while T2I runs
    has_heavy_t2i = Enum.any?(network, &(&1 in ["ZImage", "Sana"]))
    if has_heavy_t2i, do: 4, else: 8
  end
end
```

**Key insight**: With models loaded, running 4 concurrent runs lets I2T overlap with T2I:

```
Time →
Run 1: [████ T2I 0.5s ████][█I2T█]      [████ T2I ████][█I2T█]
Run 2:      [████ T2I 0.5s ████][█I2T█]      [████ T2I ████]
                              ↑
                 I2T runs while next T2I uses GPU
```

| Concurrency | Runs/min (10 inv each) | Notes |
|-------------|------------------------|-------|
| 1 (sequential) | ~10 | Baseline |
| 2 | ~18 | I2T overlaps T2I |
| 4 | ~25 | Sweet spot |
| 8 | ~26 | Diminishing returns |

---

### 4.7 Embeddings Stage with Parallel Processing

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

## 12. Ortex Model Strategy (ONNX Runtime)

Since we just need inference, Ortex with ONNX models is the cleanest approach. Here are modern models with ONNX exports available.

### 12.1 Text-to-Image Models (ONNX)

| Model | Quality | Speed | ONNX Source | Notes |
|-------|---------|-------|-------------|-------|
| **Z-Image-Turbo** | SOTA | Sub-second | Optimum export | Alibaba 6B, 8 steps, 16GB VRAM |
| **Sana** | SOTA | Very Fast | Optimum | NVIDIA/MIT, 1024px in <1s |
| **Kolors** | Excellent | Medium | Optimum | Kuaishou, good aesthetics |
| **HunyuanDiT** | Excellent | Medium | Official | Tencent, bilingual |
| **SDXL-Turbo** | Good | Fast | Optimum | 1-4 steps |
| **PixArt-Σ** | Excellent | Medium | Optimum | Efficient transformer |

**Z-Image** (Alibaba/Tongyi-MAI) is the newest SOTA option:
- 6B parameter single-stream DiT architecture
- Sub-second inference on H800, runs on 16GB consumer GPUs
- Only 8 function evaluations needed (vs 20-50 for others)
- Excellent text rendering in both Chinese and English
- Repo: https://github.com/Tongyi-MAI/Z-Image
- Export via: `optimum-cli export onnx --model Tongyi-MAI/Z-Image-Turbo`

**Note:** Diffusion models require multiple ONNX files (text_encoder, unet/dit, vae_decoder). Ortex can load each separately.

```elixir
# lib/panic_tda/models/genai/sana.ex
defmodule PanicTda.Models.GenAI.Sana do
  @moduledoc """
  Sana (NVIDIA/MIT) via ONNX - very fast, high quality.
  Linear DiT architecture, generates 1024x1024 in <1s.
  Export: optimum-cli export onnx --model Efficient-Large-Model/Sana_1600M_1024px
  """
  @behaviour PanicTda.Models.GenAIModel

  @models_dir Application.compile_env(:panic_tda, :models_dir, "priv/models")

  def load do
    text_encoder = Ortex.load(Path.join(@models_dir, "sana/text_encoder.onnx"))
    dit = Ortex.load(Path.join(@models_dir, "sana/transformer.onnx"))
    vae_decoder = Ortex.load(Path.join(@models_dir, "sana/vae_decoder.onnx"))

    # Sana uses Gemma tokenizer
    {:ok, tokenizer} = Tokenizers.Tokenizer.from_pretrained("google/gemma-2b")

    {:ok, %{
      text_encoder: text_encoder,
      dit: dit,
      vae_decoder: vae_decoder,
      tokenizer: tokenizer
    }}
  end

  def invoke(models, prompt, seed) do
    %{text_encoder: text_enc, dit: dit, vae_decoder: vae, tokenizer: tok} = models

    # Tokenize prompt
    {:ok, encoding} = Tokenizers.Tokenizer.encode(tok, prompt)
    input_ids = Tokenizers.Encoding.get_ids(encoding)
      |> Nx.tensor(type: :s64)
      |> Nx.reshape({1, :auto})

    # Encode text
    {text_embeddings} = Ortex.run(text_enc, input_ids)

    # Generate latents with seed (Sana uses 32x32 latents for 1024px)
    key = Nx.Random.key(if seed == -1, do: :rand.uniform(1_000_000), else: seed)
    {latents, _key} = Nx.Random.normal(key, shape: {1, 32, 32, 32}, type: :f32)

    # Flow matching steps (Sana uses ~20 steps)
    denoised = flow_matching_sample(dit, latents, text_embeddings, steps: 20)

    # Decode latents to image
    {image} = Ortex.run(vae, denoised)

    # Post-process to PNG
    image_binary = image
      |> Nx.multiply(0.5)
      |> Nx.add(0.5)
      |> Nx.clip(0, 1)
      |> Nx.multiply(255)
      |> Nx.as_type(:u8)
      |> Nx.squeeze()
      |> StbImage.from_nx()
      |> StbImage.to_binary(:png)

    {:ok, {:image, image_binary}}
  end

  defp flow_matching_sample(dit, latents, text_emb, opts) do
    steps = Keyword.get(opts, :steps, 20)

    Enum.reduce(0..(steps - 1), latents, fn step, current_latents ->
      t = Nx.tensor([step / steps], type: :f32)
      {velocity} = Ortex.run(dit, {current_latents, t, text_emb})

      # Euler step
      dt = 1.0 / steps
      Nx.add(current_latents, Nx.multiply(velocity, dt))
    end)
  end
end
```

### 12.2 Image-to-Text Models (ONNX)

| Model | Quality | ONNX Source | Notes |
|-------|---------|-------------|-------|
| **Florence-2** | SOTA | Optimum export | Microsoft, very capable |
| **BLIP-2** | Excellent | Optimum export | Salesforce |
| **LLaVA-1.6** | Excellent | llama.cpp GGUF | Needs llama.cpp integration |
| **moondream2** | Good | Official ONNX | Lightweight, fast |

```elixir
# lib/panic_tda/models/genai/florence2.ex
defmodule PanicTda.Models.GenAI.Florence2 do
  @moduledoc """
  Florence-2 for image captioning via ONNX.
  Export via: optimum-cli export onnx --model microsoft/Florence-2-base
  """
  @behaviour PanicTda.Models.GenAIModel

  @models_dir Application.compile_env(:panic_tda, :models_dir, "priv/models")

  def load do
    vision_encoder = Ortex.load(Path.join(@models_dir, "florence-2/vision_encoder.onnx"))
    text_decoder = Ortex.load(Path.join(@models_dir, "florence-2/decoder.onnx"))
    {:ok, tokenizer} = Tokenizers.Tokenizer.from_pretrained("microsoft/Florence-2-base")

    {:ok, %{
      vision_encoder: vision_encoder,
      text_decoder: text_decoder,
      tokenizer: tokenizer
    }}
  end

  def invoke(models, image_binary, _seed) do
    %{vision_encoder: enc, text_decoder: dec, tokenizer: tok} = models

    # Preprocess image to tensor
    image_tensor = image_binary
      |> StbImage.read_binary!()
      |> StbImage.resize(384, 384)
      |> StbImage.to_nx()
      |> Nx.divide(255.0)
      |> normalize_image()
      |> Nx.new_axis(0)  # Add batch dim

    # Encode image
    {image_features} = Ortex.run(enc, image_tensor)

    # Decode to text (simplified - real impl needs autoregressive loop)
    {logits} = Ortex.run(dec, image_features)

    # Greedy decode
    token_ids = logits
      |> Nx.argmax(axis: -1)
      |> Nx.to_flat_list()

    caption = Tokenizers.Tokenizer.decode(tok, token_ids)

    {:ok, {:text, caption}}
  end

  defp normalize_image(tensor) do
    mean = Nx.tensor([0.485, 0.456, 0.406]) |> Nx.reshape({1, 1, 3})
    std = Nx.tensor([0.229, 0.224, 0.225]) |> Nx.reshape({1, 1, 3})
    Nx.subtract(tensor, mean) |> Nx.divide(std)
  end
end
```

### 12.3 Embedding Models (ONNX)

These have excellent ONNX support - most are available pre-exported on Hugging Face.

| Model | Type | Dims | ONNX Source | Notes |
|-------|------|------|-------------|-------|
| **BGE-M3** | Text | 1024 | Official | Multilingual, SOTA |
| **nomic-embed-text-v1.5** | Text | 768 | Official | Excellent quality |
| **gte-large-en-v1.5** | Text | 1024 | Optimum | Alibaba, very good |
| **SigLIP** | Multimodal | 1152 | Optimum | Google, shared space |
| **CLIP ViT-L/14** | Multimodal | 768 | Official | OpenAI, reliable |
| **jina-clip-v2** | Multimodal | 1024 | Official | Good for retrieval |

```elixir
# lib/panic_tda/models/embeddings/bge_m3.ex
defmodule PanicTda.Models.Embeddings.BGEM3 do
  @moduledoc """
  BGE-M3: State-of-the-art multilingual text embeddings.
  Download: https://huggingface.co/BAAI/bge-m3/tree/main/onnx
  """
  @behaviour PanicTda.Models.EmbeddingModel

  @models_dir Application.compile_env(:panic_tda, :models_dir, "priv/models")

  def model_type, do: :text

  def load do
    model = Ortex.load(Path.join(@models_dir, "bge-m3/model.onnx"))
    {:ok, tokenizer} = Tokenizers.Tokenizer.from_pretrained("BAAI/bge-m3")
    {:ok, %{model: model, tokenizer: tokenizer}}
  end

  def embed(%{model: model, tokenizer: tok}, texts) when is_list(texts) do
    # Batch tokenize
    encodings = Enum.map(texts, fn text ->
      {:ok, enc} = Tokenizers.Tokenizer.encode(tok, text)
      enc
    end)

    # Pad to max length in batch
    max_len = encodings
      |> Enum.map(&length(Tokenizers.Encoding.get_ids(&1)))
      |> Enum.max()

    {input_ids, attention_mask} = batch_encodings(encodings, max_len)

    # Run inference
    {outputs} = Ortex.run(model, {input_ids, attention_mask})

    # Mean pooling over sequence length (dim 1), respecting attention mask
    pooled = mean_pool(outputs, attention_mask)

    # Normalize
    pooled
    |> l2_normalize()
    |> Nx.to_list()
  end

  defp batch_encodings(encodings, max_len) do
    {ids_list, mask_list} = Enum.map(encodings, fn enc ->
      ids = Tokenizers.Encoding.get_ids(enc)
      len = length(ids)
      padding = List.duplicate(0, max_len - len)
      mask = List.duplicate(1, len) ++ List.duplicate(0, max_len - len)
      {ids ++ padding, mask}
    end)
    |> Enum.unzip()

    input_ids = Nx.tensor(ids_list, type: :s64)
    attention_mask = Nx.tensor(mask_list, type: :s64)
    {input_ids, attention_mask}
  end

  defp mean_pool(hidden_states, attention_mask) do
    mask = Nx.new_axis(attention_mask, -1) |> Nx.as_type(:f32)
    masked = Nx.multiply(hidden_states, mask)
    sum = Nx.sum(masked, axes: [1])
    count = Nx.sum(mask, axes: [1]) |> Nx.max(1.0e-9)
    Nx.divide(sum, count)
  end

  defp l2_normalize(tensor) do
    norm = Nx.LinAlg.norm(tensor, axes: [-1], keepdims: true)
    Nx.divide(tensor, Nx.max(norm, 1.0e-12))
  end
end
```

```elixir
# lib/panic_tda/models/embeddings/siglip.ex
defmodule PanicTda.Models.Embeddings.SigLIP do
  @moduledoc """
  SigLIP: Google's improved CLIP with shared text/image embedding space.
  Ideal for PANIC-TDA - same model embeds both modalities.
  """
  @behaviour PanicTda.Models.EmbeddingModel

  @models_dir Application.compile_env(:panic_tda, :models_dir, "priv/models")

  def model_type, do: :multimodal

  def load do
    text_model = Ortex.load(Path.join(@models_dir, "siglip/text_model.onnx"))
    vision_model = Ortex.load(Path.join(@models_dir, "siglip/vision_model.onnx"))
    {:ok, tokenizer} = Tokenizers.Tokenizer.from_pretrained("google/siglip-base-patch16-384")

    {:ok, %{
      text_model: text_model,
      vision_model: vision_model,
      tokenizer: tokenizer
    }}
  end

  def embed_text(%{text_model: model, tokenizer: tok}, texts) do
    encodings = Enum.map(texts, fn text ->
      {:ok, enc} = Tokenizers.Tokenizer.encode(tok, text)
      enc
    end)

    {input_ids, attention_mask} = batch_encodings(encodings, 64)
    {embeddings} = Ortex.run(model, {input_ids, attention_mask})

    embeddings |> l2_normalize() |> Nx.to_list()
  end

  def embed_image(%{vision_model: model}, images) do
    # images is list of binaries or Nx tensors
    batch = images
      |> Enum.map(&preprocess_image/1)
      |> Nx.stack()

    {embeddings} = Ortex.run(model, batch)

    embeddings |> l2_normalize() |> Nx.to_list()
  end

  defp preprocess_image(image_binary) when is_binary(image_binary) do
    image_binary
    |> StbImage.read_binary!()
    |> StbImage.resize(384, 384)
    |> StbImage.to_nx()
    |> Nx.divide(255.0)
    |> normalize_imagenet()
  end

  defp normalize_imagenet(tensor) do
    mean = Nx.tensor([0.485, 0.456, 0.406]) |> Nx.reshape({1, 1, 3})
    std = Nx.tensor([0.229, 0.224, 0.225]) |> Nx.reshape({1, 1, 3})
    Nx.subtract(tensor, mean) |> Nx.divide(std)
  end

  # ... batch_encodings and l2_normalize same as BGEM3
end
```

### 12.4 Recommended Model Set (Ortex/ONNX)

| Category | Primary Model | Alt Model | Why Primary |
|----------|--------------|-----------|-------------|
| **Text-to-Image** | Z-Image-Turbo | Sana | SOTA quality, 8 steps, sub-second |
| **Image-to-Text** | Florence-2-base | moondream2 | SOTA captioning, reasonable size |
| **Embeddings** | SigLIP | BGE-M3 + CLIP | Single model for both modalities |

**Recommendations:**
- Use **Z-Image-Turbo** for T2I (primary) or **Sana** as fallback - both are SOTA and fast
- Use **Florence-2** for I2T (primary) or **moondream2** for lightweight/fast captioning
- Use **SigLIP** for embeddings - single model embeds both text and images into the same space, ideal for TDA analysis comparing trajectories

```elixir
# lib/panic_tda/models/genai/moondream.ex
defmodule PanicTda.Models.GenAI.Moondream do
  @moduledoc """
  moondream2 for fast, lightweight image captioning via ONNX.
  ~1.8B params, very efficient. Good alternative to Florence-2.
  Export: optimum-cli export onnx --model vikhyatk/moondream2
  """
  @behaviour PanicTda.Models.GenAIModel

  @models_dir Application.compile_env(:panic_tda, :models_dir, "priv/models")

  def load do
    vision_encoder = Ortex.load(Path.join(@models_dir, "moondream2/vision_encoder.onnx"))
    text_decoder = Ortex.load(Path.join(@models_dir, "moondream2/text_model.onnx"))
    {:ok, tokenizer} = Tokenizers.Tokenizer.from_pretrained("vikhyatk/moondream2")

    {:ok, %{
      vision_encoder: vision_encoder,
      text_decoder: text_decoder,
      tokenizer: tokenizer
    }}
  end

  def invoke(models, image_binary, _seed) do
    %{vision_encoder: enc, text_decoder: dec, tokenizer: tok} = models

    # Preprocess image
    image_tensor = image_binary
      |> StbImage.read_binary!()
      |> StbImage.resize(378, 378)
      |> StbImage.to_nx()
      |> Nx.divide(255.0)
      |> normalize_image()
      |> Nx.new_axis(0)

    # Encode image
    {image_features} = Ortex.run(enc, image_tensor)

    # Autoregressive decoding for caption
    caption = autoregressive_decode(dec, tok, image_features, max_tokens: 50)

    {:ok, {:text, caption}}
  end

  defp autoregressive_decode(decoder, tokenizer, image_features, opts) do
    max_tokens = Keyword.get(opts, :max_tokens, 50)
    eos_token_id = 2  # Typical EOS token

    # Start with BOS token
    tokens = [1]

    tokens = Enum.reduce_while(1..max_tokens, tokens, fn _step, acc_tokens ->
      input_ids = Nx.tensor([acc_tokens], type: :s64)
      {logits} = Ortex.run(decoder, {image_features, input_ids})

      # Get last token prediction
      next_token = logits
        |> Nx.slice_along_axis(-1, 1, axis: 1)
        |> Nx.squeeze()
        |> Nx.argmax()
        |> Nx.to_number()

      if next_token == eos_token_id do
        {:halt, acc_tokens}
      else
        {:cont, acc_tokens ++ [next_token]}
      end
    end)

    {:ok, text} = Tokenizers.Tokenizer.decode(tokenizer, tokens)
    String.trim(text)
  end

  defp normalize_image(tensor) do
    mean = Nx.tensor([0.5, 0.5, 0.5]) |> Nx.reshape({1, 1, 3})
    std = Nx.tensor([0.5, 0.5, 0.5]) |> Nx.reshape({1, 1, 3})
    Nx.subtract(tensor, mean) |> Nx.divide(std)
  end
end
```

```elixir
# lib/panic_tda/models/genai/z_image.ex
defmodule PanicTda.Models.GenAI.ZImage do
  @moduledoc """
  Z-Image-Turbo (Alibaba/Tongyi-MAI) via ONNX.
  6B params, 8 NFEs, sub-second on H800, runs on 16GB consumer GPUs.
  Export: optimum-cli export onnx --model Tongyi-MAI/Z-Image-Turbo
  """
  @behaviour PanicTda.Models.GenAIModel

  @models_dir Application.compile_env(:panic_tda, :models_dir, "priv/models")

  def load do
    # Z-Image uses single-stream DiT with unified token processing
    text_encoder = Ortex.load(Path.join(@models_dir, "z-image/text_encoder.onnx"))
    dit = Ortex.load(Path.join(@models_dir, "z-image/transformer.onnx"))
    vae_decoder = Ortex.load(Path.join(@models_dir, "z-image/vae_decoder.onnx"))

    {:ok, tokenizer} = Tokenizers.Tokenizer.from_pretrained("Tongyi-MAI/Z-Image-Turbo")

    {:ok, %{
      text_encoder: text_encoder,
      dit: dit,
      vae_decoder: vae_decoder,
      tokenizer: tokenizer
    }}
  end

  def invoke(models, prompt, seed) do
    %{text_encoder: text_enc, dit: dit, vae_decoder: vae, tokenizer: tok} = models

    # Tokenize
    {:ok, encoding} = Tokenizers.Tokenizer.encode(tok, prompt)
    input_ids = Tokenizers.Encoding.get_ids(encoding)
      |> Nx.tensor(type: :s64)
      |> Nx.reshape({1, :auto})

    # Text encoding
    {text_embeddings} = Ortex.run(text_enc, input_ids)

    # Initialize latents
    key = Nx.Random.key(if seed == -1, do: :rand.uniform(1_000_000), else: seed)
    {latents, _key} = Nx.Random.normal(key, shape: {1, 4, 128, 128}, type: :f32)

    # Z-Image uses 8 NFEs (function evaluations)
    denoised = flow_matching_sample(dit, latents, text_embeddings, steps: 8)

    # Decode
    {image} = Ortex.run(vae, denoised)

    image_binary = image
      |> Nx.multiply(0.5)
      |> Nx.add(0.5)
      |> Nx.clip(0, 1)
      |> Nx.multiply(255)
      |> Nx.as_type(:u8)
      |> Nx.squeeze()
      |> StbImage.from_nx()
      |> StbImage.to_binary(:png)

    {:ok, {:image, image_binary}}
  end

  defp flow_matching_sample(dit, latents, text_emb, opts) do
    steps = Keyword.get(opts, :steps, 8)

    Enum.reduce(0..(steps - 1), latents, fn step, current ->
      t = Nx.tensor([(step + 0.5) / steps], type: :f32)
      {velocity} = Ortex.run(dit, {current, t, text_emb})
      Nx.add(current, Nx.multiply(velocity, 1.0 / steps))
    end)
  end
end
```

### 12.5 Model Download Script

```elixir
# lib/mix/tasks/panic_tda.download_models.ex
defmodule Mix.Tasks.PanicTda.DownloadModels do
  @moduledoc "Download ONNX models from Hugging Face"
  use Mix.Task

  @models [
    {"BAAI/bge-m3", "onnx/model.onnx", "bge-m3/model.onnx"},
    {"google/siglip-base-patch16-384", "onnx/text_model.onnx", "siglip/text_model.onnx"},
    {"google/siglip-base-patch16-384", "onnx/vision_model.onnx", "siglip/vision_model.onnx"},
    # Add more as needed
  ]

  def run(_args) do
    Application.ensure_all_started(:req)
    models_dir = Application.get_env(:panic_tda, :models_dir, "priv/models")

    for {repo, file, local_path} <- @models do
      dest = Path.join(models_dir, local_path)
      File.mkdir_p!(Path.dirname(dest))

      unless File.exists?(dest) do
        url = "https://huggingface.co/#{repo}/resolve/main/#{file}"
        IO.puts("Downloading #{url}...")

        Req.get!(url, into: File.stream!(dest))
        IO.puts("  -> #{dest}")
      end
    end
  end
end
```

### 12.6 ONNX Export Commands

For models without pre-exported ONNX, use Optimum CLI:

```bash
# Install optimum
pip install optimum[exporters]

# Text-to-Image models
optimum-cli export onnx --model Tongyi-MAI/Z-Image-Turbo --task text-to-image z-image-onnx/
optimum-cli export onnx --model Efficient-Large-Model/Sana_1600M_1024px --task text-to-image sana-onnx/

# Image-to-Text models
optimum-cli export onnx --model microsoft/Florence-2-base florence-2-onnx/
optimum-cli export onnx --model vikhyatk/moondream2 moondream2-onnx/

# Embedding models
optimum-cli export onnx --model google/siglip-base-patch16-384 siglip-onnx/
optimum-cli export onnx --model BAAI/bge-m3 bge-m3-onnx/
```

### 12.7 GPU Parallelization Strategy (RTX 6000 Ada, 48GB VRAM)

With 48GB VRAM, you have significant headroom for parallel execution.

#### Memory Estimates (FP16)

| Model | VRAM (FP16) | VRAM (FP32) | Notes |
|-------|-------------|-------------|-------|
| Z-Image-Turbo (6B) | ~12GB | ~24GB | Largest model |
| Sana (1.6B) | ~4GB | ~8GB | More memory efficient |
| Florence-2-base | ~1.5GB | ~3GB | Small |
| moondream2 (1.8B) | ~4GB | ~8GB | Efficient |
| SigLIP | ~0.5GB | ~1GB | Very small |
| BGE-M3 | ~1GB | ~2GB | Small |

**With Z-Image-Turbo + Florence-2 + SigLIP loaded simultaneously: ~14GB FP16**

This leaves **~34GB free** for:
- Batch processing (multiple images in flight)
- Running multiple experiment pipelines in parallel
- KV cache for autoregressive decoding

#### Parallelization Strategies

```elixir
# lib/panic_tda/models/gpu_scheduler.ex
defmodule PanicTda.Models.GPUScheduler do
  @moduledoc """
  Manages GPU resources for parallel model execution.
  Supports multiple concurrent experiments on RTX 6000 Ada (48GB).
  """
  use GenServer

  # Conservative estimates - can be tuned based on actual usage
  @total_vram_gb 48
  @model_vram %{
    "ZImage" => 12,
    "Sana" => 4,
    "Florence2" => 2,
    "Moondream" => 4,
    "SigLIP" => 1,
    "BGEM3" => 1
  }

  defstruct [:loaded_models, :available_vram, :model_refs]

  def start_link(_opts) do
    GenServer.start_link(__MODULE__, %{}, name: __MODULE__)
  end

  def init(_) do
    state = %__MODULE__{
      loaded_models: %{},
      available_vram: @total_vram_gb,
      model_refs: %{}
    }
    {:ok, state}
  end

  @doc """
  Request access to a model. Loads if not already loaded.
  Blocks if insufficient VRAM until space is available.
  """
  def acquire(model_name) do
    GenServer.call(__MODULE__, {:acquire, model_name}, :infinity)
  end

  def release(model_name) do
    GenServer.cast(__MODULE__, {:release, model_name})
  end

  # Keep hot models loaded, evict LRU when needed
  def handle_call({:acquire, model_name}, _from, state) do
    vram_needed = Map.get(@model_vram, model_name, 2)

    state = maybe_evict_models(state, vram_needed)

    case Map.get(state.loaded_models, model_name) do
      nil ->
        # Load the model
        {:ok, model_ref} = load_model(model_name)
        new_state = %{state |
          loaded_models: Map.put(state.loaded_models, model_name, {model_ref, 1}),
          available_vram: state.available_vram - vram_needed,
          model_refs: Map.put(state.model_refs, model_name, model_ref)
        }
        {:reply, {:ok, model_ref}, new_state}

      {model_ref, count} ->
        # Increment reference count
        new_state = %{state |
          loaded_models: Map.put(state.loaded_models, model_name, {model_ref, count + 1})
        }
        {:reply, {:ok, model_ref}, new_state}
    end
  end

  defp load_model(model_name) do
    module = Module.concat([PanicTda.Models.GenAI, model_name])
    module.load()
  end

  defp maybe_evict_models(state, needed) do
    if state.available_vram >= needed do
      state
    else
      # Evict models with 0 reference count, LRU first
      # Implementation: track last_used timestamp, evict oldest
      state
    end
  end
end
```

#### Parallel Experiment Execution

```elixir
# lib/panic_tda/reactors/parallel_experiments.ex
defmodule PanicTda.Reactors.ParallelExperiments do
  @moduledoc """
  Run multiple experiments in parallel, sharing GPU resources.
  With 48GB VRAM, can typically run 2-3 experiments simultaneously.
  """

  def run_parallel(experiment_ids, opts \\ []) do
    max_concurrent = Keyword.get(opts, :max_concurrent, 3)

    experiment_ids
    |> Task.async_stream(
      fn exp_id ->
        Reactor.run(PanicTda.Reactors.PerformExperiment, %{experiment_id: exp_id})
      end,
      max_concurrency: max_concurrent,
      timeout: :infinity,
      ordered: false
    )
    |> Enum.to_list()
  end
end
```

#### Pipeline Parallelism Within a Run

The three-stage pipeline can overlap:

```elixir
defmodule PanicTda.Pipeline.Overlapped do
  @moduledoc """
  Overlapped pipeline execution - while run N is computing embeddings,
  run N+1 can start generating images.

  Timeline with 48GB VRAM:
  ┌─────────────────────────────────────────────────────────────────┐
  │ Run 1: [T2I ████████] [I2T ███] [Embed ██] [TDA █]             │
  │ Run 2:        [T2I ████████] [I2T ███] [Embed ██] [TDA █]      │
  │ Run 3:               [T2I ████████] [I2T ███] [Embed ██] [TDA] │
  └─────────────────────────────────────────────────────────────────┘
  """

  def run_with_overlap(run_ids, opts \\ []) do
    overlap = Keyword.get(opts, :overlap, 2)  # How many runs can overlap

    run_ids
    |> Stream.chunk_every(overlap, 1, :discard)
    |> Stream.map(fn chunk ->
      # Start all runs in chunk concurrently
      tasks = Enum.map(chunk, fn run_id ->
        Task.async(fn -> execute_run_pipeline(run_id) end)
      end)

      # Wait for first to complete before starting next chunk
      [first | _rest] = tasks
      Task.await(first, :infinity)
    end)
    |> Stream.run()
  end
end
```

#### Batch Inference for Embeddings

The embedding stage can process many items in parallel:

```elixir
defmodule PanicTda.Models.BatchEmbedding do
  @moduledoc """
  With 48GB VRAM and SigLIP (~0.5GB), we can process large batches.
  """

  @batch_size 128  # Can go higher with 48GB

  def embed_all(invocations, model) do
    invocations
    |> Stream.chunk_every(@batch_size)
    |> Stream.map(fn batch ->
      contents = Enum.map(batch, &extract_content/1)
      vectors = PanicTda.Models.Embeddings.SigLIP.embed(model, contents)
      Enum.zip(batch, vectors)
    end)
    |> Enum.to_list()
    |> List.flatten()
  end
end
```

#### Ortex CUDA Configuration

```elixir
# config/runtime.exs
config :ortex,
  # Use CUDA execution provider
  execution_providers: [:cuda, :cpu],

  # GPU device ID (0 for single GPU)
  cuda_device_id: 0,

  # Memory arena configuration for 48GB
  cuda_arena_extend_strategy: :same_as_alloc,
  cuda_gpu_mem_limit: 45 * 1024 * 1024 * 1024,  # 45GB limit (leave 3GB headroom)

  # Enable memory pattern optimization
  cuda_enable_cuda_graph: true,

  # Thread configuration
  intra_op_num_threads: 8,
  inter_op_num_threads: 4
```

#### Expected Throughput

With RTX 6000 Ada (48GB) and optimal configuration:

| Operation | Single | Parallel (3 experiments) |
|-----------|--------|--------------------------|
| T2I (Z-Image, 8 steps) | ~0.5s/image | ~1.2s/image (memory bound) |
| T2I (Sana, 20 steps) | ~0.8s/image | ~1.5s/image |
| I2T (Florence-2) | ~0.1s/image | ~0.15s/image |
| Embeddings (SigLIP, batch 128) | ~0.05s/item | ~0.06s/item |
| Full run (10 invocations) | ~6s | ~8s |

**Bottom line:** Yes, you can run multiple experiments in parallel. With 48GB VRAM:
- 2-3 concurrent experiments work well
- Main bottleneck is T2I model (Z-Image at 12GB)
- Using Sana (4GB) instead allows 4-5 concurrent experiments
- Embedding stage is highly parallelizable (small models, big batches)

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

Porting PANIC-TDA to Elixir is **highly feasible** with a pure Elixir approach using Ortex/ONNX models:

### Recommended Stack

| Layer | Technology |
|-------|------------|
| Data Model | Ash 3.x + AshSqlite |
| Database | SQLite (same as Python) |
| Orchestration | Reactor |
| T2I Model | Z-Image-Turbo (Ortex/ONNX) |
| I2T Model | Florence-2 (Ortex/ONNX) |
| Embeddings | SigLIP (Ortex/ONNX, multimodal) |
| TDA | Rustler NIF wrapping ripser |
| Export | FFmpeg (unchanged) + Image library |

### Benefits of Elixir Port

- **No Python dependency** - Ortex runs ONNX models natively
- **Better concurrency** via OTP supervision trees
- **Cleaner orchestration** via Reactor's declarative steps
- **Type safety** with Ash validations and constraints
- **Distributed execution** across BEAM nodes (future)
- **Hot code reloading** for development
- **Modern models** - Z-Image-Turbo, Florence-2, SigLIP are all SOTA

### Migration Path

1. Set up Ash resources (map 1:1 from SQLModel)
2. Export ONNX models (Z-Image-Turbo, Florence-2, SigLIP)
3. Implement Ortex model wrappers with tokenizer handling
4. Build Reactor pipelines (direct translation of Ray workflow)
5. Add Rustler NIF for TDA (or Python port initially)
6. Port export functionality (straightforward)
7. Run migration script for existing data

### Estimated Effort (Pure Elixir)

| Component | Effort |
|-----------|--------|
| Ash data model | 1 week |
| Ortex model wrappers | 1-2 weeks |
| Reactor pipelines | 1-2 weeks |
| TDA (Rustler) | 1 week |
| Export | 3-4 days |
| CLI | 2-3 days |
| Migration script | 2-3 days |
| **Total** | **5-7 weeks** |
