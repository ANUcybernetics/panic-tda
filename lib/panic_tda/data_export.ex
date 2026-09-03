defmodule PanicTda.DataExport do
  @moduledoc """
  Exports experiment data — everything except the image bytes — to parquet
  files via Explorer.

  One file is written per table (`experiments`, `runs`, `invocations`,
  `embeddings`, `persistence_diagrams`, `clustering_results`,
  `embedding_clusters`, `lyapunov_results`). Array and nested attributes
  (`networks`, `network`, `prompts`, `embedding_models`) and the persistence
  diagram payload (`diagram_data`) are serialised to JSON strings so the output
  is portable; embedding vectors become a `list[f32]` column. Timestamps are
  ISO 8601 strings.

  Clustering is global — a clustering run pools every embedding for a model
  across all experiments — so the two cluster tables are scoped rather than
  copied wholesale: `embedding_clusters` covers only the embeddings belonging to
  the exported experiments, and `clustering_results` only the runs for the
  embedding models those experiments used. Two consequences for downstream
  analysis: a cluster's `medoid_embedding_id` may name an embedding that lives
  in an experiment outside the export (treat it as an opaque cluster label, not
  a joinable key), and cluster sizes computed from the export are counts within
  the export, not within the clustering run that produced them. A null
  `medoid_embedding_id` marks an outlier.

  The output is intended to be read back with polars/pandas for downstream
  analysis; see `analysis/load_with_polars.py` for an example wide join.
  """

  require Ash.Query

  alias PanicTda.Models.Embeddings

  @doc """
  Export `experiment_ids` to parquet files under `output_dir`.

  Options:

    * `:embedding_models` — list of embedding-model names to include. Filters
      both the `embeddings` and `persistence_diagrams` tables. Defaults to all
      models.

    * `:embed_prompts` — when `true`, embed each run's `initial_prompt` and emit
      it as a synthetic `sequence_number == -1` row in both the `invocations`
      and `embeddings` tables (the input text state, `t_0`). This lets analysis
      distinguish drift from the original prompt versus drift from the first
      generated text state. Requires the Python interpreter (loads the text
      embedding model). Defaults to `false`.

    * `:env` — an existing Snex env to reuse for `:embed_prompts`. When absent, a
      fresh interpreter is started.

  Returns `{:ok, [{table, path, row_count}]}`.
  """
  def export(experiment_ids, output_dir, opts \\ []) when is_list(experiment_ids) do
    File.mkdir_p!(output_dir)
    models = Keyword.get(opts, :embedding_models)

    extra_rows =
      if Keyword.get(opts, :embed_prompts, false) do
        prompt_synthetic_rows(experiment_ids, models, opts)
      else
        %{}
      end

    results =
      Enum.map(specs(experiment_ids, models), fn {table, records, columns} ->
        records = records ++ Map.get(extra_rows, table, [])
        df = build_frame(records, columns)
        path = Path.join(output_dir, "#{table}.parquet")
        Explorer.DataFrame.to_parquet!(df, path, compression: {:zstd, nil})
        {table, path, Explorer.DataFrame.n_rows(df)}
      end)

    {:ok, results}
  end

  # Build synthetic `t_0` invocation + embedding rows for each run's initial
  # prompt. One invocation row per run (sequence -1) holds the prompt text; one
  # embedding row per (run, text embedding model) holds its vector. Ids are
  # prefixed `prompt-` so they never collide with real uuid_v7 ids and are easy
  # to filter out. Returns `%{invocations: [...], embeddings: [...]}`.
  defp prompt_synthetic_rows(experiment_ids, models, opts) do
    runs = read_runs(experiment_ids)
    text_models = prompt_text_models(experiment_ids, models)

    if runs == [] or text_models == [] do
      %{}
    else
      env = ensure_env(opts)
      prompts = runs |> Enum.map(& &1.initial_prompt) |> Enum.uniq()

      embedding_rows =
        Enum.flat_map(text_models, fn model ->
          {:ok, vectors} = Embeddings.embed(env, model, prompts)
          vector_by_prompt = Map.new(Enum.zip(prompts, vectors))

          Enum.map(runs, fn run ->
            binary = Map.fetch!(vector_by_prompt, run.initial_prompt)
            prompt_embedding_row(run, model, binary)
          end)
        end)

      %{
        invocations: Enum.map(runs, &prompt_invocation_row/1),
        embeddings: embedding_rows
      }
    end
  end

  # The embedding spaces to place prompts in: the requested `--embedding-model`
  # filter if given, else every model the experiments actually used. Retired
  # embedders named by historical experiments are skipped.
  defp prompt_text_models(experiment_ids, models) do
    candidates =
      models || experiment_ids |> read_experiments() |> Enum.flat_map(& &1.embedding_models)

    candidates
    |> Enum.uniq()
    |> Enum.filter(&Embeddings.registered?/1)
  end

  defp prompt_invocation_row(run) do
    %{
      id: "prompt-" <> run.id,
      run_id: run.id,
      input_invocation_id: nil,
      sequence_number: -1,
      type: :text,
      model: nil,
      output_text: run.initial_prompt,
      started_at: nil,
      completed_at: nil,
      inserted_at: nil,
      updated_at: nil
    }
  end

  defp prompt_embedding_row(run, model, binary) do
    %{
      id: "prompt-" <> run.id <> "-" <> model,
      invocation_id: "prompt-" <> run.id,
      embedding_model: model,
      vector: Nx.from_binary(binary, :f32),
      started_at: nil,
      completed_at: nil,
      inserted_at: nil,
      updated_at: nil
    }
  end

  defp ensure_env(opts) do
    case Keyword.get(opts, :env) do
      nil ->
        {:ok, interpreter} = PanicTda.Models.PythonInterpreter.start_link()
        {:ok, env} = Snex.make_env(interpreter)
        env

      env ->
        env
    end
  end

  # Each spec is `{table, records, columns}` where `columns` is a list of
  # `{name, extractor, dtype}`.
  defp specs(experiment_ids, models) do
    [
      {:experiments, read_experiments(experiment_ids),
       [
         {"id", & &1.id, :string},
         {"networks", &json(&1.networks), :string},
         {"num_runs", & &1.num_runs, {:s, 64}},
         {"prompts", &json(&1.prompts), :string},
         {"embedding_models", &json(&1.embedding_models), :string},
         {"max_length", & &1.max_length, {:s, 64}},
         {"i2t_max_new_tokens", & &1.i2t_max_new_tokens, {:s, 64}},
         {"started_at", &iso(&1.started_at), :string},
         {"completed_at", &iso(&1.completed_at), :string},
         {"inserted_at", &iso(&1.inserted_at), :string},
         {"updated_at", &iso(&1.updated_at), :string}
       ]},
      {:runs, read_runs(experiment_ids),
       [
         {"id", & &1.id, :string},
         {"experiment_id", & &1.experiment_id, :string},
         {"network", &json(&1.network), :string},
         {"run_number", & &1.run_number, {:s, 64}},
         {"max_length", & &1.max_length, {:s, 64}},
         {"initial_prompt", & &1.initial_prompt, :string},
         {"inserted_at", &iso(&1.inserted_at), :string},
         {"updated_at", &iso(&1.updated_at), :string}
       ]},
      {:invocations, read_invocations(experiment_ids),
       [
         {"id", & &1.id, :string},
         {"run_id", & &1.run_id, :string},
         {"input_invocation_id", & &1.input_invocation_id, :string},
         {"sequence_number", & &1.sequence_number, {:s, 64}},
         {"type", &to_string(&1.type), :string},
         {"model", & &1.model, :string},
         {"output_text", & &1.output_text, :string},
         {"started_at", &iso(&1.started_at), :string},
         {"completed_at", &iso(&1.completed_at), :string},
         {"inserted_at", &iso(&1.inserted_at), :string},
         {"updated_at", &iso(&1.updated_at), :string}
       ]},
      {:embeddings, read_embeddings(experiment_ids, models),
       [
         {"id", & &1.id, :string},
         {"invocation_id", & &1.invocation_id, :string},
         {"embedding_model", & &1.embedding_model, :string},
         {"vector", &vector_to_list(&1.vector), {:list, {:f, 32}}},
         {"started_at", &iso(&1.started_at), :string},
         {"completed_at", &iso(&1.completed_at), :string},
         {"inserted_at", &iso(&1.inserted_at), :string},
         {"updated_at", &iso(&1.updated_at), :string}
       ]},
      {:persistence_diagrams, read_persistence_diagrams(experiment_ids, models),
       [
         {"id", & &1.id, :string},
         {"run_id", & &1.run_id, :string},
         {"embedding_model", & &1.embedding_model, :string},
         {"diagram_data", &json(&1.diagram_data), :string},
         {"started_at", &iso(&1.started_at), :string},
         {"completed_at", &iso(&1.completed_at), :string},
         {"inserted_at", &iso(&1.inserted_at), :string},
         {"updated_at", &iso(&1.updated_at), :string}
       ]},
      {:clustering_results, read_clustering_results(experiment_ids, models),
       [
         {"id", & &1.id, :string},
         {"embedding_model", & &1.embedding_model, :string},
         {"algorithm", & &1.algorithm, :string},
         {"parameters", &json(&1.parameters), :string},
         {"layer", & &1.layer, {:s, 64}},
         {"started_at", &iso(&1.started_at), :string},
         {"completed_at", &iso(&1.completed_at), :string},
         {"inserted_at", &iso(&1.inserted_at), :string},
         {"updated_at", &iso(&1.updated_at), :string}
       ]},
      {:embedding_clusters, read_embedding_clusters(experiment_ids, models),
       [
         {"id", & &1.id, :string},
         {"embedding_id", & &1.embedding_id, :string},
         {"clustering_result_id", & &1.clustering_result_id, :string},
         {"medoid_embedding_id", & &1.medoid_embedding_id, :string},
         {"inserted_at", &iso(&1.inserted_at), :string},
         {"updated_at", &iso(&1.updated_at), :string}
       ]},
      {:lyapunov_results, read_lyapunov_results(experiment_ids, models),
       [
         {"id", & &1.id, :string},
         {"experiment_id", & &1.experiment_id, :string},
         {"embedding_model", & &1.embedding_model, :string},
         {"network", &json(&1.network), :string},
         {"prompt", & &1.prompt, :string},
         {"exponent", &lyapunov_field(&1, :exponent), {:f, 64}},
         {"r_squared", &lyapunov_field(&1, :r_squared), {:f, 64}},
         {"num_pairs", &lyapunov_field(&1, :num_pairs), {:s, 64}},
         {"num_timesteps", &lyapunov_field(&1, :num_timesteps), {:s, 64}},
         {"divergence_curve", &lyapunov_field(&1, :divergence_curve), {:list, {:f, 64}}},
         {"started_at", &iso(&1.started_at), :string},
         {"completed_at", &iso(&1.completed_at), :string},
         {"inserted_at", &iso(&1.inserted_at), :string},
         {"updated_at", &iso(&1.updated_at), :string}
       ]}
    ]
  end

  defp read_experiments(experiment_ids) do
    PanicTda.Experiment
    |> Ash.Query.filter(id in ^experiment_ids)
    |> Ash.read!()
  end

  defp read_runs(experiment_ids) do
    PanicTda.Run
    |> Ash.Query.filter(experiment_id in ^experiment_ids)
    |> Ash.read!()
  end

  defp read_invocations(experiment_ids) do
    PanicTda.Invocation
    |> Ash.Query.filter(run.experiment_id in ^experiment_ids)
    |> Ash.Query.select([
      :id,
      :run_id,
      :input_invocation_id,
      :sequence_number,
      :type,
      :model,
      :output_text,
      :started_at,
      :completed_at,
      :inserted_at,
      :updated_at
    ])
    |> Ash.Query.sort([:run_id, :sequence_number])
    |> Ash.read!()
  end

  defp read_embeddings(experiment_ids, models) do
    PanicTda.Embedding
    |> Ash.Query.filter(invocation.run.experiment_id in ^experiment_ids)
    |> filter_models(models)
    |> Ash.read!()
  end

  defp read_persistence_diagrams(experiment_ids, models) do
    PanicTda.PersistenceDiagram
    |> Ash.Query.filter(run.experiment_id in ^experiment_ids)
    |> filter_models(models)
    |> Ash.read!()
  end

  # Clustering runs are global, so there is no experiment to filter on — scope
  # instead to the embedding models the exported experiments actually used.
  defp read_clustering_results(experiment_ids, models) do
    model_names = models || exported_embedding_models(experiment_ids)

    PanicTda.ClusteringResult
    |> Ash.Query.filter(embedding_model in ^model_names)
    |> Ash.read!()
  end

  defp read_embedding_clusters(experiment_ids, models) do
    PanicTda.EmbeddingCluster
    |> Ash.Query.filter(embedding.invocation.run.experiment_id in ^experiment_ids)
    |> filter_cluster_models(models)
    |> Ash.read!()
  end

  defp read_lyapunov_results(experiment_ids, models) do
    PanicTda.LyapunovResult
    |> Ash.Query.filter(experiment_id in ^experiment_ids)
    |> filter_models(models)
    |> Ash.read!()
  end

  defp exported_embedding_models(experiment_ids) do
    experiment_ids
    |> read_experiments()
    |> Enum.flat_map(& &1.embedding_models)
    |> Enum.uniq()
  end

  defp filter_models(query, nil), do: query

  defp filter_models(query, models) when is_list(models),
    do: Ash.Query.filter(query, embedding_model in ^models)

  # `embedding_clusters` has no `embedding_model` of its own; it inherits one
  # from the clustering run it belongs to.
  defp filter_cluster_models(query, nil), do: query

  defp filter_cluster_models(query, models) when is_list(models),
    do: Ash.Query.filter(query, clustering_result.embedding_model in ^models)

  defp build_frame(records, columns) do
    data =
      Map.new(columns, fn {name, extractor, _dtype} ->
        {name, Enum.map(records, extractor)}
      end)

    dtypes = Enum.map(columns, fn {name, _extractor, dtype} -> {name, dtype} end)

    Explorer.DataFrame.new(data, dtypes: dtypes)
  end

  defp iso(nil), do: nil
  defp iso(%DateTime{} = dt), do: DateTime.to_iso8601(dt)

  defp json(nil), do: nil
  defp json(term), do: Jason.encode!(term)

  defp vector_to_list(nil), do: nil
  defp vector_to_list(%Nx.Tensor{} = tensor), do: Nx.to_flat_list(tensor)

  defp lyapunov_field(%{lyapunov_data: nil}, _key), do: nil
  defp lyapunov_field(%{lyapunov_data: data}, key), do: Map.get(data, key)
end
