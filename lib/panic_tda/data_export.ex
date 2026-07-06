defmodule PanicTda.DataExport do
  @moduledoc """
  Exports experiment data — everything except the image bytes — to parquet
  files via Explorer.

  One file is written per table (`experiments`, `runs`, `invocations`,
  `embeddings`, `persistence_diagrams`). Array and nested attributes
  (`networks`, `network`, `prompts`, `embedding_models`) and the persistence
  diagram payload (`diagram_data`) are serialised to JSON strings so the output
  is portable; embedding vectors become a `list[f32]` column. Timestamps are
  ISO 8601 strings.

  The output is intended to be read back with polars/pandas for downstream
  analysis; see `db/load_with_polars.py` for an example wide join.
  """

  require Ash.Query

  @doc """
  Export `experiment_ids` to parquet files under `output_dir`.

  Options:

    * `:embedding_models` — list of embedding-model names to include. Filters
      both the `embeddings` and `persistence_diagrams` tables. Defaults to all
      models.

  Returns `{:ok, [{table, path, row_count}]}`.
  """
  def export(experiment_ids, output_dir, opts \\ []) when is_list(experiment_ids) do
    File.mkdir_p!(output_dir)
    models = Keyword.get(opts, :embedding_models)

    results =
      Enum.map(specs(experiment_ids, models), fn {table, records, columns} ->
        df = build_frame(records, columns)
        path = Path.join(output_dir, "#{table}.parquet")
        Explorer.DataFrame.to_parquet!(df, path, compression: {:zstd, nil})
        {table, path, Explorer.DataFrame.n_rows(df)}
      end)

    {:ok, results}
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

  defp filter_models(query, nil), do: query

  defp filter_models(query, models) when is_list(models),
    do: Ash.Query.filter(query, embedding_model in ^models)

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
end
