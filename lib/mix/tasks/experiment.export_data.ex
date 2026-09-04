defmodule Mix.Tasks.Experiment.ExportData do
  @shortdoc "Export experiment data (everything except images) to parquet files"

  @moduledoc """
  Dumps one or more experiments to parquet files — everything except the image
  bytes. One file per table (`experiments`, `runs`, `invocations`,
  `embeddings`, `persistence_diagrams`, `clustering_results`,
  `embedding_clusters`), suitable for loading back into
  polars/pandas for analysis.

      $ mix experiment.export_data <experiment-id-prefix> [<id-prefix> ...] [--output dir] [--embedding-model NAME] [--embed-prompts]

  Array/nested attributes (`networks`, `network`, `prompts`) and the
  persistence-diagram payload (`diagram_data`) are written as JSON strings;
  embedding vectors become a `list[f32]` column; timestamps are ISO 8601
  strings.

  ## Options

    - `--output` — output directory (default: `./<short-id>_parquet`, or
      `./panic_tda_parquet` when more than one experiment is given)
    - `--embedding-model` — only include this embedding model (repeatable);
      filters `embeddings`, `persistence_diagrams`, `clustering_results`,
      `embedding_clusters`. Defaults to all models.
    - `--embed-prompts` — also embed each run's `initial_prompt` and emit it as a
      synthetic `sequence_number == -1` row (`t_0`) in the `invocations` and
      `embeddings` tables. Requires the GPU/Python interpreter (loads the text
      embedding model). Off by default.
  """

  use Mix.Task

  @impl Mix.Task
  def run(args) do
    {opts, prefixes, _} =
      OptionParser.parse(args,
        strict: [output: :string, embedding_model: :keep, embed_prompts: :boolean],
        aliases: [o: :output]
      )

    if prefixes == [] do
      Mix.raise(
        "Usage: mix experiment.export_data <experiment-id-prefix> [...] [--output dir] [--embedding-model NAME]"
      )
    end

    Mix.Task.run("ecto.create", ["--quiet"])
    Mix.Task.run("ecto.migrate", ["--quiet"])
    Mix.Task.run("app.start")

    experiments = Enum.map(prefixes, &find_experiment/1) |> Enum.uniq_by(& &1.id)
    experiment_ids = Enum.map(experiments, & &1.id)

    output_dir =
      Keyword.get(opts, :output) ||
        case experiments do
          [experiment] -> "./#{short_id(experiment.id)}_parquet"
          _ -> "./panic_tda_parquet"
        end

    models =
      case Keyword.get_values(opts, :embedding_model) do
        [] -> nil
        list -> list
      end

    embed_prompts? = Keyword.get(opts, :embed_prompts, false)

    Mix.shell().info(
      "Exporting #{length(experiments)} experiment(s) to #{output_dir}/" <>
        if(models, do: " (embedding models: #{Enum.join(models, ", ")})", else: "") <>
        if(embed_prompts?, do: " [embedding initial prompts]", else: "")
    )

    {:ok, results} =
      PanicTda.DataExport.export(experiment_ids, output_dir,
        embedding_models: models,
        embed_prompts: embed_prompts?
      )

    for {table, path, rows} <- results do
      size_mb = File.stat!(path).size / 1024 / 1024
      Mix.shell().info("  #{pad(table)} #{format_int(rows)} rows  #{format_mb(size_mb)} MB")
    end

    Mix.shell().info("Done. Parquet files in #{output_dir}/")
  end

  defp find_experiment(id_prefix) do
    PanicTda.list_experiments!()
    |> Enum.find(&String.starts_with?(&1.id, id_prefix)) ||
      Mix.raise("No experiment found matching '#{id_prefix}'")
  end

  defp short_id(id), do: String.slice(id, 0, 8)

  defp pad(table), do: String.pad_trailing(to_string(table), 22)

  defp format_int(n) do
    n
    |> Integer.to_string()
    |> String.reverse()
    |> String.replace(~r/(\d{3})(?=\d)/, "\\1,")
    |> String.reverse()
    |> String.pad_leading(9)
  end

  defp format_mb(mb), do: :erlang.float_to_binary(mb, decimals: 1) |> String.pad_leading(7)
end
