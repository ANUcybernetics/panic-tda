defmodule Mix.Tasks.Experiment.Status do
  @shortdoc "Show detailed status of a PANIC-TDA experiment"

  @moduledoc """
  Shows detailed status for a specific experiment including config summary,
  timestamps, and stage progress counts.

      $ mix experiment.status <experiment-id>
  """

  use Mix.Task

  require Ash.Query

  @impl Mix.Task
  def run([id_prefix]) do
    Mix.Task.run("ecto.create", ["--quiet"])
    Mix.Task.run("ecto.migrate", ["--quiet"])
    Mix.Task.run("app.start")

    experiment = find_experiment(id_prefix)
    counts = experiment_counts(experiment)
    activity = last_activity(experiment)

    Mix.shell().info("""
    Experiment: #{experiment.id}
    Status:     #{status(experiment, activity)}

    Config:
      Networks:         #{inspect(experiment.networks)}
      Num runs:         #{experiment.num_runs}
      Prompts:          #{inspect(experiment.prompts)}
      Embedding models: #{inspect(experiment.embedding_models)}
      Max length:       #{experiment.max_length}
      I2T token ceiling: #{experiment.i2t_max_new_tokens || "model default"}

    Timestamps:
      Created:       #{format_time(experiment.inserted_at)}
      Started:       #{format_time(experiment.started_at)}
      Completed:     #{format_time(experiment.completed_at)}
      Last activity: #{format_time(activity)}

    Progress:
      Runs:                 #{counts.runs}
      Embeddings:           #{counts.embeddings}
      Persistence diagrams: #{counts.persistence_diagrams}
      Clustering layers:    #{format_layers(counts.clustering_layers)}

    Caption truncation (share not ending in terminal punctuation):
    #{format_truncation(caption_truncation(experiment))}
    """)
  end

  def run(_args) do
    Mix.raise("Usage: mix experiment.status <experiment-id>")
  end

  # A caption that hit its generation ceiling is cut off mid-sentence, so the
  # share without terminal punctuation is a direct data-quality check.
  defp caption_truncation(experiment) do
    PanicTda.Invocation
    |> Ash.Query.filter(run.experiment_id == ^experiment.id and type == :text)
    |> Ash.Query.select([:model, :output_text])
    |> Ash.read!()
    |> Enum.group_by(& &1.model, & &1.output_text)
    |> Enum.map(fn {model, captions} ->
      truncated = Enum.count(captions, &(not String.match?(String.trim(&1), ~r/[.!?"”)]$/u)))
      max_words = captions |> Enum.map(&length(String.split(&1))) |> Enum.max()
      {model, length(captions), truncated, max_words}
    end)
    |> Enum.sort()
  end

  defp format_truncation([]), do: "  (no captions yet)"

  defp format_truncation(rows) do
    Enum.map_join(rows, "\n", fn {model, n, truncated, max_words} ->
      pct = Float.round(100 * truncated / n, 1)
      "  #{model}: #{truncated}/#{n} (#{pct}%), longest #{max_words} words"
    end)
  end

  defp format_layers(layers) when layers == %{}, do: "(none)"

  defp format_layers(layers) do
    layers
    |> Enum.sort_by(fn {model, _} -> model end)
    |> Enum.map_join(", ", fn {model, count} -> "#{model}=#{count}" end)
  end

  defp find_experiment(id_prefix) do
    experiments = PanicTda.list_experiments!()

    Enum.find(experiments, fn e -> String.starts_with?(e.id, id_prefix) end) ||
      Mix.raise("No experiment found matching '#{id_prefix}'")
  end

  defp experiment_counts(experiment) do
    %{
      runs:
        PanicTda.Run
        |> Ash.Query.filter(experiment_id == ^experiment.id)
        |> Ash.count!(),
      embeddings:
        PanicTda.Embedding
        |> Ash.Query.filter(invocation.run.experiment_id == ^experiment.id)
        |> Ash.count!(),
      persistence_diagrams:
        PanicTda.PersistenceDiagram
        |> Ash.Query.filter(run.experiment_id == ^experiment.id)
        |> Ash.count!(),
      clustering_layers: clustering_layers(experiment)
    }
  end

  defp clustering_layers(experiment) do
    PanicTda.ClusteringResult
    |> Ash.Query.filter(embedding_model in ^experiment.embedding_models)
    |> Ash.read!()
    |> Enum.group_by(& &1.embedding_model)
    |> Map.new(fn {model, results} -> {model, length(results)} end)
  end

  defp last_activity(experiment) do
    case PanicTda.Invocation
         |> Ash.Query.filter(run.experiment_id == ^experiment.id)
         |> Ash.Query.sort(completed_at: :desc)
         |> Ash.Query.limit(1)
         |> Ash.read_one!() do
      nil -> experiment.started_at
      inv -> inv.completed_at || inv.started_at
    end
  end

  defp status(%{completed_at: %DateTime{}}, _last_activity), do: "completed"

  defp status(%{started_at: %DateTime{}}, last_activity) do
    if stalled?(last_activity), do: "stalled", else: "running"
  end

  defp status(_, _last_activity), do: "pending"

  defp stalled?(nil), do: false

  defp stalled?(last_activity) do
    DateTime.diff(DateTime.utc_now(), last_activity, :second) > 3600
  end

  defp format_time(nil), do: "-"
  defp format_time(dt), do: Calendar.strftime(dt, "%Y-%m-%d %H:%M:%S")
end
