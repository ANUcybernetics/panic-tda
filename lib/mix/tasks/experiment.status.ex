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

    Mix.shell().info("""
    Experiment: #{experiment.id}
    Status:     #{status(experiment)}

    Config:
      Networks:         #{inspect(experiment.networks)}
      Num runs:         #{experiment.num_runs}
      Prompts:          #{inspect(experiment.prompts)}
      Embedding models: #{inspect(experiment.embedding_models)}
      Max length:       #{experiment.max_length}

    Timestamps:
      Created:   #{format_time(experiment.inserted_at)}
      Started:   #{format_time(experiment.started_at)}
      Completed: #{format_time(experiment.completed_at)}

    Progress:
      Runs:                 #{counts.runs}
      Embeddings:           #{counts.embeddings}
      Persistence diagrams: #{counts.persistence_diagrams}
      Clustering results:   #{counts.clustering_results}
    """)
  end

  def run(_args) do
    Mix.raise("Usage: mix experiment.status <experiment-id>")
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
      clustering_results:
        PanicTda.ClusteringResult
        |> Ash.Query.filter(experiment_id == ^experiment.id)
        |> Ash.count!()
    }
  end

  defp status(%{completed_at: %DateTime{}}), do: "completed"
  defp status(%{started_at: %DateTime{}}), do: "running"
  defp status(_), do: "pending"

  defp format_time(nil), do: "-"
  defp format_time(dt), do: Calendar.strftime(dt, "%Y-%m-%d %H:%M:%S")
end
