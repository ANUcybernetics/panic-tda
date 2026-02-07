defmodule Mix.Tasks.Experiment.Resume do
  @shortdoc "Resume an interrupted PANIC-TDA experiment"

  @moduledoc """
  Resumes a previously started but incomplete experiment.

      $ mix experiment.resume <experiment-id-prefix>

  The experiment must have been started (has `started_at`) but not completed
  (no `completed_at`). The task picks up where the experiment left off:
  missing runs are created, partial runs are continued, missing embeddings
  and persistence diagrams are computed, and clustering is recomputed.
  """

  use Mix.Task

  require Ash.Query

  @impl Mix.Task
  def run([id_prefix]) do
    setup_db()
    Mix.Task.run("app.start")

    experiment = find_experiment(id_prefix)
    Mix.shell().info("Resuming experiment #{short_id(experiment.id)}")

    case PanicTda.Engine.resume_experiment(experiment.id) do
      {:ok, experiment} ->
        print_summary(experiment)

      {:error, :not_started} ->
        Mix.raise("Experiment #{short_id(experiment.id)} has not been started yet")

      {:error, :already_completed} ->
        Mix.raise("Experiment #{short_id(experiment.id)} is already completed")
    end
  end

  def run(_args) do
    Mix.raise("Usage: mix experiment.resume <experiment-id-prefix>")
  end

  defp setup_db do
    Mix.Task.run("ecto.create", ["--quiet"])
    Mix.Task.run("ecto.migrate", ["--quiet"])
  end

  defp find_experiment(id_prefix) do
    experiments = PanicTda.list_experiments!()

    Enum.find(experiments, fn e -> String.starts_with?(e.id, id_prefix) end) ||
      Mix.raise("No experiment found matching '#{id_prefix}'")
  end

  defp print_summary(experiment) do
    counts = experiment_counts(experiment)

    Mix.shell().info("""

    Experiment #{short_id(experiment.id)} resumed and completed successfully.

      Runs:                 #{counts.runs}
      Embeddings:           #{counts.embeddings}
      Persistence diagrams: #{counts.persistence_diagrams}
      Clustering results:   #{counts.clustering_results}
    """)
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

  defp short_id(id), do: String.slice(id, 0, 8)
end
