defmodule Mix.Tasks.Experiment.Delete do
  @shortdoc "Delete a PANIC-TDA experiment and all its data"

  @moduledoc """
  Deletes an experiment and all associated data (runs, invocations, embeddings,
  persistence diagrams, clustering results).

      $ mix experiment.delete <experiment-id-prefix>
      $ mix experiment.delete <experiment-id-prefix> --force

  Use `--force` to skip the confirmation prompt.
  """

  use Mix.Task

  require Ash.Query

  @impl Mix.Task
  def run(args) do
    {opts, rest} = OptionParser.parse!(args, strict: [force: :boolean])

    case rest do
      [id_prefix] ->
        Mix.Task.run("ecto.create", ["--quiet"])
        Mix.Task.run("ecto.migrate", ["--quiet"])
        Mix.Task.run("app.start")

        experiment = find_experiment(id_prefix)

        if opts[:force] || confirm_deletion(experiment) do
          delete_experiment(experiment)
          Mix.shell().info("Deleted experiment #{short_id(experiment.id)}")
        else
          Mix.shell().info("Aborted.")
        end

      _ ->
        Mix.raise("Usage: mix experiment.delete <experiment-id-prefix> [--force]")
    end
  end

  defp find_experiment(id_prefix) do
    experiments = PanicTda.list_experiments!()

    Enum.find(experiments, fn e -> String.starts_with?(e.id, id_prefix) end) ||
      Mix.raise("No experiment found matching '#{id_prefix}'")
  end

  defp confirm_deletion(experiment) do
    Mix.shell().yes?("Delete experiment #{short_id(experiment.id)} (#{experiment.id})? [yn]")
  end

  defp delete_experiment(experiment) do
    PanicTda.EmbeddingCluster
    |> Ash.Query.filter(clustering_result.experiment_id == ^experiment.id)
    |> Ash.bulk_destroy!(:destroy, %{})

    PanicTda.ClusteringResult
    |> Ash.Query.filter(experiment_id == ^experiment.id)
    |> Ash.bulk_destroy!(:destroy, %{})

    PanicTda.Embedding
    |> Ash.Query.filter(invocation.run.experiment_id == ^experiment.id)
    |> Ash.bulk_destroy!(:destroy, %{})

    PanicTda.PersistenceDiagram
    |> Ash.Query.filter(run.experiment_id == ^experiment.id)
    |> Ash.bulk_destroy!(:destroy, %{})

    PanicTda.Invocation
    |> Ash.Query.filter(run.experiment_id == ^experiment.id)
    |> Ash.bulk_destroy!(:destroy, %{})

    PanicTda.Run
    |> Ash.Query.filter(experiment_id == ^experiment.id)
    |> Ash.bulk_destroy!(:destroy, %{})

    Ash.destroy!(experiment)
  end

  defp short_id(id), do: String.slice(id, 0, 8)
end
