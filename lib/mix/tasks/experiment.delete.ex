defmodule Mix.Tasks.Experiment.Delete do
  @shortdoc "Delete a PANIC-TDA experiment and all its data"

  @moduledoc """
  Deletes an experiment and all associated data (runs, invocations, embeddings,
  persistence diagrams, embedding-cluster assignments). The global
  `ClusteringResult` rows are left intact; rerun `mix cluster.recompute`
  to refresh them.

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
    case PanicTda.find_experiment(id_prefix) do
      {:ok, experiment} -> experiment
      {:error, _} -> Mix.raise("No experiment found matching '#{id_prefix}'")
    end
  end

  defp confirm_deletion(experiment) do
    Mix.shell().yes?("Delete experiment #{short_id(experiment.id)} (#{experiment.id})? [yn]")
  end

  # Spelled out rather than `cascade_destroy` on the resources: that change
  # needs keyset pagination on every primary read action, and it could not
  # honour the newest-first order that `input_invocation_id` (a foreign key
  # onto the same table) forces on invocations.
  defp delete_experiment(experiment) do
    PanicTda.EmbeddingCluster
    |> Ash.Query.filter(embedding.invocation.run.experiment_id == ^experiment.id)
    |> Ash.bulk_destroy!(:destroy, %{})

    PanicTda.Embedding
    |> Ash.Query.filter(invocation.run.experiment_id == ^experiment.id)
    |> Ash.bulk_destroy!(:destroy, %{})

    PanicTda.PersistenceDiagram
    |> Ash.Query.filter(run.experiment_id == ^experiment.id)
    |> Ash.bulk_destroy!(:destroy, %{})

    PanicTda.Invocation
    |> Ash.Query.filter(run.experiment_id == ^experiment.id)
    |> Ash.Query.sort(sequence_number: :desc)
    |> Ash.bulk_destroy!(:destroy, %{})

    PanicTda.Run
    |> Ash.Query.filter(experiment_id == ^experiment.id)
    |> Ash.bulk_destroy!(:destroy, %{})

    Ash.destroy!(experiment)
  end

  defp short_id(id), do: String.slice(id, 0, 8)
end
