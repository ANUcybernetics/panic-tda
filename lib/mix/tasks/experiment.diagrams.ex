defmodule Mix.Tasks.Experiment.Diagrams do
  @shortdoc "Compute an experiment's missing persistence diagrams"

  @moduledoc """
  Computes a Vietoris-Rips persistence diagram for every run of an experiment
  that lacks one, for each of its embedding models. Diagrams are not part of
  the experiment pipeline (see `backlog/docs/tda-keep-kill.md`), so this is the
  only way they get made.

      $ mix experiment.diagrams <experiment-id-prefix>
  """

  use Mix.Task

  require Ash.Query

  @impl Mix.Task
  def run([id_prefix]) do
    Mix.Task.run("ecto.create", ["--quiet"])
    Mix.Task.run("ecto.migrate", ["--quiet"])
    Mix.Task.run("app.start")

    experiment =
      case PanicTda.find_experiment(id_prefix) do
        {:ok, experiment} -> experiment
        {:error, _} -> Mix.raise("No experiment found matching '#{id_prefix}'")
      end

    :ok = PanicTda.Engine.compute_persistence_diagrams(experiment.id)

    count =
      PanicTda.PersistenceDiagram
      |> Ash.Query.filter(run.experiment_id == ^experiment.id)
      |> Ash.count!()

    Mix.shell().info(
      "Experiment #{String.slice(experiment.id, 0, 8)}: #{count} persistence diagrams"
    )
  end

  def run(_), do: Mix.raise("Usage: mix experiment.diagrams <experiment-id-prefix>")
end
