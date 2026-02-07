defmodule Mix.Tasks.Experiment.Run do
  @shortdoc "Run a PANIC-TDA experiment from a JSON config file"

  @moduledoc """
  Runs a PANIC-TDA experiment pipeline from a JSON configuration file.

      $ mix experiment.run config/experiment.example.json

  The JSON config must contain the following keys:

    - `networks` - list of model network lists (e.g. `[["DummyT2I", "DummyI2T"]]`)
    - `seeds` - list of integer seeds
    - `prompts` - list of initial prompt strings
    - `embedding_models` - list of embedding model names
    - `max_length` - integer trajectory length (must be > 0)

  The task ensures the database is created and migrated before running.
  """

  use Mix.Task

  @impl Mix.Task
  def run([config_path]) do
    setup_db()
    Mix.Task.run("app.start")

    config = read_config(config_path)
    experiment = PanicTda.create_experiment!(config)
    Mix.shell().info("Created experiment #{short_id(experiment.id)}")

    try do
      {:ok, experiment} = PanicTda.Engine.perform_experiment(experiment.id)
      print_summary(experiment)
    rescue
      e -> Mix.raise("Experiment failed: #{Exception.message(e)}")
    end
  end

  def run(_args) do
    Mix.raise("Usage: mix experiment.run <config.json>")
  end

  defp setup_db do
    Mix.Task.run("ecto.create", ["--quiet"])
    Mix.Task.run("ecto.migrate", ["--quiet"])
  end

  @key_mapping %{
    "networks" => :networks,
    "seeds" => :seeds,
    "prompts" => :prompts,
    "embedding_models" => :embedding_models,
    "max_length" => :max_length
  }

  defp read_config(path) do
    json = path |> File.read!() |> Jason.decode!()

    Map.new(@key_mapping, fn {str_key, atom_key} ->
      {atom_key, Map.fetch!(json, str_key)}
    end)
  end

  defp print_summary(experiment) do
    require Ash.Query

    counts = experiment_counts(experiment)

    Mix.shell().info("""

    Experiment #{short_id(experiment.id)} completed successfully.

      Runs:                 #{counts.runs}
      Embeddings:           #{counts.embeddings}
      Persistence diagrams: #{counts.persistence_diagrams}
      Clustering results:   #{counts.clustering_results}
    """)
  end

  defp experiment_counts(experiment) do
    require Ash.Query

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
