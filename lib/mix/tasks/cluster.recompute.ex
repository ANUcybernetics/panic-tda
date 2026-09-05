defmodule Mix.Tasks.Cluster.Recompute do
  @shortdoc "Re-run global EVoC clustering across every embedding model"

  @moduledoc """
  Wipes and rewrites the global clustering for every embedding model in
  use across the database. Cluster identities are shared across
  experiments — cluster N in any experiment refers to the same
  semantic region.

      $ mix cluster.recompute
      $ mix cluster.recompute --model Nomic --model Qwen3Embed

  Without `--model` flags, clusters every embedding model that has
  embeddings in the DB.
  """

  use Mix.Task

  alias PanicTda.Engine.ClusteringStage

  @impl Mix.Task
  def run(args) do
    {opts, _, _} = OptionParser.parse(args, strict: [model: :keep])

    Mix.Task.run("ecto.create", ["--quiet"])
    Mix.Task.run("ecto.migrate", ["--quiet"])
    Mix.Task.run("app.start")

    {:ok, interpreter} = PanicTda.Models.PythonInterpreter.start_link()
    {:ok, env} = Snex.make_env(interpreter)

    try do
      models =
        case Keyword.get_values(opts, :model) do
          [] -> ClusteringStage.all_embedding_models()
          requested -> requested
        end

      total = length(models)
      Mix.shell().info("Reclustering #{total} embedding model(s): #{Enum.join(models, ", ")}")

      models
      |> Enum.with_index(1)
      |> Enum.each(fn {model, i} ->
        Mix.shell().info("[#{i}/#{total}] #{model}")
        :ok = ClusteringStage.recompute(env, [model])
      end)

      Mix.shell().info("Done.")
    after
      PanicTda.Models.PythonInterpreter.stop_or_kill(interpreter)
    end
  end
end
