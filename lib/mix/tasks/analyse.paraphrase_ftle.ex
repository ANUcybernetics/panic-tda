defmodule Mix.Tasks.Analyse.ParaphraseFtle do
  @shortdoc "Compute paraphrase-FTLE for an experiment and produce report artefacts"

  @moduledoc """
  Post-hoc analysis: reads identical-prompt FTLEs from existing
  LyapunovResult rows and computes paraphrase (cross-prompt) FTLEs by
  re-reading the stored embeddings. Writes a CSV + two plots + a
  report.md stub under an output directory.

      $ mix analyse.paraphrase_ftle <experiment-id-prefix> [--out analysis/my_dir]
      $ mix analyse.paraphrase_ftle <experiment-id-prefix> --cell "SD35Medium|Moondream,Nomic"
  """

  use Mix.Task

  require Ash.Query

  @impl Mix.Task
  def run(args) do
    {opts, positional, _} =
      OptionParser.parse(args, strict: [out: :string, cell: :string])

    id_prefix =
      case positional do
        [prefix] -> prefix
        _ -> Mix.raise("Usage: mix analyse.paraphrase_ftle <experiment-id-prefix> [--out path] [--cell NETWORK,EMBEDDING]")
      end

    Mix.Task.run("ecto.create", ["--quiet"])
    Mix.Task.run("ecto.migrate", ["--quiet"])
    Mix.Task.run("app.start")

    experiment = find_experiment(id_prefix)

    out_dir = opts[:out] || Path.join(["analysis", experiment_slug(experiment)])
    File.mkdir_p!(out_dir)

    Mix.shell().info("Analysing experiment #{experiment.id}")
    Mix.shell().info("Output directory: #{out_dir}")

    :ok
  end

  defp find_experiment(id_prefix) do
    PanicTda.list_experiments!()
    |> Enum.find(fn e -> String.starts_with?(e.id, id_prefix) end) ||
      Mix.raise("No experiment found matching '#{id_prefix}'")
  end

  defp experiment_slug(experiment) do
    prompts = experiment.prompts || []
    base = prompts |> List.first() |> Kernel.||("experiment")
    base
    |> String.downcase()
    |> String.replace(~r/[^a-z0-9]+/, "_")
    |> String.trim("_")
    |> String.slice(0, 40)
  end
end
