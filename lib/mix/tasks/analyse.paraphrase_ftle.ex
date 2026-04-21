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

    identical_rows = load_identical_rows(experiment)

    csv_path = Path.join(out_dir, "ftle_values.csv")
    write_csv(csv_path, identical_rows)

    Mix.shell().info("Wrote #{length(identical_rows)} identical-prompt rows to #{csv_path}")
    :ok
  end

  defp find_experiment(id_prefix) do
    PanicTda.list_experiments!()
    |> Enum.find(fn e -> String.starts_with?(e.id, id_prefix) end) ||
      Mix.raise("No experiment found matching '#{id_prefix}'")
  end

  defp load_identical_rows(experiment) do
    PanicTda.LyapunovResult
    |> Ash.Query.filter(experiment_id == ^experiment.id)
    |> Ash.read!()
    |> Enum.map(fn r ->
      %{
        experiment_id: experiment.id,
        network: Enum.join(r.network, "|"),
        embedding_model: r.embedding_model,
        category: "identical",
        prompt_or_pair: r.prompt,
        ftle: r.lyapunov_data.exponent,
        r_squared: r.lyapunov_data.r_squared,
        num_pairs: r.lyapunov_data.num_pairs,
        num_timesteps: r.lyapunov_data.num_timesteps
      }
    end)
  end

  @csv_headers ~w(experiment_id network embedding_model category prompt_or_pair ftle r_squared num_pairs num_timesteps)

  defp write_csv(path, rows) do
    header_line = Enum.join(@csv_headers, ",") <> "\n"

    body =
      rows
      |> Enum.map(fn row ->
        @csv_headers
        |> Enum.map(fn h -> csv_escape(Map.get(row, String.to_atom(h))) end)
        |> Enum.join(",")
      end)
      |> Enum.join("\n")

    File.write!(path, header_line <> body <> "\n")
  end

  defp csv_escape(nil), do: ""
  defp csv_escape(v) when is_number(v), do: to_string(v)
  defp csv_escape(v) when is_binary(v) do
    if String.contains?(v, [",", "\"", "\n"]) do
      "\"" <> String.replace(v, "\"", "\"\"") <> "\""
    else
      v
    end
  end
  defp csv_escape(v), do: inspect(v)

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
