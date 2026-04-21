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

    {:ok, interpreter} = PanicTda.Models.PythonInterpreter.start_link()
    {:ok, env} = Snex.make_env(interpreter)
    {:ok, _} = ensure_penguin_analysis_loaded(env)

    identical_rows = load_identical_rows(experiment)
    paraphrase_rows = compute_paraphrase_rows(env, experiment)
    all_rows = identical_rows ++ paraphrase_rows

    csv_path = Path.join(out_dir, "ftle_values.csv")
    write_csv(csv_path, all_rows)

    Mix.shell().info("""
    Identical-prompt rows: #{length(identical_rows)}
    Paraphrase rows:       #{length(paraphrase_rows)}
    Wrote CSV: #{csv_path}
    """)

    :ok
  end

  defp ensure_penguin_analysis_loaded(env) do
    priv_python =
      :code.priv_dir(:panic_tda) |> to_string() |> Path.join("python")

    Snex.pyeval(
      env,
      """
      import sys
      if _priv_python not in sys.path:
          sys.path.insert(0, _priv_python)
      import penguin_analysis
      return True
      """,
      %{"_priv_python" => priv_python}
    )
  end

  defp compute_paraphrase_rows(env, experiment) do
    runs =
      PanicTda.Run
      |> Ash.Query.filter(experiment_id == ^experiment.id)
      |> Ash.read!()

    embedding_models = experiment.embedding_models || []

    runs_by_network = Enum.group_by(runs, fn r -> r.network end)

    for {network, network_runs} <- runs_by_network,
        embedding_model <- embedding_models,
        row <- rows_for_network(env, experiment, network, network_runs, embedding_model) do
      row
    end
  end

  defp rows_for_network(env, experiment, network, network_runs, embedding_model) do
    trajectories_by_prompt =
      network_runs
      |> Enum.group_by(& &1.initial_prompt)
      |> Enum.map(fn {prompt, runs} ->
        {prompt, Enum.map(runs, &load_trajectory(&1, embedding_model))}
      end)
      |> Enum.reject(fn {_prompt, trajs} -> length(Enum.reject(trajs, &(&1 == []))) < 2 end)
      |> Enum.into(%{})

    prompts = Map.keys(trajectories_by_prompt)

    prompts
    |> pairs()
    |> Enum.map(fn {p1, p2} ->
      trajs_a = trajectories_by_prompt |> Map.fetch!(p1) |> Enum.reject(&(&1 == []))
      trajs_b = trajectories_by_prompt |> Map.fetch!(p2) |> Enum.reject(&(&1 == []))

      min_length =
        (trajs_a ++ trajs_b)
        |> Enum.map(&length/1)
        |> Enum.min()

      if min_length < 2 do
        nil
      else
        trajs_a_trunc = Enum.map(trajs_a, &Enum.take(&1, min_length))
        trajs_b_trunc = Enum.map(trajs_b, &Enum.take(&1, min_length))

        dimension = trajs_a_trunc |> hd() |> hd() |> Nx.size()

        a_binary = trajs_a_trunc |> List.flatten() |> Nx.stack() |> Nx.to_binary()
        b_binary = trajs_b_trunc |> List.flatten() |> Nx.stack() |> Nx.to_binary()

        {:ok, result} =
          Snex.pyeval(
            env,
            "return penguin_analysis.cross_prompt_ftle(a_b64, b_b64, num_a, num_b, num_ts, dim)",
            %{
              "a_b64" => Base.encode64(a_binary),
              "b_b64" => Base.encode64(b_binary),
              "num_a" => length(trajs_a_trunc),
              "num_b" => length(trajs_b_trunc),
              "num_ts" => min_length,
              "dim" => dimension
            }
          )

        %{
          experiment_id: experiment.id,
          network: Enum.join(network, "|"),
          embedding_model: embedding_model,
          category: "paraphrase",
          prompt_or_pair: "#{p1} || #{p2}",
          ftle: result["exponent"],
          r_squared: result["r_squared"],
          num_pairs: result["num_pairs"],
          num_timesteps: result["num_timesteps"]
        }
      end
    end)
    |> Enum.reject(&is_nil/1)
  end

  defp load_trajectory(run, embedding_model) do
    PanicTda.Embedding
    |> Ash.Query.filter(invocation.run_id == ^run.id and embedding_model == ^embedding_model)
    |> Ash.Query.load(:invocation)
    |> Ash.read!()
    |> Enum.sort_by(& &1.invocation.sequence_number)
    |> Enum.map(& &1.vector)
  end

  defp pairs(list) do
    for {a, i} <- Enum.with_index(list),
        {b, j} <- Enum.with_index(list),
        i < j,
        do: {a, b}
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
