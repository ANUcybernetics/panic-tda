defmodule Mix.Tasks.Analyse.ThreeRegimeOverlay do
  @shortdoc "Plot divergence curves across three prompt-variation regimes"

  @moduledoc """
  Produces a three-regime divergence-curve overlay for a single
  (network, embedding_model) cell, loading data from two experiments:

    - `--close EXPERIMENT_ID` — experiment with close paraphrases of one
      scenario (e.g., penguin_campfire). Supplies the identical-prompt
      baseline and the paraphrase curve.
    - `--far EXPERIMENT_ID` — experiment with semantically distant prompts
      (e.g., the three-category pilot). Supplies the distant-topic curve.
    - `--network NETWORK` — e.g., "SD35Medium|Pixtral"
    - `--embedding-model MODEL` — e.g., "Qwen3Embed"
    - `--out PATH` — output PDF path.

  Identical-prompt curves are averaged over existing LyapunovResult rows.
  Close/far curves are computed fresh by picking a representative
  prompt-pair and running the Python cross_prompt_ftle helper.
  """

  use Mix.Task

  require Ash.Query

  @impl Mix.Task
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        strict: [
          close: :string,
          far: :string,
          network: :string,
          embedding_model: :string,
          out: :string
        ]
      )

    close_id = opts[:close] || Mix.raise("--close <experiment-id> is required")
    far_id = opts[:far] || Mix.raise("--far <experiment-id> is required")
    network_str = opts[:network] || Mix.raise("--network NETWORK is required")
    embedding_model = opts[:embedding_model] || Mix.raise("--embedding-model MODEL is required")
    out_path = (opts[:out] || "analysis/comparison/three_regime_overlay.pdf") |> Path.expand()

    Mix.Task.run("ecto.create", ["--quiet"])
    Mix.Task.run("ecto.migrate", ["--quiet"])
    Mix.Task.run("app.start")

    close_exp = find_experiment(close_id)
    far_exp = find_experiment(far_id)

    File.mkdir_p!(Path.dirname(out_path))

    {:ok, interpreter} = PanicTda.Models.PythonInterpreter.start_link()
    {:ok, env} = Snex.make_env(interpreter)
    {:ok, _} = ensure_loaded(env)

    network = String.split(network_str, "|")

    identical_curve = identical_mean_curve(close_exp, network, embedding_model)
    close_curve = representative_cross_curve(env, close_exp, network, embedding_model)
    far_curve = representative_cross_curve(env, far_exp, network, embedding_model)

    Mix.shell().info("""
    Three-regime overlay for #{network_str} · #{embedding_model}
    Identical curve length:  #{length(identical_curve)}  (from #{close_exp.id})
    Close paraphrase length: #{length(close_curve)}  (from #{close_exp.id})
    Distant-topic length:    #{length(far_curve)}  (from #{far_exp.id})
    """)

    {:ok, true} =
      Snex.pyeval(
        env,
        """
        penguin_analysis.plot_three_regime_overlay(
            out_path,
            network=network,
            embedding_model=embedding_model,
            identical_curve=identical_curve,
            close_curve=close_curve,
            far_curve=far_curve,
        )
        return True
        """,
        %{
          "out_path" => out_path,
          "network" => network_str,
          "embedding_model" => embedding_model,
          "identical_curve" => identical_curve,
          "close_curve" => close_curve,
          "far_curve" => far_curve
        }
      )

    Mix.shell().info("Wrote #{out_path}")
    :ok
  end

  defp find_experiment(id_prefix) do
    PanicTda.list_experiments!()
    |> Enum.find(fn e -> String.starts_with?(e.id, id_prefix) end) ||
      Mix.raise("No experiment found matching '#{id_prefix}'")
  end

  defp ensure_loaded(env) do
    priv_python = :code.priv_dir(:panic_tda) |> to_string() |> Path.join("python")

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

  defp identical_mean_curve(experiment, network, embedding_model) do
    curves =
      PanicTda.LyapunovResult
      |> Ash.Query.filter(
        experiment_id == ^experiment.id and
          embedding_model == ^embedding_model and
          network == ^network
      )
      |> Ash.read!()
      |> Enum.map(& &1.lyapunov_data.divergence_curve)

    case curves do
      [] ->
        Mix.raise(
          "No LyapunovResult rows for network=#{inspect(network)} / embedding_model=#{embedding_model} in experiment #{experiment.id}"
        )

      _ ->
        mean_curves(curves)
    end
  end

  defp representative_cross_curve(env, experiment, network, embedding_model) do
    runs =
      PanicTda.Run
      |> Ash.Query.filter(experiment_id == ^experiment.id and network == ^network)
      |> Ash.read!()

    prompts = runs |> Enum.map(& &1.initial_prompt) |> Enum.uniq()

    case pairs(prompts) do
      [] ->
        Mix.raise(
          "Only one prompt for network=#{inspect(network)} in experiment #{experiment.id}; need >= 2"
        )

      [{p1, p2} | _] ->
        compute_cross_curve(env, runs, p1, p2, embedding_model)
    end
  end

  defp compute_cross_curve(env, runs, p1, p2, embedding_model) do
    trajs_a =
      runs
      |> Enum.filter(&(&1.initial_prompt == p1))
      |> Enum.map(&load_trajectory(&1, embedding_model))
      |> Enum.reject(&(&1 == []))

    trajs_b =
      runs
      |> Enum.filter(&(&1.initial_prompt == p2))
      |> Enum.map(&load_trajectory(&1, embedding_model))
      |> Enum.reject(&(&1 == []))

    if Enum.empty?(trajs_a) or Enum.empty?(trajs_b) do
      Mix.raise(
        "No embeddings for one of the prompts (#{inspect(p1)} / #{inspect(p2)}) with embedding_model=#{embedding_model}"
      )
    end

    min_length = (trajs_a ++ trajs_b) |> Enum.map(&length/1) |> Enum.min()

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

    result["divergence_curve"]
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

  defp mean_curves(curves) do
    min_length = curves |> Enum.map(&length/1) |> Enum.min()

    for t <- 0..(min_length - 1) do
      sum = Enum.reduce(curves, 0.0, fn c, acc -> acc + Enum.at(c, t) end)
      sum / length(curves)
    end
  end
end
