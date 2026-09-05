defmodule PanicTda.Engine do
  @moduledoc """
  Main engine for running PANIC-TDA experiments.
  Orchestrates the three-stage pipeline:
  1. Runs stage - batch execute model networks
  2. Embeddings stage - compute embeddings for outputs
  3. Persistence diagrams stage - TDA computation

  Clustering (global EVoC clustering of raw embeddings, pooled across
  experiments) is not part of this pipeline; run it separately via
  `mix cluster.recompute`.
  """

  require Ash.Query

  alias PanicTda.Engine.{RunExecutor, EmbeddingsStage, PdStage}
  alias PanicTda.Models.{PythonBridge, PythonSession}

  def perform_experiment(experiment_id, opts \\ []) do
    experiment = PanicTda.get_experiment!(experiment_id)
    experiment = PanicTda.start_experiment!(experiment)

    {:ok, session} = session_from_opts(opts, experiment)

    try do
      runs = init_runs(experiment)

      runs
      |> cells_in_config_order(experiment)
      |> Enum.each(fn cell ->
        :ok = PythonBridge.unload_all_models(PythonSession.env(session))
        :ok = RunExecutor.execute_batch(session, cell)
        finish_cell(session, cell, experiment, &EmbeddingsStage.compute/3, &PdStage.compute/3)
      end)

      experiment = PanicTda.complete_experiment!(experiment)
      {:ok, experiment}
    after
      PythonSession.stop(session)
    end
  end

  def resume_experiment(experiment_id, opts \\ []) do
    experiment = PanicTda.get_experiment!(experiment_id)
    force? = Keyword.get(opts, :force, false)

    cond do
      is_nil(experiment.started_at) ->
        {:error, :not_started}

      not is_nil(experiment.completed_at) and not force? ->
        {:error, :already_completed}

      not is_nil(experiment.completed_at) and force? ->
        experiment |> PanicTda.reopen_experiment!() |> do_resume(opts)

      true ->
        do_resume(experiment, opts)
    end
  end

  defp do_resume(experiment, opts) do
    {:ok, session} = session_from_opts(opts, experiment)

    try do
      runs = find_or_create_runs(experiment)

      runs
      |> cells_in_config_order(experiment)
      |> Enum.each(fn cell ->
        :ok = PythonBridge.unload_all_models(PythonSession.env(session))
        :ok = RunExecutor.resume_batch(session, cell)
        finish_cell(session, cell, experiment, &EmbeddingsStage.resume/3, &PdStage.resume/3)
      end)

      experiment = PanicTda.complete_experiment!(experiment)
      {:ok, experiment}
    after
      PythonSession.stop(session)
    end
  end

  # A cell is one network's runs, executed in lockstep. Cells run in the order
  # the config lists them (map iteration would put them in term order), so the
  # fast generators can go first and their cells be analysed while the slow
  # ones run; each cell is embedded and given its persistence diagrams as soon
  # as its runs finish, for the same reason.
  defp cells_in_config_order(runs, experiment) do
    by_network = Enum.group_by(runs, & &1.network)
    Enum.map(experiment.networks, &Map.fetch!(by_network, &1))
  end

  defp finish_cell(session, cell, experiment, embed, pd) do
    env = PythonSession.env(session)
    :ok = PythonBridge.unload_all_models(env)

    Enum.each(cell, fn run ->
      :ok = embed.(env, run, experiment.embedding_models)
      :ok = pd.(env, run, experiment.embedding_models)
    end)
  end

  def init_runs(experiment) do
    for network <- experiment.networks,
        prompt <- experiment.prompts,
        run_number <- 0..(experiment.num_runs - 1) do
      PanicTda.create_run!(%{
        network: network,
        run_number: run_number,
        max_length: experiment.max_length,
        initial_prompt: prompt,
        experiment_id: experiment.id
      })
    end
  end

  # The experiment's image-to-text ceiling is Python-side state, so it is
  # re-applied through `on_start` whenever the session boots a fresh
  # interpreter — including after a retry restarts one mid-run.
  defp session_from_opts(opts, experiment) do
    on_start = &PythonBridge.set_i2t_max_new_tokens(&1, experiment.i2t_max_new_tokens)

    case Keyword.get(opts, :env) do
      nil -> PythonSession.start(on_start)
      env -> PythonSession.wrap(env, on_start)
    end
  end

  def find_or_create_runs(experiment) do
    existing_runs =
      PanicTda.Run
      |> Ash.Query.filter(experiment_id == ^experiment.id)
      |> Ash.read!()

    existing_keys =
      MapSet.new(existing_runs, fn run ->
        {run.network, run.initial_prompt, run.run_number}
      end)

    new_runs =
      for network <- experiment.networks,
          prompt <- experiment.prompts,
          run_number <- 0..(experiment.num_runs - 1),
          not MapSet.member?(existing_keys, {network, prompt, run_number}) do
        PanicTda.create_run!(%{
          network: network,
          run_number: run_number,
          max_length: experiment.max_length,
          initial_prompt: prompt,
          experiment_id: experiment.id
        })
      end

    existing_runs ++ new_runs
  end
end
