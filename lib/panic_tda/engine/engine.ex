defmodule PanicTda.Engine do
  @moduledoc """
  Main engine for running PANIC-TDA experiments.
  Orchestrates the four-stage pipeline:
  1. Runs stage - batch execute model networks
  2. Embeddings stage - compute embeddings for outputs
  3. Persistence diagrams stage - TDA computation
  4. Clustering stage - cluster persistence diagrams
  """

  require Ash.Query

  alias PanicTda.Engine.{RunExecutor, EmbeddingsStage, PdStage, ClusteringStage}
  alias PanicTda.Models.PythonBridge

  def perform_experiment(experiment_id) do
    experiment = PanicTda.get_experiment!(experiment_id)
    experiment = PanicTda.start_experiment!(experiment)

    {:ok, interpreter} = PanicTda.Models.PythonInterpreter.start_link()
    {:ok, env} = Snex.make_env(interpreter)

    try do
      runs = init_runs(experiment)

      runs
      |> Enum.group_by(& &1.network)
      |> Enum.each(fn {_network, group} ->
        :ok = RunExecutor.execute_batch(env, group)
      end)

      :ok = PythonBridge.unload_all_models(env)

      Enum.each(runs, fn run ->
        :ok = EmbeddingsStage.compute(env, run, experiment.embedding_models)
        :ok = PdStage.compute(env, run, experiment.embedding_models)
      end)

      :ok = PythonBridge.unload_all_models(env)
      :ok = ClusteringStage.compute(env, experiment, experiment.embedding_models)

      experiment = PanicTda.complete_experiment!(experiment)
      {:ok, experiment}
    after
      GenServer.stop(interpreter)
    end
  end

  def resume_experiment(experiment_id) do
    experiment = PanicTda.get_experiment!(experiment_id)

    cond do
      is_nil(experiment.started_at) ->
        {:error, :not_started}

      not is_nil(experiment.completed_at) ->
        {:error, :already_completed}

      true ->
        do_resume(experiment)
    end
  end

  defp do_resume(experiment) do
    {:ok, interpreter} = PanicTda.Models.PythonInterpreter.start_link()
    {:ok, env} = Snex.make_env(interpreter)

    try do
      runs = find_or_create_runs(experiment)

      runs
      |> Enum.group_by(& &1.network)
      |> Enum.each(fn {_network, group} ->
        :ok = RunExecutor.resume_batch(env, group)
      end)

      :ok = PythonBridge.unload_all_models(env)

      Enum.each(runs, fn run ->
        :ok = EmbeddingsStage.resume(env, run, experiment.embedding_models)
        :ok = PdStage.resume(env, run, experiment.embedding_models)
      end)

      :ok = PythonBridge.unload_all_models(env)
      :ok = ClusteringStage.resume(env, experiment, experiment.embedding_models)

      experiment = PanicTda.complete_experiment!(experiment)
      {:ok, experiment}
    after
      GenServer.stop(interpreter)
    end
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
