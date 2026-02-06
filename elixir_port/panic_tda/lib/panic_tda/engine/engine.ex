defmodule PanicTda.Engine do
  @moduledoc """
  Main engine for running PANIC-TDA experiments.
  Orchestrates the three-stage pipeline:
  1. Runs stage - execute model networks
  2. Embeddings stage - compute embeddings for outputs
  3. Persistence diagrams stage - TDA computation
  """

  alias PanicTda.Engine.{RunExecutor, EmbeddingsStage, PdStage, ClusteringStage}

  def perform_experiment(experiment_id) do
    experiment = PanicTda.get_experiment!(experiment_id)
    experiment = PanicTda.start_experiment!(experiment)

    {:ok, interpreter} = PanicTda.Models.PythonInterpreter.start_link()
    {:ok, env} = Snex.make_env(interpreter)

    try do
      runs = init_runs(experiment)

      Enum.each(runs, fn run ->
        :ok = RunExecutor.execute(env, run)
        :ok = EmbeddingsStage.compute(env, run, experiment.embedding_models)
        :ok = PdStage.compute(env, run, experiment.embedding_models)
      end)

      :ok = ClusteringStage.compute(env, experiment, experiment.embedding_models)

      experiment = PanicTda.complete_experiment!(experiment)
      {:ok, experiment}
    after
      GenServer.stop(interpreter)
    end
  end

  def init_runs(experiment) do
    for network <- experiment.networks,
        seed <- experiment.seeds,
        prompt <- experiment.prompts do
      PanicTda.create_run!(%{
        network: network,
        seed: seed,
        max_length: experiment.max_length,
        initial_prompt: prompt,
        experiment_id: experiment.id
      })
    end
  end
end
