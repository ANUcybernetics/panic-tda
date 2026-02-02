defmodule PanicTda.Engine do
  @moduledoc """
  Main engine for running PANIC-TDA experiments.
  Orchestrates the three-stage pipeline:
  1. Runs stage - execute model networks
  2. Embeddings stage - compute embeddings for outputs
  3. Persistence diagrams stage - TDA computation (TODO)
  """

  alias PanicTda.Engine.{RunExecutor, EmbeddingsStage}

  def perform_experiment(experiment_id) do
    experiment = Ash.get!(PanicTda.Experiment, experiment_id)

    {:ok, experiment} =
      experiment
      |> Ash.Changeset.for_update(:start)
      |> Ash.update()

    {:ok, interpreter} = PanicTda.Models.PythonInterpreter.start_link()
    {:ok, env} = Snex.make_env(interpreter)

    try do
      runs = init_runs(experiment)

      Enum.each(runs, fn run ->
        {:ok, _invocation_ids} = RunExecutor.execute(env, run)
        :ok = EmbeddingsStage.compute(env, run, experiment.embedding_models)
      end)

      {:ok, experiment} =
        experiment
        |> Ash.Changeset.for_update(:complete)
        |> Ash.update()

      {:ok, experiment}
    after
      GenServer.stop(interpreter)
    end
  end

  def init_runs(experiment) do
    for network <- experiment.networks,
        seed <- experiment.seeds,
        prompt <- experiment.prompts do
      {:ok, run} =
        PanicTda.Run
        |> Ash.Changeset.for_create(:create, %{
          network: network,
          seed: seed,
          max_length: experiment.max_length,
          initial_prompt: prompt,
          experiment_id: experiment.id
        })
        |> Ash.create()

      run
    end
  end
end
