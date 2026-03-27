defmodule PanicTda.Engine.LyapunovStage do
  @moduledoc """
  Computes finite-time Lyapunov exponents from multi-run trajectory divergence.
  Groups runs by (network, prompt) and measures pairwise embedding divergence
  over time. Requires num_runs >= 2.
  """

  require Ash.Query
  require Logger

  alias PanicTda.Models.Lyapunov

  def compute(env, experiment, embedding_models) do
    runs =
      PanicTda.Run
      |> Ash.Query.filter(experiment_id == ^experiment.id)
      |> Ash.read!()

    groups = Enum.group_by(runs, fn run -> {run.network, run.initial_prompt} end)

    Enum.each(groups, fn {{network, prompt}, group_runs} ->
      if length(group_runs) < 2 do
        Logger.warning(
          "Lyapunov: skipping #{inspect(network)} / #{inspect(prompt)} — need >= 2 runs, got #{length(group_runs)}"
        )
      else
        Enum.each(embedding_models, fn embedding_model ->
          :ok = compute_for_group(env, experiment, network, prompt, group_runs, embedding_model)
        end)
      end
    end)

    :ok
  end

  def resume(env, experiment, embedding_models) do
    delete_existing(experiment)
    compute(env, experiment, embedding_models)
  end

  defp delete_existing(experiment) do
    PanicTda.LyapunovResult
    |> Ash.Query.filter(experiment_id == ^experiment.id)
    |> Ash.read!()
    |> Enum.each(&PanicTda.destroy_lyapunov_result!(&1))
  end

  defp compute_for_group(env, experiment, network, prompt, runs, embedding_model) do
    trajectories =
      runs
      |> Enum.map(fn run -> load_trajectory(run, embedding_model) end)
      |> Enum.reject(&(&1 == []))

    if length(trajectories) < 2 do
      Logger.warning(
        "Lyapunov: skipping #{inspect(network)} / #{inspect(prompt)} / #{embedding_model} — fewer than 2 trajectories with embeddings"
      )

      :ok
    else
      min_length = trajectories |> Enum.map(&length/1) |> Enum.min()

      if min_length < 2 do
        Logger.warning(
          "Lyapunov: skipping #{inspect(network)} / #{inspect(prompt)} / #{embedding_model} — trajectory length #{min_length} < 2"
        )

        :ok
      else
        truncated = Enum.map(trajectories, &Enum.take(&1, min_length))
        started_at = DateTime.utc_now()

        dimension = hd(hd(truncated)) |> Nx.size()
        num_trajectories = length(truncated)

        stacked_binary =
          truncated
          |> List.flatten()
          |> Nx.stack()
          |> Nx.to_binary()

        {:ok, lyapunov_data} =
          Lyapunov.compute_ftle(env, stacked_binary, num_trajectories, min_length, dimension)

        completed_at = DateTime.utc_now()

        PanicTda.create_lyapunov_result!(%{
          embedding_model: embedding_model,
          network: network,
          prompt: prompt,
          lyapunov_data: lyapunov_data,
          started_at: started_at,
          completed_at: completed_at,
          experiment_id: experiment.id
        })

        :ok
      end
    end
  end

  defp load_trajectory(run, embedding_model) do
    PanicTda.Embedding
    |> Ash.Query.filter(invocation.run_id == ^run.id and embedding_model == ^embedding_model)
    |> Ash.Query.load(:invocation)
    |> Ash.read!()
    |> Enum.sort_by(& &1.invocation.sequence_number)
    |> Enum.map(& &1.vector)
  end
end
