defmodule PanicTda.LyapunovStageTest do
  use ExUnit.Case

  alias PanicTda.Engine

  setup do
    :ok = Ecto.Adapters.SQL.Sandbox.checkout(PanicTda.Repo)
    :ok
  end

  describe "Lyapunov stage" do
    test "computes FTLE for groups with num_runs >= 2" do
      experiment =
        PanicTda.create_experiment!(%{
          networks: [["DummyT2I", "DummyI2T"]],
          num_runs: 3,
          prompts: ["test lyapunov"],
          embedding_models: ["DummyText"],
          max_length: 6
        })

      {:ok, _} = Engine.perform_experiment(experiment.id)

      results = PanicTda.list_lyapunov_results!()
      assert length(results) == 1

      result = hd(results)
      assert result.embedding_model == "DummyText"
      assert result.network == ["DummyT2I", "DummyI2T"]
      assert result.prompt == "test lyapunov"
      assert result.started_at != nil
      assert result.completed_at != nil

      data = result.lyapunov_data
      assert is_float(data.exponent)
      assert is_list(data.divergence_curve)
      assert data.num_pairs == 3
      assert data.num_timesteps == 3
    end

    test "skips groups with num_runs < 2" do
      experiment =
        PanicTda.create_experiment!(%{
          networks: [["DummyT2I", "DummyI2T"]],
          num_runs: 1,
          prompts: ["solo run"],
          embedding_models: ["DummyText"],
          max_length: 6
        })

      {:ok, _} = Engine.perform_experiment(experiment.id)

      results = PanicTda.list_lyapunov_results!()
      assert results == []
    end

    test "computes separate results per prompt" do
      experiment =
        PanicTda.create_experiment!(%{
          networks: [["DummyT2I", "DummyI2T"]],
          num_runs: 2,
          prompts: ["prompt alpha", "prompt beta"],
          embedding_models: ["DummyText"],
          max_length: 6
        })

      {:ok, _} = Engine.perform_experiment(experiment.id)

      results = PanicTda.list_lyapunov_results!()
      assert length(results) == 2

      prompts = Enum.map(results, & &1.prompt) |> Enum.sort()
      assert prompts == ["prompt alpha", "prompt beta"]
    end
  end
end
