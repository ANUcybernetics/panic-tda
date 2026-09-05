defmodule ResourceQueryTest do
  use ExUnit.Case

  setup do
    :ok = Ecto.Adapters.SQL.Sandbox.checkout(PanicTda.Repo)
    :ok
  end

  defp create_experiment do
    PanicTda.create_experiment!(%{
      networks: [["DummyT2I", "DummyI2T"]],
      prompts: ["test prompt"],
      embedding_models: ["DummyText"],
      max_length: 4
    })
  end

  defp create_run(experiment) do
    PanicTda.create_run!(%{
      network: ["DummyT2I", "DummyI2T"],
      run_number: 0,
      max_length: 4,
      initial_prompt: "test prompt",
      experiment_id: experiment.id
    })
  end

  defp create_clustering_result(algorithm) do
    PanicTda.create_clustering_result!(%{
      embedding_model: "DummyText",
      algorithm: algorithm,
      started_at: DateTime.utc_now()
    })
  end

  defp create_invocation(run, seq) do
    PanicTda.create_invocation!(%{
      model: "DummyT2I",
      type: :text,
      sequence_number: seq,
      output_text: "hello #{seq}",
      started_at: DateTime.utc_now(),
      completed_at: DateTime.utc_now(),
      run_id: run.id
    })
  end

  describe "find_experiment by id prefix" do
    test "returns the one experiment whose id starts with the prefix" do
      experiment = create_experiment()
      _other = create_experiment()

      # The first digits of a UUIDv7 are a timestamp, so two experiments made
      # in the same millisecond share them; twenty characters is unambiguous.
      assert {:ok, %{id: id}} = PanicTda.find_experiment(String.slice(experiment.id, 0, 20))
      assert id == experiment.id
    end

    test "errors when nothing matches" do
      create_experiment()
      assert {:error, %Ash.Error.Invalid{}} = PanicTda.find_experiment("zzzzzzzz")
    end

    test "errors when the prefix is ambiguous rather than picking one" do
      create_experiment()
      create_experiment()

      # UUIDv7 ids created in the same millisecond share their first digits.
      assert {:error, %Ash.Error.Invalid{}} = PanicTda.find_experiment("0")
    end
  end

  describe "calculations" do
    # AshSqlite has no aggregate support, so these are the calculations the
    # data layer can actually evaluate: fragments and is_nil, no count/2.
    test "invocation.duration is the wall-clock seconds between its timestamps" do
      run = create_experiment() |> create_run()
      started = ~U[2026-01-01 00:00:00.000000Z]

      invocation =
        PanicTda.create_invocation!(%{
          model: "DummyT2I",
          type: :text,
          sequence_number: 0,
          output_text: "hello",
          started_at: started,
          completed_at: DateTime.add(started, 1500, :millisecond),
          run_id: run.id
        })

      assert %{duration: duration} = Ash.load!(invocation, :duration)
      assert_in_delta duration, 1.5, 0.001
    end

    test "embedding.dimension is the vector length" do
      run = create_experiment() |> create_run()

      embedding =
        PanicTda.create_embedding!(%{
          embedding_model: "DummyText",
          vector: [1.0, 2.0, 3.0],
          started_at: DateTime.utc_now(),
          completed_at: DateTime.utc_now(),
          invocation_id: create_invocation(run, 0).id
        })

      assert %{dimension: 3} = Ash.load!(embedding, :dimension)
    end

    test "embedding_cluster.is_outlier is true without a medoid" do
      run = create_experiment() |> create_run()

      embedding =
        PanicTda.create_embedding!(%{
          embedding_model: "DummyText",
          vector: [1.0, 2.0, 3.0],
          started_at: DateTime.utc_now(),
          completed_at: DateTime.utc_now(),
          invocation_id: create_invocation(run, 0).id
        })

      result =
        PanicTda.create_clustering_result!(%{
          embedding_model: "DummyText",
          algorithm: "dummy",
          started_at: DateTime.utc_now()
        })

      outlier =
        PanicTda.create_embedding_cluster!(%{
          embedding_id: embedding.id,
          clustering_result_id: result.id,
          medoid_embedding_id: nil
        })

      clustered =
        PanicTda.create_embedding_cluster!(%{
          embedding_id: embedding.id,
          clustering_result_id: create_clustering_result("dummy2").id,
          medoid_embedding_id: embedding.id
        })

      assert %{is_outlier: true} = Ash.load!(outlier, :is_outlier)
      assert %{is_outlier: false} = Ash.load!(clustered, :is_outlier)
    end
  end
end
