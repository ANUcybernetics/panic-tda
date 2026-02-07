defmodule ResourceValidationTest do
  use ExUnit.Case

  setup do
    :ok = Ecto.Adapters.SQL.Sandbox.checkout(PanicTda.Repo)
    :ok
  end

  defp create_experiment(overrides \\ %{}) do
    defaults = %{
      networks: [["DummyT2I", "DummyI2T"]],
      seeds: [42],
      prompts: ["test prompt"],
      embedding_models: ["DummyText"],
      max_length: 4
    }

    PanicTda.create_experiment!(Map.merge(defaults, overrides))
  end

  defp create_run(experiment, overrides \\ %{}) do
    defaults = %{
      network: ["DummyT2I", "DummyI2T"],
      seed: 42,
      max_length: 4,
      initial_prompt: "test prompt",
      experiment_id: experiment.id
    }

    PanicTda.create_run!(Map.merge(defaults, overrides))
  end

  describe "experiment validations" do
    test "rejects empty networks array" do
      assert_raise Ash.Error.Invalid, fn ->
        create_experiment(%{networks: []})
      end
    end

    test "rejects empty sub-arrays in networks" do
      assert_raise Ash.Error.Invalid, fn ->
        create_experiment(%{networks: [[]]})
      end
    end

    test "rejects empty seeds array" do
      assert_raise Ash.Error.Invalid, fn ->
        create_experiment(%{seeds: []})
      end
    end

    test "rejects empty prompts array" do
      assert_raise Ash.Error.Invalid, fn ->
        create_experiment(%{prompts: []})
      end
    end

    test "rejects empty embedding_models array" do
      assert_raise Ash.Error.Invalid, fn ->
        create_experiment(%{embedding_models: []})
      end
    end

    test "rejects max_length of 0" do
      assert_raise Ash.Error.Invalid, fn ->
        create_experiment(%{max_length: 0})
      end
    end

    test "rejects negative max_length" do
      assert_raise Ash.Error.Invalid, fn ->
        create_experiment(%{max_length: -1})
      end
    end

    test "accepts max_length of 1" do
      experiment = create_experiment(%{max_length: 1})
      assert experiment.max_length == 1
    end
  end

  describe "run validations" do
    test "rejects empty network array" do
      experiment = create_experiment()

      assert_raise Ash.Error.Invalid, fn ->
        create_run(experiment, %{network: []})
      end
    end

    test "rejects empty initial_prompt" do
      experiment = create_experiment()

      assert_raise Ash.Error.Invalid, fn ->
        create_run(experiment, %{initial_prompt: ""})
      end
    end

    test "rejects seed less than -1" do
      experiment = create_experiment()

      assert_raise Ash.Error.Invalid, fn ->
        create_run(experiment, %{seed: -2})
      end
    end

    test "accepts seed of -1" do
      experiment = create_experiment(%{seeds: [-1]})
      run = create_run(experiment, %{seed: -1})
      assert run.seed == -1
    end

    test "accepts seed of 0" do
      experiment = create_experiment(%{seeds: [0]})
      run = create_run(experiment, %{seed: 0})
      assert run.seed == 0
    end

    test "rejects duplicate run within experiment" do
      experiment = create_experiment()
      _run1 = create_run(experiment)

      assert_raise Ash.Error.Invalid, fn ->
        create_run(experiment)
      end
    end
  end

  describe "invocation validations" do
    test "rejects negative sequence_number" do
      experiment = create_experiment()
      run = create_run(experiment)
      now = DateTime.utc_now()

      assert_raise Ash.Error.Invalid, fn ->
        PanicTda.create_invocation!(%{
          model: "DummyT2I",
          type: :image,
          seed: 42,
          sequence_number: -1,
          output_image: <<0, 1, 2, 3>>,
          started_at: now,
          completed_at: now,
          run_id: run.id
        })
      end
    end

    test "rejects duplicate sequence_number within run" do
      experiment = create_experiment()
      run = create_run(experiment)
      now = DateTime.utc_now()

      PanicTda.create_invocation!(%{
        model: "DummyT2I",
        type: :image,
        seed: 42,
        sequence_number: 0,
        output_image: <<0, 1, 2, 3>>,
        started_at: now,
        completed_at: now,
        run_id: run.id
      })

      assert_raise Ash.Error.Invalid, fn ->
        PanicTda.create_invocation!(%{
          model: "DummyT2I",
          type: :image,
          seed: 42,
          sequence_number: 0,
          output_image: <<4, 5, 6, 7>>,
          started_at: now,
          completed_at: now,
          run_id: run.id
        })
      end
    end

    test "rejects text invocation with output_image set" do
      experiment = create_experiment()
      run = create_run(experiment)
      now = DateTime.utc_now()

      assert_raise Ash.Error.Invalid, fn ->
        PanicTda.create_invocation!(%{
          model: "DummyI2T",
          type: :text,
          seed: 42,
          sequence_number: 0,
          output_text: "hello",
          output_image: <<0, 1, 2, 3>>,
          started_at: now,
          completed_at: now,
          run_id: run.id
        })
      end
    end

    test "rejects image invocation with output_text set" do
      experiment = create_experiment()
      run = create_run(experiment)
      now = DateTime.utc_now()

      assert_raise Ash.Error.Invalid, fn ->
        PanicTda.create_invocation!(%{
          model: "DummyT2I",
          type: :image,
          seed: 42,
          sequence_number: 0,
          output_text: "hello",
          output_image: <<0, 1, 2, 3>>,
          started_at: now,
          completed_at: now,
          run_id: run.id
        })
      end
    end
  end

  describe "timestamp ordering" do
    test "rejects invocation where completed_at is before started_at" do
      experiment = create_experiment()
      run = create_run(experiment)
      started = ~U[2024-01-01 12:00:00.000000Z]
      completed = ~U[2024-01-01 11:00:00.000000Z]

      assert_raise Ash.Error.Invalid, fn ->
        PanicTda.create_invocation!(%{
          model: "DummyT2I",
          type: :image,
          seed: 42,
          sequence_number: 0,
          output_image: <<0, 1, 2, 3>>,
          started_at: started,
          completed_at: completed,
          run_id: run.id
        })
      end
    end

    test "accepts invocation where completed_at equals started_at" do
      experiment = create_experiment()
      run = create_run(experiment)
      now = ~U[2024-01-01 12:00:00.000000Z]

      inv =
        PanicTda.create_invocation!(%{
          model: "DummyT2I",
          type: :image,
          seed: 42,
          sequence_number: 0,
          output_image: <<0, 1, 2, 3>>,
          started_at: now,
          completed_at: now,
          run_id: run.id
        })

      assert inv.started_at == now
      assert inv.completed_at == now
    end

    test "rejects embedding where completed_at is before started_at" do
      experiment = create_experiment()
      run = create_run(experiment)
      now = DateTime.utc_now()

      inv =
        PanicTda.create_invocation!(%{
          model: "DummyT2I",
          type: :image,
          seed: 42,
          sequence_number: 0,
          output_image: <<0, 1, 2, 3>>,
          started_at: now,
          completed_at: now,
          run_id: run.id
        })

      vector = [0.1, 0.2, 0.3, 0.4] |> Nx.tensor(type: :f32) |> Nx.to_binary()

      assert_raise Ash.Error.Invalid, fn ->
        PanicTda.create_embedding!(%{
          embedding_model: "DummyText",
          vector: vector,
          started_at: ~U[2024-01-01 12:00:00.000000Z],
          completed_at: ~U[2024-01-01 11:00:00.000000Z],
          invocation_id: inv.id
        })
      end
    end

    test "rejects persistence diagram where completed_at is before started_at" do
      experiment = create_experiment()
      run = create_run(experiment)

      assert_raise Ash.Error.Invalid, fn ->
        PanicTda.create_persistence_diagram!(%{
          embedding_model: "DummyText",
          diagram_data: %{dgms: [], entropy: []},
          started_at: ~U[2024-01-01 12:00:00.000000Z],
          completed_at: ~U[2024-01-01 11:00:00.000000Z],
          run_id: run.id
        })
      end
    end

    test "rejects clustering result where completed_at is before started_at" do
      experiment = create_experiment()

      assert_raise Ash.Error.Invalid, fn ->
        PanicTda.create_clustering_result!(%{
          embedding_model: "DummyText",
          algorithm: "hdbscan",
          started_at: ~U[2024-01-01 12:00:00.000000Z],
          completed_at: ~U[2024-01-01 11:00:00.000000Z],
          experiment_id: experiment.id
        })
      end
    end
  end
end
