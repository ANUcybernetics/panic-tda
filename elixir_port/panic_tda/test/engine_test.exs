defmodule PanicTda.EngineTest do
  use ExUnit.Case

  require Ash.Query

  alias PanicTda.Engine
  alias PanicTda.Models.{GenAI, Embeddings, PythonInterpreter}

  setup do
    :ok = Ecto.Adapters.SQL.Sandbox.checkout(PanicTda.Repo)
    :ok
  end

  describe "Python interop" do
    test "DummyT2I generates deterministic images" do
      {:ok, interpreter} = PythonInterpreter.start_link()
      {:ok, env} = Snex.make_env(interpreter)

      {:ok, image1} = GenAI.invoke(env, "DummyT2I", "A test prompt", 42)
      {:ok, image2} = GenAI.invoke(env, "DummyT2I", "A test prompt", 42)

      assert is_binary(image1)
      assert is_binary(image2)
      assert image1 == image2

      GenServer.stop(interpreter)
    end

    test "DummyI2T generates deterministic captions" do
      {:ok, interpreter} = PythonInterpreter.start_link()
      {:ok, env} = Snex.make_env(interpreter)

      {:ok, image} = GenAI.invoke(env, "DummyT2I", "A test prompt", 42)
      {:ok, caption1} = GenAI.invoke(env, "DummyI2T", image, 42)
      {:ok, caption2} = GenAI.invoke(env, "DummyI2T", image, 42)

      assert is_binary(caption1)
      assert is_binary(caption2)
      assert caption1 == caption2
      assert String.starts_with?(caption1, "dummy caption:")

      GenServer.stop(interpreter)
    end

    test "DummyText generates embeddings" do
      {:ok, interpreter} = PythonInterpreter.start_link()
      {:ok, env} = Snex.make_env(interpreter)

      {:ok, [emb1, emb2]} = Embeddings.embed(env, "DummyText", ["hello", "world"])

      assert is_binary(emb1)
      assert is_binary(emb2)
      assert byte_size(emb1) == 768 * 4
      assert byte_size(emb2) == 768 * 4

      GenServer.stop(interpreter)
    end

    test "DummyVision generates embeddings from images" do
      {:ok, interpreter} = PythonInterpreter.start_link()
      {:ok, env} = Snex.make_env(interpreter)

      {:ok, image} = GenAI.invoke(env, "DummyT2I", "test", 42)
      {:ok, [emb]} = Embeddings.embed(env, "DummyVision", [image])

      assert is_binary(emb)
      assert byte_size(emb) == 768 * 4

      GenServer.stop(interpreter)
    end
  end

  describe "full pipeline" do
    test "executes a simple T2I -> I2T trajectory" do
      {:ok, experiment} =
        PanicTda.Experiment
        |> Ash.Changeset.for_create(:create, %{
          networks: [["DummyT2I", "DummyI2T"]],
          seeds: [42],
          prompts: ["A beautiful sunset"],
          embedding_models: ["DummyText"],
          max_length: 4
        })
        |> Ash.create()

      {:ok, completed_experiment} = Engine.perform_experiment(experiment.id)

      assert completed_experiment.started_at != nil
      assert completed_experiment.completed_at != nil

      completed_experiment = Ash.load!(completed_experiment, runs: [:invocations])

      assert length(completed_experiment.runs) == 1

      run = hd(completed_experiment.runs)
      assert length(run.invocations) == 4

      [inv0, inv1, inv2, inv3] = run.invocations
      assert inv0.type == :image
      assert inv0.model == "DummyT2I"
      assert inv0.output_image != nil

      assert inv1.type == :text
      assert inv1.model == "DummyI2T"
      assert inv1.output_text != nil

      assert inv2.type == :image
      assert inv2.model == "DummyT2I"

      assert inv3.type == :text
      assert inv3.model == "DummyI2T"
    end

    test "creates embeddings for text invocations" do
      {:ok, experiment} =
        PanicTda.Experiment
        |> Ash.Changeset.for_create(:create, %{
          networks: [["DummyT2I", "DummyI2T"]],
          seeds: [42],
          prompts: ["Test prompt"],
          embedding_models: ["DummyText"],
          max_length: 4
        })
        |> Ash.create()

      {:ok, _} = Engine.perform_experiment(experiment.id)

      embeddings =
        PanicTda.Embedding
        |> Ash.Query.filter(embedding_model == ^"DummyText")
        |> Ash.read!()

      assert length(embeddings) == 2

      Enum.each(embeddings, fn emb ->
        assert %Nx.Tensor{} = emb.vector
        assert Nx.shape(emb.vector) == {768}
      end)
    end

    test "creates embeddings for image invocations with DummyVision" do
      {:ok, experiment} =
        PanicTda.Experiment
        |> Ash.Changeset.for_create(:create, %{
          networks: [["DummyT2I", "DummyI2T"]],
          seeds: [42],
          prompts: ["Test prompt"],
          embedding_models: ["DummyVision"],
          max_length: 4
        })
        |> Ash.create()

      {:ok, _} = Engine.perform_experiment(experiment.id)

      embeddings =
        PanicTda.Embedding
        |> Ash.Query.filter(embedding_model == ^"DummyVision")
        |> Ash.read!()

      assert length(embeddings) == 2

      Enum.each(embeddings, fn emb ->
        assert %Nx.Tensor{} = emb.vector
        assert Nx.shape(emb.vector) == {768}
      end)
    end

    test "handles multiple embedding models" do
      {:ok, experiment} =
        PanicTda.Experiment
        |> Ash.Changeset.for_create(:create, %{
          networks: [["DummyT2I", "DummyI2T"]],
          seeds: [42],
          prompts: ["Test"],
          embedding_models: ["DummyText", "DummyVision"],
          max_length: 4
        })
        |> Ash.create()

      {:ok, _} = Engine.perform_experiment(experiment.id)

      text_embeddings =
        PanicTda.Embedding
        |> Ash.Query.filter(embedding_model == ^"DummyText")
        |> Ash.read!()

      vision_embeddings =
        PanicTda.Embedding
        |> Ash.Query.filter(embedding_model == ^"DummyVision")
        |> Ash.read!()

      assert length(text_embeddings) == 2
      assert length(vision_embeddings) == 2
    end

    test "handles multiple seeds and prompts" do
      {:ok, experiment} =
        PanicTda.Experiment
        |> Ash.Changeset.for_create(:create, %{
          networks: [["DummyT2I", "DummyI2T"]],
          seeds: [42, 123],
          prompts: ["Prompt A", "Prompt B"],
          embedding_models: ["DummyText"],
          max_length: 2
        })
        |> Ash.create()

      {:ok, completed} = Engine.perform_experiment(experiment.id)
      completed = Ash.load!(completed, :runs)

      assert length(completed.runs) == 4
    end

    test "deterministic outputs for same seed" do
      {:ok, exp1} =
        PanicTda.Experiment
        |> Ash.Changeset.for_create(:create, %{
          networks: [["DummyT2I", "DummyI2T"]],
          seeds: [42],
          prompts: ["Test"],
          embedding_models: ["DummyText"],
          max_length: 2
        })
        |> Ash.create()

      {:ok, exp2} =
        PanicTda.Experiment
        |> Ash.Changeset.for_create(:create, %{
          networks: [["DummyT2I", "DummyI2T"]],
          seeds: [42],
          prompts: ["Test"],
          embedding_models: ["DummyText"],
          max_length: 2
        })
        |> Ash.create()

      {:ok, _} = Engine.perform_experiment(exp1.id)
      {:ok, _} = Engine.perform_experiment(exp2.id)

      runs1 = Ash.load!(exp1, runs: [:invocations]).runs
      runs2 = Ash.load!(exp2, runs: [:invocations]).runs

      run1 = hd(runs1)
      run2 = hd(runs2)

      assert hd(run1.invocations).output_image == hd(run2.invocations).output_image

      text_inv1 = Enum.at(run1.invocations, 1)
      text_inv2 = Enum.at(run2.invocations, 1)
      assert text_inv1.output_text == text_inv2.output_text
    end
  end
end
