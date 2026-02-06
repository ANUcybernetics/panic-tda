defmodule PanicTda.EngineTest do
  use ExUnit.Case

  alias PanicTda.Engine
  alias PanicTda.Models.{GenAI, Embeddings, PythonInterpreter}

  setup do
    :ok = Ecto.Adapters.SQL.Sandbox.checkout(PanicTda.Repo)
    :ok
  end

  describe "Python interop" do
    setup do
      {:ok, interpreter} = PythonInterpreter.start_link()
      {:ok, env} = Snex.make_env(interpreter)
      on_exit(fn ->
        if Process.alive?(interpreter), do: GenServer.stop(interpreter)
      end)
      %{env: env}
    end

    test "DummyT2I generates deterministic images", %{env: env} do
      {:ok, image1} = GenAI.invoke(env, "DummyT2I", "A test prompt", 42)
      {:ok, image2} = GenAI.invoke(env, "DummyT2I", "A test prompt", 42)

      assert is_binary(image1)
      assert is_binary(image2)
      assert image1 == image2
    end

    test "DummyI2T generates deterministic captions", %{env: env} do
      {:ok, image} = GenAI.invoke(env, "DummyT2I", "A test prompt", 42)
      {:ok, caption1} = GenAI.invoke(env, "DummyI2T", image, 42)
      {:ok, caption2} = GenAI.invoke(env, "DummyI2T", image, 42)

      assert is_binary(caption1)
      assert is_binary(caption2)
      assert caption1 == caption2
      assert String.starts_with?(caption1, "dummy caption:")
    end

    test "DummyText generates embeddings", %{env: env} do
      {:ok, [emb1, emb2]} = Embeddings.embed(env, "DummyText", ["hello", "world"])

      assert is_binary(emb1)
      assert is_binary(emb2)
      assert byte_size(emb1) == 768 * 4
      assert byte_size(emb2) == 768 * 4
    end

    test "DummyVision generates embeddings from images", %{env: env} do
      {:ok, image} = GenAI.invoke(env, "DummyT2I", "test", 42)
      {:ok, [emb]} = Embeddings.embed(env, "DummyVision", [image])

      assert is_binary(emb)
      assert byte_size(emb) == 768 * 4
    end
  end

  describe "full pipeline" do
    test "executes a simple T2I -> I2T trajectory" do
      experiment =
        PanicTda.create_experiment!(%{
          networks: [["DummyT2I", "DummyI2T"]],
          seeds: [42],
          prompts: ["A beautiful sunset"],
          embedding_models: ["DummyText"],
          max_length: 4
        })

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
      experiment =
        PanicTda.create_experiment!(%{
          networks: [["DummyT2I", "DummyI2T"]],
          seeds: [42],
          prompts: ["Test prompt"],
          embedding_models: ["DummyText"],
          max_length: 4
        })

      {:ok, _} = Engine.perform_experiment(experiment.id)

      embeddings = PanicTda.list_embeddings!(query: [filter: [embedding_model: "DummyText"]])

      assert length(embeddings) == 2

      Enum.each(embeddings, fn emb ->
        assert %Nx.Tensor{} = emb.vector
        assert Nx.shape(emb.vector) == {768}
      end)
    end

    test "creates embeddings for image invocations with DummyVision" do
      experiment =
        PanicTda.create_experiment!(%{
          networks: [["DummyT2I", "DummyI2T"]],
          seeds: [42],
          prompts: ["Test prompt"],
          embedding_models: ["DummyVision"],
          max_length: 4
        })

      {:ok, _} = Engine.perform_experiment(experiment.id)

      embeddings = PanicTda.list_embeddings!(query: [filter: [embedding_model: "DummyVision"]])

      assert length(embeddings) == 2

      Enum.each(embeddings, fn emb ->
        assert %Nx.Tensor{} = emb.vector
        assert Nx.shape(emb.vector) == {768}
      end)
    end

    test "creates persistence diagrams via giotto-ph" do
      experiment =
        PanicTda.create_experiment!(%{
          networks: [["DummyT2I", "DummyI2T"]],
          seeds: [42],
          prompts: ["Test prompt"],
          embedding_models: ["DummyText"],
          max_length: 4
        })

      {:ok, _} = Engine.perform_experiment(experiment.id)

      pds = PanicTda.list_persistence_diagrams!()

      assert length(pds) == 1
      pd = hd(pds)

      assert pd.embedding_model == "DummyText"
      assert pd.diagram_data != nil
      assert is_map(pd.diagram_data)
      assert Map.has_key?(pd.diagram_data, :dgms)
      assert Map.has_key?(pd.diagram_data, :entropy)
      assert is_list(pd.diagram_data.dgms)
      assert length(pd.diagram_data.dgms) == 3
    end

    test "handles multiple embedding models" do
      experiment =
        PanicTda.create_experiment!(%{
          networks: [["DummyT2I", "DummyI2T"]],
          seeds: [42],
          prompts: ["Test"],
          embedding_models: ["DummyText", "DummyVision"],
          max_length: 4
        })

      {:ok, _} = Engine.perform_experiment(experiment.id)

      text_embeddings =
        PanicTda.list_embeddings!(query: [filter: [embedding_model: "DummyText"]])

      vision_embeddings =
        PanicTda.list_embeddings!(query: [filter: [embedding_model: "DummyVision"]])

      assert length(text_embeddings) == 2
      assert length(vision_embeddings) == 2
    end

    test "handles multiple seeds and prompts" do
      experiment =
        PanicTda.create_experiment!(%{
          networks: [["DummyT2I", "DummyI2T"]],
          seeds: [42, 123],
          prompts: ["Prompt A", "Prompt B"],
          embedding_models: ["DummyText"],
          max_length: 2
        })

      {:ok, completed} = Engine.perform_experiment(experiment.id)
      completed = Ash.load!(completed, :runs)

      assert length(completed.runs) == 4
    end

    test "creates clustering results across experiment" do
      experiment =
        PanicTda.create_experiment!(%{
          networks: [["DummyT2I", "DummyI2T"]],
          seeds: [42, 123, 456],
          prompts: ["Alpha", "Beta", "Gamma"],
          embedding_models: ["DummyText"],
          max_length: 6
        })

      {:ok, _} = Engine.perform_experiment(experiment.id)

      clustering_results = PanicTda.list_clustering_results!()
      assert length(clustering_results) == 1

      cr = hd(clustering_results)
      assert cr.embedding_model == "DummyText"
      assert cr.algorithm == "hdbscan"
      assert is_map(cr.parameters)
      assert cr.parameters["epsilon"] == 0.4
      assert cr.started_at != nil
      assert cr.completed_at != nil

      embedding_clusters = PanicTda.list_embedding_clusters!()
      assert length(embedding_clusters) > 0

      all_embeddings =
        PanicTda.list_embeddings!(query: [filter: [embedding_model: "DummyText"]])

      assert length(embedding_clusters) == length(all_embeddings)

      Enum.each(embedding_clusters, fn ec ->
        assert ec.clustering_result_id == cr.id
      end)
    end

    test "deterministic outputs for same seed" do
      exp1 =
        PanicTda.create_experiment!(%{
          networks: [["DummyT2I", "DummyI2T"]],
          seeds: [42],
          prompts: ["Test"],
          embedding_models: ["DummyText"],
          max_length: 2
        })

      exp2 =
        PanicTda.create_experiment!(%{
          networks: [["DummyT2I", "DummyI2T"]],
          seeds: [42],
          prompts: ["Test"],
          embedding_models: ["DummyText"],
          max_length: 2
        })

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
