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

    test "DummyT2I generates images", %{env: env} do
      {:ok, image} = GenAI.invoke(env, "DummyT2I", "A test prompt")

      assert is_binary(image)
      assert byte_size(image) > 0
    end

    test "DummyI2T generates captions", %{env: env} do
      {:ok, image} = GenAI.invoke(env, "DummyT2I", "A test prompt")
      {:ok, caption} = GenAI.invoke(env, "DummyI2T", image)

      assert is_binary(caption)
      assert String.starts_with?(caption, "dummy caption:")
    end

    test "DummyText generates embeddings", %{env: env} do
      {:ok, [emb1, emb2]} = Embeddings.embed(env, "DummyText", ["hello", "world"])

      assert is_binary(emb1)
      assert is_binary(emb2)
      assert byte_size(emb1) == 256 * 4
      assert byte_size(emb2) == 256 * 4
    end
  end

  describe "full pipeline" do
    test "executes a simple T2I -> I2T trajectory" do
      experiment =
        PanicTda.create_experiment!(%{
          networks: [["DummyT2I", "DummyI2T"]],
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

    test "records a seed on every text-to-image invocation and none on captions" do
      experiment =
        PanicTda.create_experiment!(%{
          networks: [["DummyT2I", "DummyI2T"]],
          prompts: ["A beautiful sunset", "A quiet harbour"],
          embedding_models: ["DummyText"],
          max_length: 4,
          num_runs: 2
        })

      {:ok, completed} = Engine.perform_experiment(experiment.id)
      completed = Ash.load!(completed, runs: [:invocations])
      invocations = Enum.flat_map(completed.runs, & &1.invocations)

      {images, texts} = Enum.split_with(invocations, &(&1.type == :image))

      assert images != []
      assert Enum.all?(images, &is_integer(&1.seed)), "every T2I step must record its seed"
      assert Enum.all?(texts, &is_nil(&1.seed)), "I2T steps have no seed"

      # Seeds are drawn per invocation, not once per run or per step, so a
      # batch of runs at the same step must not share one.
      seeds = Enum.map(images, & &1.seed)
      assert length(Enum.uniq(seeds)) == length(seeds)
    end

    test "applies i2t_max_new_tokens to the Python model registry" do
      {:ok, interpreter} = PythonInterpreter.start_link()
      {:ok, env} = Snex.make_env(interpreter)
      on_exit(fn -> if Process.alive?(interpreter), do: GenServer.stop(interpreter) end)

      read_override = fn ->
        {:ok, value} = Snex.pyeval(env, "return panic_models._I2T_MAX_NEW_TOKENS_OVERRIDE", %{})
        value
      end

      capped =
        PanicTda.create_experiment!(%{
          networks: [["DummyT2I", "DummyI2T"]],
          prompts: ["A beautiful sunset"],
          embedding_models: ["DummyText"],
          max_length: 2,
          i2t_max_new_tokens: 512
        })

      {:ok, _} = Engine.perform_experiment(capped.id, env: env)
      assert read_override.() == 512

      uncapped =
        PanicTda.create_experiment!(%{
          networks: [["DummyT2I", "DummyI2T"]],
          prompts: ["A beautiful sunset"],
          embedding_models: ["DummyText"],
          max_length: 2
        })

      {:ok, _} = Engine.perform_experiment(uncapped.id, env: env)
      assert read_override.() == nil
    end

    test "creates embeddings for text invocations" do
      experiment =
        PanicTda.create_experiment!(%{
          networks: [["DummyT2I", "DummyI2T"]],
          prompts: ["Test prompt"],
          embedding_models: ["DummyText"],
          max_length: 4
        })

      {:ok, _} = Engine.perform_experiment(experiment.id)

      embeddings = PanicTda.list_embeddings!(query: [filter: [embedding_model: "DummyText"]])

      assert length(embeddings) == 2

      Enum.each(embeddings, fn emb ->
        assert %Nx.Tensor{} = emb.vector
        assert Nx.shape(emb.vector) == {256}
      end)
    end

    test "creates persistence diagrams via giotto-ph" do
      experiment =
        PanicTda.create_experiment!(%{
          networks: [["DummyT2I", "DummyI2T"]],
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
          prompts: ["Test"],
          embedding_models: ["DummyText", "DummyText2"],
          max_length: 4
        })

      {:ok, _} = Engine.perform_experiment(experiment.id)

      first =
        PanicTda.list_embeddings!(query: [filter: [embedding_model: "DummyText"]])

      second =
        PanicTda.list_embeddings!(query: [filter: [embedding_model: "DummyText2"]])

      assert length(first) == 2
      assert length(second) == 2
    end

    test "handles multiple runs and prompts" do
      experiment =
        PanicTda.create_experiment!(%{
          networks: [["DummyT2I", "DummyI2T"]],
          num_runs: 2,
          prompts: ["Prompt A", "Prompt B"],
          embedding_models: ["DummyText"],
          max_length: 2
        })

      {:ok, completed} = Engine.perform_experiment(experiment.id)
      completed = Ash.load!(completed, :runs)

      assert length(completed.runs) == 4
    end

    test "global clustering across all embeddings of a model" do
      experiment =
        PanicTda.create_experiment!(%{
          networks: [["DummyT2I", "DummyI2T"]],
          num_runs: 3,
          prompts: ["Alpha", "Beta", "Gamma"],
          embedding_models: ["DummyText"],
          max_length: 6
        })

      {:ok, _} = Engine.perform_experiment(experiment.id)

      assert PanicTda.list_clustering_results!() == []

      {:ok, interpreter} = PythonInterpreter.start_link()
      {:ok, env} = Snex.make_env(interpreter)

      try do
        :ok = PanicTda.Engine.ClusteringStage.recompute(env, ["DummyText"])
      after
        GenServer.stop(interpreter)
      end

      clustering_results = PanicTda.list_clustering_results!()
      assert length(clustering_results) >= 1

      Enum.each(clustering_results, fn cr ->
        assert cr.embedding_model == "DummyText"
        assert cr.algorithm == "evoc"
        assert is_map(cr.parameters)
        assert cr.parameters["metric"] == "euclidean_on_normalised"
        assert is_integer(cr.layer) and cr.layer >= 0
        assert cr.started_at != nil
        assert cr.completed_at != nil
      end)

      layer_indices = clustering_results |> Enum.map(& &1.layer) |> Enum.sort()
      assert layer_indices == Enum.to_list(0..(length(clustering_results) - 1))

      all_embeddings =
        PanicTda.list_embeddings!(query: [filter: [embedding_model: "DummyText"]])

      embedding_clusters = PanicTda.list_embedding_clusters!()
      assert length(embedding_clusters) == length(all_embeddings) * length(clustering_results)

      Enum.each(clustering_results, fn cr ->
        per_layer = Enum.filter(embedding_clusters, &(&1.clustering_result_id == cr.id))
        assert length(per_layer) == length(all_embeddings)
      end)
    end
  end
end
