defmodule PanicTdaTest do
  use ExUnit.Case

  setup do
    :ok = Ecto.Adapters.SQL.Sandbox.checkout(PanicTda.Repo)
    :ok
  end

  test "creates an experiment with runs" do
    {:ok, experiment} =
      PanicTda.Experiment
      |> Ash.Changeset.for_create(:create, %{
        networks: [["DummyT2I", "DummyI2T"]],
        seeds: [42, 123],
        prompts: ["A beautiful sunset"],
        embedding_models: ["Nomic"],
        max_length: 10
      })
      |> Ash.create()

    assert experiment.id != nil
    assert experiment.networks == [["DummyT2I", "DummyI2T"]]
    assert experiment.seeds == [42, 123]
    assert experiment.max_length == 10

    {:ok, run} =
      PanicTda.Run
      |> Ash.Changeset.for_create(:create, %{
        network: ["DummyT2I", "DummyI2T"],
        seed: 42,
        max_length: 10,
        initial_prompt: "A beautiful sunset",
        experiment_id: experiment.id
      })
      |> Ash.create()

    assert run.id != nil
    assert run.network == ["DummyT2I", "DummyI2T"]
    assert run.seed == 42
  end

  test "creates an invocation with embedding" do
    {:ok, experiment} =
      PanicTda.Experiment
      |> Ash.Changeset.for_create(:create, %{
        networks: [["DummyT2I"]],
        seeds: [42],
        prompts: ["Test"],
        embedding_models: ["Nomic"],
        max_length: 5
      })
      |> Ash.create()

    {:ok, run} =
      PanicTda.Run
      |> Ash.Changeset.for_create(:create, %{
        network: ["DummyT2I"],
        seed: 42,
        max_length: 5,
        initial_prompt: "Test",
        experiment_id: experiment.id
      })
      |> Ash.create()

    now = DateTime.utc_now()

    {:ok, invocation} =
      PanicTda.Invocation
      |> Ash.Changeset.for_create(:create, %{
        model: "DummyT2I",
        type: :image,
        seed: 42,
        sequence_number: 0,
        output_image: <<0, 1, 2, 3>>,
        started_at: now,
        completed_at: now,
        run_id: run.id
      })
      |> Ash.create()

    assert invocation.id != nil
    assert invocation.type == :image
    assert invocation.output_image == <<0, 1, 2, 3>>

    vector = [0.1, 0.2, 0.3, 0.4] |> Nx.tensor(type: :f32) |> Nx.to_binary()

    {:ok, embedding} =
      PanicTda.Embedding
      |> Ash.Changeset.for_create(:create, %{
        embedding_model: "Nomic",
        vector: vector,
        started_at: now,
        completed_at: now,
        invocation_id: invocation.id
      })
      |> Ash.create()

    assert embedding.id != nil
    assert %Nx.Tensor{} = embedding.vector
    assert Nx.shape(embedding.vector) == {4}
  end

  test "creates clustering result with embedding clusters" do
    {:ok, experiment} =
      PanicTda.Experiment
      |> Ash.Changeset.for_create(:create, %{
        networks: [["DummyT2I"]],
        seeds: [42],
        prompts: ["Test"],
        embedding_models: ["Nomic"],
        max_length: 5
      })
      |> Ash.create()

    {:ok, run} =
      PanicTda.Run
      |> Ash.Changeset.for_create(:create, %{
        network: ["DummyT2I"],
        seed: 42,
        max_length: 5,
        initial_prompt: "Test",
        experiment_id: experiment.id
      })
      |> Ash.create()

    now = DateTime.utc_now()

    {:ok, inv} =
      PanicTda.Invocation
      |> Ash.Changeset.for_create(:create, %{
        model: "DummyT2I",
        type: :text,
        seed: 42,
        sequence_number: 0,
        output_text: "Result",
        started_at: now,
        completed_at: now,
        run_id: run.id
      })
      |> Ash.create()

    vector = [0.1, 0.2, 0.3, 0.4] |> Nx.tensor(type: :f32) |> Nx.to_binary()

    {:ok, emb} =
      PanicTda.Embedding
      |> Ash.Changeset.for_create(:create, %{
        embedding_model: "Nomic",
        vector: vector,
        started_at: now,
        completed_at: now,
        invocation_id: inv.id
      })
      |> Ash.create()

    {:ok, clustering} =
      PanicTda.ClusteringResult
      |> Ash.Changeset.for_create(:create, %{
        embedding_model: "Nomic",
        algorithm: "hdbscan",
        parameters: %{min_cluster_size: 5},
        started_at: now,
        completed_at: now
      })
      |> Ash.create()

    {:ok, cluster_assignment} =
      PanicTda.EmbeddingCluster
      |> Ash.Changeset.for_create(:create, %{
        embedding_id: emb.id,
        clustering_result_id: clustering.id,
        medoid_embedding_id: emb.id
      })
      |> Ash.create()

    assert cluster_assignment.id != nil
    assert cluster_assignment.medoid_embedding_id == emb.id
  end

  test "creates persistence diagram" do
    {:ok, experiment} =
      PanicTda.Experiment
      |> Ash.Changeset.for_create(:create, %{
        networks: [["DummyT2I"]],
        seeds: [42],
        prompts: ["Test"],
        embedding_models: ["Nomic"],
        max_length: 5
      })
      |> Ash.create()

    {:ok, run} =
      PanicTda.Run
      |> Ash.Changeset.for_create(:create, %{
        network: ["DummyT2I"],
        seed: 42,
        max_length: 5,
        initial_prompt: "Test",
        experiment_id: experiment.id
      })
      |> Ash.create()

    now = DateTime.utc_now()
    diagram_data = %{dgms: [[{0.0, 0.5}, {0.1, 0.8}]], entropy: [0.5]}

    {:ok, pd} =
      PanicTda.PersistenceDiagram
      |> Ash.Changeset.for_create(:create, %{
        embedding_model: "Nomic",
        diagram_data: diagram_data,
        started_at: now,
        completed_at: now,
        run_id: run.id
      })
      |> Ash.create()

    assert pd.id != nil

    loaded = Ash.get!(PanicTda.PersistenceDiagram, pd.id)
    assert loaded.diagram_data == diagram_data
  end
end
