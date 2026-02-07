defmodule PanicTdaTest do
  use ExUnit.Case

  setup do
    :ok = Ecto.Adapters.SQL.Sandbox.checkout(PanicTda.Repo)
    :ok
  end

  test "creates an experiment with runs" do
    experiment =
      PanicTda.create_experiment!(%{
        network: ["DummyT2I", "DummyI2T"],
        num_runs: 2,
        prompts: ["A beautiful sunset"],
        embedding_models: ["Nomic"],
        max_length: 10
      })

    assert experiment.id != nil
    assert experiment.network == ["DummyT2I", "DummyI2T"]
    assert experiment.num_runs == 2
    assert experiment.max_length == 10

    run =
      PanicTda.create_run!(%{
        network: ["DummyT2I", "DummyI2T"],
        run_number: 0,
        max_length: 10,
        initial_prompt: "A beautiful sunset",
        experiment_id: experiment.id
      })

    assert run.id != nil
    assert run.network == ["DummyT2I", "DummyI2T"]
    assert run.run_number == 0
  end

  test "creates an invocation with embedding" do
    experiment =
      PanicTda.create_experiment!(%{
        network: ["DummyT2I"],
        prompts: ["Test"],
        embedding_models: ["Nomic"],
        max_length: 5
      })

    run =
      PanicTda.create_run!(%{
        network: ["DummyT2I"],
        run_number: 0,
        max_length: 5,
        initial_prompt: "Test",
        experiment_id: experiment.id
      })

    now = DateTime.utc_now()

    invocation =
      PanicTda.create_invocation!(%{
        model: "DummyT2I",
        type: :image,
        sequence_number: 0,
        output_image: <<0, 1, 2, 3>>,
        started_at: now,
        completed_at: now,
        run_id: run.id
      })

    assert invocation.id != nil
    assert invocation.type == :image
    assert invocation.output_image == <<0, 1, 2, 3>>

    vector = [0.1, 0.2, 0.3, 0.4] |> Nx.tensor(type: :f32) |> Nx.to_binary()

    embedding =
      PanicTda.create_embedding!(%{
        embedding_model: "Nomic",
        vector: vector,
        started_at: now,
        completed_at: now,
        invocation_id: invocation.id
      })

    assert embedding.id != nil
    assert %Nx.Tensor{} = embedding.vector
    assert Nx.shape(embedding.vector) == {4}
  end

  test "creates clustering result with embedding clusters" do
    experiment =
      PanicTda.create_experiment!(%{
        network: ["DummyT2I"],
        prompts: ["Test"],
        embedding_models: ["Nomic"],
        max_length: 5
      })

    run =
      PanicTda.create_run!(%{
        network: ["DummyT2I"],
        run_number: 0,
        max_length: 5,
        initial_prompt: "Test",
        experiment_id: experiment.id
      })

    now = DateTime.utc_now()

    inv =
      PanicTda.create_invocation!(%{
        model: "DummyT2I",
        type: :text,
        sequence_number: 0,
        output_text: "Result",
        started_at: now,
        completed_at: now,
        run_id: run.id
      })

    vector = [0.1, 0.2, 0.3, 0.4] |> Nx.tensor(type: :f32) |> Nx.to_binary()

    emb =
      PanicTda.create_embedding!(%{
        embedding_model: "Nomic",
        vector: vector,
        started_at: now,
        completed_at: now,
        invocation_id: inv.id
      })

    clustering =
      PanicTda.create_clustering_result!(%{
        embedding_model: "Nomic",
        algorithm: "hdbscan",
        parameters: %{min_cluster_size: 5},
        started_at: now,
        completed_at: now,
        experiment_id: experiment.id
      })

    cluster_assignment =
      PanicTda.create_embedding_cluster!(%{
        embedding_id: emb.id,
        clustering_result_id: clustering.id,
        medoid_embedding_id: emb.id
      })

    assert cluster_assignment.id != nil
    assert cluster_assignment.medoid_embedding_id == emb.id
  end

  test "creates persistence diagram" do
    experiment =
      PanicTda.create_experiment!(%{
        network: ["DummyT2I"],
        prompts: ["Test"],
        embedding_models: ["Nomic"],
        max_length: 5
      })

    run =
      PanicTda.create_run!(%{
        network: ["DummyT2I"],
        run_number: 0,
        max_length: 5,
        initial_prompt: "Test",
        experiment_id: experiment.id
      })

    now = DateTime.utc_now()
    diagram_data = %{dgms: [[{0.0, 0.5}, {0.1, 0.8}]], entropy: [0.5]}

    pd =
      PanicTda.create_persistence_diagram!(%{
        embedding_model: "Nomic",
        diagram_data: diagram_data,
        started_at: now,
        completed_at: now,
        run_id: run.id
      })

    assert pd.id != nil

    loaded = PanicTda.get_persistence_diagram!(pd.id)
    assert loaded.diagram_data == diagram_data
  end
end
