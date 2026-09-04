defmodule EmbeddingsRecomputeTest do
  @moduledoc """
  The recompute exists to change vectors without disturbing anything that
  points at them: `embedding_clusters` reference embeddings by id, so writing
  new rows instead of updating in place would orphan every cluster assignment
  in the database.
  """

  use ExUnit.Case, async: false

  alias PanicTda.Embeddings.Recompute
  alias PanicTda.Models.{Embeddings, PythonInterpreter}

  # One interpreter for the module: it holds no database connection, and
  # starting five of them costs more than the tests themselves.
  setup_all do
    {:ok, interpreter} = PythonInterpreter.start_link()
    {:ok, env} = Snex.make_env(interpreter)
    on_exit(fn -> if Process.alive?(interpreter), do: GenServer.stop(interpreter) end)
    %{env: env}
  end

  setup do
    :ok = Ecto.Adapters.SQL.Sandbox.checkout(PanicTda.Repo)
    :ok
  end

  defp fixture(texts) do
    experiment =
      PanicTda.create_experiment!(%{
        networks: [["DummyT2I", "DummyI2T"]],
        prompts: ["test prompt"],
        embedding_models: ["DummyText"],
        max_length: 4
      })

    run =
      PanicTda.create_run!(%{
        network: ["DummyT2I", "DummyI2T"],
        run_number: 0,
        max_length: 4,
        initial_prompt: "test prompt",
        experiment_id: experiment.id
      })

    for {text, i} <- Enum.with_index(texts) do
      invocation =
        PanicTda.create_invocation!(%{
          model: "DummyI2T",
          type: :text,
          sequence_number: i,
          output_text: text,
          started_at: DateTime.utc_now(),
          completed_at: DateTime.utc_now(),
          run_id: run.id
        })

      # a deliberately wrong vector, standing in for the mean-pooled ones
      PanicTda.create_embedding!(%{
        embedding_model: "DummyText",
        vector: List.duplicate(0.0, 256),
        started_at: DateTime.utc_now(),
        completed_at: DateTime.utc_now(),
        invocation_id: invocation.id
      })
    end
  end

  defp stored_vectors do
    PanicTda.list_embeddings!()
    |> Enum.sort_by(& &1.id)
    |> Enum.map(&{&1.id, Nx.to_binary(&1.vector)})
  end

  defp expected(env, texts) do
    {:ok, vectors} = Embeddings.embed(env, "DummyText", texts)
    vectors
  end

  test "replaces every vector while keeping the embedding's id", %{env: env} do
    texts = ["alpha", "beta", "gamma"]
    fixture(texts)
    before = stored_vectors()

    Recompute.run(env, models: ["DummyText"], batch: 2)

    now = stored_vectors()
    assert Enum.map(before, &elem(&1, 0)) == Enum.map(now, &elem(&1, 0))
    assert Enum.map(now, &elem(&1, 1)) == expected(env, texts)
    refute Enum.any?(before, fn {id, v} -> {id, v} in now end)
  end

  test "leaves cluster assignments pointing at their embeddings", %{env: env} do
    fixture(["alpha", "beta"])
    [embedding | _] = Enum.sort_by(PanicTda.list_embeddings!(), & &1.id)

    clustering =
      PanicTda.create_clustering_result!(%{
        embedding_model: "DummyText",
        algorithm: "dummy",
        started_at: DateTime.utc_now()
      })

    assignment =
      PanicTda.create_embedding_cluster!(%{
        embedding_id: embedding.id,
        clustering_result_id: clustering.id
      })

    Recompute.run(env, models: ["DummyText"])

    reloaded = Ash.get!(PanicTda.EmbeddingCluster, assignment.id, load: [:embedding])
    assert reloaded.embedding.id == embedding.id
    assert Nx.to_binary(reloaded.embedding.vector) == hd(expected(env, ["alpha"]))
  end

  test "dry run writes nothing", %{env: env} do
    fixture(["alpha", "beta"])
    before = stored_vectors()

    Recompute.run(env, models: ["DummyText"], dry_run: true)

    assert stored_vectors() == before
  end

  test "resumes after a given id, leaving earlier rows alone", %{env: env} do
    fixture(["alpha", "beta", "gamma"])
    [{first_id, first_vector} | rest] = stored_vectors()

    Recompute.run(env, models: ["DummyText"], after: first_id)

    now = stored_vectors()
    assert {first_id, first_vector} == hd(now)
    refute Enum.any?(rest, fn {id, v} -> {id, v} in now end)
  end

  test "reports progress with the id needed to resume", %{env: env} do
    fixture(["alpha", "beta", "gamma"])
    test = self()

    Recompute.run(env,
      models: ["DummyText"],
      batch: 2,
      on_progress: &send(test, {:progress, &1})
    )

    assert_received {:progress, %{model: "DummyText", total: 3, done: 0}}
    assert_received {:progress, %{done: 2, after: _}}
    assert_received {:progress, %{done: 3, after: last}}
    assert last == PanicTda.list_embeddings!() |> Enum.map(& &1.id) |> Enum.max()
  end
end
