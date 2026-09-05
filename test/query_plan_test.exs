defmodule QueryPlanTest do
  @moduledoc """
  Every query the engine issues must find its rows through an index.

  ash_sql renders a comparison on a typed column as
  `CAST(col AS TEXT) = CAST(? AS TEXT)`, which SQLite cannot answer from a
  plain index, so every filter on a uuid column and every atomic update by
  primary key would otherwise scan the table --- invisible on a test database,
  about 175 ms a query on the real one (backlog/docs/ash-sql-cast-issue.md).
  The resources declare expression indexes on `CAST(col AS TEXT)`; this
  asserts the plans rather than the SQL, so any other route to an indexed
  query passes too.
  """

  use ExUnit.Case, async: false
  require Ash.Query
  require Ash.Sort

  setup do
    :ok = Ecto.Adapters.SQL.Sandbox.checkout(PanicTda.Repo)
    :ok
  end

  # Runs `fun` and returns the SQL and params of every statement it issued
  # that starts with `prefix`.
  defp captured(prefix, fun) do
    ref = make_ref()
    test = self()
    handler = {__MODULE__, ref}

    :telemetry.attach(
      handler,
      [:panic_tda, :repo, :query],
      fn _event, _measure, meta, _cfg ->
        if String.starts_with?(meta.query, prefix) do
          send(test, {ref, meta.query, meta.params})
        end
      end,
      nil
    )

    try do
      fun.()
    after
      :telemetry.detach(handler)
    end

    case collect(ref, []) do
      [] -> flunk("no #{String.trim(prefix)} was issued")
      statements -> statements
    end
  end

  defp collect(ref, acc) do
    receive do
      {^ref, query, params} -> collect(ref, [{query, params} | acc])
    after
      0 -> Enum.reverse(acc)
    end
  end

  defp plan(query, params) do
    %{rows: rows} =
      Ecto.Adapters.SQL.query!(PanicTda.Repo, "EXPLAIN QUERY PLAN " <> query, params)

    rows |> Enum.map(&List.last/1) |> Enum.join("\n")
  end

  # SQLite names the alias rather than the table in a plan (`SCAN i0`), so
  # match on the access method: SEARCH uses an index, SCAN reads every row.
  defp assert_indexed(prefix, fun, label) do
    for {query, params} <- captured(prefix, fun) do
      detail = plan(query, params)

      assert detail =~ "SEARCH",
             """
             #{label} full-scans instead of using an index.

             plan:  #{detail}
             query: #{query}
             """

      refute detail =~ "SCAN", "unexpected scan in #{label}: #{detail}"
    end
  end

  defp assert_indexed_update(fun, table), do: assert_indexed("UPDATE ", fun, "update on #{table}")
  defp assert_indexed_read(fun, label), do: assert_indexed("SELECT ", fun, label)

  defp experiment do
    PanicTda.create_experiment!(%{
      networks: [["DummyT2I", "DummyI2T"]],
      prompts: ["test prompt"],
      embedding_models: ["DummyText"],
      max_length: 4
    })
  end

  defp run do
    PanicTda.create_run!(%{
      network: ["DummyT2I", "DummyI2T"],
      run_number: 0,
      max_length: 4,
      initial_prompt: "test prompt",
      experiment_id: experiment().id
    })
  end

  defp invocation, do: invocation_for(run(), 0)

  defp invocation_for(run, seq) do
    PanicTda.create_invocation!(%{
      model: "DummyT2I",
      type: :text,
      sequence_number: seq,
      output_text: "hello",
      started_at: DateTime.utc_now(),
      completed_at: DateTime.utc_now(),
      run_id: run.id
    })
  end

  defp embedding, do: embedding_for(invocation())

  defp embedding_for(invocation) do
    PanicTda.create_embedding!(%{
      embedding_model: "DummyText",
      vector: [1.0, 2.0, 3.0],
      started_at: DateTime.utc_now(),
      completed_at: DateTime.utc_now(),
      invocation_id: invocation.id
    })
  end

  defp clustering_result do
    PanicTda.create_clustering_result!(%{
      embedding_model: "DummyText",
      algorithm: "dummy",
      started_at: DateTime.utc_now()
    })
  end

  defp update!(record, attrs) do
    record
    |> Ash.Changeset.for_update(:update, attrs)
    |> Ash.update!()
  end

  describe "updates" do
    test "experiment" do
      e = experiment()
      assert_indexed_update(fn -> update!(e, %{max_length: 8}) end, "experiments")
    end

    test "run" do
      r = run()
      assert_indexed_update(fn -> update!(r, %{run_number: 1}) end, "runs")
    end

    test "invocation" do
      i = invocation()
      assert_indexed_update(fn -> update!(i, %{output_text: "goodbye"}) end, "invocations")
    end

    test "embedding" do
      e = embedding()
      assert_indexed_update(fn -> update!(e, %{vector: [4.0, 5.0, 6.0]}) end, "embeddings")
    end

    test "persistence diagram" do
      d =
        PanicTda.create_persistence_diagram!(%{
          embedding_model: "DummyText",
          started_at: DateTime.utc_now(),
          run_id: run().id
        })

      assert_indexed_update(
        fn -> update!(d, %{completed_at: DateTime.utc_now()}) end,
        "persistence_diagrams"
      )
    end

    test "clustering result" do
      c = clustering_result()

      assert_indexed_update(
        fn -> update!(c, %{parameters: %{"k" => 3}}) end,
        "clustering_results"
      )
    end

    test "embedding cluster" do
      e = embedding()

      ec =
        PanicTda.create_embedding_cluster!(%{
          embedding_id: e.id,
          clustering_result_id: clustering_result().id
        })

      assert_indexed_update(
        fn -> update!(ec, %{medoid_embedding_id: e.id}) end,
        "embedding_clusters"
      )
    end
  end

  describe "reads the engine issues per run" do
    setup do
      r = run()
      i = invocation_for(r, 0)
      invocation_for(r, 1)
      e = embedding_for(i)
      %{run: r, embedding: e}
    end

    test "experiment by id", %{run: r} do
      assert_indexed_read(fn -> PanicTda.get_experiment!(r.experiment_id) end, "get_experiment")
    end

    test "runs of an experiment", %{run: r} do
      assert_indexed_read(
        fn ->
          PanicTda.Run |> Ash.Query.filter(experiment_id == ^r.experiment_id) |> Ash.read!()
        end,
        "runs by experiment_id"
      )
    end

    test "last invocation of a run (resume)", %{run: r} do
      assert_indexed_read(
        fn ->
          PanicTda.Invocation
          |> Ash.Query.filter(run_id == ^r.id)
          |> Ash.Query.sort(sequence_number: :desc)
          |> Ash.Query.limit(1)
          |> Ash.read!()
        end,
        "last invocation of a run"
      )
    end

    test "invocations loaded onto a run (embeddings stage)", %{run: r} do
      assert_indexed_read(fn -> Ash.load!(r, :invocations) end, "Ash.load!(run, :invocations)")
    end

    test "embeddings of a run with their invocations (pd stage)", %{run: r} do
      assert_indexed_read(
        fn ->
          PanicTda.Embedding
          |> Ash.Query.filter(invocation.run_id == ^r.id and embedding_model == "DummyText")
          |> Ash.Query.load(:invocation)
          |> Ash.read!()
        end,
        "embeddings via invocation.run_id"
      )
    end

    test "persistence diagram count for a run", %{run: r} do
      assert_indexed_read(
        fn ->
          PanicTda.PersistenceDiagram
          |> Ash.Query.filter(run_id == ^r.id and embedding_model == "DummyText")
          |> Ash.count!()
        end,
        "pd count"
      )
    end

    test "embedding count of an experiment (status)", %{run: r} do
      assert_indexed_read(
        fn ->
          PanicTda.Embedding
          |> Ash.Query.filter(invocation.run.experiment_id == ^r.experiment_id)
          |> Ash.count!()
        end,
        "embedding count via invocation.run.experiment_id"
      )
    end

    test "keyset page of embeddings (recompute)", %{embedding: e} do
      assert_indexed_read(
        fn ->
          PanicTda.Embedding
          |> Ash.Query.filter(embedding_model == "DummyText" and id > ^e.id)
          |> Ash.Query.sort([{Ash.Sort.expr_sort(fragment("?", id), :string), :asc}])
          |> Ash.Query.limit(64)
          |> Ash.read!()
        end,
        "keyset page id > after"
      )
    end
  end
end
