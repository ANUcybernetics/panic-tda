defmodule QueryPlanTest do
  @moduledoc """
  Every update must find its row through an index.

  Ash's atomic update builder writes the primary key predicate as
  `CAST(id AS TEXT) = CAST(? AS TEXT)`. SQLite cannot answer that from the
  primary key index, so it falls back to a full table scan --- which is
  invisible on a test database and costs 72ms a row on the real one, where
  `invocations` and `embeddings` are gigabytes. `require_atomic?(false)` is
  the current fix, but this asserts the outcome rather than the fix, so any
  other route to an indexed update passes too.
  """

  use ExUnit.Case, async: false

  setup do
    :ok = Ecto.Adapters.SQL.Sandbox.checkout(PanicTda.Repo)
    :ok
  end

  # Runs `fun` and returns the SQL and params of the UPDATE it issued.
  defp captured_update(fun) do
    ref = make_ref()
    test = self()
    handler = {__MODULE__, ref}

    :telemetry.attach(
      handler,
      [:panic_tda, :repo, :query],
      fn _event, _measure, meta, _cfg ->
        if String.starts_with?(meta.query, "UPDATE ") do
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

    receive do
      {^ref, query, params} -> {query, params}
    after
      0 -> flunk("no UPDATE was issued")
    end
  end

  defp plan(query, params) do
    %{rows: rows} =
      Ecto.Adapters.SQL.query!(PanicTda.Repo, "EXPLAIN QUERY PLAN " <> query, params)

    rows |> Enum.map(&List.last/1) |> Enum.join("\n")
  end

  # SQLite names the alias rather than the table in a plan (`SCAN i0`), so
  # match on the access method: SEARCH uses an index, SCAN reads every row.
  defp assert_indexed(fun, table) do
    {query, params} = captured_update(fun)
    detail = plan(query, params)

    assert detail =~ "SEARCH",
           """
           update on #{table} full-scans the table instead of using an index.

           plan:  #{detail}
           query: #{query}
           """

    refute detail =~ "SCAN", "unexpected scan updating #{table}: #{detail}"
  end

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

  defp invocation do
    PanicTda.create_invocation!(%{
      model: "DummyT2I",
      type: :text,
      sequence_number: 0,
      output_text: "hello",
      started_at: DateTime.utc_now(),
      completed_at: DateTime.utc_now(),
      run_id: run().id
    })
  end

  defp embedding do
    PanicTda.create_embedding!(%{
      embedding_model: "DummyText",
      vector: [1.0, 2.0, 3.0],
      started_at: DateTime.utc_now(),
      completed_at: DateTime.utc_now(),
      invocation_id: invocation().id
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

  test "experiment" do
    e = experiment()
    assert_indexed(fn -> update!(e, %{max_length: 8}) end, "experiments")
  end

  test "run" do
    r = run()
    assert_indexed(fn -> update!(r, %{run_number: 1}) end, "runs")
  end

  test "invocation" do
    i = invocation()
    assert_indexed(fn -> update!(i, %{output_text: "goodbye"}) end, "invocations")
  end

  test "embedding" do
    e = embedding()
    assert_indexed(fn -> update!(e, %{vector: [4.0, 5.0, 6.0]}) end, "embeddings")
  end

  test "persistence diagram" do
    d =
      PanicTda.create_persistence_diagram!(%{
        embedding_model: "DummyText",
        started_at: DateTime.utc_now(),
        run_id: run().id
      })

    assert_indexed(
      fn -> update!(d, %{completed_at: DateTime.utc_now()}) end,
      "persistence_diagrams"
    )
  end

  test "clustering result" do
    c = clustering_result()

    assert_indexed(
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

    assert_indexed(
      fn -> update!(ec, %{medoid_embedding_id: e.id}) end,
      "embedding_clusters"
    )
  end
end
