defmodule PanicTda.DataExportTest do
  use ExUnit.Case

  alias PanicTda.Engine
  alias PanicTda.Models.PythonInterpreter

  setup do
    :ok = Ecto.Adapters.SQL.Sandbox.checkout(PanicTda.Repo)
    :ok
  end

  defp run_experiment(overrides \\ %{}) do
    defaults = %{
      networks: [["DummyT2I", "DummyI2T"]],
      prompts: ["test prompt"],
      embedding_models: ["DummyText"],
      max_length: 4
    }

    experiment = PanicTda.create_experiment!(Map.merge(defaults, overrides))
    {:ok, experiment} = Engine.perform_experiment(experiment.id)
    experiment
  end

  defp tmp_dir do
    dir =
      System.tmp_dir!()
      |> Path.join("panic_tda_export_test_#{System.unique_integer([:positive])}")

    on_exit(fn -> File.rm_rf!(dir) end)
    dir
  end

  test "exports every table to parquet, dropping image bytes" do
    experiment = run_experiment()
    dir = tmp_dir()

    {:ok, results} = PanicTda.DataExport.export([experiment.id], dir)
    tables = Map.new(results, fn {table, path, rows} -> {table, {path, rows}} end)

    assert Enum.sort(Map.keys(tables)) == [
             :clustering_results,
             :embedding_clusters,
             :embeddings,
             :experiments,
             :invocations,
             :persistence_diagrams,
             :runs
           ]

    for {_table, {path, _rows}} <- tables, do: assert(File.exists?(path))

    assert {_, 1} = tables[:experiments]
    assert {_, 1} = tables[:runs]
    # 1 run × max_length invocations
    assert {_, 4} = tables[:invocations]
    # text outputs (seq 1, 3) get embedded by the one model
    assert {_, 2} = tables[:embeddings]

    # the image bytes must not be exported
    inv = Explorer.DataFrame.from_parquet!(elem(tables[:invocations], 0))
    refute "output_image" in Explorer.DataFrame.names(inv)

    # vectors round-trip as a list[f32] column
    emb = Explorer.DataFrame.from_parquet!(elem(tables[:embeddings], 0))
    assert {:list, {:f, 32}} == Explorer.DataFrame.dtypes(emb)["vector"]

    # nested/array attributes are JSON strings
    runs = Explorer.DataFrame.from_parquet!(elem(tables[:runs], 0))
    [network_json] = runs["network"] |> Explorer.Series.to_list()
    assert Jason.decode!(network_json) == ["DummyT2I", "DummyI2T"]
  end

  test "embedding_models option filters embeddings and persistence diagrams" do
    experiment = run_experiment(%{embedding_models: ["DummyText", "DummyText2"]})
    dir = tmp_dir()

    {:ok, results} =
      PanicTda.DataExport.export([experiment.id], dir, embedding_models: ["DummyText"])

    tables = Map.new(results, fn {table, _path, rows} -> {table, rows} end)

    # both models embed the 2 text outputs, but we keep only DummyText
    assert tables[:embeddings] == 2

    dir_all = tmp_dir()
    {:ok, all} = PanicTda.DataExport.export([experiment.id], dir_all)
    all_tables = Map.new(all, fn {table, _path, rows} -> {table, rows} end)
    assert all_tables[:embeddings] == 4
  end

  test "embed_prompts adds a synthetic t_0 row per run to invocations and embeddings" do
    experiment = run_experiment()
    dir = tmp_dir()

    {:ok, results} = PanicTda.DataExport.export([experiment.id], dir, embed_prompts: true)
    tables = Map.new(results, fn {table, path, rows} -> {table, {path, rows}} end)

    # 4 real invocations + 1 synthetic prompt row (1 run)
    assert {_, 5} = tables[:invocations]
    # 2 real text embeddings + 1 prompt embedding (1 run × 1 text model)
    assert {_, 3} = tables[:embeddings]

    inv_rows =
      elem(tables[:invocations], 0)
      |> Explorer.DataFrame.from_parquet!()
      |> Explorer.DataFrame.to_rows()

    prompt_rows = Enum.filter(inv_rows, &(&1["sequence_number"] == -1))

    assert [%{"output_text" => "test prompt", "type" => "text", "id" => id, "run_id" => run_id}] =
             prompt_rows

    assert id == "prompt-" <> run_id

    # the prompt embedding joins back to the synthetic invocation
    emb_rows =
      elem(tables[:embeddings], 0)
      |> Explorer.DataFrame.from_parquet!()
      |> Explorer.DataFrame.to_rows()

    prompt_emb = Enum.filter(emb_rows, &(&1["invocation_id"] == id))

    assert [%{"embedding_model" => "DummyText", "vector" => vector}] = prompt_emb
    assert is_list(vector) and vector != []
  end

  test "exports cluster assignments" do
    # enough embeddings for EVoC to cluster at all (>= 16)
    experiment =
      run_experiment(%{num_runs: 3, prompts: ["Alpha", "Beta", "Gamma"], max_length: 6})

    {:ok, interpreter} = PythonInterpreter.start_link()
    {:ok, env} = Snex.make_env(interpreter)

    try do
      :ok = PanicTda.Engine.ClusteringStage.recompute(env, ["DummyText"])
    after
      GenServer.stop(interpreter)
    end

    dir = tmp_dir()
    {:ok, results} = PanicTda.DataExport.export([experiment.id], dir)
    tables = Map.new(results, fn {table, path, rows} -> {table, {path, rows}} end)

    {_, n_layers} = tables[:clustering_results]
    {_, n_embeddings} = tables[:embeddings]
    {_, n_assignments} = tables[:embedding_clusters]

    assert n_layers >= 1
    # every embedding is assigned in every layer (outliers get a null medoid)
    assert n_assignments == n_embeddings * n_layers

    clusters =
      elem(tables[:embedding_clusters], 0)
      |> Explorer.DataFrame.from_parquet!()
      |> Explorer.DataFrame.to_rows()

    embedding_ids =
      elem(tables[:embeddings], 0)
      |> Explorer.DataFrame.from_parquet!()
      |> Explorer.DataFrame.to_rows()
      |> MapSet.new(& &1["id"])

    # assignments point at embeddings that are actually in the export
    assert Enum.all?(clusters, &MapSet.member?(embedding_ids, &1["embedding_id"]))
    # at least some rows are non-outliers, i.e. clustering did something
    assert Enum.any?(clusters, &(&1["medoid_embedding_id"] != nil))

  end

  test "exports multiple experiments into combined files" do
    a = run_experiment()
    b = run_experiment(%{prompts: ["another prompt"]})
    dir = tmp_dir()

    {:ok, results} = PanicTda.DataExport.export([a.id, b.id], dir)
    tables = Map.new(results, fn {table, _path, rows} -> {table, rows} end)

    assert tables[:experiments] == 2
    assert tables[:runs] == 2
    assert tables[:invocations] == 8
  end
end
