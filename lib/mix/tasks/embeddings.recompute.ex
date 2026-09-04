defmodule Mix.Tasks.Embeddings.Recompute do
  @shortdoc "Recompute stored embedding vectors in place"

  @moduledoc """
  Re-embeds every stored embedding's invocation text and writes the new vector
  over the old one.

      $ mix embeddings.recompute
      $ mix embeddings.recompute --model Qwen3Embed --experiment 01a060b4
      $ mix embeddings.recompute --dry-run

  Vectors are updated in place rather than destroyed and recreated, so
  embedding ids survive and `embedding_clusters` rows are not orphaned. The
  clustering is still computed from the vectors, so run `mix cluster.recompute`
  afterwards.

  Needed whenever the embedding path changes underneath stored data: every
  vector written before 2026-09-03 was mean-pooled, which Qwen3-Embedding's
  last-token pooling makes wrong (TASK-96).
  """

  use Mix.Task

  require Ash.Query

  alias PanicTda.Models.Embeddings

  @batch 64

  @impl Mix.Task
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        strict: [model: :keep, experiment: :keep, batch: :integer, dry_run: :boolean]
      )

    Mix.Task.run("ecto.create", ["--quiet"])
    Mix.Task.run("ecto.migrate", ["--quiet"])
    Mix.Task.run("app.start")

    batch = Keyword.get(opts, :batch, @batch)
    dry_run? = Keyword.get(opts, :dry_run, false)
    experiments = Keyword.get_values(opts, :experiment)

    models =
      case Keyword.get_values(opts, :model) do
        [] -> Embeddings.list_models()
        requested -> requested
      end

    {:ok, interpreter} = PanicTda.Models.PythonInterpreter.start_link()
    {:ok, env} = Snex.make_env(interpreter)

    try do
      Enum.each(models, &recompute_model(env, &1, experiments, batch, dry_run?))
    after
      GenServer.stop(interpreter)
    end
  end

  defp recompute_model(env, model, experiments, batch, dry_run?) do
    total = base_query(model, experiments) |> Ash.count!()

    if total == 0 do
      Mix.shell().info("#{model}: nothing to do")
    else
      Mix.shell().info("#{model}: #{total} embeddings#{if dry_run?, do: " (dry run)", else: ""}")
      started = System.monotonic_time(:millisecond)
      page(env, model, experiments, batch, dry_run?, 0, total, started)
    end
  end

  # Paged rather than read in one go: loading :invocation for thousands of rows
  # at once builds a query SQLite rejects at its 1000-deep expression limit.
  defp page(_env, _model, _experiments, _batch, _dry_run?, done, total, _started)
       when done >= total,
       do: :ok

  defp page(env, model, experiments, batch, dry_run?, done, total, started) do
    chunk =
      base_query(model, experiments)
      |> Ash.Query.sort(id: :asc)
      |> Ash.Query.limit(batch)
      |> Ash.Query.offset(done)
      |> Ash.Query.load(:invocation)
      |> Ash.read!()
      |> Enum.filter(&(&1.invocation && &1.invocation.output_text))

    if chunk == [] do
      :ok
    else
      recompute_chunk(env, model, chunk, dry_run?)
      done = done + length(chunk)
      elapsed = System.monotonic_time(:millisecond) - started
      rate = done * 1000 / max(elapsed, 1)

      Mix.shell().info(
        "  #{done}/#{total}  #{Float.round(rate, 1)}/s  eta #{div(round((total - done) / max(rate, 0.001)), 60)}m"
      )

      page(env, model, experiments, batch, dry_run?, done, total, started)
    end
  end

  defp base_query(model, experiments) do
    PanicTda.Embedding
    |> Ash.Query.filter(embedding_model == ^model)
    |> then(fn q ->
      Enum.reduce(experiments, q, fn prefix, acc ->
        Ash.Query.filter(acc, like(invocation.run.experiment_id, ^"#{prefix}%"))
      end)
    end)
  end

  defp recompute_chunk(env, model, chunk, dry_run?) do
    texts = Enum.map(chunk, & &1.invocation.output_text)
    {:ok, vectors} = Embeddings.embed(env, model, texts)
    completed_at = DateTime.utc_now()

    # One transaction per chunk: a separate one per row makes SQLite fsync
    # thousands of times and halves throughput.
    unless dry_run? do
      PanicTda.Repo.transaction(
        fn ->
          chunk
          |> Enum.zip(vectors)
          |> Enum.each(fn {embedding, vector} ->
            embedding
            |> Ash.Changeset.for_update(:update, %{vector: vector, completed_at: completed_at})
            |> Ash.update!()
          end)
        end,
        timeout: :infinity
      )
    end

    :ok
  end
end
