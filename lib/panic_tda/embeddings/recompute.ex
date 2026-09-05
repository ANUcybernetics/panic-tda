defmodule PanicTda.Embeddings.Recompute do
  @moduledoc """
  Re-embeds stored embeddings' invocation text and writes the new vector over
  the old one.

  Vectors are updated in place rather than destroyed and recreated, so
  embedding ids survive and the `embedding_clusters` rows pointing at them are
  not orphaned. The clustering is computed from the vectors, so it still has to
  be recomputed afterwards.

  Needed whenever the embedding path changes underneath stored data: every
  vector written before 2026-09-03 was mean-pooled, which Qwen3-Embedding's
  last-token pooling makes wrong (TASK-96).
  """

  require Ash.Query
  require Ash.Sort

  alias PanicTda.Models.Embeddings

  @batch 64

  @doc """
  Recomputes every stored vector for `models`.

  Options:

    * `:models` --- embedding models to recompute (default: all registered)
    * `:experiments` --- experiment id prefixes to restrict to
    * `:batch` --- rows per page and per embedding call
    * `:dry_run` --- read and embed, but write nothing
    * `:after` --- resume from an id, as printed by `:on_progress`
    * `:on_progress` --- called with a progress map after each page

  """
  def run(env, opts \\ []) do
    models = Keyword.get_lazy(opts, :models, &Embeddings.list_models/0)
    experiments = Keyword.get(opts, :experiments, [])
    batch = Keyword.get(opts, :batch, @batch)
    dry_run? = Keyword.get(opts, :dry_run, false)
    after_id = Keyword.get(opts, :after)
    progress = Keyword.get(opts, :on_progress, fn _ -> :ok end)

    Enum.each(models, fn model ->
      total = model |> base_query(experiments, after_id) |> Ash.count!()

      progress.(%{model: model, total: total, done: 0, dry_run: dry_run?})

      if total > 0 do
        page(env, model, experiments, batch, dry_run?, after_id, 0, total, progress)
      end
    end)
  end

  # Paged rather than read in one go: loading :invocation for thousands of rows
  # at once builds a query SQLite rejects at its 1000-deep expression limit.
  #
  # Keyset, not offset: `OFFSET n` makes SQLite walk and discard n rows of a
  # table whose every row carries a vector blob, so page cost grows with the
  # offset until the read outlives the connection timeout. Seeking on `id >`
  # keeps every page the same cost.
  defp page(env, model, experiments, batch, dry_run?, after_id, done, total, progress) do
    chunk =
      model
      |> base_query(experiments, after_id)
      |> Ash.Query.sort([{keyset_order(), :asc}])
      |> Ash.Query.limit(batch)
      |> Ash.Query.load(:invocation)
      |> Ash.read!()

    if chunk == [] do
      :ok
    else
      last_id = chunk |> List.last() |> Map.fetch!(:id)

      chunk
      |> Enum.filter(&(&1.invocation && &1.invocation.output_text))
      |> then(&recompute_chunk(env, model, &1, dry_run?))

      done = done + length(chunk)
      progress.(%{model: model, total: total, done: done, after: last_id})

      page(env, model, experiments, batch, dry_run?, last_id, done, total, progress)
    end
  end

  # ash_sql renders the seek as `CAST(id AS TEXT) > CAST(? AS TEXT)`, which
  # the expression index on `CAST(id AS TEXT)` answers --- but only if the
  # ORDER BY is the same expression, otherwise the planner walks the primary
  # key in order and re-evaluates the cast from the first row every page.
  # `fragment("?", id)` renders as exactly that cast
  # (backlog/docs/ash-sql-cast-issue.md).
  defp keyset_order, do: Ash.Sort.expr_sort(fragment("?", id), :string)

  defp base_query(model, experiments, after_id) do
    PanicTda.Embedding
    |> Ash.Query.filter(embedding_model == ^model)
    |> then(fn q ->
      if after_id, do: Ash.Query.filter(q, id > ^after_id), else: q
    end)
    |> then(fn q ->
      Enum.reduce(experiments, q, fn prefix, acc ->
        Ash.Query.filter(acc, like(invocation.run.experiment_id, ^"#{prefix}%"))
      end)
    end)
  end

  defp recompute_chunk(_env, _model, [], _dry_run?), do: :ok

  defp recompute_chunk(env, model, chunk, dry_run?) do
    texts = Enum.map(chunk, & &1.invocation.output_text)
    {:ok, vectors} = Embeddings.embed(env, model, texts)
    completed_at = DateTime.utc_now()

    # One transaction per chunk: a separate one per row makes SQLite fsync
    # thousands of times and halves throughput. Raw Ecto because
    # `Ash.transaction` is a no-op on AshSqlite (`can?(_, :transact)` is
    # false).
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
