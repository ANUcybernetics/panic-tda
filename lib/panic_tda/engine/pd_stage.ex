defmodule PanicTda.Engine.PdStage do
  @moduledoc """
  Computes persistence diagrams for all embedding models in a run.
  """

  require Ash.Query
  alias PanicTda.Models.Tda

  def compute(env, run, embedding_models) do
    Enum.each(embedding_models, fn embedding_model ->
      :ok = compute_for_model(env, run, embedding_model)
    end)

    :ok
  end

  def resume(env, run, embedding_models) do
    Enum.each(embedding_models, fn embedding_model ->
      has_pd =
        PanicTda.PersistenceDiagram
        |> Ash.Query.filter(run_id == ^run.id and embedding_model == ^embedding_model)
        |> Ash.count!()
        |> Kernel.>(0)

      unless has_pd do
        :ok = compute_for_model(env, run, embedding_model)
      end
    end)

    :ok
  end

  defp compute_for_model(env, run, embedding_model) do
    # Ordered through the run's own invocations, not by loading each
    # embedding's: that load is one `id = ?` per embedding, and SQLite rejects
    # the query once a run has a thousand of them.
    position =
      PanicTda.Invocation
      |> Ash.Query.filter(run_id == ^run.id)
      |> Ash.Query.select([:id, :sequence_number])
      |> Ash.read!()
      |> Map.new(&{&1.id, &1.sequence_number})

    embeddings =
      PanicTda.Embedding
      |> Ash.Query.filter(invocation.run_id == ^run.id and embedding_model == ^embedding_model)
      |> Ash.read!()
      |> Enum.sort_by(&Map.fetch!(position, &1.invocation_id))

    if embeddings == [] do
      :ok
    else
      started_at = DateTime.utc_now()

      vectors = Enum.map(embeddings, & &1.vector)
      dimension = Nx.size(hd(vectors))
      point_cloud_binary = vectors |> Nx.stack() |> Nx.to_binary()

      {:ok, diagram_data} = Tda.compute_persistence_diagram(env, point_cloud_binary, dimension)
      completed_at = DateTime.utc_now()

      PanicTda.create_persistence_diagram!(%{
        embedding_model: embedding_model,
        diagram_data: diagram_data,
        started_at: started_at,
        completed_at: completed_at,
        run_id: run.id
      })

      :ok
    end
  end
end
