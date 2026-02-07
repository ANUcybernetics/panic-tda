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
    embeddings =
      PanicTda.Embedding
      |> Ash.Query.filter(invocation.run_id == ^run.id and embedding_model == ^embedding_model)
      |> Ash.Query.load(:invocation)
      |> Ash.read!()
      |> Enum.sort_by(& &1.invocation.sequence_number)

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
