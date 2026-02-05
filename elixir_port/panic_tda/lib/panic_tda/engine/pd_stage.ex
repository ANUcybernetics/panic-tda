defmodule PanicTda.Engine.PdStage do
  @moduledoc """
  Computes persistence diagrams for all embedding models in a run.
  """

  require Ash.Query
  alias PanicTda.Models.Tda

  def compute(env, run, embedding_models) do
    Enum.each(embedding_models, fn embedding_model ->
      compute_for_model(env, run, embedding_model)
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

      case Tda.compute_persistence_diagram(env, point_cloud_binary, dimension) do
        {:ok, diagram_data} ->
          completed_at = DateTime.utc_now()

          PanicTda.PersistenceDiagram
          |> Ash.Changeset.for_create(:create, %{
            embedding_model: embedding_model,
            diagram_data: diagram_data,
            started_at: started_at,
            completed_at: completed_at,
            run_id: run.id
          })
          |> Ash.create!()

          :ok

        {:error, reason} ->
          {:error, reason}
      end
    end
  end
end
