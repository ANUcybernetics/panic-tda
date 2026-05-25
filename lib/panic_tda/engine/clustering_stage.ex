defmodule PanicTda.Engine.ClusteringStage do
  @moduledoc """
  Global EVoC clustering across all embeddings for a given embedding model.
  Cluster identities are shared across experiments: cluster N in any
  experiment refers to the same semantic region.

  Run explicitly via `mix cluster.recompute`; not invoked automatically
  by the per-experiment pipeline.
  """

  require Ash.Query
  alias PanicTda.Models.Clustering

  def all_embedding_models do
    import Ecto.Query

    PanicTda.Repo.all(
      from e in "embeddings", distinct: true, select: e.embedding_model, order_by: e.embedding_model
    )
  end

  def recompute(env, embedding_models) do
    Enum.each(embedding_models, fn embedding_model ->
      :ok = recompute_for_model(env, embedding_model)
    end)

    :ok
  end

  defp delete_existing_clustering(embedding_model) do
    existing =
      PanicTda.ClusteringResult
      |> Ash.Query.filter(embedding_model == ^embedding_model)
      |> Ash.Query.load(:embedding_clusters)
      |> Ash.read!()

    Enum.each(existing, fn cr ->
      Enum.each(cr.embedding_clusters, &PanicTda.destroy_embedding_cluster!(&1))
      PanicTda.destroy_clustering_result!(cr)
    end)
  end

  defp recompute_for_model(env, embedding_model) do
    delete_existing_clustering(embedding_model)

    embeddings =
      PanicTda.Embedding
      |> Ash.Query.filter(embedding_model == ^embedding_model)
      |> Ash.read!()

    if length(embeddings) < 16 do
      :ok
    else
      started_at = DateTime.utc_now()
      n_embeddings = length(embeddings)
      vectors = Enum.map(embeddings, & &1.vector)
      stacked_binary = vectors |> Nx.stack() |> Nx.to_binary()

      {:ok, %{layers: layers, base_min_cluster_size: base_min_cluster_size}} =
        Clustering.evoc(env, stacked_binary, n_embeddings)

      completed_at = DateTime.utc_now()
      dimension = Nx.size(hd(vectors))

      parameters = %{
        "noise_level" => 0.5,
        "base_min_cluster_size" => base_min_cluster_size,
        "min_samples" => 5,
        "random_state" => 42,
        "metric" => "euclidean_on_normalised",
        "dimension" => dimension,
        "n_layers" => length(layers),
        "n_embeddings" => n_embeddings
      }

      Enum.each(layers, fn %{layer: layer_idx, labels: labels, medoid_indices: medoid_indices} ->
        clustering_result =
          PanicTda.create_clustering_result!(%{
            embedding_model: embedding_model,
            algorithm: "evoc",
            parameters: parameters,
            layer: layer_idx,
            started_at: started_at,
            completed_at: completed_at
          })

        medoid_embedding_ids =
          Map.new(medoid_indices, fn {label, idx} ->
            {label, Enum.at(embeddings, idx).id}
          end)

        embeddings
        |> Enum.zip(labels)
        |> Enum.each(fn {embedding, label} ->
          medoid_embedding_id =
            if label == -1, do: nil, else: Map.get(medoid_embedding_ids, label)

          PanicTda.create_embedding_cluster!(%{
            embedding_id: embedding.id,
            clustering_result_id: clustering_result.id,
            medoid_embedding_id: medoid_embedding_id
          })
        end)
      end)

      :ok
    end
  end
end
