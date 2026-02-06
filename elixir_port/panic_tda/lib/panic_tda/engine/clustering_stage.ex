defmodule PanicTda.Engine.ClusteringStage do
  @moduledoc """
  Clusters embeddings across an entire experiment using HDBSCAN.
  Runs after all per-run stages complete.
  """

  require Ash.Query
  alias PanicTda.Models.Clustering

  def compute(env, experiment, embedding_models) do
    Enum.each(embedding_models, fn embedding_model ->
      compute_for_model(env, experiment, embedding_model)
    end)

    :ok
  end

  defp compute_for_model(env, experiment, embedding_model) do
    embeddings =
      PanicTda.Embedding
      |> Ash.Query.filter(
        embedding_model == ^embedding_model and
          invocation.run.experiment_id == ^experiment.id
      )
      |> Ash.read!()

    if length(embeddings) < 2 do
      :ok
    else
      started_at = DateTime.utc_now()
      n_embeddings = length(embeddings)
      vectors = Enum.map(embeddings, & &1.vector)
      stacked_binary = vectors |> Nx.stack() |> Nx.to_binary()

      case Clustering.hdbscan(env, stacked_binary, n_embeddings) do
        {:ok, %{labels: labels, medoid_indices: medoid_indices}} ->
          completed_at = DateTime.utc_now()

          dimension = Nx.size(hd(vectors))

          clustering_result =
            PanicTda.ClusteringResult
            |> Ash.Changeset.for_create(:create, %{
              embedding_model: embedding_model,
              algorithm: "hdbscan",
              parameters: %{
                "epsilon" => 0.4,
                "min_cluster_size_pct" => 0.001,
                "metric" => "euclidean_on_normalised",
                "dimension" => dimension
              },
              started_at: started_at,
              completed_at: completed_at
            })
            |> Ash.create!()

          medoid_embedding_ids =
            Map.new(medoid_indices, fn {label, idx} ->
              {label, Enum.at(embeddings, idx).id}
            end)

          embeddings
          |> Enum.zip(labels)
          |> Enum.each(fn {embedding, label} ->
            medoid_embedding_id =
              if label == -1, do: nil, else: Map.get(medoid_embedding_ids, label)

            PanicTda.EmbeddingCluster
            |> Ash.Changeset.for_create(:create, %{
              embedding_id: embedding.id,
              clustering_result_id: clustering_result.id,
              medoid_embedding_id: medoid_embedding_id
            })
            |> Ash.create!()
          end)

          :ok

        {:error, reason} ->
          {:error, reason}
      end
    end
  end
end
