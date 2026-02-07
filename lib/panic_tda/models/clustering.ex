defmodule PanicTda.Models.Clustering do
  @moduledoc """
  HDBSCAN clustering via Python interop.
  Clusters embeddings using scikit-learn's HDBSCAN on normalised vectors.
  """

  def hdbscan(env, embeddings_binary, n_embeddings, epsilon \\ 0.0) do
    embeddings_b64 = Base.encode64(embeddings_binary)

    case Snex.pyeval(
           env,
           """
           import numpy as np
           import base64
           from sklearn.cluster import HDBSCAN as _HDBSCAN

           raw = base64.b64decode(embeddings_b64)
           embeddings = np.frombuffer(raw, dtype=np.float32).reshape(n_embeddings, -1)

           n_samples = embeddings.shape[0]
           min_cluster_size = max(2, int(n_samples * 0.001))
           min_samples = max(2, int(n_samples * 0.001))

           norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
           embeddings_norm = embeddings / norms

           hdb = _HDBSCAN(
               min_cluster_size=min_cluster_size,
               min_samples=min_samples,
               cluster_selection_epsilon=float(epsilon),
               allow_single_cluster=True,
               metric="euclidean",
               store_centers="medoid",
               n_jobs=-1,
           ).fit(embeddings_norm.astype(np.float64))

           labels = hdb.labels_.tolist()

           medoid_indices = {}
           if hasattr(hdb, "medoids_") and hdb.medoids_ is not None and len(hdb.medoids_) > 0:
               unique_labels = sorted(set(l for l in labels if l != -1))
               for i, label in enumerate(unique_labels):
                   if i < len(hdb.medoids_):
                       normalized_medoid = hdb.medoids_[i]
                       cluster_mask = hdb.labels_ == label
                       cluster_indices = np.where(cluster_mask)[0]
                       cluster_embeddings_n = embeddings_norm[cluster_mask]
                       distances = np.sum((cluster_embeddings_n - normalized_medoid) ** 2, axis=1)
                       best_idx = np.argmin(distances)
                       medoid_indices[int(label)] = int(cluster_indices[best_idx])

           result = {"labels": labels, "medoid_indices": medoid_indices}
           """,
           %{
             "embeddings_b64" => embeddings_b64,
             "n_embeddings" => n_embeddings,
             "epsilon" => epsilon
           },
           returning: "result"
         ) do
      {:ok, result} ->
        {:ok,
         %{
           labels: result["labels"],
           medoid_indices:
             Map.new(result["medoid_indices"], fn {k, v} ->
               {if(is_binary(k), do: String.to_integer(k), else: k), v}
             end)
         }}

      error ->
        error
    end
  end
end
