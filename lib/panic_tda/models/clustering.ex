defmodule PanicTda.Models.Clustering do
  @moduledoc """
  EVoC clustering via Python interop.
  Produces a multi-layer hierarchical clustering of L2-normalised embeddings:
  layer 0 is the finest grain, later layers are progressively coarser.
  """

  def evoc(env, embeddings_binary, n_embeddings) do
    embeddings_b64 = Base.encode64(embeddings_binary)
    base_min_cluster_size = max(5, trunc(n_embeddings * 0.001))

    case Snex.pyeval(
           env,
           """
           import numpy as np
           import base64
           import evoc

           raw = base64.b64decode(embeddings_b64)
           embeddings = np.frombuffer(raw, dtype=np.float32).reshape(n_embeddings, -1)

           norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
           norms = np.clip(norms, 1e-12, None)
           embeddings_norm = (embeddings / norms).astype(np.float32)

           clusterer = evoc.EVoC(
               noise_level=0.5,
               base_min_cluster_size=base_min_cluster_size,
               min_samples=5,
               random_state=42,
           ).fit(embeddings_norm)

           cluster_layers = clusterer.cluster_layers_

           layers_out = []
           for layer_idx, labels_arr in enumerate(cluster_layers):
               labels_list = labels_arr.tolist()
               medoid_indices = {}
               unique_labels = sorted(set(l for l in labels_list if l != -1))
               for label in unique_labels:
                   mask = labels_arr == label
                   cluster_indices = np.where(mask)[0]
                   cluster_points = embeddings_norm[mask]
                   centroid = cluster_points.mean(axis=0)
                   distances = np.sum((cluster_points - centroid) ** 2, axis=1)
                   best_idx = int(np.argmin(distances))
                   medoid_indices[int(label)] = int(cluster_indices[best_idx])
               layers_out.append({
                   "layer": int(layer_idx),
                   "labels": labels_list,
                   "medoid_indices": medoid_indices,
               })

           return {"layers": layers_out, "base_min_cluster_size": int(base_min_cluster_size)}
           """,
           %{
             "embeddings_b64" => embeddings_b64,
             "n_embeddings" => n_embeddings,
             "base_min_cluster_size" => base_min_cluster_size
           },
           timeout: 600_000
         ) do
      {:ok, result} ->
        layers =
          Enum.map(result["layers"], fn layer ->
            %{
              layer: layer["layer"],
              labels: layer["labels"],
              medoid_indices:
                Map.new(layer["medoid_indices"], fn {k, v} ->
                  {if(is_binary(k), do: String.to_integer(k), else: k), v}
                end)
            }
          end)

        {:ok,
         %{
           layers: layers,
           base_min_cluster_size: result["base_min_cluster_size"]
         }}

      error ->
        error
    end
  end
end
