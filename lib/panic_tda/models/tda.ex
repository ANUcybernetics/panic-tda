defmodule PanicTda.Models.Tda do
  @moduledoc """
  Topological Data Analysis via Python interop.
  Computes persistent homology using giotto-ph's ripser_parallel.
  """

  def compute_persistence_diagram(env, point_cloud_binary, dimension, max_dim \\ 2) do
    point_cloud_b64 = Base.encode64(point_cloud_binary)

    case Snex.pyeval(
           env,
           """
           import numpy as np
           import base64

           point_cloud_bytes = base64.b64decode(point_cloud_b64)
           point_cloud = np.frombuffer(point_cloud_bytes, dtype=np.float32).reshape(-1, dimension)

           from gph import ripser_parallel
           from persim.persistent_entropy import persistent_entropy

           dgm = ripser_parallel(point_cloud, maxdim=max_dim, return_generators=False, n_threads=4)
           dgm["entropy"] = persistent_entropy(dgm["dgms"], normalize=False)

           result = {
               "dgms": [d.tolist() for d in dgm["dgms"]],
               "entropy": dgm["entropy"].tolist(),
               "num_edges": int(dgm.get("num_edges", 0))
           }
           """,
           %{
             "point_cloud_b64" => point_cloud_b64,
             "dimension" => dimension,
             "max_dim" => max_dim
           },
           returning: "result"
         ) do
      {:ok, result} ->
        {:ok,
         %{
           dgms: result["dgms"],
           entropy: result["entropy"],
           num_edges: result["num_edges"]
         }}

      error ->
        error
    end
  end
end
