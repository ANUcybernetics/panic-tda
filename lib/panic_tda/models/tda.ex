defmodule PanicTda.Models.Tda do
  @moduledoc """
  Topological Data Analysis via Python interop.
  Computes persistent homology using giotto-ph's ripser_parallel.
  """

  # The cost climbs steeply with the number of points: 0.03 s for a run of 150
  # text states, 7-11 s for 700 and up to 140 s for 1,400
  # (`analysis/pd_cost.py`). A diagram that times out is never retried into
  # success, so the ceiling sits far above anything a planned run needs.
  @pd_timeout 3_600_000

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

           return {
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
           timeout: @pd_timeout
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
