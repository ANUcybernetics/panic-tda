defmodule PanicTda.Models.Lyapunov do
  def compute_ftle(env, trajectories_binary, num_trajectories, num_timesteps, dimension) do
    trajectories_b64 = Base.encode64(trajectories_binary)

    case Snex.pyeval(
           env,
           """
           import numpy as np
           import base64
           from scipy.spatial.distance import pdist

           raw = base64.b64decode(trajectories_b64)
           trajectories = np.frombuffer(raw, dtype=np.float32).reshape(
               num_trajectories, num_timesteps, dimension
           )

           num_pairs = num_trajectories * (num_trajectories - 1) // 2
           divergence_curve = np.zeros(num_timesteps)

           for t in range(num_timesteps):
               distances = pdist(trajectories[:, t, :], metric="euclidean")
               divergence_curve[t] = np.mean(distances)

           epsilon = 1e-10
           clamped = np.maximum(divergence_curve, epsilon)
           ln_divergence = np.log(clamped)

           t_vals = np.arange(num_timesteps, dtype=np.float64)
           slope, intercept = np.polyfit(t_vals, ln_divergence, 1)

           ss_res = np.sum((ln_divergence - (slope * t_vals + intercept)) ** 2)
           ss_tot = np.sum((ln_divergence - np.mean(ln_divergence)) ** 2)
           r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else None

           return {
               "exponent": float(slope),
               "r_squared": float(r_squared) if r_squared is not None else None,
               "divergence_curve": divergence_curve.tolist(),
               "num_pairs": int(num_pairs),
               "num_timesteps": int(num_timesteps),
           }
           """,
           %{
             "trajectories_b64" => trajectories_b64,
             "num_trajectories" => num_trajectories,
             "num_timesteps" => num_timesteps,
             "dimension" => dimension
           }
         ) do
      {:ok, result} ->
        {:ok,
         %{
           exponent: result["exponent"],
           r_squared: result["r_squared"],
           divergence_curve: result["divergence_curve"],
           num_pairs: result["num_pairs"],
           num_timesteps: result["num_timesteps"]
         }}

      error ->
        error
    end
  end
end
