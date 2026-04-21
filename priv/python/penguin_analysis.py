"""
Post-hoc analysis helpers for the penguin_campfire experiment.

Invoked from Elixir via Snex; all public functions are pure Python and take
plain JSON-compatible arguments.
"""

from __future__ import annotations

import base64

import numpy as np
from scipy.spatial.distance import cdist


def cross_prompt_ftle(
    trajectories_a_b64: str,
    trajectories_b_b64: str,
    num_runs_a: int,
    num_runs_b: int,
    num_timesteps: int,
    dimension: int,
) -> dict:
    """
    Compute a cross-prompt FTLE: the slope of log(mean cross-prompt
    Euclidean distance) vs time.

    Inputs are two base64-encoded float32 arrays, each shaped
    (num_runs_x, num_timesteps, dimension). Returns a dict matching the
    shape of PanicTda.Models.Lyapunov.compute_ftle's result.
    """
    raw_a = base64.b64decode(trajectories_a_b64)
    raw_b = base64.b64decode(trajectories_b_b64)

    a = np.frombuffer(raw_a, dtype=np.float32).reshape(
        num_runs_a, num_timesteps, dimension
    )
    b = np.frombuffer(raw_b, dtype=np.float32).reshape(
        num_runs_b, num_timesteps, dimension
    )

    divergence_curve = np.zeros(num_timesteps)
    for t in range(num_timesteps):
        distances = cdist(a[:, t, :], b[:, t, :], metric="euclidean")
        divergence_curve[t] = float(distances.mean())

    epsilon = 1e-10
    clamped = np.maximum(divergence_curve, epsilon)
    ln_divergence = np.log(clamped)

    t_vals = np.arange(num_timesteps, dtype=np.float64)
    slope, intercept = np.polyfit(t_vals, ln_divergence, 1)

    ss_res = float(np.sum((ln_divergence - (slope * t_vals + intercept)) ** 2))
    ss_tot = float(np.sum((ln_divergence - np.mean(ln_divergence)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else None

    return {
        "exponent": float(slope),
        "r_squared": float(r_squared) if r_squared is not None else None,
        "divergence_curve": divergence_curve.tolist(),
        "num_pairs": int(num_runs_a * num_runs_b),
        "num_timesteps": int(num_timesteps),
    }
