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


def plot_ftle_grid(csv_path: str, out_path: str) -> None:
    """
    Read the per-value FTLE CSV and produce a per-network panel grid of
    strip plots comparing identical-prompt and paraphrase FTLEs, coloured
    by embedding model. Saves a PDF (or other format inferred from
    out_path's extension) via altair + vl-convert.
    """
    import csv

    import altair as alt

    jitter_rng = np.random.default_rng(0)

    rows = []
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(
                {
                    "network": row["network"].replace("|", " → "),
                    "embedding_model": row["embedding_model"],
                    "category": row["category"],
                    "ftle": float(row["ftle"]),
                    "jitter": float(jitter_rng.uniform(-0.2, 0.2)),
                }
            )

    data = alt.Data(values=rows)

    chart = (
        alt.Chart(data)
        .mark_circle(opacity=0.75, size=50)
        .encode(
            x=alt.X(
                "category:N",
                title=None,
                sort=["identical", "paraphrase"],
                axis=alt.Axis(labelAngle=0),
            ),
            xOffset=alt.XOffset("jitter:Q", scale=alt.Scale(domain=[-0.5, 0.5])),
            y=alt.Y("ftle:Q", title="FTLE (per step, natural log)"),
            color=alt.Color(
                "embedding_model:N", legend=alt.Legend(title="Embedding model")
            ),
            tooltip=[
                alt.Tooltip("network:N"),
                alt.Tooltip("embedding_model:N"),
                alt.Tooltip("category:N"),
                alt.Tooltip("ftle:Q", format=".4f"),
            ],
        )
        .properties(width=180, height=180)
        .facet(
            facet=alt.Facet("network:N", title=None),
            columns=3,
            title=alt.TitleParams(
                "FTLE: identical-prompt vs paraphrase (penguin_campfire)",
                anchor="middle",
            ),
        )
        .resolve_scale(y="shared")
    )

    chart.save(out_path)


def plot_divergence_curves(
    out_path: str,
    network: str,
    embedding_model: str,
    identical_curve: list,
    paraphrase_curve: list,
) -> None:
    """
    Plot two divergence curves on a log-y axis: the mean within-prompt
    divergence (identical-prompt) and the mean between-prompt divergence
    (paraphrase), both for a single (network, embedding) cell.

    Inputs are lists of per-timestep mean distances (not yet logged).
    Saves the file at out_path via altair + vl-convert (format inferred
    from the extension).
    """
    import altair as alt

    rows = [
        {"step": t, "distance": float(d), "curve": "identical prompt"}
        for t, d in enumerate(identical_curve)
    ] + [
        {"step": t, "distance": float(d), "curve": "paraphrase"}
        for t, d in enumerate(paraphrase_curve)
    ]

    data = alt.Data(values=rows)

    title = f"Divergence curve: {network.replace('|', ' → ')}  ·  {embedding_model}"

    chart = (
        alt.Chart(data)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("step:Q", title="invocation step"),
            y=alt.Y(
                "distance:Q",
                title="mean pairwise Euclidean distance (log scale)",
                scale=alt.Scale(type="log"),
            ),
            color=alt.Color("curve:N", legend=alt.Legend(title=None)),
        )
        .properties(width=600, height=340, title=title)
    )

    chart.save(out_path)
