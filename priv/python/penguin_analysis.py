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
    by embedding model. Saves a PNG to out_path.
    """
    import csv
    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(
                {
                    "network": row["network"],
                    "embedding_model": row["embedding_model"],
                    "category": row["category"],
                    "ftle": float(row["ftle"]),
                }
            )

    networks = sorted({r["network"] for r in rows})
    embeddings = sorted({r["embedding_model"] for r in rows})
    categories = ["identical", "paraphrase"]

    ncols = min(3, max(1, len(networks)))
    nrows = max(1, math.ceil(len(networks) / ncols))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows), squeeze=False, sharey=True
    )

    emb_colours = {emb: f"C{i}" for i, emb in enumerate(embeddings)}
    cat_positions = {cat: i for i, cat in enumerate(categories)}
    jitter_rng = np.random.default_rng(0)

    for idx, network in enumerate(networks):
        ax = axes[idx // ncols][idx % ncols]
        for emb in embeddings:
            for cat in categories:
                values = [
                    r["ftle"]
                    for r in rows
                    if r["network"] == network
                    and r["embedding_model"] == emb
                    and r["category"] == cat
                ]
                if not values:
                    continue
                xs = cat_positions[cat] + jitter_rng.uniform(
                    -0.1, 0.1, size=len(values)
                )
                ax.scatter(
                    xs,
                    values,
                    color=emb_colours[emb],
                    alpha=0.75,
                    s=28,
                    label=emb if idx == 0 and cat == "identical" else None,
                )
        ax.set_xticks(list(cat_positions.values()))
        ax.set_xticklabels(list(cat_positions.keys()))
        ax.set_title(network.replace("|", " -> "), fontsize=10)
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")

    for idx in range(len(networks), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    axes[0][0].set_ylabel("FTLE (per step, natural log)")
    fig.suptitle("FTLE: identical-prompt vs paraphrase (penguin_campfire)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(embeddings))
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
