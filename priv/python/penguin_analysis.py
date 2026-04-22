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


def plot_ftle_heatmap(csv_path: str, out_path: str) -> None:
    """
    Read the per-value FTLE CSV and produce a faceted per-network prompt ×
    prompt heatmap. Diagonal cells are identical-prompt FTLEs; off-diagonal
    cells are paraphrase (cross-prompt) FTLEs. Matrix is symmetric — both
    halves rendered for readability.

    Saves to out_path (format inferred from extension) via altair +
    vl-convert.
    """
    import csv

    import altair as alt

    rows = []
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(
                {
                    "network": row["network"],
                    "embedding_model": row["embedding_model"],
                    "category": row["category"],
                    "prompt_or_pair": row["prompt_or_pair"],
                    "ftle": float(row["ftle"]),
                }
            )

    prompts = sorted({r["prompt_or_pair"] for r in rows if r["category"] == "identical"})
    prompt_labels = {p: f"p{i + 1}" for i, p in enumerate(prompts)}

    cells = []
    for (network, embedding_model), _ in _grouped(
        rows, lambda r: (r["network"], r["embedding_model"])
    ):
        cell_rows = [
            r
            for r in rows
            if r["network"] == network and r["embedding_model"] == embedding_model
        ]

        identical = {
            r["prompt_or_pair"]: r["ftle"] for r in cell_rows if r["category"] == "identical"
        }
        paraphrase = {}
        for r in cell_rows:
            if r["category"] != "paraphrase":
                continue
            p1, p2 = r["prompt_or_pair"].split(" || ", 1)
            paraphrase[(p1, p2)] = r["ftle"]

        for i, p_i in enumerate(prompts):
            for j, p_j in enumerate(prompts):
                if i == j:
                    ftle = identical.get(p_i)
                else:
                    ftle = paraphrase.get((p_i, p_j)) or paraphrase.get((p_j, p_i))
                if ftle is None:
                    continue
                cells.append(
                    {
                        "network": network.replace("|", " → "),
                        "embedding_model": embedding_model,
                        "row_label": prompt_labels[p_i],
                        "col_label": prompt_labels[p_j],
                        "row_prompt": p_i,
                        "col_prompt": p_j,
                        "ftle": ftle,
                    }
                )

    if not cells:
        raise ValueError(f"No cells to plot from {csv_path}")

    max_abs = max(abs(c["ftle"]) for c in cells) or 1.0
    label_order = [prompt_labels[p] for p in prompts]

    legend_text = "  |  ".join(f"{prompt_labels[p]}: {p}" for p in prompts)

    data = alt.Data(values=cells)

    heatmap = (
        alt.Chart(data)
        .mark_rect()
        .encode(
            x=alt.X(
                "col_label:N",
                sort=label_order,
                axis=alt.Axis(labelAngle=0, title=None),
            ),
            y=alt.Y("row_label:N", sort=label_order, axis=alt.Axis(title=None)),
            color=alt.Color(
                "ftle:Q",
                scale=alt.Scale(
                    scheme="redblue",
                    domain=[-max_abs, max_abs],
                    reverse=True,
                ),
                legend=alt.Legend(title="FTLE", format=".4f"),
            ),
            tooltip=[
                alt.Tooltip("network:N"),
                alt.Tooltip("row_prompt:N"),
                alt.Tooltip("col_prompt:N"),
                alt.Tooltip("ftle:Q", format=".5f"),
            ],
        )
        .properties(width=140, height=140)
        .facet(
            facet=alt.Facet("network:N", title=None, header=alt.Header(labelFontSize=9)),
            columns=3,
            title=alt.TitleParams(
                "FTLE heatmap (diagonal = identical prompt, off-diagonal = paraphrase pair)",
                subtitle=legend_text,
                anchor="middle",
                subtitleFontSize=9,
            ),
        )
        .resolve_scale(color="shared")
    )

    heatmap.save(out_path)


def plot_three_regime_overlay(
    out_path: str,
    network: str,
    embedding_model: str,
    identical_curve: list,
    close_curve: list,
    far_curve: list,
) -> None:
    """
    Overlay three divergence curves on a log-y axis for one
    (network, embedding) cell, showing how distance evolves under three
    regimes of prompt variation: identical prompt (stochastic noise only),
    close paraphrase, and distant-topic prompts.

    Each input is a list of per-timestep mean pairwise Euclidean distances
    (unlogged). The three curves may have different lengths; each is plotted
    for its own duration.
    """
    import altair as alt

    rows = []
    for label, curve in [
        ("1. identical prompt (noise)", identical_curve),
        ("2. close paraphrase", close_curve),
        ("3. distant topic", far_curve),
    ]:
        rows.extend(
            {"step": t, "distance": float(d), "regime": label}
            for t, d in enumerate(curve)
        )

    data = alt.Data(values=rows)

    title = (
        f"Three-regime divergence: {network.replace('|', ' → ')}  ·  {embedding_model}"
    )

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
            color=alt.Color(
                "regime:N",
                legend=alt.Legend(title=None, orient="bottom"),
                scale=alt.Scale(
                    domain=[
                        "1. identical prompt (noise)",
                        "2. close paraphrase",
                        "3. distant topic",
                    ],
                    range=["#4c72b0", "#dd8452", "#55a868"],
                ),
            ),
        )
        .properties(width=640, height=360, title=title)
    )

    chart.save(out_path)


def _grouped(items, key):
    """Lightweight groupby that preserves first-seen order."""
    seen = {}
    for item in items:
        k = key(item)
        seen.setdefault(k, []).append(item)
    return seen.items()


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
