#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["polars>=1.0"]
# ///
"""Load a `mix experiment.export_data` parquet dump into polars.

    ./db/load_with_polars.py 019f3645_parquet

`wide_frame` is the join you probably want: one row per embedding, carrying the
run and experiment it belongs to, the text that was embedded, and the cluster it
landed in at each clustering layer.

Two things to know about the cluster columns. Clustering is global — it pools
every embedding for a model across all experiments — so `medoid_embedding_id` is
an opaque cluster label, not a key you can join back to `embeddings.id` (the
medoid may live in an experiment that isn't in this export). A null medoid means
EVoC called that point an outlier.

Rows with `sequence_number == -1` are synthetic: they hold each run's initial
prompt embedded into the same space as the trajectory (the t_0 state), and only
appear if the dump was made with `--embed-prompts`. They are embedded at export
time and so were never part of the clustering run — their `cluster_layer_*`
values are always null. Drop them for anything that needs every state to carry a
cluster label (a symbol sequence, say), and keep them when you want to measure
drift away from the original prompt.
"""

import sys
from pathlib import Path

import polars as pl

TABLES = (
    "experiments",
    "runs",
    "invocations",
    "embeddings",
    "persistence_diagrams",
    "clustering_results",
    "embedding_clusters",
    "lyapunov_results",
)


def load(export_dir: Path) -> dict[str, pl.DataFrame]:
    """Read every parquet file in the dump, keyed by table name."""
    return {name: pl.read_parquet(export_dir / f"{name}.parquet") for name in TABLES}


def wide_frame(tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """One row per embedding, with its run, text, and per-layer cluster label."""
    base = (
        tables["embeddings"]
        .rename({"id": "embedding_id"})
        .join(
            tables["invocations"].select(
                "id", "run_id", "sequence_number", "model", "output_text"
            ),
            left_on="invocation_id",
            right_on="id",
        )
        .join(
            tables["runs"].select("id", "experiment_id", "network", "initial_prompt"),
            left_on="run_id",
            right_on="id",
        )
    )

    # one cluster-label column per layer, e.g. `cluster_layer_0`
    layers = tables["clustering_results"].select("id", "layer", "embedding_model")
    assignments = (
        tables["embedding_clusters"]
        .join(layers, left_on="clustering_result_id", right_on="id")
        .select("embedding_id", "layer", "medoid_embedding_id")
        .pivot(on="layer", index="embedding_id", values="medoid_embedding_id")
    )
    assignments.columns = [
        c if c == "embedding_id" else f"cluster_layer_{c}" for c in assignments.columns
    ]

    return base.join(assignments, on="embedding_id", how="left")


def main() -> None:
    export_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    tables = load(export_dir)

    for name, frame in tables.items():
        print(f"{name:24} {frame.height:>8,} rows  {len(frame.columns):>3} cols")

    wide = wide_frame(tables)
    print(f"\nwide frame: {wide.height:,} rows x {len(wide.columns)} cols")
    print(wide.select(pl.exclude("vector")).head())


if __name__ == "__main__":
    main()
