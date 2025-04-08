import json
import logging

import holoviews as hv
import hvplot.polars  # noqa: F401  # Register hvplot with polars DataFrames
import polars as pl
from holoviews import opts
from sqlmodel import Session

from trajectory_tracer.analysis import load_runs_df

# Set up HoloViews rendering options for high-quality output
hv.extension("plotly")
hv.output(size=200)

# Default plot options
PLOT_WIDTH = 600
PLOT_HEIGHT = 400
POINT_SIZE = 5

## visualisation


def save(chart, file):
    raise


def create_persistence_diagram_chart(df: pl.DataFrame) -> hv.Overlay:
    """
    Create a base persistence diagram chart for a single run.

    Args:
        df: DataFrame containing run data with persistence homology information

    Returns:
        A HoloViews plot object for the persistence diagram
    """
    # Extract initial prompt, or indicate if there are multiple prompts
    unique_prompts = df["initial_prompt"].unique()
    if len(unique_prompts) > 1:
        _initial_prompt = "multiple prompts"
    else:
        _initial_prompt = unique_prompts[0]

    # Get entropy values per dimension if they exist
    dim_entropy_pairs = df.select(["homology_dimension", "entropy"]).unique(
        subset=["homology_dimension", "entropy"]
    )

    # Sort by homology dimension and format entropy values
    entropy_values = []
    for row in dim_entropy_pairs.sort("homology_dimension").iter_rows(named=True):
        entropy_values.append(f"{row['homology_dimension']}: {row['entropy']:.3f}")

    # Join entropy values into subtitle
    _subtitle = "Entropy " + ", ".join(entropy_values)

    # Create scatter plot
    scatter_plot = df.hvplot.scatter(
        x="birth",
        y="persistence",
        by="homology_dimension",
        color="homology_dimension",
        alpha=0.1,
        legend="top",
        width=300,
        height=300,
        xlim=(-0.1, None),
        ylim=(-0.1, None),
        xlabel="Feature Appearance",
        ylabel="Feature Persistence",
        hover_cols=["homology_dimension", "birth", "persistence"],
    )

    return scatter_plot


def plot_persistence_diagram(
    df: pl.DataFrame, output_file: str = "output/vis/persistence_diagram.html"
) -> None:
    """
    Create and save a visualization of a single persistence diagram for the given DataFrame.

    Args:
        df: DataFrame containing run data with persistence homology information
        output_file: Path to save the visualization
    """
    # Create the chart
    chart = create_persistence_diagram_chart(df)

    # Save chart
    saved_file = save(chart, output_file)

    logging.info(f"Saved single persistence diagram to {saved_file}")


def plot_persistence_diagram_faceted(
    df: pl.DataFrame,
    output_file: str = "output/vis/persistence_diagram.html",
    num_cols: int = 2,
) -> None:
    """
    Create and save a visualization of persistence diagrams for runs in the DataFrame,
    creating a grid of charts (one per run).

    Args:
        df: DataFrame containing run data with persistence homology information
        output_file: Path to save the visualization
        num_cols: Number of columns in the grid layout
    """
    # Create faceted plot by text_model and image_model
    chart = df.hvplot.scatter(
        x="birth",
        y="persistence",
        by="homology_dimension",
        color="homology_dimension",
        alpha=0.1,
        width=300,
        height=300,
        xlim=(-0.1, None),
        ylim=(-0.1, None),
        xlabel="Feature Appearance",
        ylabel="Feature Persistence",
        hover_cols=["homology_dimension", "birth", "persistence"],
        groupby=["text_model", "image_model"],
        row="text_model",
        col="image_model",
        subplots=True,
    ).opts(opts.Scatter(width=300, height=300, show_grid=True))

    # Save chart
    saved_file = save(chart, output_file)
    logging.info(f"Saved persistence diagrams to {saved_file}")


def plot_persistence_diagram_by_run(
    df: pl.DataFrame,
    cols: int,
    output_file: str = "output/vis/persistence_diagram.html",
    num_cols: int = 2,
) -> None:
    """
    Create and save a visualization of persistence diagrams for runs in the DataFrame,
    creating a grid of charts (one per run).

    Args:
        df: DataFrame containing run data with persistence homology information
        output_file: Path to save the visualization
        num_cols: Number of columns in the grid layout
    """
    # Create faceted plot by run_id
    chart = df.hvplot.scatter(
        x="birth",
        y="persistence",
        by="homology_dimension",
        color="homology_dimension",
        alpha=0.1,
        width=300,
        height=300,
        xlim=(-0.1, None),
        ylim=(-0.1, None),
        xlabel="Feature Appearance",
        ylabel="Feature Persistence",
        hover_cols=["homology_dimension", "birth", "persistence"],
        groupby=["run_id"],
        cols=cols,
    ).opts(opts.Scatter(width=300, height=300, show_grid=True))

    # Save chart
    saved_file = save(chart, output_file)
    logging.info(f"Saved persistence diagrams to {saved_file}")


def plot_persistence_entropy(
    df: pl.DataFrame,
    output_file: str = "output/vis/persistence_entropy.html",
) -> None:
    """
    Create and save a visualization of entropy distributions with:
    - entropy on x axis
    - homology_dimension on y axis
    - embedding_model as color

    Args:
        df: DataFrame containing runs data with homology_dimension and entropy
        output_file: Path to save the visualization
    """
    # Create boxplot chart with embedding_model as color
    chart = df.hvplot.box(
        y="entropy",  # The values to display in the boxes
        by=["homology_dimension", "embedding_model"],  # Group by both dimensions
        cmap="Category10",  # Use a categorical color palette instead of direct color mapping
        width=600,
        height=300,
        legend="top",
        xlabel="Homology Dimension / Model",
        ylabel="Persistence Entropy",
        alpha=0.7,
        hover_cols=["homology_dimension", "entropy", "initial_prompt", "run_id"],
    )

    # Save the chart using the save function
    hvplot.save(chart, output_file)
    logging.info(f"Saved persistence entropy plot to {output_file}")


def plot_loop_length_by_prompt(df: pl.DataFrame, output_file: str) -> None:
    """
    Create a faceted histogram of loop length by initial prompt.

    Args:
        df: a Polars DataFrame
        output_file: Path to save the visualization
    """
    # Filter to only include rows with loop_length
    df_filtered = df.filter(pl.col("loop_length").is_not_null())

    # Create faceted histogram
    chart = df_filtered.hvplot.hist(
        "loop_length",
        by="initial_prompt",
        width=500,
        height=300,
        subplots=True,
        row="initial_prompt",
        xlabel="Loop Length",
        ylabel="Count",
    ).opts(opts.Histogram(width=500, height=300))

    # Save the chart
    saved_file = save(chart, output_file)
    logging.info(f"Saved loop length plot to {saved_file}")


def plot_semantic_drift(
    df: pl.DataFrame, output_file: str = "output/vis/semantic_drift.html"
) -> None:
    """
    Create a line plot showing semantic drift over sequence number,
    faceted by initial prompt and model.

    Args:
        df: DataFrame containing embedding data with semantic_drift and sequence_number
        output_file: Path to save the visualization
    """
    # Check if we have the required columns
    required_columns = {
        "semantic_drift",
        "sequence_number",
        "run_id",
        "initial_prompt",
        "embedding_model",  # Added for faceting
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        logging.info(
            f"Required columns not found in DataFrame: {', '.join(missing_columns)}"
        )
        return

    # Filter to only include rows with drift measure
    df_filtered = df.filter(pl.col("semantic_drift").is_not_null())

    # Create a faceted line plot
    chart = df_filtered.hvplot.line(
        x="sequence_number",
        y="semantic_drift",
        by="run_id",
        groupby=["initial_prompt", "embedding_model"],
        row="initial_prompt",
        col="embedding_model",
        width=300,
        height=200,
        xlabel="Sequence Number",
        ylabel="Semantic Drift",
        line_alpha=0.9,
        hover_cols=[
            "sequence_number",
            "semantic_drift",
            "embedding_model",
            "initial_prompt",
        ],
    ).opts(opts.Curve(width=300, height=200, show_grid=True))

    # Save the chart
    saved_file = save(chart, output_file)
    logging.info(f"Saved semantic drift plot to {saved_file}")


def persistance_diagram_benchmark_vis(benchmark_file: str) -> None:
    """
    Visualize PD (Giotto PH) benchmark data from a JSON file using hvPlot.

    Args:
        benchmark_file: Path to the JSON benchmark file
    """
    # Load the benchmark data
    with open(benchmark_file, "r") as f:
        data = json.load(f)

    # Extract benchmark results
    benchmark_data = []
    for benchmark in data["benchmarks"]:
        benchmark_data.append({
            "n_points": benchmark["params"]["n_points"],
            "mean": benchmark["stats"]["mean"],
            "min": benchmark["stats"]["min"],
            "max": benchmark["stats"]["max"],
            "stddev": benchmark["stats"]["stddev"],
        })

    # Convert to DataFrame
    df = pl.DataFrame(benchmark_data)

    # Create bar chart with error bars
    base_chart = df.hvplot.bar(
        x="n_points",
        y="mean",
        width=600,
        height=400,
        title="Giotto PH wall-clock time",
        xlabel="Number of Points",
        ylabel="Time (seconds)",
        hover_cols=["n_points", "mean", "min", "max", "stddev"],
    )

    # Add error bars using HoloViews errorbars
    error_bars = hv.ErrorBars(
        [
            (row["n_points"], row["mean"], row["min"], row["max"])
            for row in df.iter_rows(named=True)
        ],
        kdims=["n_points"],
        vdims=["mean", "min", "max"],
    )

    # Combine charts
    combined_chart = base_chart * error_bars

    # Save the chart to a file
    saved_file = save(combined_chart, "output/vis/giotto_benchmark.html")
    logging.info(f"Saved Giotto benchmark plot to {saved_file}")


def paper_charts(session: Session) -> None:
    """
    Generate charts for paper publications.
    """
    # embeddings_df = load_embeddings_df(session, use_cache=True)
    # embeddings_df = embeddings_df.filter(
    #     (pl.col("experiment_id") == "067ed16c-e9a4-7bec-9378-9325a6fb10f7")
    #     | (pl.col("experiment_id") == "067ee281-70f5-774a-b09f-e199840304d0")
    # )
    # plot_semantic_drift(embeddings_df, "output/vis/semantic_drift.html")

    runs_df = load_runs_df(session, use_cache=True)
    runs_df = runs_df.filter(
        (pl.col("experiment_id") == "067ed16c-e9a4-7bec-9378-9325a6fb10f7")
        | (pl.col("experiment_id") == "067ee281-70f5-774a-b09f-e199840304d0")
    )
    # plot_persistence_diagram_faceted(runs_df, "output/vis/persistence_diagram_faceted.html")
    # plot_persistence_diagram_by_run(
    #     runs_df, 16, "output/vis/persistence_diagram_by_run.html"
    # )
    plot_persistence_entropy(
        runs_df, "output/vis/persistence_entropy_faceted.json"
    )
