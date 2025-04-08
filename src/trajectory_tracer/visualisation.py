import json
import logging
import os

import altair as alt
import polars as pl
from sqlmodel import Session

from trajectory_tracer.analysis import load_embeddings_df, load_runs_df

CHART_SCALE_FACTOR = 4.0

## visualisation


def save(chart: alt.Chart, filename: str) -> str:
    """
    Save an Altair chart to both HTML and PNG formats with the given scale factor.

    Args:
        chart: Altair chart to save
        filename: Path to save the chart

    Returns:
        Path to the saved HTML file
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(filename)
    os.makedirs(output_dir, exist_ok=True)

    # Get file base and extension
    base, ext = os.path.splitext(filename)

    # Save as HTML
    html_filename = f"{base}.html"
    chart.save(html_filename, scale_factor=CHART_SCALE_FACTOR)

    # Save as PNG
    png_filename = f"{base}.png"
    chart.save(png_filename, scale_factor=CHART_SCALE_FACTOR)

    return html_filename


def create_persistence_diagram_chart(df: pl.DataFrame) -> alt.Chart:
    """
    Create a base persistence diagram chart for a single run.

    Args:
        df: DataFrame containing run data with persistence homology information

    Returns:
        An Altair chart object for the persistence diagram
    """
    # Extract initial prompt, or indicate if there are multiple prompts
    unique_prompts = df["initial_prompt"].unique()
    if len(unique_prompts) > 1:
        _initial_prompt = "multiple prompts"
    else:
        _initial_prompt = unique_prompts[0]

    # Create a scatterplot for the persistence diagram
    points_chart = (
        alt.Chart(df)
        .mark_point(filled=True, opacity=0.1)
        .encode(
            x=alt.X("birth:Q").title("Feature Appearance").scale(domainMin=-0.1),
            y=alt.Y("persistence:Q").title("Feature Persistence").scale(domainMin=-0.1),
            color=alt.Color("homology_dimension:N").title("Dimension"),
            tooltip=["homology_dimension:N", "birth:Q", "persistence:Q"],
        )
    )

    # Get entropy values per dimension if they exist
    # Get unique homology dimensions and their entropy values
    dim_entropy_pairs = df.select(["homology_dimension", "entropy"]).unique(
        subset=["homology_dimension", "entropy"]
    )

    # Sort by homology dimension and format entropy values
    entropy_values = []
    for row in dim_entropy_pairs.sort("homology_dimension").iter_rows(named=True):
        entropy_values.append(f"{row['homology_dimension']}: {row['entropy']:.3f}")

    # Join entropy values into subtitle
    _subtitle = "Entropy " + ", ".join(entropy_values)

    # Set title/subtitle
    combined_chart = points_chart.properties(
        width=300,
        height=300,
        # title={"text": f"Prompt: {initial_prompt}", "subtitle": subtitle}
    ).interactive()

    return combined_chart


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

    # Save chart with high resolution
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
    # Create the base chart then facet by run_id
    chart = create_persistence_diagram_chart(df).encode(
        alt.Row("text_model:N").title("text model").header(labelAngle=0),
        alt.Column("image_model:N").title("image model"),
    )

    # Save chart with high resolution
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
    # Create the base chart then facet by run_id
    chart = create_persistence_diagram_chart(df).encode(
        alt.Facet("run_id:N", columns=cols, title="Run ID")
    )

    # Save chart with high resolution
    saved_file = save(chart, output_file)
    logging.info(f"Saved persistence diagrams to {saved_file}")


def create_persistence_entropy_chart(df: pl.DataFrame) -> alt.Chart:
    """
    Create a base boxplot showing the distribution of entropy values
    with homology_dimension on the y-axis.

    Args:
        df: DataFrame containing runs data with homology_dimension and entropy

    Returns:
        An Altair chart object for the entropy distribution
    """
    # Create a boxplot chart
    base_chart = (
        alt.Chart(df)
        .mark_boxplot(opacity=0.7)
        .encode(
            x=alt.X("entropy:Q").title("Persistence Entropy").scale(zero=False),
            y=alt.Y("homology_dimension:N")
            .title("Homology dimension")
            .axis(
                labelExpr="'h' + (datum.label == '0' ? '₀' : (datum.label == '1' ? '₁' : '₂'))"
            ),
            color=alt.Color("embedding_model:N").title("Embedding model"),
            yOffset=alt.YOffset(
                "embedding_model:N"
            ),  # Offset boxplots by embedding model
            tooltip=[
                "homology_dimension:N",
                "entropy:Q",
                "initial_prompt:N",
                "run_id:N",
            ],
        )
        .properties(width=300, height=120)
    )

    return base_chart


def plot_persistence_entropy(
    df: pl.DataFrame, output_file: str = "output/vis/persistence_entropy.html"
) -> None:
    """
    Create and save a visualization of entropy distribution across different homology dimensions.

    Args:
        df: DataFrame containing runs data with homology_dimension and entropy
        output_file: Path to save the visualization
    """
    # Check if we have the required columns
    required_columns = {"homology_dimension", "entropy"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        logging.info(
            f"Required columns not found in DataFrame: {', '.join(missing_columns)}"
        )
        return

    # If the DataFrame is empty, return early
    if df.is_empty():
        logging.info("ERROR: DataFrame is empty - no entropy data to plot")
        return

    # Create the chart
    chart = create_persistence_entropy_chart(df)

    # Save chart with high resolution
    saved_file = save(chart, output_file)
    logging.info(f"Saved persistence entropy plot to {saved_file}")


def plot_persistence_entropy_faceted(
    df: pl.DataFrame, output_file: str = "output/vis/persistence_entropy_faceted.html"
) -> None:
    """
    Create and save a visualization of entropy distributions with:
    - entropy on x axis
    - homology_dimension on y axis
    - embedding_model as color
    - faceted by text_model (rows) and image_model (columns)

    Args:
        df: DataFrame containing runs data with homology_dimension and entropy
        output_file: Path to save the visualization
    """
    # Check if we have the required columns
    required_columns = {
        "homology_dimension",
        "entropy",
        "embedding_model",
        "text_model",
        "image_model",
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        logging.info(
            f"Required columns not found in DataFrame: {', '.join(missing_columns)}"
        )
        return

    # If the DataFrame is empty, return early
    if df.is_empty():
        logging.info("ERROR: DataFrame is empty - no entropy data to plot")
        return

    # Create the base chart
    base_chart = create_persistence_entropy_chart(df)

    # Add faceting for text_model (rows) and image_model (columns)
    faceted_chart = base_chart.encode(
        row=alt.Row("text_model:N").title("Text Model"),
        column=alt.Column("image_model:N").title("Image Model"),
    )

    # Save the chart
    saved_file = save(faceted_chart, output_file)
    logging.info(f"Saved faceted persistence entropy plots to {saved_file}")


def plot_loop_length_by_prompt(df: pl.DataFrame, output_file: str) -> None:
    """
    Create a faceted histogram of loop length by initial prompt.

    Args:
        df: a Polars DataFrame
        output_file: Path to save the visualization
    """
    # Filter to only include rows with loop_length
    df_filtered = df.filter(pl.col("loop_length").is_not_null())

    # Create Altair faceted histogram chart
    chart = (
        alt.Chart(df_filtered)
        .mark_bar()
        .encode(
            x=alt.X("loop_length:Q", title="Loop Length", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("count()", title=None),
            row=alt.Row("initial_prompt:N", title=None),
        )
        .properties(
            width=500,
            height=300,  # 3x as wide as tall
        )
    )

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

    # Create a single chart with faceting
    chart = (
        alt.Chart(df_filtered)
        .mark_line(opacity=0.9)
        .encode(
            x=alt.X("sequence_number:Q", title="Sequence Number"),
            y=alt.Y("semantic_drift:Q").title("Semantic Drift"),
            color=alt.Color("run_id:N").title("Run ID"),
            tooltip=["sequence_number", "semantic_drift", "embedding_model", "initial_prompt"],
            row=alt.Row("initial_prompt:N").title("Prompt"),
            column=alt.Column("embedding_model:N").title("Model"),
        )
        .properties(
            width=300,
            height=200,
        )
    )

    # Save the chart
    saved_file = save(chart, output_file)
    logging.info(f"Saved semantic drift plot to {saved_file}")


def persistance_diagram_benchmark_vis(benchmark_file: str) -> None:
    """
    Visualize PD (Giotto PH) benchmark data from a JSON file using Altair.

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

    # Create Altair chart
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("n_points:O", title="Number of Points"),
            y=alt.Y("mean:Q", title="Time (seconds)"),
            tooltip=["n_points", "mean", "min", "max", "stddev"],
        )
        .properties(title="Giotto PH wall-clock time", width=600, height=400)
    )

    # Add error bars
    error_bars = (
        alt.Chart(df)
        .mark_errorbar()
        .encode(
            x=alt.X("n_points:O"),
            y=alt.Y("min:Q", title="Time (seconds)"),
            y2=alt.Y2("max:Q"),
        )
    )

    # Combine the bar chart and error bars
    combined_chart = (chart + error_bars).configure_axis(
        labelFontSize=12, titleFontSize=14
    )

    # Save the chart to a file
    saved_file = save(combined_chart, "output/vis/giotto_benchmark.html")
    logging.info(f"Saved Giotto benchmark plot to {saved_file}")


def paper_charts(session: Session) -> None:
    """
    Generate charts for paper publications.
    """
    embeddings_df = load_embeddings_df(session, use_cache=True)
    # Filter to only rows in specified experiments
    embeddings_df = embeddings_df.filter(
        (pl.col("experiment_id") == "067ed16c-e9a4-7bec-9378-9325a6fb10f7")
        | (pl.col("experiment_id") == "067ee281-70f5-774a-b09f-e199840304d0")
    )
    plot_semantic_drift(embeddings_df, "output/vis/semantic_drift.html")

    # runs_df = load_runs_df(session, use_cache=True)
    # plot_persistence_diagram_faceted(df, "output/vis/persistence_diagram_faceted.html")
    # plot_persistence_diagram_by_run(
    #     df, 16, "output/vis/persistence_diagram_by_run.html"
    # )
    # plot_persistence_entropy_faceted(df, "output/vis/persistence_entropy_faceted.html")
