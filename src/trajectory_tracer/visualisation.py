import json
import logging
import os

import altair as alt
import polars as pl

## visualisation

def create_persistence_diagram_chart(df: pl.DataFrame, include_entropy: bool = True) -> alt.Chart:
    """
    Create a single persistence diagram chart (unfaceted) from a DataFrame.

    Args:
        df: DataFrame containing run data with persistence homology information
        include_entropy: Whether to include entropy text on the chart

    Returns:
        An Altair chart object for the persistence diagram
    """
    # Create a scatterplot for the persistence diagram
    points_chart = alt.Chart(df).mark_point(
        filled=True, opacity=0.4
    ).encode(
        x=alt.X("birth:Q", title="Birth", scale=alt.Scale(domainMin=-0.1)),
        y=alt.Y("death:Q", title="Death", scale=alt.Scale(domainMin=-0.1)),
        color=alt.Color("homology_dimension:N", title="Dimension"),
        tooltip=["homology_dimension:N", "birth:Q", "death:Q", "persistence:Q"]
    )

    # Add text labels for initial prompts in fixed position
    prompt_text = alt.Chart(df).mark_text(
        align='left',
        baseline='top',
        fontSize=16,
        dx=5,
        dy=5
    ).encode(
        text=alt.Text('initial_prompt:N'),
        x=alt.value(10),  # Fixed position from left
        y=alt.value(50)   # Fixed position from top
    )

    # Create the chart with prompt text
    combined_chart = (points_chart + prompt_text)

    # Add entropy text only if requested
    if include_entropy:
        # Add text labels for entropy in fixed position
        entropy_text = alt.Chart(df).mark_text(
            align='left',
            baseline='middle',
            fontSize=16,
            dx=5,
            dy=5
        ).encode(
            text=alt.Text('entropy:Q', format='.3f'),  # Round to 3 decimal places
            x=alt.value(10),  # Fixed position from left
            y=alt.value(80)   # Fixed position from top
        )
        combined_chart = combined_chart + entropy_text

    # Set properties and make interactive
    combined_chart = combined_chart.properties(
        width=400,
        height=400
    ).interactive()

    return combined_chart


def plot_persistence_diagram(df: pl.DataFrame, output_file: str = "output/vis/persistence_diagram_single.html") -> None:
    """
    Create and save a visualization of a single persistence diagram for the given DataFrame.

    Args:
        df: DataFrame containing run data with persistence homology information
        output_file: Path to save the visualization
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    # Check if we have persistence diagram data
    required_columns = {"homology_dimension", "birth", "death", "initial_prompt"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        logging.info(f"Required columns not found in DataFrame: {', '.join(missing_columns)}")
        return

    # If the DataFrame is empty, return early
    if df.is_empty():
        logging.info("ERROR: DataFrame is empty - no persistence diagram data to plot")
        return

    # Create the unfaceted chart without entropy
    chart = create_persistence_diagram_chart(df, include_entropy=False).properties(
        title="Persistence Diagram",
        width=600,
        height=600
    )

    # Save chart with high resolution
    chart.save(output_file, scale_factor=2.0)

    logging.info(f"Saved single persistence diagram to {output_file}")


def plot_persistence_diagram_faceted(df: pl.DataFrame, output_file: str = "output/vis/persistence_diagram.html") -> None:
    """
    Create and save a visualization of persistence diagrams for runs in the DataFrame,
    faceted by homology dimension (columns) and run_id (rows).

    Args:
        df: DataFrame containing run data with persistence homology information
        output_file: Path to save the visualization
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    # Check if we have persistence diagram data
    required_columns = {"homology_dimension", "birth", "death", "run_id", "initial_prompt"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        logging.info(f"Required columns not found in DataFrame: {', '.join(missing_columns)}")
        return

    # If the DataFrame is empty, return early
    if df.is_empty():
        logging.info("ERROR: DataFrame is empty - no persistence diagram data to plot")
        return

    # Create the unfaceted chart with entropy
    unfaceted_chart = create_persistence_diagram_chart(df, include_entropy=True)

    # Apply faceting
    faceted_chart = unfaceted_chart.facet(
        column=alt.Column("homology_dimension:N", title="Homology Dimension"),
        row=alt.Row("run_id:N", title="Run ID")
    )

    # Save chart with high resolution
    faceted_chart.save(output_file, scale_factor=4.0)

    logging.info(f"Saved persistence diagram to {output_file}")


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
            row=alt.Row("initial_prompt:N", title=None)
        )
        .properties(
            width=500,
            height=300  # 3x as wide as tall
        )
    )

    # Save the chart
    chart.save(output_file)


def plot_semantic_drift(df: pl.DataFrame, output_file: str = "output/vis/semantic_drift.html") -> None:
    """
    Create a line plot showing semantic drift (both euclidean and cosine) over sequence number,
    faceted by run_id with dual axes for different drift metrics.

    Args:
        df: DataFrame containing embedding data with drift_euclidean, drift_cosine and sequence_number
        output_file: Path to save the visualization
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    # Check if we have the required columns
    required_columns = {"drift_euclidean", "drift_cosine", "sequence_number", "run_id", "initial_prompt"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        logging.info(f"Required columns not found in DataFrame: {', '.join(missing_columns)}")
        return

    # Filter to only include rows with at least one drift measure
    df_filtered = df.filter(
        pl.col("drift_euclidean").is_not_null() | pl.col("drift_cosine").is_not_null()
    )

    # Get unique run IDs for faceting
    run_ids = df_filtered["run_id"].unique().to_list()

    # Create a separate chart for each run ID
    charts = []

    for run_id in run_ids:
        # Filter data for this run
        run_df = df_filtered.filter(pl.col("run_id") == run_id)

        # Get the initial prompt for this run (should be the same for all rows with the same run_id)
        initial_prompt = run_df["initial_prompt"].unique()[0]

        # Create base chart
        base = alt.Chart(run_df).encode(
            x=alt.X("sequence_number:Q", title="Sequence Number")
        )

        # Create euclidean distance line
        euclidean_line = base.mark_line(color="#57A44C", opacity=0.7).encode(
            alt.Y("drift_euclidean:Q").axis(
                title="Euclidean Distance",
                titleColor="#57A44C"
            ),
            tooltip=["sequence_number", "drift_euclidean", "embedding_model"]
        )

        # Add points to the euclidean line
        euclidean_points = base.mark_point(color="#57A44C").encode(
            alt.Y("drift_euclidean:Q"),
            tooltip=["sequence_number", "drift_euclidean", "embedding_model"]
        )

        # Create cosine distance line
        cosine_line = base.mark_line(color="#5276A7", strokeDash=[3, 3], opacity=0.7).encode(
            alt.Y("drift_cosine:Q").axis(
                title="Cosine Distance",
                titleColor="#5276A7"
            ),
            tooltip=["sequence_number", "drift_cosine", "embedding_model"]
        )

        # Add points to the cosine line
        cosine_points = base.mark_point(color="#5276A7").encode(
            alt.Y("drift_cosine:Q"),
            tooltip=["sequence_number", "drift_cosine", "embedding_model"]
        )

        # Combine the charts with dual axis
        combined = alt.layer(
            euclidean_line + euclidean_points,
            cosine_line + cosine_points
        ).resolve_scale(
            y="independent"
        ).properties(
            width=400,
            height=300,
            title=f"Prompt: {initial_prompt}"
        )

        charts.append(combined)

    # Arrange charts in a grid
    if len(charts) > 1:
        # Determine the number of columns (default to 2)
        cols = 2
        # Split the charts into rows with the specified number of columns
        chart_rows = [charts[i:i+cols] for i in range(0, len(charts), cols)]

        # Create a row of charts for each group
        rows = []
        for row_charts in chart_rows:
            row = alt.hconcat(*row_charts)
            rows.append(row)

        # Combine all rows vertically
        final_chart = alt.vconcat(*rows).properties(
            title="Semantic Dispersion Measures by Run"
        )
    else:
        final_chart = charts[0].properties(
            title="Semantic Dispersion Measures"
        )

    # Save the chart
    final_chart.save(output_file)
    logging.info(f"Saved semantic drift plot to {output_file}")


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
    combined_chart.save("output/vis/giotto_benchmark.html")
