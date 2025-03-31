import json
import os

import altair as alt
import polars as pl

## visualisation

def plot_persistence_diagram(df: pl.DataFrame, cols: int = 3, output_file: str = "output/vis/persistence_diagram.html") -> None:
    """
    Create and save a visualization of persistence diagrams for runs in the DataFrame.

    Args:
        df: DataFrame containing run data with persistence homology information
        cols: Number of columns for faceting the charts (default: 3)
        output_file: Path to save the visualization
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    # Check if we have persistence diagram data
    required_columns = {"homology_dimension", "birth", "death", "run_id", "initial_prompt"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        print(f"Required columns not found in DataFrame: {', '.join(missing_columns)}")
        return

    # Handle infinity values in death column
    # For visualization, replace inf with a large finite value
    # Create a copy of the dataframe to avoid modifying the original
    vis_df = df.clone()

    # Get the maximum finite value in both birth and death columns
    max_birth = vis_df["birth"].max()

    # For death column, get max of finite values only
    max_finite_death = vis_df.filter(pl.col("death").is_finite())["death"].max()

    # Use the larger of max_birth and max_finite_death as the max value
    max_value = max(max_birth, max_finite_death) * 1.2  # Add 20% margin

    # Replace infinity with a value slightly above max_value for visualization
    vis_df = vis_df.with_columns(
        pl.when(pl.col("death").is_infinite())
        .then(max_value * 0.95)  # Place just below max_value
        .otherwise(pl.col("death"))
        .alias("death_vis")
    )

    # Get unique run IDs for faceting
    run_ids = vis_df["run_id"].unique().to_list()

    # Create a separate chart for each run ID
    charts = []

    for run_id in run_ids:
        # Filter data for this run
        run_df = vis_df.filter(pl.col("run_id") == run_id)

        # Get the initial prompt for this run (should be the same for all rows with the same run_id)
        initial_prompt = run_df["initial_prompt"].unique()[0]

        # Create the main scatter plot for this run
        scatter = alt.Chart(run_df).mark_point(filled=True, opacity=0.7).encode(
            x=alt.X("birth:Q", title="Birth", scale=alt.Scale(domain=[0, max_value])),
            y=alt.Y("death_vis:Q", title="Death", scale=alt.Scale(domain=[0, max_value])),
            color=alt.Color("homology_dimension:N", title="Dimension",
                          scale=alt.Scale(scheme="category10")),
            tooltip=["homology_dimension:N", "birth:Q", "death:Q", "persistence:Q"]
        )

        # Create diagonal reference line data
        diagonal_data = {
            "x": [0.0, float(max_value)],
            "y": [0.0, float(max_value)]
        }
        diagonal_df = pl.DataFrame(diagonal_data, schema={"x": pl.Float64, "y": pl.Float64})

        # Create diagonal line
        diagonal = alt.Chart(diagonal_df).mark_line(
            color="gray", strokeDash=[5, 5]
        ).encode(
            x="x:Q",
            y="y:Q"
        )

        # Layer the charts
        combined = alt.layer(scatter, diagonal).properties(
            width=400,
            height=400,
            title=f"Prompt: {initial_prompt}"
        )

        charts.append(combined)

    # Concatenate the charts with the specified number of columns
    if len(charts) > 1:
        # Split the charts into rows with the specified number of columns
        chart_rows = [charts[i:i+cols] for i in range(0, len(charts), cols)]

        # Create a row of charts for each group
        rows = []
        for row_charts in chart_rows:
            row = alt.hconcat(*row_charts)
            rows.append(row)

        # Combine all rows vertically
        final_chart = alt.vconcat(*rows).properties(
            title="Persistence Diagrams"
        ).resolve_scale(
            x='shared',
            y='shared'
        )
    else:
        final_chart = charts[0].properties(
            title="Persistence Diagram"
        )

    # Save chart
    final_chart.save(output_file)
    print(f"Saved persistence diagram to {output_file}")


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
    Create a line plot showing semantic drift (both euclidean and cosine) over sequence number.

    Args:
        df: DataFrame containing embedding data with drift_euclidean, drift_cosine and sequence_number
        output_file: Path to save the visualization
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    # Check if we have the required columns
    required_columns = {"drift_euclidean", "drift_cosine", "sequence_number", "run_id"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        print(f"Required columns not found in DataFrame: {', '.join(missing_columns)}")
        return

    # Filter to only include rows with at least one drift measure
    df_filtered = df.filter(
        pl.col("drift_euclidean").is_not_null() | pl.col("drift_cosine").is_not_null()
    )

    # Create two separate charts with different y-scales
    euclidean_chart = (
        alt.Chart(df_filtered)
        .mark_line(point=True)
        .encode(
            x=alt.X("sequence_number:Q", title="Sequence Number"),
            y=alt.Y("drift_euclidean:Q", title="Euclidean Distance"),
            color=alt.Color("run_id:N", title="Run ID"),
            tooltip=["run_id", "sequence_number", "drift_euclidean", "embedding_model"]
        )
        .properties(
            width=800,
            height=250,
            title="Euclidean Distance Over Sequence"
        )
    )

    cosine_chart = (
        alt.Chart(df_filtered)
        .mark_line(point=True, strokeDash=[3, 3])  # Make the cosine lines dashed
        .encode(
            x=alt.X("sequence_number:Q", title="Sequence Number"),
            y=alt.Y("drift_cosine:Q", title="Cosine Distance", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("run_id:N", title="Run ID"),
            tooltip=["run_id", "sequence_number", "drift_cosine", "embedding_model"]
        )
        .properties(
            width=800,
            height=250,
            title="Cosine Distance Over Sequence"
        )
    )

    # Combine the charts vertically
    combined_chart = alt.vconcat(
        euclidean_chart,
        cosine_chart
    ).resolve_scale(
        color='shared'  # Use the same color scheme for runs across both charts
    ).properties(
        title="Semantic Dispersion Measures"
    )

    # Save the chart
    combined_chart.save(output_file)
    print(f"Saved semantic drift plot to {output_file}")


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
