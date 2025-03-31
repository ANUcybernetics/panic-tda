import json
import os

import altair as alt
import polars as pl

## visualisation

def plot_persistence_diagram(df: pl.DataFrame, output_file: str = "output/vis/persistence_diagram.html") -> None:
    """
    Create and save a visualization of persistence diagrams for runs in the DataFrame.

    Args:
        df: DataFrame containing run data with persistence homology information
        output_file: Path to save the visualization
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    # Check if we have persistence diagram data
    if "homology_dimension" not in df.columns or "birth" not in df.columns or "death" not in df.columns or "run_id" not in df.columns:
        print("Required columns not found in DataFrame")
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
            title=f"Run ID: {run_id}"
        )

        charts.append(combined)

    # Concatenate the charts horizontally
    if len(charts) > 1:
        final_chart = alt.hconcat(*charts).properties(
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
