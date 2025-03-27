import json

import altair as alt
import polars as pl
from sqlmodel import Session

from trajectory_tracer.db import list_embeddings, list_runs

## load the DB objects into dataframes


def load_embeddings_df(session: Session) -> pl.DataFrame:
    """
    Load all embeddings from the database and flatten them into a polars DataFrame.

    Args:
        session: SQLModel database session

    Returns:
        A polars DataFrame containing all embedding data
    """
    embeddings = list_embeddings(session)

    # Convert the embeddings to a format suitable for a DataFrame
    data = []
    for embedding in embeddings:
        invocation = embedding.invocation

        row = {
            "id": embedding.id,
            "invocation_id": invocation.id,
            "embedding_started_at": embedding.started_at,
            "embedding_completed_at": embedding.completed_at,
            "invocation_started_at": invocation.started_at,
            "invocation_completed_at": invocation.completed_at,
            "duration": embedding.duration,
            "run_id": invocation.run_id,
            "type": invocation.type,
            "initial_prompt": invocation.run.initial_prompt,
            "seed": invocation.run.seed,
            "model": invocation.model,
            "sequence_number": invocation.sequence_number,
            "embedding_model": embedding.embedding_model,
        }
        data.append(row)

    # Create a polars DataFrame
    return pl.DataFrame(data)


def load_runs_df(session: Session) -> pl.DataFrame:
    """
    Load all runs from the database and flatten them into a polars DataFrame.

    Args:
        session: SQLModel database session

    Returns:
        A polars DataFrame containing all run data
    """

    runs = list_runs(session)
    # Print the number of runs
    print(f"Number of runs: {len(runs)}")

    data = []
    for run in runs:
        # Skip runs with no invocations
        if not run.invocations:
            continue
        row = {
            "run_id": run.id,
            "network": run.network,
            "initial_prompt": run.initial_prompt,
            "seed": run.seed,
            "max_length": run.max_length,
            "num_invocations": len(run.invocations)
        }

        # Process stop_reason to separate into reason and loop_length
        stop_reason_value = run.stop_reason
        loop_length = None

        if isinstance(stop_reason_value, tuple) and stop_reason_value[0] == "duplicate":
            stop_reason = "duplicate"
            loop_length = stop_reason_value[1]
        else:
            stop_reason = stop_reason_value

        row["stop_reason"] = stop_reason
        row["loop_length"] = loop_length

        data.append(row)

    # Create a polars DataFrame with explicit schema for loop_length
    return pl.DataFrame(data, schema_overrides={"loop_length": pl.Int64})


## visualisation

def plot_loop_length_by_prompt(session: Session, output_file: str) -> None:
    """
    Create a bar plot of loop length by initial prompt with jittered points.

    Args:
        session: SQLModel database session
        output_file: Path to save the visualization
    """
    # Load embeddings data
    df = load_embeddings_df(session)

    # Filter to only include rows with loop_length
    df_filtered = df.filter(pl.col("loop_length").is_not_null())

    # Create Altair chart
    # Bar chart for averages
    bar_chart = (
        alt.Chart(df_filtered)
        .mark_bar(opacity=0.5, color="steelblue")
        .encode(
            x=alt.X("initial_prompt:N", title="Initial Prompt"),
            y=alt.Y("mean(loop_length):Q", title="Loop Length")
        )
    )

    # Point chart with jitter for individual data points
    point_chart = (
        alt.Chart(df_filtered)
        .mark_circle(size=60)
        .encode(
            x=alt.X(
                "initial_prompt:N",
                title="Initial Prompt"
            ),
            y=alt.Y("loop_length:Q", title="Loop Length"),
            color=alt.Color("run_id:N", legend=None),
            tooltip=["run_id", "loop_length", "model", "embedding_model"]
        )
    )

    # Add jitter to points
    jittered_points = point_chart.transform_calculate(
        jitter="random() * 0.5"
    ).encode(
        x=alt.X(
            "initial_prompt:N",
            title="Initial Prompt",
            band=0.5,
            axis=alt.Axis(labelAngle=-45)
        )
    )

    # Combine the charts
    combined_chart = (
        (bar_chart + jittered_points)
        .properties(
            title="Loop Length by Initial Prompt",
            width=800,
            height=500
        )
        .configure_title(fontSize=16)
        .configure_axis(labelFontSize=12, titleFontSize=14)
    )

    # Save the chart
    combined_chart.save(output_file)


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
