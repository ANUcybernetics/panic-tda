import json

import altair as alt
import polars as pl
from sqlmodel import Session

from trajectory_tracer.db import list_embeddings, list_persistence_diagrams

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
            "stop_reason": invocation.run.stop_reason,
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


def load_persistence_diagram_df(session: Session) -> pl.DataFrame:
    """
    Load all persistence diagrams from the database and flatten them into a polars DataFrame.

    Args:
        session: SQLModel database session

    Returns:
        A polars DataFrame containing all persistence diagram data
    """

    persistence_diagrams = list_persistence_diagrams(session)

    data = []
    for diagram in persistence_diagrams:
        # Count the number of generators
        num_generators = len(diagram.generators) if diagram.generators else 0

        row = {
            "id": diagram.id,
            "run_id": diagram.run_id,
            "started_at": diagram.started_at,
            "completed_at": diagram.completed_at,
            "embedding_model": diagram.embedding_model,
            "num_generators": num_generators,
            "network": diagram.run.network,
            "initial_prompt": diagram.run.initial_prompt,
            "seed": diagram.run.seed,
        }
        data.append(row)

    # Create a polars DataFrame
    return pl.DataFrame(data)


## visualisation


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
