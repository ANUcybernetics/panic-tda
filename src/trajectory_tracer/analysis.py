import json
import os

import altair as alt
import numpy as np
import polars as pl
from sqlmodel import Session

from trajectory_tracer.db import list_embeddings, list_runs
from trajectory_tracer.schemas import Run

## load the DB objects into dataframes



def load_embeddings_df(session: Session, use_cache: bool = True) -> pl.DataFrame:
    """
    Load all embeddings from the database and flatten them into a polars DataFrame.

    Args:
        session: SQLModel database session
        use_cache: Whether to use cached dataframe if available

    Returns:
        A polars DataFrame containing all embedding data
    """
    cache_path = "output/cache/embeddings.parquet"

    # Check if cache exists and should be used
    if use_cache and os.path.exists(cache_path):
        print(f"Loading embeddings from cache: {cache_path}")
        return pl.read_parquet(cache_path)

    print("Loading embeddings from database...")
    embeddings = list_embeddings(session)

    # Convert the embeddings to a format suitable for a DataFrame
    data = []
    for embedding in embeddings:
        invocation = embedding.invocation
        run = invocation.run

        row = {
            "id": str(embedding.id),
            "invocation_id": str(invocation.id),
            "embedding_started_at": embedding.started_at,
            "embedding_completed_at": embedding.completed_at,
            "invocation_started_at": invocation.started_at,
            "invocation_completed_at": invocation.completed_at,
            "duration": embedding.duration,
            "run_id": str(invocation.run_id),
            "experiment_id": str(run.experiment_id),
            "type": invocation.type,
            "initial_prompt": invocation.run.initial_prompt,
            "seed": invocation.run.seed,
            "model": invocation.model,
            "sequence_number": invocation.sequence_number,
            "embedding_model": embedding.embedding_model,
        }
        data.append(row)

    # Create a polars DataFrame
    df = pl.DataFrame(data)

    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.write_parquet(cache_path)
    print(f"Saved embeddings to cache: {cache_path}")

    return df


def load_runs_df(session: Session, use_cache: bool = True) -> pl.DataFrame:
    """
    Load all runs from the database and flatten them into a polars DataFrame.
    Includes persistence diagrams and generators for each run.

    Args:
        session: SQLModel database session
        use_cache: Whether to use cached dataframe if available

    Returns:
        A polars DataFrame containing all run data with persistence diagrams
    """
    cache_path = "output/cache/runs.parquet"

    # Check if cache exists and should be used
    if use_cache and os.path.exists(cache_path):
        print(f"Loading runs from cache: {cache_path}")
        return pl.read_parquet(cache_path)

    print("Loading runs from database...")
    runs = list_runs(session)
    # Print the number of runs
    print(f"Number of runs: {len(runs)}")

    data = []
    for run in runs:
        # Skip runs with no invocations
        if not run.invocations:
            continue

        # Process stop_reason to separate into reason and loop_length
        stop_reason_value = run.stop_reason
        loop_length = None

        if isinstance(stop_reason_value, tuple) and stop_reason_value[0] == "duplicate":
            stop_reason = "duplicate"
            loop_length = stop_reason_value[1]
        else:
            stop_reason = stop_reason_value

        # Base run information
        base_row = {
            "run_id": str(run.id),
            "experiment_id": str(run.experiment_id),
            "network": run.network,
            "initial_prompt": run.initial_prompt,
            "seed": run.seed,
            "max_length": run.max_length,
            "num_invocations": len(run.invocations),
            "stop_reason": stop_reason,
            "loop_length": loop_length
        }

        # Only include runs with persistence diagrams
        if run.persistence_diagrams:
            for pd in run.persistence_diagrams:
                row = base_row.copy()
                row["persistence_diagram_id"] = str(pd.id)
                row["embedding_model"] = pd.embedding_model
                row["persistence_diagram_started_at"] = pd.started_at
                row["persistence_diagram_completed_at"] = pd.completed_at
                row["persistence_diagram_duration"] = pd.duration

                # Only include persistence diagrams with diagram_data
                if pd.diagram_data and "dgms" in pd.diagram_data:
                    # Always create rows for dimensions 0, 1, and 2
                    for dim in range(3):  # Dimensions 0, 1, 2
                        homology_row = row.copy()
                        homology_row["homology_dimension"] = dim

                        # Add entropy for this dimension if available
                        if "entropy" in pd.diagram_data and dim < len(pd.diagram_data["entropy"]):
                            homology_row["entropy"] = float(pd.diagram_data["entropy"][dim])
                        else:
                            homology_row["entropy"] = 0.0  # Default value

                        # Add feature count for this dimension
                        if dim < len(pd.diagram_data["dgms"]):
                            homology_row["feature_count"] = len(pd.diagram_data["dgms"][dim])
                        else:
                            homology_row["feature_count"] = 0  # Empty for this dimension

                        # Add generator count for this dimension
                        if "gens" in pd.diagram_data and dim < len(pd.diagram_data["gens"]):
                            homology_row["generator_count"] = len(pd.diagram_data["gens"][dim])
                        else:
                            homology_row["generator_count"] = 0  # Empty for this dimension

                        data.append(homology_row)

    # Create a polars DataFrame with explicit schema for loop_length
    schema_overrides = {"loop_length": pl.Int64}

    # Add schema overrides for entropy columns
    for i in range(5):  # Assuming max 5 dimensions
        schema_overrides[f"entropy_dim_{i}"] = pl.Float64
        schema_overrides[f"feature_count_dim_{i}"] = pl.Int64
        schema_overrides[f"generator_count_dim_{i}"] = pl.Int64

    # Only create DataFrame if we have data
    if data:
        df = pl.DataFrame(data, schema_overrides=schema_overrides)

        # Save to cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.write_parquet(cache_path)
        print(f"Saved runs to cache: {cache_path}")

        return df
    else:
        print("No valid persistence diagram data found")
        return pl.DataFrame()


## visualisation

def plot_persistence_diagram(df: pl.DataFrame, output_dir: str = "output/vis/persistence") -> None:
    """
    Create and save a visualization of persistence diagrams for runs in the DataFrame.

    Args:
        df: DataFrame containing run data with persistence diagram information
        output_dir: Directory to save the visualizations
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Filter to only rows with persistence diagram IDs
    if "persistence_diagram_id" not in df.columns:
        print("No persistence diagram data found in DataFrame")
        return

    # Process each run in the DataFrame
    for row in df.filter(pl.col("persistence_diagram_id").is_not_null()).iter_rows(named=True):
        run_id = row["run_id"]
        embedding_model = row["embedding_model"]
        diagram_data = row.get("diagram_data")

        if not diagram_data or "dgms" not in diagram_data:
            print(f"No valid diagram data for run {run_id}, embedding {embedding_model}")
            continue

        # Convert persistence diagram data to DataFrame
        records = []

        for dim, diagram in enumerate(diagram_data["dgms"]):
            for i, (birth, death) in enumerate(diagram):
                # Skip infinite death values for visualization
                if np.isinf(death):
                    continue

                # Add record for each feature
                records.append({
                    "dimension": dim,
                    "feature_id": i,
                    "birth": float(birth),
                    "death": float(death),
                    "persistence": float(death - birth)
                })

                # Add generator information if available
                if "gens" in diagram_data and dim < len(diagram_data["gens"]) and i < len(diagram_data["gens"][dim]):
                    generator = diagram_data["gens"][dim][i]
                    # Convert generator to a list if it's a numpy array
                    if hasattr(generator, "tolist"):
                        generator = generator.tolist()
                    records[-1]["generator"] = str(generator)
                    records[-1]["generator_size"] = len(generator) if isinstance(generator, list) else 1

        if not records:
            print(f"No finite features in persistence diagram for run {run_id}, embedding {embedding_model}")
            continue

        # Create DataFrame
        feature_df = pl.DataFrame(records)

        # Create diagonal reference line data
        max_value = max(feature_df["death"].max(), feature_df["birth"].max())
        diagonal_df = pl.DataFrame({
            "x": [0, max_value],
            "y": [0, max_value]
        })

        # Create the persistence diagram visualization
        diagram = alt.Chart(feature_df).mark_point(filled=True, size=100, opacity=0.7).encode(
            x=alt.X("birth:Q", title="Birth", scale=alt.Scale(domain=[0, max_value])),
            y=alt.Y("death:Q", title="Death", scale=alt.Scale(domain=[0, max_value])),
            color=alt.Color("dimension:N", title="Dimension",
                            scale=alt.Scale(scheme="category10")),
            size=alt.Size("persistence:Q", title="Persistence",
                        scale=alt.Scale(range=[20, 400])),
            tooltip=["dimension:N", "birth:Q", "death:Q", "persistence:Q", "generator:N"]
        ).properties(
            width=500,
            height=500,
            title=f"Persistence Diagram - {embedding_model}"
        )

        # Add diagonal line
        diagonal = alt.Chart(diagonal_df).mark_line(
            color="gray", strokeDash=[5, 5]
        ).encode(
            x="x:Q",
            y="y:Q"
        )

        # Combine chart with diagonal
        combined = diagram + diagonal

        # Save chart
        output_file = f"{output_dir}/persistence_diagram_{run_id}_{embedding_model}.html"
        combined.save(output_file)
        print(f"Saved persistence diagram to {output_file}")


def plot_barcode(run_id: str, session: Session, output_dir: str = "output/vis/barcodes") -> None:
    """
    Create and save a barcode visualization for a specific run.

    Args:
        run_id: ID of the run to visualize
        session: SQLModel database session
        output_dir: Directory to save the visualization
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get the run and its persistence diagrams
    run = session.get(Run, run_id)
    if not run or not run.persistence_diagrams:
        print(f"No persistence diagrams found for run {run_id}")
        return

    for pd in run.persistence_diagrams:
        if not pd.diagram_data or "dgms" not in pd.diagram_data:
            continue

        # Convert persistence diagram data to DataFrame for barcode viz
        records = []

        for dim, diagram in enumerate(pd.diagram_data["dgms"]):
            for i, (birth, death) in enumerate(diagram):
                # For infinite death values, use a large value for visualization
                if np.isinf(death):
                    death = birth + (np.max(diagram[:, 1][~np.isinf(diagram[:, 1])]) if np.any(~np.isinf(diagram[:, 1])) else 10.0)

                # Add record for each feature
                records.append({
                    "dimension": f"H{dim}",
                    "feature_id": f"{dim}_{i}",
                    "birth": float(birth),
                    "death": float(death),
                    "persistence": float(death - birth)
                })

        if not records:
            continue

        # Create DataFrame
        df = pl.DataFrame(records)

        # Sort by persistence (descending) within each dimension
        df = df.sort(by=["dimension", "persistence"], descending=[False, True])

        # Create the barcode visualization
        barcode = alt.Chart(df).mark_rule(strokeWidth=3).encode(
            y=alt.Y(
                "feature_id:N",
                title=None,
                sort=alt.EncodingSortField(field="feature_id", order="ascending"),
                axis=alt.Axis(labels=False, ticks=False)
            ),
            x=alt.X("birth:Q", title="Parameter value"),
            x2="death:Q",
            color=alt.Color("dimension:N", title="Homology Dimension"),
            tooltip=["dimension:N", "birth:Q", "death:Q", "persistence:Q"]
        ).properties(
            width=600,
            height=alt.Step(10),  # Controls the bar thickness
            title=f"Barcode - {pd.embedding_model}"
        ).facet(
            row=alt.Row("dimension:N", title=None, header=alt.Header(labelAngle=0))
        )

        # Save chart
        output_file = f"{output_dir}/barcode_{run_id}_{pd.embedding_model}.html"
        barcode.save(output_file)
        print(f"Saved barcode to {output_file}")

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
