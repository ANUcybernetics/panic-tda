import altair as alt
import polars as pl

## visualisation

def plot_persistence_diagram(df: pl.DataFrame, output_dir: str = "output/vis/persistence") -> None:
    """
    Create and save a visualization of persistence diagrams for runs in the DataFrame.

    Args:
        df: DataFrame containing run data with persistence homology information
        output_dir: Directory to save the visualizations
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Check if we have persistence diagram data
    if "persistence_diagram_id" not in df.columns or "homology_dimension" not in df.columns:
        print("No persistence diagram data found in DataFrame")
        return

    # Group by run_id and embedding_model to process each unique combination
    unique_diagrams = df.select(["run_id", "embedding_model", "persistence_diagram_id"]).unique()

    for diagram_row in unique_diagrams.iter_rows(named=True):
        run_id = diagram_row["run_id"]
        embedding_model = diagram_row["embedding_model"]
        pd_id = diagram_row["persistence_diagram_id"]

        # Get all birth/death pairs for this diagram
        diagram_data = df.filter(
            (pl.col("run_id") == run_id) &
            (pl.col("embedding_model") == embedding_model) &
            (pl.col("persistence_diagram_id") == pd_id)
        )

        if diagram_data.height == 0:
            print(f"No data for run {run_id}, embedding {embedding_model}")
            continue

        # Find maximum birth/death values for setting chart domain
        max_value = max(
            diagram_data["birth"].max(),
            diagram_data["death"].max()
        ) * 1.1  # Add 10% margin

        # Create diagonal reference line data
        diagonal_df = pl.DataFrame({
            "x": [0, max_value],
            "y": [0, max_value]
        })

        # Create the persistence diagram visualization
        diagram = alt.Chart(diagram_data).mark_point(filled=True, opacity=0.7).encode(
            x=alt.X("birth:Q", title="Birth", scale=alt.Scale(domain=[0, max_value])),
            y=alt.Y("death:Q", title="Death", scale=alt.Scale(domain=[0, max_value])),
            color=alt.Color("homology_dimension:N", title="Dimension",
                           scale=alt.Scale(scheme="category10")),
            size=alt.Size("persistence:Q", title="Persistence",
                         scale=alt.Scale(range=[20, 400])),
            tooltip=["homology_dimension:N", "birth:Q", "death:Q", "persistence:Q", "entropy:Q"]
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

        # Combine chart elements
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
