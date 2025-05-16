import json
import logging
import os
from typing import Dict
from uuid import UUID

import polars as pl
from plotnine import (
    aes,
    coord_flip,
    element_blank,
    element_text,
    facet_grid,
    facet_wrap,
    geom_bar,
    geom_boxplot,
    geom_line,
    geom_point,
    geom_text,
    geom_violin,
    ggplot,
    labs,
    scale_x_continuous,
    scale_y_continuous,
    theme,
)
from plotnine.options import set_option
from sqlmodel import Session

from panic_tda.data_prep import calculate_cluster_transitions, filter_top_n_clusters
from panic_tda.export import export_mosaic_image

## datavis

set_option("limitsize", False)


def save(plot, filename: str) -> str:
    """
    Save a plotnine plot to file formats.

    Args:
        plot: plotnine plot to save
        filename: Path to save the chart

    Returns:
        Path to the saved file
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(filename)
    os.makedirs(output_dir, exist_ok=True)

    # Save the plot with high resolution
    plot.save(filename, dpi=300, verbose=False)

    return filename


def write_label_map(
    cluster_labels: pl.Series,
    output_path: str = "output/vis/cluster_label_map.json",
) -> Dict[str, int]:
    """
    Map string cluster labels to integers and writes the mapping to a JSON file and a LaTeX file.

    Args:
        cluster_labels: A polars Series of string labels
        output_path: Path to save the JSON mapping file (LaTeX file will use same path with .tex extension)

    Returns:
        Dictionary mapping string labels to integers (0 for "OUTLIER", 1+ for others)
    """
    # Get unique labels
    unique_labels = cluster_labels.unique().to_list()

    # Separate OUTLIER from other labels and sort the rest
    other_labels = sorted([
        label for label in unique_labels if label is not None and label != "OUTLIER"
    ])

    # Create mapping dictionary
    label_map = {}

    # Handle OUTLIER first if it exists
    if "OUTLIER" in unique_labels:
        label_map["OUTLIER"] = 0

    # Assign IDs to sorted labels
    next_id = 1
    for label in other_labels:
        label_map[label] = next_id
        next_id += 1

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Write mapping to JSON file
    with open(output_path, "w") as f:
        json.dump(label_map, f, indent=2)

    # Create LaTeX file path by replacing .json with .tex
    latex_output_path = output_path.replace(".json", ".tex")

    # Create LaTeX content for the table
    latex_content = [
        "\\documentclass{article}",
        "\\usepackage{booktabs}",
        "\\usepackage{longtable}",
        "\\usepackage{array}",
        "\\begin{document}",
        "\\begin{longtable}{|c|p{12cm}|}",
        "\\hline",
        "\\textbf{Index} & \\textbf{Cluster Label} \\\\",
        "\\hline",
    ]

    # Sort items by index value for the table
    sorted_items = sorted(label_map.items(), key=lambda x: x[1])

    # Add rows to the table
    for label, index in sorted_items:
        # Escape special LaTeX characters
        escaped_label = (
            label.replace("_", "\\_")
            .replace("#", "\\#")
            .replace("$", "\\$")
            .replace("%", "\\%")
            .replace("&", "\\&")
            .replace("{", "\\{")
            .replace("}", "\\}")
        )
        latex_content.append(f"{index} & {escaped_label} \\\\")
        latex_content.append("\\hline")

    # Close the table and document
    latex_content.append("\\end{longtable}")
    latex_content.append("\\end{document}")

    # Write LaTeX file
    with open(latex_output_path, "w") as f:
        f.write("\n".join(latex_content))

    return label_map


def read_existing_label_map(column_name: str, input_path: str) -> pl.Expr:
    """
    Read the mapping of string cluster labels to integers from a JSON file
    and return a polars expression to use in with_columns.

    Args:
        column_name: Name of the column to transform
        input_path: Path to the JSON mapping file

    Returns:
        Polars expression that can be used in a with_columns call
    """
    # Read mapping from JSON file
    with open(input_path, "r") as f:
        label_map = json.load(f)

    # Return a polars expression for use in with_columns
    return pl.col(column_name).replace_strict(label_map)


def create_persistence_diagram_chart(df: pl.DataFrame):
    """
    Create a base persistence diagram chart for a single run.

    Args:
        df: DataFrame containing run data with persistence homology information

    Returns:
        A plotnine plot object for the persistence diagram
    """
    # Convert polars DataFrame to pandas for plotnine
    pandas_df = df.to_pandas()

    plot = (
        ggplot(
            pandas_df,
            aes(x="birth", y="persistence", color="factor(homology_dimension)"),
        )
        + geom_point(alpha=0.02)
        + labs(
            x="feature appearance", y="feature persistence", color="homology dimension"
        )
        + theme(figure_size=(10, 5))  # Roughly equivalent to width/height 300px
    )

    return plot


def plot_persistence_diagram(
    df: pl.DataFrame, output_file: str = "output/vis/persistence_diagram.png"
) -> None:
    """
    Create and save a visualization of a single persistence diagram for the given DataFrame.

    Args:
        df: DataFrame containing run data with persistence homology information
        output_file: Path to save the visualization
    """
    # Create the chart
    plot = create_persistence_diagram_chart(df)

    # Save plot with high resolution
    saved_file = save(plot, output_file)

    logging.info(f"Saved single persistence diagram to {saved_file}")


def plot_persistence_diagram_faceted(
    df: pl.DataFrame,
    output_file: str = "output/vis/persistence_diagram.png",
) -> None:
    """
    Create and save a visualization of persistence diagrams for runs in the DataFrame,
    creating a grid of charts (one per run).

    Args:
        df: DataFrame containing run data with persistence homology information
        output_file: Path to save the visualization
    """
    # Create the base plot using the existing function
    plot = create_persistence_diagram_chart(df)

    # Add faceting to the plot
    plot = (
        plot
        + facet_grid("text_model ~ image_model + homology_dimension")
        + theme(figure_size=(15, 5), strip_text=element_text(size=10))
    )

    # Save plot with high resolution
    saved_file = save(plot, output_file)
    logging.info(f"Saved persistence diagrams to {saved_file}")


def plot_persistence_diagram_by_prompt(
    df: pl.DataFrame,
    output_file: str = "output/vis/persistence_diagram.png",
) -> None:
    """
    Create and save a visualization of persistence diagrams by prompt,
    creating a grid of charts (one per prompt).

    Args:
        df: DataFrame containing run data with persistence homology information
        output_file: Path to save the visualization
    """
    # Convert polars DataFrame to pandas for plotnine
    pandas_df = df.to_pandas()

    # Create the base plot with faceting by run_id
    plot = (
        ggplot(pandas_df, aes(x="birth", y="persistence", color="homology_dimension"))
        + geom_point(alpha=0.1)
        + scale_x_continuous(name="Feature Appearance")
        + scale_y_continuous(name="Feature Persistence")
        + labs(color="Dimension")
        + facet_wrap("~ initial_prompt")
        + theme(figure_size=(16, 10), strip_text=element_text(size=8))
    )

    # Save plot with high resolution
    saved_file = save(plot, output_file)
    logging.info(f"Saved persistence diagrams to {saved_file}")


def plot_persistence_entropy(
    df: pl.DataFrame, output_file: str = "output/vis/persistence_entropy.pdf"
) -> None:
    """
    Create and save a visualization of entropy distributions with:
    - entropy on x axis
    - homology_dimension on y axis (treated as a factor)
    - embedding_model as color
    - faceted by text_model (rows) and image_model (columns)

    Args:
        df: DataFrame containing runs data with homology_dimension and entropy
        output_file: Path to save the visualization
    """
    # # Convert homology_dimension to h with subscripts
    # subscripts = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

    # # Transform homology_dimension column to use h with subscripts
    # df = df.with_columns(
    #     pl.concat_str(
    #         pl.lit("h"),
    #         pl.col("homology_dimension").cast(pl.Utf8).str.replace_all(
    #             dict(zip("0123456789", "₀₁₂₃₄₅₆₇₈₉"))
    #         )
    #     ).alias("homology_dimension")
    # )

    # Convert polars DataFrame to pandas for plotnine
    pandas_df = df.to_pandas()

    plot = (
        ggplot(
            pandas_df,
            aes(x="embedding_model", y="entropy", fill="embedding_model"),
        )
        + geom_violin()
        + geom_boxplot(fill="white", width=0.5, alpha=0.5)
        + labs(x="embedding model", y="distribution of persistence entropy")
        + facet_grid("homology_dimension ~ image_model + text_model")
        + theme(
            figure_size=(10, 6),
            strip_text=element_text(size=10),
            axis_ticks_major_x=element_blank(),
            axis_text_x=element_blank(),
        )
    )

    # Save plot with high resolution
    saved_file = save(plot, output_file)
    logging.info(f"Saved persistence entropy plot to {saved_file}")


def plot_persistence_entropy_by_prompt(
    df: pl.DataFrame, output_file: str = "output/vis/persistence_entropy_by_prompt.pdf"
) -> None:
    """
    Create and save a visualization of entropy distributions faceted by prompt and models.
    - entropy on x axis
    - homology_dimension on y axis (treated as a factor)
    - embedding_model as color
    - faceted by initial_prompt (rows) and text_model + image_model (columns)

    Args:
        df: DataFrame containing runs data with homology_dimension and entropy
        output_file: Path to save the visualization
    """
    # Convert polars DataFrame to pandas for plotnine
    pandas_df = df.to_pandas()
    # pandas_df = df.filter(pl.col("embedding_model") == "Nomic").to_pandas()

    # Create the plot with faceting
    # Define subscript translator for x-axis labels
    # subscripts = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

    plot = (
        ggplot(
            pandas_df,
            aes(x="initial_prompt", y="entropy", fill="embedding_model"),
        )
        # + geom_violin(alpha=0.7, width=0.7)
        + geom_boxplot()
        + labs(x="initial prompt", y="entropy", fill="embedding model")
        + coord_flip()
        + facet_grid("~ homology_dimension + image_model + text_model")
        + theme(
            figure_size=(30, 20), strip_text=element_text(size=8)
        )  # Adjust size as needed
    )

    # Save plot with high resolution
    saved_file = save(plot, output_file)
    logging.info(f"Saved persistence entropy by prompt plot to {saved_file}")


def plot_semantic_drift(
    df: pl.DataFrame, output_file: str = "output/vis/semantic_drift.pdf"
) -> None:
    """
    Create a line plot showing semantic drift over sequence number,
    faceted by initial prompt and model.

    Args:
        df: DataFrame containing embedding data with semantic_drift and sequence_number
        output_file: Path to save the visualization
    """
    # Scale drift columns between 1.0 and 10.0
    # df = df.with_columns(
    #     (
    #         1.0
    #         + (pl.col("drift_euclid") - pl.col("drift_euclid").min())
    #         * 9.0
    #         / (pl.col("drift_euclid").max() - pl.col("drift_euclid").min())
    #     ).alias("drift_euclid")
    # ).with_columns(
    #     (
    #         1.0
    #         + (pl.col("drift_cosine") - pl.col("drift_cosine").min())
    #         * 9.0
    #         / (pl.col("drift_cosine").max() - pl.col("drift_cosine").min())
    #     ).alias("drift_cosine")
    # )

    # Unpivot the drift columns
    pandas_df = df.unpivot(
        ["drift_euclid", "drift_cosine"],
        index=list(set(df.columns) - set(["drift_euclid", "drift_cosine"])),
        variable_name="drift_metric",
        value_name="drift_value",
    ).to_pandas()

    # Create a single chart with faceting
    plot = (
        ggplot(
            pandas_df,
            aes(
                x="sequence_number",
                y="drift_value",
                # size="drift_value",
                color="initial_prompt",
            ),
        )
        + geom_line()
        # + scale_color_manual(values=["black", "red"])
        + labs(x="sequence number", y="semantic drift")
        + facet_grid(
            "initial_prompt + drift_metric + embedding_model ~",
            labeller="label_context",
        )
        + theme(
            figure_size=(20, 10),
            strip_text=element_text(size=10),
        )
    )

    # Save the chart
    saved_file = save(plot, output_file)
    logging.info(f"Saved semantic drift plot to {saved_file}")


def plot_cluster_timelines(
    df: pl.DataFrame,
    output_file: str = "output/vis/cluster_timelines.pdf",
    label_map_path: str = "output/vis/cluster_label_map.json",
) -> None:
    """
    Create a faceted scatter plot showing cluster labels over sequence number for each run,
    faceted by initial prompt.

    Args:
        df: DataFrame containing embedding data with cluster_label and sequence_number
        output_file: Path to save the visualization
        label_map_path: Path to the JSON file containing cluster label mappings
    """
    # Filter out null cluster labels
    filtered_df = df.filter(pl.col("cluster_label").is_not_null())

    # Convert string cluster labels to indices
    indexed_df = filtered_df.with_columns(
        read_existing_label_map("cluster_label", label_map_path).alias("cluster_index")
    )

    # Calculate the number of unique facet combinations to determine figure height
    unique_facets_count = (
        indexed_df.select("initial_prompt", "embedding_model", "network")
        .unique()
        .height
    )

    # Set a base height per facet (adjust as needed)
    base_height_per_facet = 3
    figure_height = max(10, unique_facets_count * base_height_per_facet)

    # Convert polars DataFrame to pandas for plotnine
    pandas_df = indexed_df.to_pandas()

    # Create the plot
    plot = (
        ggplot(
            pandas_df,
            aes(
                x="sequence_number",
                y="run_id",
                color="cluster_index",
                alpha="cluster_index != 0",  # 0 is now the OUTLIER value
            ),
        )
        + geom_line(colour="black", alpha=0.5)
        + geom_point(size=8, show_legend=False)
        + geom_text(aes(label="cluster_index"), color="black", size=8)
        + labs(x="sequence number", y="run id", color="cluster")
        + facet_wrap(
            "~ initial_prompt + embedding_model + network", scales="free", ncol=4
        )
        + theme(
            figure_size=(50, figure_height),  # Use calculated adaptive height
            strip_text=element_text(size=10),
            axis_text_y=element_blank(),
        )
    )

    # Save the chart
    saved_file = save(plot, output_file)
    logging.info(f"Saved cluster timelines plot to {saved_file}")


def plot_cluster_bubblegrid(
    df: pl.DataFrame,
    include_outliers: bool = False,
    output_file: str = "output/vis/cluster_bubblegrid.pdf",
    label_map_path: str = "output/vis/cluster_label_map.json",
) -> None:
    """
    Create a bubble grid visualization of cluster label frequencies.

    Uses a grid where x is cluster_label, y is initial_prompt, and bubble size
    represents count. Uses facet_grid by embedding_model and network.

    Args:
        df: DataFrame containing embedding data with cluster_label and initial_prompt
        include_outliers: If False, filter out all "OUTLIER" cluster labels
        output_file: Path to save the visualization
        label_map_path: Path to the JSON file containing cluster label mappings
    """
    # Filter out null cluster labels
    filtered_df = df.filter(pl.col("cluster_label").is_not_null())

    # Filter out outliers if specified
    if not include_outliers:
        filtered_df = filtered_df.filter(pl.col("cluster_label") != "OUTLIER")

    # Group by and count occurrences
    counts_df = (
        filtered_df.group_by([
            "cluster_label",
            "initial_prompt",
            "embedding_model",
            "network",
        ])
        .agg(pl.len().alias("count"))
        .sort(["embedding_model", "network", "cluster_label", "initial_prompt"])
    )

    # Convert string cluster labels to their integer values using the label mapping
    counts_df = counts_df.with_columns(
        read_existing_label_map("cluster_label", label_map_path).alias("cluster_index"),
    )

    # Convert to pandas for plotting
    pandas_df = counts_df.to_pandas()

    # Add display_label column that only shows labels for counts > 500
    pandas_df["display_label"] = pandas_df.apply(
        lambda row: row["cluster_index"] if row["count"] > 600 else "", axis=1
    )

    # Create the bubble grid visualization
    plot = (
        ggplot(
            pandas_df,
            aes(
                x="cluster_index",
                y="initial_prompt",
                size="count",
                fill="cluster_label",
            ),
        )
        + geom_point(alpha=0.8, show_legend=False)
        + geom_text(aes(label="display_label"), color="black", size=6)
        + labs(
            x="Cluster",
            y="Prompt",
            size="Count",
            fill="Count",
        )
        + facet_grid("embedding_model ~ network")
        + theme(
            figure_size=(30, 20),
            strip_text=element_text(size=10),
            axis_text_y=element_text(size=8),
        )
    )

    # Save the plot
    saved_file = save(plot, output_file)
    logging.info(f"Saved cluster bubble grid to {saved_file}")


def plot_cluster_histograms(
    df: pl.DataFrame,
    include_outliers: bool = False,
    output_file: str = "output/vis/cluster_histograms.pdf",
) -> None:
    """
    Create a faceted histogram showing counts of each cluster label,
    faceted by embedding model and network.

    Args:
        df: DataFrame containing embedding data with cluster_label
        include_outliers: If False, filter out all "OUTLIER" cluster labels
        output_file: Path to save the visualization
    """
    # Convert polars DataFrame to pandas for plotnine
    df = df.filter(pl.col("cluster_label").is_not_null())
    if not include_outliers:
        df = df.filter(pl.col("cluster_label") != "OUTLIER")

    # Calculate the number of unique cluster labels to determine figure height
    num_unique_clusters = df.get_column("cluster_label").n_unique()
    # Dynamic height based on number of unique clusters
    figure_height = max(10, num_unique_clusters * 0.8)

    # Create the plot
    plot = (
        ggplot(
            df.to_pandas(),
            aes(x="cluster_label", fill="embedding_model"),
        )
        + geom_bar()
        + labs(x="cluster_label", y="count")
        + facet_wrap(
            "~ embedding_model + network",
            # labeller="label_context",
            # scales="free_y",
            ncol=4,
        )
        + theme(
            figure_size=(22, figure_height),
            strip_text=element_text(size=10),
        )
        + coord_flip()
    )

    # Save the chart
    saved_file = save(plot, output_file)
    logging.info(f"Saved cluster histograms plot to {saved_file}")


def plot_cluster_histograms_top_n(
    df: pl.DataFrame,
    top_n: int,
    include_outliers: bool = False,
    output_file: str = "output/vis/cluster_histograms_top_n.pdf",
) -> None:
    """
    Create a faceted histogram showing counts of the top N most frequent cluster labels,
    faceted by embedding model and network.

    Args:
        df: DataFrame containing embedding data with cluster_label
        top_n: Number of top clusters to display
        include_outliers: If False, filter out all "OUTLIER" cluster labels
        output_file: Path to save the visualization
    """
    # Filter out null cluster labels
    df = df.filter(pl.col("cluster_label").is_not_null())
    if not include_outliers:
        df = df.filter(pl.col("cluster_label") != "OUTLIER")

    # Filter to top N clusters
    df = filter_top_n_clusters(df, top_n, ["embedding_model", "network"])

    # Calculate the number of unique cluster labels to determine figure height
    num_unique_clusters = df.get_column("cluster_label").n_unique()
    # Dynamic height based on number of unique clusters
    figure_height = max(10, num_unique_clusters * 2)

    # Create the plot
    plot = (
        ggplot(
            df.to_pandas(),
            aes(x="cluster_label", fill="embedding_model"),
        )
        + geom_bar()
        + labs(x="cluster_label", y="count")
        + facet_wrap(
            "~ embedding_model + network",
            # labeller="label_context",
            scales="free_x",
            ncol=1,
        )
        + theme(
            figure_size=(22, figure_height),
            strip_text=element_text(size=10),
        )
        + coord_flip()
    )

    # Save the chart
    saved_file = save(plot, output_file)
    logging.info(f"Saved top {top_n} cluster histograms plot to {saved_file}")


def plot_cluster_transitions(
    df: pl.DataFrame,
    include_outliers: bool,
    output_file: str = "output/vis/cluster_transitions.pdf",
    label_map_path: str = "output/vis/cluster_label_map.json",
) -> None:
    """
    Create a visualization of cluster transitions within runs.

    For each run_id, calculates transitions between consecutive cluster labels
    and displays them as a grid of points where point size represents transition frequency.

    Args:
        df: DataFrame containing embedding data with cluster_label, run_id, and sequence_number
        include_outliers: If False, filter out all "OUTLIER" cluster labels
        output_file: Path to save the visualization
        label_map_path: Path to the JSON file containing cluster label mappings
    """
    # Use calculate_cluster_transitions to get transition counts
    transition_counts = calculate_cluster_transitions(
        df, ["embedding_model", "network"], include_outliers
    )

    # Transform cluster labels to their integer values using the label mapping
    transition_counts = transition_counts.with_columns([
        read_existing_label_map("from_cluster", label_map_path).alias("from_cluster"),
        read_existing_label_map("to_cluster", label_map_path).alias("to_cluster"),
    ])

    # Convert to pandas for plotting
    pandas_df = transition_counts.to_pandas()

    # Create point grid visualization
    plot = (
        ggplot(
            pandas_df,
            aes(
                x="to_cluster",
                y="from_cluster",
                size="transition_count",
                fill="transition_count",
            ),
        )
        + geom_point()
        + labs(
            x="to cluster",
            y="from cluster",
            size="transition count",
            fill="transition count",
        )
        + facet_grid("embedding_model ~ network")
        + theme(
            figure_size=(25, 14),
            strip_text=element_text(size=10),
        )
    )

    # Save plot
    saved_file = save(plot, output_file)
    logging.info(f"Saved cluster transition matrix to {saved_file}")


def export_cluster_counts_to_json(
    df: pl.DataFrame,
    output_file: str = "output/vis/cluster_counts.json",
    top_n: int = 10,
) -> None:
    """
    Count occurrences of each cluster label grouped by initial_prompt, embedding_model,
    text_model and cluster_label, then export to a JSON file sorted in decreasing order.
    The output JSON has an array of objects with embedding_model and cluster_label keys,
    and a "counts" object mapping text to count values.

    Args:
        df: DataFrame containing embedding data with cluster_label
        output_file: Path to save the JSON file
        top_n: Number of top text entries to include for each cluster (default: 10)
    """
    # Group by the required columns and count occurrences
    counts_df = (
        df.filter(pl.col("cluster_label").is_not_null())
        .group_by(["embedding_model", "cluster_label", "text"])
        .agg(pl.len().alias("count"))
        .sort(
            ["embedding_model", "cluster_label", "count"],
            descending=[False, False, True],
        )
    )

    # Convert to pandas for easier manipulation (NOTE: that's Claude's comment, not mine. TODO find a polars way to do this, which is probably better)
    counts_pd = counts_df.to_pandas()

    # Create the result structure with nested counts
    result = []

    # Get unique combinations of embedding_model and cluster_label
    unique_groups = counts_pd[["embedding_model", "cluster_label"]].drop_duplicates()

    for _, group in unique_groups.iterrows():
        embedding_model = group["embedding_model"]
        cluster_label = group["cluster_label"]

        # Get the data for this group
        group_data = counts_pd[
            (counts_pd["embedding_model"] == embedding_model)
            & (counts_pd["cluster_label"] == cluster_label)
        ]

        # Calculate the total count for this cluster
        cluster_total_count = group_data["count"].sum()

        # Take the top N rows
        top_group_data = group_data.head(top_n)

        # Create a counts dictionary
        counts_dict = {
            row["text"]: row["count"] for _, row in top_group_data.iterrows()
        }

        # Add to result
        result.append({
            "embedding_model": embedding_model,
            "cluster_label": cluster_label,
            "cluster_count": int(
                cluster_total_count
            ),  # Convert numpy.int64 to int for JSON serialization
            "counts": counts_dict,
        })

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    # Write to JSON file
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    logging.info(f"Saved cluster counts to {output_file}")


def plot_cluster_example_images(
    df: pl.DataFrame,
    num_examples: int,
    embedding_model: str,
    session: Session,
    output_file: str = "output/vis/cluster_examples.jpg",
) -> None:
    """
    Create a mosaic image showing example images for each cluster.

    Args:
        df: DataFrame containing embedding data with cluster_label and invocation_id
        num_examples: Number of example images to include for each cluster
        embedding_model: The embedding model to filter by
        output_file: Path to save the visualization
    """

    # Filter dataframe to get only rows with the specified embedding model
    # and where cluster_label is not null and not0
    filtered_df = df.filter(
        (pl.col("embedding_model") == embedding_model)
        & (pl.col("cluster_label").is_not_null())
        & (pl.col("cluster_label") != "OUTLIER")
    )

    # Group by cluster_label and collect the first num_examples invocation_ids for each
    cluster_examples = {}

    # Convert to pandas for easier groupby operations
    pandas_df = filtered_df.to_pandas()

    # Group by cluster_label
    for cluster_label, group in pandas_df.groupby("cluster_label"):
        # Get the first num_examples invocation_ids
        invocation_ids = group["invocation_id"].head(num_examples).tolist()
        cluster_examples[str(cluster_label)] = [UUID(id) for id in invocation_ids]

    # Use export_mosaic_image to create the visualization
    if cluster_examples:
        export_mosaic_image(cluster_examples, session, output_file=output_file)
        logging.info(f"Saved cluster example images to {output_file}")
    else:
        logging.warning(
            f"No cluster examples found for embedding model {embedding_model}"
        )


def plot_invocation_duration(
    df: pl.DataFrame, output_file: str = "output/vis/invocation_duration.pdf"
) -> None:
    """
    Create and save a visualization of invocation duration distribution,
    with model on x-axis and duration on y-axis.

    Args:
        df: DataFrame containing embedding data with duration
        output_file: Path to save the visualization
    """
    # Convert polars DataFrame to pandas for plotnine
    pandas_df = df.to_pandas()

    # Create the plot with model on x-axis and duration on y-axis
    plot = (
        ggplot(
            pandas_df,
            aes(x="model", y="duration", fill="model"),
        )
        + geom_boxplot()
        + labs(x="Model", y="Invocation Duration")
        + theme(figure_size=(15, 8), axis_text_x=element_text(angle=45, hjust=1))
    )

    # Save plot with high resolution
    saved_file = save(plot, output_file)
    logging.info(f"Saved invocation duration plot to {saved_file}")
