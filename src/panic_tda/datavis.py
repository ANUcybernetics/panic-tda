import json
import logging
import os
from typing import Optional
from uuid import UUID

import pandas as pd
import polars as pl
from plotnine import (
    aes,
    coord_flip,
    element_blank,
    element_text,
    facet_grid,
    facet_wrap,
    geom_area,
    geom_bar,
    geom_boxplot,
    geom_line,
    geom_point,
    geom_text,
    geom_violin,
    ggplot,
    labs,
    scale_color_brewer,
    scale_size_continuous,
    scale_x_continuous,
    scale_y_continuous,
    theme,
)
from plotnine.options import set_option
from sqlmodel import Session

from panic_tda.data_prep import (
    calculate_cluster_run_lengths,
    calculate_cluster_transitions,
    filter_top_n_clusters,
)
from panic_tda.export import export_mosaic_image

set_option("limitsize", False)

# these things required for IEEE SMC camera-ready
# matplotlib.rcParams["pdf.fonttype"] = 42  # For TrueType
# matplotlib.rcParams["ps.fonttype"] = 42  # For PostScript Type 1

## datavis


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


def format_label_map_as_markdown(
    map_df: pl.DataFrame,
    embedding_model: Optional[str] = None,
    max_label_length: int = 70,
) -> str:
    """
    Format a cluster label mapping as a markdown table.

    Args:
        map_df: DataFrame with embedding_model, cluster_label, and cluster_index columns
        embedding_model: Optional specific model to filter for
        max_label_length: Maximum length for cluster labels before truncation

    Returns:
        Markdown-formatted table string
    """
    from panic_tda.utils import polars_to_markdown

    # Filter for specific model if provided
    if embedding_model:
        filtered_df = map_df.filter(pl.col("embedding_model") == embedding_model)
    else:
        filtered_df = map_df

    # Sort by cluster_index and select only the columns we need
    table_df = filtered_df.sort("cluster_index").select([
        "cluster_index",
        "cluster_label",
    ])

    # Use the utility function with custom headers
    return polars_to_markdown(
        table_df, max_col_width=max_label_length, headers=["Index", "Cluster Label"]
    )


def create_label_map_df(
    clusters_df: pl.DataFrame,
    print_mapping: bool = False,
) -> pl.DataFrame:
    """
    Creates a mapping of cluster labels to sequential integer indices.

    This function takes clusters data and creates a mapping where each unique
    cluster gets a sequential integer index starting from 1. Outliers (cluster_id == -1)
    are excluded from the mapping. Clusters are ordered by size (largest first) within
    each embedding model.

    Args:
        clusters_df: DataFrame containing cluster data
        print_mapping: If True, print the mapping to console as markdown tables

    Returns:
        DataFrame with embedding_model, cluster_label, and cluster_index columns for joining
    """

    # Filter out outliers (cluster_id == -1)
    filtered_df = clusters_df.filter(pl.col("cluster_id") != -1)

    # Calculate cluster sizes by counting occurrences
    cluster_sizes = filtered_df.group_by(["embedding_model", "cluster_label"]).agg(
        pl.len().alias("cluster_size")
    )

    # Get unique combinations of embedding_model and cluster_label
    # Use cluster_size to order by importance (larger clusters first)
    unique_clusters = cluster_sizes.sort(
        ["embedding_model", "cluster_size", "cluster_label"],
        descending=[False, True, False],
    )

    # Add cluster_index column with increasing integers starting at 1
    # Group by embedding_model to restart numbering for each model
    map_df = unique_clusters.with_columns(
        pl.col("cluster_label")
        .rank(method="dense")
        .over("embedding_model")
        .cast(pl.Int64)
        .alias("cluster_index")
    ).drop("cluster_size")

    # Print mapping if requested
    if print_mapping:
        print("\nCluster Label Mapping:")
        print("=" * 80)
        for embedding_model in map_df["embedding_model"].unique().sort():
            print(f"\n## {embedding_model}\n")
            markdown_table = format_label_map_as_markdown(map_df, embedding_model)
            print(markdown_table)
            print()

    # Return the mapping dataframe for joining
    return map_df.select("embedding_model", "cluster_label", "cluster_index")


def create_persistence_diagram_chart(pd_df: pl.DataFrame):
    """
    Create a base persistence diagram chart.

    Args:
        pd_df: DataFrame from load_pd_df() containing persistence diagram data
               with columns: birth, persistence, homology_dimension

    Returns:
        A plotnine plot object for the persistence diagram
    """
    # Convert polars DataFrame to pandas for plotnine
    pandas_df = pd_df.to_pandas()

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
    pd_df: pl.DataFrame, output_file: str = "output/vis/persistence_diagram.png"
) -> None:
    """
    Create and save a visualization of a single persistence diagram for the given DataFrame.

    Args:
        pd_df: DataFrame from load_pd_df() containing persistence diagram data
        output_file: Path to save the visualization
    """
    # Create the chart
    plot = create_persistence_diagram_chart(pd_df)

    # Save plot with high resolution
    saved_file = save(plot, output_file)

    logging.info(f"Saved single persistence diagram to {saved_file}")


def plot_persistence_diagram_faceted(
    pd_df: pl.DataFrame,
    output_file: str = "output/vis/persistence_diagram.png",
) -> None:
    """
    Create and save a visualization of persistence diagrams for runs in the DataFrame,
    creating a grid of charts (one per run).

    Args:
        pd_df: DataFrame from load_pd_df() containing persistence diagram data
        output_file: Path to save the visualization
    """
    # Create the base plot using the existing function
    plot = create_persistence_diagram_chart(pd_df)

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
    pd_df: pl.DataFrame,
    output_file: str = "output/vis/persistence_diagram.png",
) -> None:
    """
    Create and save a visualization of persistence diagrams by prompt,
    creating a grid of charts (one per prompt).

    Args:
        pd_df: DataFrame from load_pd_df() containing persistence diagram data
        output_file: Path to save the visualization
    """
    # Convert polars DataFrame to pandas for plotnine
    pandas_df = pd_df.to_pandas()

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
    # # Transform homology_dimension column to use h with subscripts
    df = df.with_columns(
        pl.col("homology_dimension")
        .replace_strict({0: "h₀", 1: "h₁", 2: "h₂"})
        .alias("homology_dimension")
    )

    # Convert polars DataFrame to pandas for plotnine
    pandas_df = df.to_pandas()

    plot = (
        ggplot(
            pandas_df,
            aes(x="embedding_model", y="entropy", fill="embedding_model"),
        )
        + geom_violin()
        + geom_boxplot(fill="white", width=0.5, alpha=0.5)
        + labs(y="persistence entropy", fill="embedding model")
        + facet_grid("homology_dimension ~ network")
        + theme(
            figure_size=(6.8, 5),
            plot_margin=0.0,
            # strip_text=element_text(size=10),
            axis_ticks_major_x=element_blank(),
            axis_text_x=element_blank(),
            axis_title_x=element_blank(),
            legend_position="bottom",
        )
    )

    # Save plot with high resolution
    saved_file = save(plot, output_file)
    logging.info(f"Saved persistence entropy plot to {saved_file}")


def plot_cluster_run_lengths(
    clusters_df: pl.DataFrame, output_file: str = "output/vis/cluster_run_lengths.pdf"
) -> None:
    """
    Create and save a visualization of cluster run length distributions with:
    - run_length on x axis
    - count on y axis
    - network as color
    - embedding_model represented by point shape

    Args:
        clusters_df: DataFrame containing cluster data with sequence information
        output_file: Path to save the visualization
    """

    # Calculate run lengths using calculate_cluster_run_lengths
    run_lengths_df = calculate_cluster_run_lengths(clusters_df)

    # Filter out null or OUTLIER cluster labels if needed
    run_lengths_df = run_lengths_df.filter(
        (pl.col("cluster_label").is_not_null()) & (pl.col("cluster_label") != "OUTLIER")
    )

    # Aggregate to get counts by run_length, embedding_model, and network
    count_df = (
        run_lengths_df.group_by(["run_length", "embedding_model", "network"])
        .agg(pl.len().alias("count"))
        .sort(["run_length", "embedding_model", "network"])
    )

    # Convert polars DataFrame to pandas for plotnine
    pandas_df = count_df.to_pandas()

    plot = (
        ggplot(
            pandas_df,
            aes(x="run_length", y="count", color="network"),
        )
        + geom_point(size=2, stroke=1, fill="none")
        + geom_line()
        + scale_color_brewer(type="qual", palette=2)
        # + scale_x_continuous(limits=[0, 12], breaks=range(0, 13))
        # + scale_y_log10()  # Log scale for y-axis
        + labs(x="run length", y="# runs", color="")
        + facet_wrap("~ embedding_model", ncol=1)
        + theme(
            figure_size=(8, 6),
            plot_margin=0.0,
            legend_position="top",
        )
    )

    # Save plot with high resolution
    saved_file = save(plot, output_file)
    logging.info(f"Saved cluster run lengths plot to {saved_file}")


def plot_cluster_run_length_violin(
    clusters_df: pl.DataFrame,
    include_outliers: bool = False,
    output_file: str = "output/vis/cluster_run_length_violin.pdf",
) -> None:
    """
    Create and save a violin plot of cluster run length distributions.

    - run_length on y-axis
    - embedding_model on x-axis and as fill color
    - faceted by network

    Args:
        clusters_df: DataFrame containing cluster data
        include_outliers: If False, filter out outliers
        output_file: Path to save the visualization
    """

    run_lengths_df = calculate_cluster_run_lengths(clusters_df, include_outliers)

    # Filter out null or OUTLIER cluster labels
    run_lengths_df = run_lengths_df.filter(
        (pl.col("cluster_label").is_not_null()) & (pl.col("cluster_label") != "OUTLIER")
    )

    # Convert polars DataFrame to pandas for plotnine
    pandas_df = run_lengths_df.to_pandas()

    plot = (
        ggplot(
            pandas_df,
            aes(x="embedding_model", y="run_length", fill="embedding_model"),
        )
        + geom_violin()
        + geom_boxplot(
            width=0.5, fill="white", alpha=0.5
        )  # Add a narrow boxplot inside, hide its outliers
        + labs(
            y="run length distribution", fill="embedding model"
        )  # X-axis label from aes, fill legend suppressed
        + facet_grid("~ network")  # Facet by network on columns
        + theme(
            figure_size=(6.8, 2.5),  # Adjust as needed
            plot_margin=0.0,
            axis_text_x=element_blank(),
            axis_title_x=element_blank(),
            legend_position="bottom",  # Fill color legend is redundant with x-axis
        )
    )

    # Save plot with high resolution
    saved_file = save(plot, output_file)
    logging.info(f"Saved cluster run length violin plot to {saved_file}")


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
    Create a ridgeline plot showing cosine semantic drift,
    with rows binned by sequence number and columns faceted by network.

    Args:
        df: DataFrame containing embedding data with semantic_drift and sequence_number
        output_file: Path to save the visualization
    """
    import numpy as np
    from scipy.stats import gaussian_kde

    # Create 10 evenly spaced bins using qcut for equal-sized bins
    n_bins = 10
    bin_labels = [f"{i * 10}-{(i + 1) * 10}%" for i in range(n_bins)]

    # Bin sequence_number into discrete quantiles
    df = df.with_columns(
        pl.col("sequence_number")
        .qcut(quantiles=n_bins, labels=bin_labels, allow_duplicates=True)
        .alias("sequence_bin")
    )

    # Convert to pandas for processing
    pandas_df = df.to_pandas()

    # Create density data for ridgeline plot
    ridge_data = []
    # Use full cosine distance range [0, 2]
    min_drift = 0
    max_drift = 2
    x_grid = np.linspace(min_drift, max_drift, 200)  # Grid for density evaluation

    # Process each network and bin combination
    for network in pandas_df["network"].unique():
        for i, bin_label in enumerate(bin_labels):
            # Get data for this bin and network
            bin_data = pandas_df[
                (pandas_df["network"] == network)
                & (pandas_df["sequence_bin"] == bin_label)
            ]["semantic_drift"].values

            if len(bin_data) > 1:
                # Calculate density
                kde = gaussian_kde(bin_data, bw_method=0.3)
                density = kde(x_grid)

                # Scale and offset density for ridgeline effect
                y_offset = i  # Base y position
                # Scale density to fit within ~80% of the space between ridges
                density_scaled = (
                    density * 0.8 / density.max()
                    if density.max() > 0
                    else density * 0.8
                )

                # Create data points for the ridge
                for j, (x, d) in enumerate(zip(x_grid, density_scaled)):
                    ridge_data.append({
                        "x": x,
                        "y": y_offset + d,
                        "y_base": y_offset,
                        "sequence_bin": bin_label,
                        "sequence_bin_num": i,
                        "network": network,
                        "density": d,
                    })

    # Create DataFrame for plotting
    ridge_df = pd.DataFrame(ridge_data)

    # Create ordered categorical for proper display
    ridge_df["sequence_bin"] = pd.Categorical(
        ridge_df["sequence_bin"],
        categories=bin_labels[::-1],  # Reverse for top-to-bottom
        ordered=True,
    )

    # Create the ridgeline plot
    plot = (
        ggplot(ridge_df, aes(x="x", y="y", group="sequence_bin"))
        # Draw the ridge areas with no fill, just black outline
        + geom_area(
            position="identity",
            fill="white",
            color="black",
            size=0.5,
        )
        # Add baseline
        + geom_line(
            aes(y="y_base"),
            color="gray",
            size=0.3,
            alpha=0.5,
        )
        + scale_y_continuous(
            breaks=list(range(n_bins)),
            labels=bin_labels[::-1],
            expand=(0.01, 0),
        )
        + scale_x_continuous(
            expand=(0.01, 0),
            limits=(0, 2),  # Full cosine distance range
        )
        + labs(x="cosine drift", y="sequence progress")
        + facet_wrap(
            "~ network",
            labeller="label_context",
            ncol=2,
        )
        + theme(
            figure_size=(12, 8),
            strip_text=element_text(size=12),
            panel_grid_major=element_blank(),
            panel_background=element_blank(),
            axis_ticks_major_y=element_blank(),
        )
    )

    # Save the chart
    saved_file = save(plot, output_file)
    logging.info(f"Saved semantic drift plot to {saved_file}")


def plot_cluster_timelines(
    df: pl.DataFrame,
    clusters_df: pl.DataFrame,
    output_file: str = "output/vis/cluster_timelines.pdf",
) -> None:
    """
    Create a faceted scatter plot showing cluster labels over sequence number for each run,
    faceted by initial prompt.

    Args:
        df: DataFrame containing embedding data with cluster_label and sequence_number
        clusters_df: DataFrame containing cluster data for label mapping
        output_file: Path to save the visualization
    """
    # Create label mapping
    label_map_df = create_label_map_df(clusters_df)

    # Filter out null cluster labels
    filtered_df = df.filter(pl.col("cluster_label").is_not_null())

    # Join with the label_map_df to add the cluster_index column
    indexed_df = filtered_df.join(label_map_df, on=["embedding_model", "cluster_label"])

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
        + geom_line(colour="black", alpha=0.5, show_legend=False)
        + geom_point(size=8, show_legend=False)
        + geom_text(
            aes(label="cluster_index"), color="black", size=8, show_legend=False
        )
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
    clusters_df: pl.DataFrame,
    include_outliers: bool = False,
    output_file: str = "output/vis/cluster_bubblegrid.pdf",
) -> None:
    """
    Create a bubble grid visualization of cluster label frequencies.

    Uses a grid where x is cluster_label, y is initial_prompt, and bubble size
    represents count. Uses facet_grid by embedding_model and network.

    Args:
        clusters_df: DataFrame containing cluster data
        include_outliers: If False, filter out all "OUTLIER" cluster labels
        output_file: Path to save the visualization
    """
    # Create label mapping
    label_map_df = create_label_map_df(clusters_df)

    # Filter out outliers if requested
    if not include_outliers:
        filtered_df = clusters_df.filter(pl.col("cluster_id") != -1)
    else:
        filtered_df = clusters_df

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

    # Join with the label_map_df to add the cluster_index column
    counts_df = counts_df.join(label_map_df, on=["embedding_model", "cluster_label"])

    # Define the list of cluster indices to display labels for
    displayable_cluster_indices = [1, 2, 17, 27, 37]

    # Create display_label column.
    counts_df = counts_df.with_columns(
        pl.when(
            (pl.col("cluster_index").is_in(displayable_cluster_indices))
            & (
                pl.col("count")
                == pl.max("count").over(["embedding_model", "network", "cluster_index"])
            )
        )
        .then(pl.col("cluster_index").cast(pl.Utf8))
        .otherwise(pl.lit(""))
        .alias("display_label")
    )

    # print out all the "labelled" (i.e. top 10%) cluster labels to LaTeX
    # print(
    #     counts_df.filter(pl.col("display_label") != pl.lit(""))
    #     .select("cluster_index", "cluster_label")
    #     .unique()
    #     .to_pandas()
    #     .to_latex(index=False)
    # )

    # Convert to pandas for plotting (only at the end)
    pandas_df = counts_df.to_pandas()

    # Determine the range and breaks for the x-axis
    # Safely get max cluster_index, defaulting to 0 if empty or all NaN
    if pandas_df["cluster_index"].empty:  # Check if the Series is empty
        max_ci_int = 0
    else:
        _max_val = pandas_df["cluster_index"].max()  # Get max value
        # Check if _max_val is NaN (which can happen if all values are NaN, or if the series is empty and .max() returns NaN for some pandas versions)
        # A standard way to check for NaN is (value != value)
        if isinstance(_max_val, float) and _max_val != _max_val:
            max_ci_int = 0
        else:
            try:
                max_ci_int = int(_max_val)  # Convert to int
            except (
                ValueError,
                TypeError,
            ):  # Catch potential errors during conversion if _max_val is not int-convertible
                max_ci_int = 0  # Default to 0 on error

    # Define tick positions (every 2)
    # The range goes up to max_ci_int + 2 to ensure the last tick is at or after max_ci_int
    x_axis_ticks = list(range(0, max_ci_int + 2, 2))

    # Define labels for these ticks: show label if tick is a multiple of 10, otherwise empty string
    x_axis_labels = [str(tick) if tick % 10 == 0 else "" for tick in x_axis_ticks]

    plot = (
        ggplot(
            pandas_df,
            aes(
                x="cluster_index",
                y="initial_prompt",
                size="count",
                fill="cluster_index",
            ),
        )
        + geom_point(alpha=0.8, show_legend=False)
        + geom_text(aes(label="display_label"), color="black", size=10)
        + scale_size_continuous(range=(1, 15))
        + scale_x_continuous(
            breaks=x_axis_ticks, labels=x_axis_labels
        )  # Apply new ticks and labels
        + labs(
            x="cluster index",
            y="initial prompt",
            size="count",
            fill="cluster index",
        )
        + facet_grid("embedding_model ~ network", labeller="label_context")
        + theme(
            figure_size=(22, 11),
            plot_margin=0.0,
            strip_text=element_text(size=20),
            axis_text=element_text(size=14),
            axis_title=element_text(size=20),
        )
    )

    # Save the plot
    saved_file = save(plot, output_file)
    logging.info(f"Saved cluster bubble grid to {saved_file}")


def plot_cluster_run_length_bubblegrid(
    clusters_df: pl.DataFrame,
    include_outliers: bool = False,
    output_file: str = "output/vis/cluster_run_length_bubblegrid.pdf",
) -> None:
    """
    Create a bubble grid visualization of average cluster run lengths.

    Uses a grid where x is cluster_label, y is initial_prompt, and bubble size
    represents average run length. Uses facet_grid by embedding_model and network.

    Args:
        clusters_df: DataFrame containing cluster data
        include_outliers: If False, filter out all "OUTLIER" cluster labels
        output_file: Path to save the visualization
    """
    # Create label mapping
    label_map_df = create_label_map_df(clusters_df)

    # Filter based on outliers
    if not include_outliers:
        filtered_df = clusters_df.filter(pl.col("cluster_id") != -1)
    else:
        filtered_df = clusters_df

    # Calculate run lengths using calculate_cluster_run_lengths
    run_lengths_df = calculate_cluster_run_lengths(filtered_df)

    # Group by and calculate average run length
    avg_run_lengths_df = (
        run_lengths_df.group_by([
            "cluster_label",
            "initial_prompt",
            "embedding_model",
            "network",
        ])
        .agg(pl.mean("run_length").alias("avg_run_length"))
        .sort(["embedding_model", "network", "cluster_label", "initial_prompt"])
    )

    # Join with the label_map_df to add the cluster_index column
    avg_run_lengths_df = avg_run_lengths_df.join(
        label_map_df, on=["embedding_model", "cluster_label"]
    )

    # Convert to pandas for plotting
    pandas_df = avg_run_lengths_df.to_pandas()

    # Add display_label column that only shows labels for high average run lengths
    threshold = pandas_df["avg_run_length"].quantile(0.75)  # Show top 25% by default
    pandas_df["display_label"] = pandas_df.apply(
        lambda row: row["cluster_index"] if row["avg_run_length"] > threshold else "",
        axis=1,
    )

    # Create the bubble grid visualization
    plot = (
        ggplot(
            pandas_df,
            aes(
                x="cluster_index",
                y="initial_prompt",
                size="avg_run_length",
                fill="cluster_label",
            ),
        )
        + geom_point(alpha=0.8, show_legend=False)
        + geom_text(aes(label="display_label"), color="black", size=10)
        + scale_size_continuous(range=(1, 12))
        + labs(
            x="Cluster",
            y="Prompt",
            size="Avg Run Length",
            fill="Cluster",
        )
        + facet_wrap("~ network", ncol=4)
        + theme(
            figure_size=(30, 12),
            strip_text=element_text(size=16),
            axis_text_x=element_text(size=12),
            axis_text_y=element_text(size=14),
        )
    )

    # Save the plot
    saved_file = save(plot, output_file)
    logging.info(f"Saved cluster run length bubble grid to {saved_file}")


def plot_sense_check_histograms(
    df: pl.DataFrame,
    output_file: str = "output/vis/sense_check_histograms.pdf",
) -> None:
    """
    Create two separate histograms showing the distribution of:
    1. Initial prompts
    2. Networks
    Each facet-wrapped by embedding_model.

    Args:
        df: DataFrame containing embedding data with initial_prompt, network, and embedding_model
        output_file: Path to save the visualization
    """
    # Convert to pandas for plotting
    pandas_df = df.to_pandas()

    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    # Create base filename for the two output files
    base_file = os.path.splitext(output_file)[0]

    # Create and save the first chart for Initial Prompts
    prompt_plot = (
        ggplot(pandas_df, aes(x="initial_prompt", fill="embedding_model"))
        + geom_bar()
        + labs(x="Initial Prompt", y="Count")
        + facet_wrap("~ embedding_model", scales="free_x")
        + coord_flip()  # Flip coordinates to make prompts more readable
        + theme(
            figure_size=(15, 8),
            strip_text=element_text(size=10),
        )
    )

    # Save the prompt plot
    prompt_output = f"{base_file}_prompts.pdf"
    saved_file = save(prompt_plot, prompt_output)
    logging.info(f"Saved prompt histogram to {saved_file}")

    # Create and save the second chart for Networks
    network_plot = (
        ggplot(pandas_df, aes(x="network", fill="embedding_model"))
        + geom_bar()
        + labs(x="Network", y="Count")
        + facet_wrap("~ embedding_model", scales="free_x")
        + theme(
            figure_size=(15, 8),
            strip_text=element_text(size=10),
            axis_text_x=element_text(angle=45, hjust=1),
        )
    )

    # Save the network plot
    network_output = f"{base_file}_networks.pdf"
    saved_file = save(network_plot, network_output)
    logging.info(f"Saved network histogram to {saved_file}")


def plot_cluster_histograms(
    clusters_df: pl.DataFrame,
    include_outliers: bool = False,
    output_file: str = "output/vis/cluster_histograms.pdf",
) -> None:
    """
    Create a faceted histogram showing counts of each cluster label,
    faceted by embedding model and network.

    Args:
        clusters_df: DataFrame containing cluster data
        include_outliers: If False, filter out all "OUTLIER" cluster labels
        output_file: Path to save the visualization
    """

    # Filter based on outliers
    if not include_outliers:
        df = clusters_df.filter(pl.col("cluster_id") != -1)
    else:
        df = clusters_df

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
    df = filter_top_n_clusters(df, top_n, ["network"])

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
    clusters_df: pl.DataFrame,
    include_outliers: bool,
    output_file: str = "output/vis/cluster_transitions.pdf",
) -> None:
    """
    Create a visualization of cluster transitions within runs.

    For each run_id, calculates transitions between consecutive cluster labels
    and displays them as a grid of points where point size represents transition frequency.

    Args:
        clusters_df: DataFrame containing cluster data with sequence information
        include_outliers: If False, filter out all "OUTLIER" cluster labels
        output_file: Path to save the visualization
    """
    # Create label mapping
    label_map_df = create_label_map_df(clusters_df)
    # Use calculate_cluster_transitions to get transition counts
    transition_counts = calculate_cluster_transitions(
        clusters_df, ["embedding_model", "network"], include_outliers
    )

    # Join with label_map_df to get cluster_index for from_cluster
    transition_counts = transition_counts.join(
        label_map_df.rename({"cluster_label": "from_cluster"}),
        on=["embedding_model", "from_cluster"],
    )

    # Join with label_map_df to get cluster_index for to_cluster
    transition_counts = transition_counts.join(
        label_map_df.rename({
            "cluster_label": "to_cluster",
            "cluster_index": "to_cluster_index",
        }),
        on=["embedding_model", "to_cluster"],
    )

    # Convert to pandas for plotting
    pandas_df = transition_counts.to_pandas()

    # Create point grid visualization
    plot = (
        ggplot(
            pandas_df,
            aes(
                x="to_cluster_index",
                y="cluster_index",
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
    examples_per_row: int = None,
    output_file: str = "output/vis/cluster_examples.jpg",
    rescale: Optional[float] = None,
) -> None:
    """
    Create a mosaic image showing example images for each cluster.

    Args:
        df: DataFrame containing embedding data with cluster_label and invocation_id
        num_examples: Number of example images to include for each cluster
        embedding_model: The embedding model to filter by
        session: Session object for database access (needed for export_mosaic_image)
        examples_per_row: Maximum number of images per row before wrapping (default: num_examples)
        output_file: Path to save the visualization
        rescale: Optional scaling factor for the output image dimensions (default: None)
    """
    # Set default examples_per_row if not provided
    if examples_per_row is None:
        examples_per_row = num_examples

    # Filter dataframe to get only rows with the specified embedding model
    # and where cluster_label is not null and not0
    filtered_df = df.filter(
        (pl.col("embedding_model") == embedding_model)
        & (pl.col("cluster_label").is_not_null())
        & (pl.col("cluster_label") != "OUTLIER")
    )

    # Group by cluster_label and collect the first num_examples invocation_ids for each
    cluster_examples = []

    # If the dataframe has a 'count' column, use it to order clusters by their count
    if "count" in filtered_df.columns:
        # Get unique clusters ordered by count (descending)
        cluster_order = (
            filtered_df.select(["cluster_label", "count"])
            .unique()
            .sort("count", descending=True)
            .get_column("cluster_label")
            .to_list()
        )
    else:
        # Otherwise, get unique cluster labels in order of their first appearance
        pandas_df = filtered_df.to_pandas()
        cluster_order = pandas_df["cluster_label"].drop_duplicates().tolist()

    # Convert to pandas for easier groupby operations
    pandas_df = filtered_df.to_pandas()

    # Group by cluster_label and process in the desired order
    grouped = pandas_df.groupby("cluster_label")
    for cluster_label in cluster_order:
        if cluster_label in grouped.groups:
            group = grouped.get_group(cluster_label)
            # Get the first num_examples invocation_ids
            invocation_ids = group["invocation_id"].head(num_examples).tolist()
            invocation_uuids = [UUID(id) for id in invocation_ids]

            # If we need to wrap rows, split the invocations into multiple rows
            if (
                examples_per_row < num_examples
                and len(invocation_uuids) > examples_per_row
            ):
                # Split into multiple rows
                rows = []
                for i in range(0, len(invocation_uuids), examples_per_row):
                    rows.append(invocation_uuids[i : i + examples_per_row])
                cluster_examples.append((str(cluster_label), rows))
            else:
                # Even for single row, wrap in a list to maintain consistent structure
                cluster_examples.append((str(cluster_label), [invocation_uuids]))

    # Use export_mosaic_image to create the visualization
    if cluster_examples:
        export_mosaic_image(
            cluster_examples, session, output_file=output_file, rescale=rescale
        )
        logging.info(f"Saved cluster example images to {output_file}")
    else:
        logging.warning(
            f"No cluster examples found for embedding model {embedding_model}"
        )


def plot_invocation_duration(
    invocation_df: pl.DataFrame, output_file: str = "output/vis/invocation_duration.pdf"
) -> None:
    """
    Create and save a visualization of invocation duration distribution,
    with model on x-axis and duration on y-axis.

    Args:
        df: DataFrame containing embedding data with duration
        output_file: Path to save the visualization
    """
    # Convert polars DataFrame to pandas for plotnine
    pandas_df = invocation_df.to_pandas()

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
