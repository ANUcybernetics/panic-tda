import json
import logging
import os
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
    geom_violin,
    ggplot,
    labs,
    scale_x_continuous,
    scale_y_continuous,
    theme,
)
from plotnine.options import set_option
from sqlmodel import Session

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
    df: pl.DataFrame, output_file: str = "output/vis/cluster_timelines.pdf"
) -> None:
    """
    Create a faceted scatter plot showing cluster labels over sequence number for each run,
    faceted by initial prompt.

    Args:
        df: DataFrame containing embedding data with cluster_label and sequence_number
        output_file: Path to save the visualization
    """
    # Convert polars DataFrame to pandas for plotnine
    pandas_df = df.filter(pl.col("cluster_label").is_not_null()).to_pandas()

    # Create the plot
    plot = (
        ggplot(
            pandas_df,
            aes(
                x="sequence_number",
                y="run_id",
                color="factor(cluster_label)",
            ),
        )
        + geom_line(colour="black", alpha=0.7)
        + geom_point(size=3)
        + labs(x="sequence number", y="run id", color="cluster")
        + facet_wrap(
            "~ initial_prompt + embedding_model",
            labeller="label_context",
            ncol=3,
            scales="free",
        )
        + theme(
            figure_size=(20, 100),
            strip_text=element_text(size=10),
            axis_text_y=element_blank(),
        )
    )

    # Save the chart
    saved_file = save(plot, output_file)
    logging.info(f"Saved cluster timelines plot to {saved_file}")


def plot_cluster_histograms(
    df: pl.DataFrame, output_file: str = "output/vis/cluster_histograms.pdf"
) -> None:
    """
    Create a faceted histogram showing counts of each cluster label,
    faceted by initial prompt and embedding model.

    Args:
        df: DataFrame containing embedding data with cluster_label
        output_file: Path to save the visualization
    """
    # Convert polars DataFrame to pandas for plotnine
    pandas_df = df.filter(
        pl.col("cluster_label").is_not_null(), pl.col("cluster_label") != "OUTLIER"
    ).to_pandas()

    # Create the plot
    plot = (
        ggplot(
            pandas_df,
            aes(x="cluster_label", fill="embedding_model"),
        )
        + geom_bar()
        + labs(x="cluster_label", y="count")
        + facet_wrap(
            "~ embedding_model",
            # labeller="label_context",
            # scales="free_y",
            ncol=3,
        )
        + theme(
            figure_size=(20, 40),
            strip_text=element_text(size=10),
        )
        + coord_flip()
    )

    # Save the chart
    saved_file = save(plot, output_file)
    logging.info(f"Saved cluster histograms plot to {saved_file}")


def export_cluster_counts_to_json(
    df: pl.DataFrame, output_file: str = "output/vis/cluster_counts.json"
) -> None:
    """
    Count occurrences of each cluster label grouped by initial_prompt, embedding_model,
    text_model and cluster_label, then export to a JSON file sorted in decreasing order.

    Args:
        df: DataFrame containing embedding data with cluster_label
        output_file: Path to save the JSON file
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
        .group_by(["embedding_model", "cluster_label"])
        .head(10)
    )

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    # Write to JSON file

    with open(output_file, "w") as f:
        json.dump(counts_df.to_dicts(), f, indent=2)

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
    # and where cluster_label is not null and not -1
    filtered_df = df.filter(
        (pl.col("embedding_model") == embedding_model)
        & (pl.col("cluster_label").is_not_null())
        & (pl.col("cluster_label") != -1)
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


def paper_charts(session: Session) -> None:
    """
    Generate charts for paper publications.
    """
    # from panic_tda.local import prompt_timeline_run_ids
    #
    # selected_ids = prompt_timeline_run_ids(session)
    # from panic_tda.export import export_timeline

    # Export the timeline image
    # export_timeline(
    #     run_ids=selected_ids,
    #     session=session,
    #     images_per_run=8,
    #     output_file="output/vis/selected_prompts_timeline.jpg",
    # )
    #
    ### CACHING
    #
    from panic_tda.analysis import cache_dfs

    cache_dfs(session, runs=False, embeddings=True, invocations=False)
    ### INVOCATIONS
    #
    from panic_tda.analysis import load_invocations_from_cache

    invocations_df = load_invocations_from_cache()
    print(invocations_df.select("run_id", "sequence_number", "type").head(50))
    # plot_invocation_duration(invocations_df, "output/vis/invocation_duration.png")
    #
    ### EMBEDDINGS
    #
    # from panic_tda.analysis import load_embeddings_from_cache

    # embeddings_df = load_embeddings_from_cache()

    # pl.Config.set_tbl_rows(1000)
    # selected_embeddings_df = embeddings_df.filter(pl.col("run_id").is_in(selected_ids))
    # print(selected_embeddings_df.select("initial_prompt", "sequence_number", "text", "drift_cosine", "drift_euclid").head(50))
    # plot_semantic_drift(selected_embeddings_df, "output/vis/semantic_drift.pdf")
    #
    ### RUNS
    #
    # from panic_tda.analysis import load_runs_from_cache

    # runs_df = load_runs_from_cache()
    #
    # plot_persistence_diagram_faceted(
    #     runs_df, "output/vis/persistence_diagram_faceted.png"
    # )
    #
    # plot_persistence_entropy(runs_df, "output/vis/persistence_entropy.pdf")
    #
    # plot_persistence_entropy_by_prompt(
    #     nruns_df.filter(pl.col("embedding_model") == "Nomic"), "output/vis/persistence_entropy_by_prompt.png"
    # )
