import logging
import os

import polars as pl
from plotnine import (
    aes,
    coord_flip,
    element_text,
    facet_grid,
    facet_wrap,
    geom_boxplot,
    geom_line,
    geom_point,
    ggplot,
    labs,
    scale_color_manual,
    scale_x_continuous,
    scale_x_discrete,
    scale_y_continuous,
    theme,
)
from plotnine.options import set_option
from sqlmodel import Session

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
    df: pl.DataFrame, output_file: str = "output/vis/persistence_entropy.png"
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
    # Convert polars DataFrame to pandas for plotnine
    pandas_df = df.to_pandas()

    # Create the plot with faceting
    # Define subscript translator for x-axis labels
    subscripts = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

    plot = (
        ggplot(
            pandas_df,
            aes(x="factor(homology_dimension)", y="entropy", fill="embedding_model"),
        )
        # + geom_violin(alpha=0.7, width=0.7)
        + geom_boxplot()
        + labs(x="homology dimension", y="entropy", fill="embedding model")
        + scale_x_discrete(
            labels=lambda x: [f"h{str(val).translate(subscripts)}" for val in x]
        )
        + facet_grid("text_model ~ image_model", labeller="label_both")
        + theme(figure_size=(10, 6), strip_text=element_text(size=10))
    )

    # Save plot with high resolution
    saved_file = save(plot, output_file)
    logging.info(f"Saved persistence entropy plot to {saved_file}")


def plot_persistence_entropy_by_prompt(
    df: pl.DataFrame, output_file: str = "output/vis/persistence_entropy_by_prompt.png"
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
    subscripts = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

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
        + theme(figure_size=(30, 20), strip_text=element_text(size=8)) # Adjust size as needed
    )

    # Save plot with high resolution
    saved_file = save(plot, output_file)
    logging.info(f"Saved persistence entropy by prompt plot to {saved_file}")


def plot_semantic_drift(
    df: pl.DataFrame, output_file: str = "output/vis/semantic_drift.png"
) -> None:
    """
    Create a line plot showing semantic drift over sequence number,
    faceted by initial prompt and model.

    Args:
        df: DataFrame containing embedding data with semantic_drift and sequence_number
        output_file: Path to save the visualization
    """
    pandas_df = df.to_pandas()

    # Create a single chart with faceting
    plot = (
        ggplot(
            pandas_df,
            aes(
                x="sequence_number",
                y="semantic_drift_instantaneous",
                color="semantic_drift_instantaneous > 0.5",
            ),
        )
        + geom_line(
            aes(group="run_id"), color="black"
        )  # Add black line connecting points by run_id
        + geom_point(alpha=0.8)
        + scale_color_manual(values=["black", "red"])
        + labs(x="sequence number", y="semantic drift")
        + facet_grid("initial_prompt ~ embedding_model")
        + theme(
            figure_size=(30, 80),
            strip_text=element_text(size=10),
            legend_position="none",
        )
    )

    # Save the chart
    saved_file = save(plot, output_file)
    logging.info(f"Saved semantic drift plot to {saved_file}")


def plot_invocation_duration(
    df: pl.DataFrame, output_file: str = "output/vis/invocation_duration.png"
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
    from panic_tda.analysis import warm_caches

    warm_caches(session, runs=True, embeddings=True, invocations=True)

    # INVOCATIONS
    #
    # from panic_tda.analysis import load_invocations_df

    # invocations_df = load_invocations_df(session, use_cache=True)
    # plot_invocation_duration(invocations_df, "output/vis/invocation_duration.png")

    # EMBEDDINGS
    #
    # from panic_tda.analysis import load_embeddings_df
    #
    # embeddings_df = load_embeddings_df(session, use_cache=True)
    # plot_semantic_drift(embeddings_df, "output/vis/semantic_drift.png")

    # RUNS
    #
    # from panic_tda.analysis import load_runs_df
    #
    # runs_df = load_runs_df(session, use_cache=True)
    # plot_persistence_diagram_faceted(
    #     runs_df, "output/vis/persistence_diagram_faceted.png"
    # )
    # plot_persistence_entropy(runs_df, "output/vis/persistence_entropy.png")
    # plot_persistence_diagram_by_run(
    #     runs_df, "output/vis/persistence_diagram_by_run.png"
    # )
