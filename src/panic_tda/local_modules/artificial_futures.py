"""Charts and visualizations for Ben's artificial futures slides."""

import polars as pl
from sqlmodel import Session

from panic_tda.data_prep import (
    calculate_cluster_bigrams,
    load_clusters_from_cache,
    load_embeddings_from_cache,
    load_runs_from_cache,
)
from panic_tda.datavis import plot_cluster_bubblegrid, plot_cluster_example_images
from panic_tda.export import export_timeline
from panic_tda.utils import print_polars_as_markdown


def artificial_futures_slides_charts(session: Session) -> None:
    """
    Generate charts for Ben's artificial futures slides.

    TODO list:

        - bigram charts (most popular transitions, by network)
        - most popular cluster (in each decile, grouped by prompt - or a subset of the prompts)
        - persistence entropy violin plots (maybe need to re-do? or just use a figure from the paper)
    """
    # Load all dataframes from cache at the top
    runs_df = load_runs_from_cache()
    embeddings_df = load_embeddings_from_cache()
    clusters_df = load_clusters_from_cache()

    # Filter embeddings to Nomic model
    embeddings_df = embeddings_df.filter(pl.col("embedding_model") == "Nomic")

    # Filter clusters to only the specified clustering result
    clusters_df = clusters_df.filter(pl.col("embedding_model") == "Nomic")
    # clusters_df = clusters_df.filter(pl.col("clustering_result_id") == "result-id")

    # Join embeddings with clusters to get cluster labels
    # First deduplicate clusters_df to avoid multiple rows per embedding
    unique_clusters = clusters_df.select([
        "embedding_id",
        "cluster_id",
        "cluster_label",
    ]).unique()
    embeddings_df = embeddings_df.join(
        unique_clusters, left_on="id", right_on="embedding_id", how="inner"
    )

    # Filter out outliers
    embeddings_df = embeddings_df.filter(
        (pl.col("cluster_label").is_not_null()) & (pl.col("cluster_label") != "OUTLIER")
    )

    # Get top 10 most popular clusters with their counts
    cluster_counts = (
        embeddings_df.group_by("cluster_label")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(10)
    )

    top_cluster_labels = cluster_counts.get_column("cluster_label")

    # Filter to only top 10 clusters
    embeddings_df = embeddings_df.filter(
        pl.col("cluster_label").is_in(top_cluster_labels)
    )

    # Join with cluster counts to add count column (but don't sort the entire df)
    embeddings_df = embeddings_df.join(cluster_counts, on="cluster_label", how="left")

    # cluster examples
    plot_cluster_example_images(
        embeddings_df,
        24,
        "Nomic",
        session,
        output_file="output/vis/cluster-examples-1.jpg",
        rescale=0.5,
    )
    plot_cluster_example_images(
        embeddings_df,
        180,
        "Nomic",
        session,
        examples_per_row=60,
        output_file="output/vis/cluster-examples-2.jpg",
        rescale=0.25,
    )

    # print "top 10 clusters" table as md (for marp slides)
    total_non_outlier = embeddings_df.height
    top_clusters_table = (
        cluster_counts.with_row_index("rank", offset=1)  # Add rank column starting at 1
        .with_columns(
            (pl.col("count") / total_non_outlier * 100).round(1).alias("percentage")
        )
        .select(["rank", "cluster_label", "percentage"])
    )
    markdown_table = top_clusters_table.to_pandas().to_markdown(index=False)
    print(markdown_table)

    # # Create ridgeline plot for semantic drift by network
    # from panic_tda.datavis import plot_semantic_drift

    # # Join with clusters to filter to our specific clustering run
    # ridgeline_df = embeddings_df.join(
    #     clusters_df.select(["embedding_id"]).unique(),
    #     left_on="id",
    #     right_on="embedding_id",
    #     how="inner",
    # )

    # plot_semantic_drift(ridgeline_df, output_file="output/vis/semantic-drift.pdf")

    # sample 20 runs at random, and then use export_timeline (with 10 images per run) to show some of the invocations from that run
    # Use the runs_df loaded at the top
    # Filter to only rows where initial prompt is "a red circle on a black background"
    filtered_runs_df = runs_df.filter(
        pl.col("initial_prompt") == "a red circle on a black background"
    )

    # Sample 20 random run IDs
    red_circle_run_ids = filtered_runs_df.get_column("run_id").unique().to_list()

    # Convert to strings (assuming export_timeline expects string IDs)
    red_circle_run_ids_str = [str(run_id) for run_id in red_circle_run_ids]

    # Export timeline with 10 images per run
    export_timeline(
        run_ids=red_circle_run_ids_str,
        session=session,
        images_per_run=7,
        output_file="output/vis/red-circle-runs.jpg",
        rescale=0.5,
    )

    # Calculate cluster bigrams and print top 10 most common bigrams by network
    # Calculate bigrams for all clusters
    bigrams_df = calculate_cluster_bigrams(clusters_df, include_outliers=False)

    # Join with runs to get network information, converting network list to string
    bigrams_with_network = bigrams_df.join(
        runs_df.select([
            "run_id",
            pl.col("network").list.join(" -> ").alias("network_str"),
        ]),
        on="run_id",
        how="inner",
    )

    # Count bigrams by network
    bigram_counts = bigrams_with_network.group_by([
        "network_str",
        "from_cluster",
        "to_cluster",
    ]).agg(pl.len().alias("count"))

    # Calculate total bigrams per network (including self-transitions) for percentages
    total_bigrams_by_network = bigram_counts.group_by("network_str").agg(
        pl.col("count").sum().alias("total_count")
    )

    # Join with totals and calculate percentages
    bigram_counts_with_pct = bigram_counts.join(
        total_bigrams_by_network, on="network_str", how="left"
    ).with_columns(
        (100.0 * pl.col("count") / pl.col("total_count")).round(1).alias("percentage")
    )

    # Filter out self-transitions and add rank
    top_bigrams_by_network = (
        bigram_counts_with_pct.filter(
            pl.col("from_cluster") != pl.col("to_cluster")
        )  # Filter out self-transitions
        .with_columns(
            pl.col("count")
            .rank(method="ordinal", descending=True)
            .over("network_str")
            .alias("rank")
        )
        .filter(pl.col("rank") <= 10)
        .sort(["network_str", "rank"])
        .select([
            pl.col("network_str").alias("network"),
            "from_cluster",
            "to_cluster",
            (pl.col("percentage").cast(pl.Utf8) + "%").alias("percentage"),
        ])
    )

    # Print results using markdown formatting
    print_polars_as_markdown(
        top_bigrams_by_network,
        title="Top 10 most common cluster bigrams by network",
        max_col_width=60,
        headers=["Network", "From Cluster", "To Cluster", "Percentage"],
    )

    plot_cluster_bubblegrid(
        clusters_df,
        False,
        "output/vis/cluster-bubblegrid.pdf",
    )
