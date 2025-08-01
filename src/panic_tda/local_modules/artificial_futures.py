"""Charts and visualizations for Ben's artificial futures slides."""

import polars as pl
from sqlmodel import Session

from panic_tda.data_prep import (
    calculate_cluster_bigrams,
    load_clusters_from_cache,
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
    clusters_df = load_clusters_from_cache()

    # Filter clusters to only the specified clustering result
    clusters_df = clusters_df.filter(
        pl.col("clustering_result_id") == "0688ac5e-ee51-7a9b-9b0a-017376f3fbb5"
    )

    # Filter out outliers
    clusters_df = clusters_df.filter(
        (pl.col("cluster_label").is_not_null()) & (pl.col("cluster_label") != "OUTLIER")
    )

    # Get top 10 most popular clusters with their counts
    cluster_counts = (
        clusters_df.group_by("cluster_label")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(10)
    )

    top_cluster_labels = cluster_counts.get_column("cluster_label")

    # Filter to only top 10 clusters
    top_clusters_df = clusters_df.filter(
        pl.col("cluster_label").is_in(top_cluster_labels)
    )

    # Join with cluster counts to add count column
    top_clusters_df = top_clusters_df.join(cluster_counts, on="cluster_label", how="left")

    # cluster examples
    plot_cluster_example_images(
        top_clusters_df,
        24,
        "Nomic",
        session,
        output_file="output/vis/cluster-examples-1.jpg",
        rescale=0.5,
    )
    plot_cluster_example_images(
        top_clusters_df,
        180,
        "Nomic",
        session,
        examples_per_row=60,
        output_file="output/vis/cluster-examples-2.jpg",
        rescale=0.25,
    )

    # Group clusters by network and get top 5 per network
    # The network column is already a string, so we can use it directly
    
    # Count clusters by network and cluster_label
    cluster_counts_by_network = (
        clusters_df.group_by(["network", "cluster_label"])
        .agg(pl.len().alias("count"))
    )
    
    # Calculate total per network for percentages
    total_by_network = clusters_df.group_by("network").agg(
        pl.len().alias("total_count")
    )
    
    # Join with totals and calculate percentages
    cluster_counts_with_pct = cluster_counts_by_network.join(
        total_by_network, on="network", how="left"
    ).with_columns(
        (100.0 * pl.col("count") / pl.col("total_count")).round(1).alias("percentage")
    )
    
    # Add rank within each network and filter to top 5
    top_clusters_by_network = (
        cluster_counts_with_pct
        .with_columns(
            pl.col("count")
            .rank(method="ordinal", descending=True)
            .over("network")
            .alias("rank")
        )
        .filter(pl.col("rank") <= 5)
        .sort(["network", "rank"])
        .select([
            "network",
            "cluster_label",
            (pl.col("percentage").cast(pl.Utf8) + "%").alias("percentage"),
        ])
    )
    
    # Print results using markdown formatting
    print_polars_as_markdown(
        top_clusters_by_network,
        title="Top 5 most common clusters by network",
        max_col_width=60,
        headers=["Network", "Cluster Label", "Percentage"],
    )

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

    # Join with runs to get network information
    # The network column in runs_df is also already a string with "→" separator
    bigrams_with_network = bigrams_df.join(
        runs_df.select([
            "run_id",
            pl.col("network").alias("network_str"),
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
