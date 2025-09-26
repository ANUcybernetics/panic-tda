"""WIP: charts and visualizations for (maybe) Cybernetics '26."""

import polars as pl
from sqlmodel import Session, select
import numpy as np
from typing import Dict, List, Tuple

from panic_tda.data_prep import (
    cache_dfs,
    load_pd_from_cache,
    load_pd_df,
    create_paired_pd_df,
    filter_paired_pd_df,
    add_pairing_metadata,
    calculate_paired_wasserstein_distances,
    load_runs_df_with_pd_metadata,
    create_optimised_wasserstein_pairs,
    calculate_optimised_wasserstein_distances,
)
from panic_tda.datavis import plot_wasserstein_violin
from panic_tda.local_modules.shared import cluster_counts, example_run_ids, run_counts
from panic_tda.schemas import Run, PersistenceDiagram


def cybernetics_26_charts(session: Session) -> None:
    """
    Generate charts for paper publications.

    TODO this has actually been modified a bit (for exploring new options), so
    if we wanted to re-create the paper charts exactly you'd probably have to
    un-comment some lines and re-comment some others.
    """
    # pl.Config.set_tbl_rows(10)
    ### CACHING
    #
    # Commenting out caching for now - just do the analysis directly
    # cache_dfs(
    #     session,
    #     runs=False,
    #     embeddings=False,
    #     invocations=False,
    #     persistence_diagrams=True,
    # )
    # cache_dfs(session, runs=True, embeddings=True, invocations=True)
    #
    # TODO all the code from here is copy-pased from the ieee_smc.py module and needs to be changed
    #
    # # DATA SELECTION
    # selected_ids = TOP4_RUN_IDS
    # #
    # # EXAMPLE IMAGES
    # #
    # export_timeline(
    #     run_ids=example_run_ids(session),
    #     session=session,
    #     images_per_run=5,
    #     output_file="output/vis/paper/fig1.jpg",
    # )
    # ### INVOCATIONS
    # #
    # # from panic_tda.data_prep import load_invocations_from_cache
    # # from panic_tda.datavis import plot_invocation_duration

    # # invocations_df = load_invocations_from_cache()
    # # invocations_df = invocations_df.filter(pl.col("run_id").is_in(selected_ids))

    # # plot_invocation_duration(invocations_df)

    # # print(
    # #     f"Total unique initial_prompts in invocations_df: {invocations_df.select(pl.col('initial_prompt').n_unique()).item()}"
    # # )

    # ### EMBEDDINGS
    # #
    # from panic_tda.data_prep import (
    #     calculate_cluster_run_lengths,
    #     load_embeddings_from_cache,
    # )
    # from panic_tda.datavis import (
    #     create_label_map_df,
    #     plot_cluster_bubblegrid,
    #     plot_cluster_run_length_violin,
    # )

    # embeddings_df = load_embeddings_from_cache()

    # embeddings_df = embeddings_df.filter(
    #     pl.col("run_id").is_in(selected_ids)
    # ).with_columns(
    #     pl.col("network").str.replace_all(" → ", "→", literal=True).alias("network")
    # )

    # # With the new approach, many visualization functions only need the session!
    # # They load the clusters data internally.

    # # plot_sense_check_histograms(embeddings_df)  # This one still needs embeddings
    # plot_cluster_bubblegrid(
    #     session,  # Just pass the session - no DataFrame needed!
    #     False,
    #     "output/vis/paper/fig2.pdf",
    # )

    # plot_cluster_run_length_violin(
    #     session,  # Just pass the session - no DataFrame needed!
    #     True,
    #     "output/vis/paper/fig3.pdf",
    # )

    # run_length_df = calculate_cluster_run_lengths(embeddings_df, True)
    # run_length_df = run_length_df.join(
    #     label_df, on=["embedding_model", "cluster_label"], how="left"
    # )
    # print(
    #     run_length_df.filter(
    #         (pl.col("run_length") == 50) & (pl.col("cluster_label") != "OUTLIER")
    #     ).sort("initial_prompt")
    # )
    # print(
    #     run_length_df.group_by("embedding_model").agg([
    #         pl.col("run_length").quantile(0.25).alias("run_length_q25"),
    #         pl.col("run_length").quantile(0.50).alias("run_length_median"),
    #         pl.col("run_length").quantile(0.75).alias("run_length_q75"),
    #     ])
    # )
    # print(
    #     run_length_df.group_by("network").agg([
    #         pl.col("run_length").quantile(0.25).alias("run_length_q25"),
    #         pl.col("run_length").quantile(0.50).alias("run_length_median"),
    #         pl.col("run_length").quantile(0.75).alias("run_length_q75"),
    #     ])
    # )
    # print(
    #     run_length_df.group_by("initial_prompt").agg([
    #         pl.col("run_length").quantile(0.25).alias("run_length_q25"),
    #         pl.col("run_length").quantile(0.50).alias("run_length_median"),
    #         pl.col("run_length").quantile(0.75).alias("run_length_q75"),
    #     ])
    # )

    # print(cluster_counts(embeddings_df, 1))
    # print(
    #     cluster_counts(embeddings_df.filter(pl.col("embedding_model") == "Nomic"), 6)
    #     .join(label_df, on=["embedding_model", "cluster_label"], how="left")
    #     .select("cluster_label", "cluster_index", "percentage")
    #     .with_columns(pl.col("percentage").round(1).alias("percentage"))
    #     .to_pandas()
    #     .to_latex(index=False)
    # )

    # print(
    #     embeddings_df.group_by("embedding_model")
    #     .agg(pl.col("id").n_unique().alias("embedding_count"))
    #     .sort("embedding_count", descending=True)
    # )

    # print(cluster_counts(embeddings_df, 3).to_pandas().to_latex())

    ### RUNS

    # from panic_tda.data_prep import load_runs_from_cache

    # runs_df = load_runs_from_cache()
    # print(runs_df.columns)
    # print(runs_df.head())
    # runs_df = runs_df.filter(pl.col("run_id").is_in(selected_ids))

    # print(run_counts(runs_df, ["network"]))

    # plot_persistence_entropy(runs_df, "output/vis/paper/fig4.pdf")

    ### Persistence Diagram DF
    # Commenting out for now - we'll work directly with the database
    # pd_df = load_pd_from_cache()
    # print(pd_df.columns)
    # print(pd_df.head())

    # Wasserstein distance analysis using the optimized four-group pipeline
    print("\n=== Optimized Wasserstein Distance Analysis: Four-Group Comparison ===")
    print(
        "Group 1: Same IC, different EM - Same initial conditions, different embedding models (Nomic vs NomicVision from different runs)"
    )
    print(
        "Group 2: Same IC, Nomic - Same initial conditions, both using Nomic embedding"
    )
    print(
        "Group 3: Same IC, NomicVision - Same initial conditions, both using NomicVision embedding"
    )
    print(
        "Group 4: Same run, different EM - Same run, different embedding models (Nomic vs NomicVision from same run)"
    )

    # Step 1: Load runs with persistence diagram metadata (much more efficient)
    print("\nStep 1: Loading runs with persistence diagram metadata...")
    embedding_models = ["Nomic", "NomicVision"]
    runs_df = load_runs_df_with_pd_metadata(session, embedding_models)

    # Filter to only runs that have PDs for both embedding models
    valid_runs_df = runs_df.filter(pl.col("has_all_requested_pds") == True)
    print(f"Found {valid_runs_df.height} runs with PDs for both Nomic and NomicVision")

    if valid_runs_df.height == 0:
        print("No runs available with PDs for both Nomic and NomicVision")
        return

    # Get unique initial conditions sets
    unique_ics = valid_runs_df.select(["initial_prompt", "network"]).unique()
    print(f"Found {unique_ics.height} unique initial condition sets")

    # Show summary statistics
    print("\nSummary:")
    print(f"  Total runs with both models: {valid_runs_df.height}")
    print(f"  Unique initial condition sets: {unique_ics.height}")

    # Expected pair counts (for validation)
    # Group 1: n_runs × n_runs × 2 (Nomic→NomicVision + NomicVision→Nomic pairs per IC set)
    # Groups 2&3: n_runs × (n_runs-1)/2 × 2 (unique pairs for each model per IC set)
    # Group 4: n_runs per IC set (one pair per run comparing its Nomic vs NomicVision)
    runs_per_ic = (
        valid_runs_df.height // unique_ics.height if unique_ics.height > 0 else 0
    )
    if runs_per_ic > 0:
        expected_group1 = (
            unique_ics.height * runs_per_ic * runs_per_ic * 2
        )  # Both directions
        expected_group2_3 = (
            unique_ics.height * (runs_per_ic * (runs_per_ic - 1) // 2) * 2
        )  # Both models
        expected_group4 = unique_ics.height * runs_per_ic  # One pair per run
        print(f"  Runs per IC set: ~{runs_per_ic}")
        print(f"  Expected Group 1 pairs: ~{expected_group1}")
        print(f"  Expected Groups 2&3 pairs: ~{expected_group2_3}")
        print(f"  Expected Group 4 pairs: ~{expected_group4}")

    # Step 2: Create optimized pairs (no full PD data loading)
    print("\nStep 2: Creating optimized run pairs...")
    pairs_df = create_optimised_wasserstein_pairs(valid_runs_df, embedding_models)
    print(f"Created {pairs_df.height} run pairs for four-group analysis")

    if pairs_df.height == 0:
        print("No valid pairs created")
        return

    # Show breakdown of the four comparison groups
    group_counts = (
        pairs_df.group_by("grouping").agg(pl.count().alias("count")).sort("grouping")
    )

    print("  Group breakdown:")
    for row in group_counts.iter_rows(named=True):
        print(f"    - {row['grouping']}: {row['count']} pairs")

    # Step 3: Calculate Wasserstein distances with on-demand PD fetching
    print("\nStep 3: Calculating Wasserstein distances with on-demand PD fetching...")
    print(
        "This fetches only the required persistence diagrams (H1 data) instead of loading all PD data"
    )

    distance_df = calculate_optimised_wasserstein_distances(
        pairs_df, session, homology_dimension=1
    )
    print(f"Successfully calculated {distance_df.height} distances")

    if distance_df.height > 0:
        # Print summary statistics for the four groups
        print("\n=== Summary Statistics ===")

        stats = (
            distance_df.group_by("grouping")
            .agg([
                pl.count("distance").alias("count"),
                pl.mean("distance").alias("mean_distance"),
                pl.std("distance").alias("std_distance"),
                pl.min("distance").alias("min_distance"),
                pl.max("distance").alias("max_distance"),
            ])
            .sort("grouping")
        )

        print("\nStatistics by comparison group:")
        for row in stats.iter_rows(named=True):
            print(f"\n{row['grouping']} ({row['count']} pairs):")
            print(f"  Mean: {row['mean_distance']:.4f}")
            print(f"  Std: {row['std_distance']:.4f}")
            print(f"  Min: {row['min_distance']:.4f}")
            print(f"  Max: {row['max_distance']:.4f}")

        # Step 5: Create visualization
        print("\n=== Creating Visualization ===")
        output_file = "output/vis/wasserstein_violin.pdf"
        plot_wasserstein_violin(distance_df, output_file)
        print(f"Saved violin plot to {output_file}")

    # #### Persistence Diagram Density Comparison
    # from panic_tda.datavis import plot_persistence_density

    # plot_persistence_density(pd_df, "output/vis/persistence-density.pdf")
