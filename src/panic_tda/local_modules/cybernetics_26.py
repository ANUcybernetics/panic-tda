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

    # Wasserstein PD analysis using the new clean pipeline
    print("\n=== Wasserstein Distance Analysis Using New Pipeline ===")

    # Step 1: Load persistence diagram data
    print("\nStep 1: Loading persistence diagram data...")
    pd_df = load_pd_df(session)
    print(f"Loaded {pd_df.height} persistence diagram entries")

    # Print unique initial prompts for reference
    unique_prompts = (
        pd_df.select("initial_prompt")
        .unique()
        .sort("initial_prompt")["initial_prompt"]
        .to_list()
    )
    print(f"Found {len(unique_prompts)} unique initial prompts in PD data")

    # For large datasets, sample a subset of initial prompts and PDs
    # Select a subset of interesting prompts for analysis
    selected_prompts = ["a cat", "a dog", "a flower", "a car", "a house"]
    # Filter to only prompts that exist in the data
    selected_prompts = [p for p in selected_prompts if p in unique_prompts]

    if not selected_prompts:
        # If none of our preferred prompts exist, use the first 5
        selected_prompts = unique_prompts[:5]

    print(
        f"\nSelecting {len(selected_prompts)} prompts for analysis: {selected_prompts}"
    )

    # Step 2: Create paired persistence diagrams (now efficiently creates only wanted pairs)
    print("\nStep 2: Creating paired persistence diagrams...")

    # Filter PD data to selected prompts and only Nomic/NomicVision models for this analysis
    filtered_pd_df = pd_df.filter(
        pl.col("initial_prompt").is_in(selected_prompts)
        & pl.col("embedding_model").is_in(["Nomic", "NomicVision"])
    )
    print(f"Filtered to {filtered_pd_df.height} PD entries for analysis")

    # Sample PDs if needed to keep memory usage reasonable
    unique_pds = filtered_pd_df.select("persistence_diagram_id").unique()
    if unique_pds.height > 500:
        print(f"Sampling 500 PDs from {unique_pds.height} total")
        sampled_pds = unique_pds.sample(n=500, seed=42)
        filtered_pd_df = filtered_pd_df.filter(
            pl.col("persistence_diagram_id").is_in(
                sampled_pds["persistence_diagram_id"]
            )
        )

    paired_df = create_paired_pd_df(filtered_pd_df)
    print(f"Created {paired_df.height} meaningful pairs")

    # Step 3: Set up for homology dimension 1 analysis
    print("\nStep 3: Preparing for H1 analysis...")
    filtered_df = filter_paired_pd_df(paired_df, homology_dimension=1)
    print(f"Marked {filtered_df.height} pairs for H1 analysis")

    # Step 4: Add pairing metadata
    print("\nStep 4: Adding pairing metadata...")
    metadata_df = add_pairing_metadata(filtered_df)
    print(f"Added metadata to {metadata_df.height} pairs")

    # Show breakdown of pair types
    if metadata_df.height > 0:
        same_ic_different_models = metadata_df.filter(
            pl.col("same_IC")
            & (pl.col("embedding_model_a") != pl.col("embedding_model_b"))
        ).height
        different_ic_same_models = metadata_df.filter(
            (~pl.col("same_IC"))
            & (pl.col("embedding_model_a") == pl.col("embedding_model_b"))
        ).height

        print(
            f"  - Same IC, different models (Nomic vs NomicVision): {same_ic_different_models}"
        )
        print(
            f"  - Different IC, same models (Nomic vs Nomic or NomicVision vs NomicVision): {different_ic_same_models}"
        )
    else:
        print("  - No pairs found for analysis")

    # Step 5: Calculate Wasserstein distances
    print("\nStep 5: Calculating Wasserstein distances...")
    if metadata_df.height > 0:
        distance_df = calculate_paired_wasserstein_distances(
            metadata_df, session, homology_dimension=1
        )
        print(f"Successfully calculated {distance_df.height} distances")
    else:
        print("No pairs available for distance calculation")
        return

    if distance_df.height > 0:
        # Print summary statistics
        print("\n=== Summary Statistics ===")
        # Create a new column that describes the pairing type
        distance_df = distance_df.with_columns([
            pl.when(
                pl.col("same_IC")
                & (pl.col("embedding_model_a") != pl.col("embedding_model_b"))
            )
            .then(pl.lit("Same IC, Different Models"))
            .when(
                (~pl.col("same_IC"))
                & (pl.col("embedding_model_a") == pl.col("embedding_model_b"))
            )
            .then(pl.lit("Different IC, Same Models"))
            .otherwise(pl.lit("Other"))  # This shouldn't happen with our filtering
            .alias("pairing_type")
        ])

        stats = (
            distance_df.group_by("pairing_type")
            .agg([
                pl.count("distance").alias("count"),
                pl.mean("distance").alias("mean_distance"),
                pl.std("distance").alias("std_distance"),
                pl.min("distance").alias("min_distance"),
                pl.max("distance").alias("max_distance"),
            ])
            .sort("pairing_type")
        )

        print("\nStatistics by pairing type:")
        for row in stats.iter_rows(named=True):
            print(f"\n{row['pairing_type']} ({row['count']} pairs):")
            print(f"  Mean: {row['mean_distance']:.4f}")
            print(f"  Std: {row['std_distance']:.4f}")
            print(f"  Min: {row['min_distance']:.4f}")
            print(f"  Max: {row['max_distance']:.4f}")

        # Statistics by embedding model
        print("\n=== Statistics by embedding model ===")
        for model in distance_df["embedding_model_a"].unique():
            model_df = distance_df.filter(pl.col("embedding_model_a") == model)
            model_distances = model_df["distance"].to_list()
            if model_distances:
                print(f"\n{model} ({len(model_distances)} pairs):")
                print(f"  Mean: {np.mean(model_distances):.4f}")
                print(f"  Std: {np.std(model_distances):.4f}")

        # Step 6: Create visualization
        print("\n=== Creating Visualization ===")
        output_file = "output/vis/wasserstein_violin.pdf"
        plot_wasserstein_violin(distance_df, output_file)
        print(f"Saved violin plot to {output_file}")

    # #### Persistence Diagram Density Comparison
    # from panic_tda.datavis import plot_persistence_density

    # plot_persistence_density(pd_df, "output/vis/persistence-density.pdf")
