"""WIP: charts and visualizations for (maybe) Cybernetics '26."""

import polars as pl
from sqlmodel import Session, select
import numpy as np
from typing import Dict, List, Tuple

from panic_tda.data_prep import (
    cache_dfs,
    load_pd_from_cache,
    calculate_wasserstein_distances,
    pd_list_to_wasserstein_df,
)
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

    # Task 49: Wasserstein PD analysis
    # Step 1: Print unique list of initial prompts from database
    print("\n=== Step 1: Unique initial prompts in database ===")
    statement = select(Run.initial_prompt).distinct()
    unique_prompts = session.exec(statement).all()
    print(f"Found {len(unique_prompts)} unique initial prompts:")
    for i, prompt in enumerate(unique_prompts, 1):
        print(f"{i}. {prompt}")

    # Step 2: Select 2 prompts and load their persistence diagrams
    print("\n=== Step 2: Selecting prompts and loading persistence diagrams ===")

    # Select "a cat" and "a dog" as our two prompts for analysis
    selected_prompts = ["a cat", "a dog"]
    print(f"Selected prompts for analysis: {selected_prompts}")

    # Get all runs for these prompts
    statement = select(Run).where(Run.initial_prompt.in_(selected_prompts))
    runs = session.exec(statement).all()
    print(f"Found {len(runs)} runs for selected prompts")

    # Get persistence diagrams for these runs, filtering for Nomic and NomicVision embedding models
    run_ids = [run.id for run in runs]
    statement = (
        select(PersistenceDiagram)
        .where(PersistenceDiagram.run_id.in_(run_ids))
        .where(PersistenceDiagram.embedding_model.in_(["Nomic", "NomicVision"]))
    )
    persistence_diagrams = session.exec(statement).all()
    print(f"Found {len(persistence_diagrams)} persistence diagrams for selected runs")

    # Group PDs by initial prompt for analysis
    pds_by_prompt = {}
    for pd in persistence_diagrams:
        # Find the run for this PD to get its initial prompt
        run = next((r for r in runs if r.id == pd.run_id), None)
        if run:
            prompt = run.initial_prompt
            if prompt not in pds_by_prompt:
                pds_by_prompt[prompt] = []
            pds_by_prompt[prompt].append(pd)

    for prompt, pds in pds_by_prompt.items():
        print(f"  {prompt}: {len(pds)} persistence diagrams")

    # Step 3: Compute pairwise Wasserstein distances using the new function
    print("\n=== Step 3: Computing pairwise Wasserstein distances ===")

    # Use the new pd_list_to_wasserstein_df function to compute all distances
    wasserstein_df = pd_list_to_wasserstein_df(persistence_diagrams)

    print(f"\nComputed {wasserstein_df.height} pairwise distances")

    if wasserstein_df.height > 0:
        # Extract distances for statistics
        all_distances = wasserstein_df["distance"].to_list()
        print("Distance statistics:")
        print(f"  Min: {np.min(all_distances):.4f}")
        print(f"  Max: {np.max(all_distances):.4f}")
        print(f"  Mean: {np.mean(all_distances):.4f}")
        print(f"  Std: {np.std(all_distances):.4f}")

        # Statistics by embedding model
        print("\n=== Statistics by embedding model ===")
        for model in wasserstein_df["embedding_model"].unique():
            model_df = wasserstein_df.filter(pl.col("embedding_model") == model)
            model_distances = model_df["distance"].to_list()
            print(f"\n{model} ({len(model_distances)} pairs):")
            print(f"  Mean: {np.mean(model_distances):.4f}")
            print(f"  Std: {np.std(model_distances):.4f}")

        # Statistics by homology dimension
        for dim in wasserstein_df["homology_dimension"].unique().sort():
            dim_df = wasserstein_df.filter(pl.col("homology_dimension") == dim)
            dim_distances = dim_df["distance"].to_list()
            print(f"\nHomology dimension {dim} ({len(dim_distances)} pairs):")
            print(f"  Mean: {np.mean(dim_distances):.4f}")
            print(f"  Std: {np.std(dim_distances):.4f}")

        # Statistics by same/different initial conditions and homology dimension
        # Add same_initial_conditions indicator to DataFrame
        wasserstein_df = wasserstein_df.with_columns(
            (pl.col("initial_conditions_a") == pl.col("initial_conditions_b")).alias(
                "same_initial_conditions"
            )
        )

        for dim in wasserstein_df["homology_dimension"].unique().sort():
            dim_same_df = wasserstein_df.filter(
                (pl.col("homology_dimension") == dim)
                & pl.col("same_initial_conditions")
            )
            dim_diff_df = wasserstein_df.filter(
                (pl.col("homology_dimension") == dim)
                & ~pl.col("same_initial_conditions")
            )

            dim_same = dim_same_df["distance"].to_list()
            dim_diff = dim_diff_df["distance"].to_list()

            print(f"\nH{dim} - Same initial conditions ({len(dim_same)} pairs):")
            if dim_same:
                print(f"  Mean: {np.mean(dim_same):.4f}")
                print(f"  Std: {np.std(dim_same):.4f}")

            print(f"H{dim} - Different initial conditions ({len(dim_diff)} pairs):")
            if dim_diff:
                print(f"  Mean: {np.mean(dim_diff):.4f}")
                print(f"  Std: {np.std(dim_diff):.4f}")

        # Statistics by embedding type and homology dimension
        print("\n=== Statistics by embedding type and homology dimension ===")
        for emb_type in wasserstein_df["embedding_type"].unique().sort():
            for dim in wasserstein_df["homology_dimension"].unique().sort():
                type_dim_df = wasserstein_df.filter(
                    (pl.col("embedding_type") == emb_type)
                    & (pl.col("homology_dimension") == dim)
                )
                type_dim_distances = type_dim_df["distance"].to_list()

                if type_dim_distances:
                    print(
                        f"\n{emb_type.capitalize()} embeddings - H{dim} ({len(type_dim_distances)} pairs):"
                    )
                    print(f"  Mean: {np.mean(type_dim_distances):.4f}")
                    print(f"  Std: {np.std(type_dim_distances):.4f}")

                    # Also break down by same/different initial conditions
                    type_dim_same_df = type_dim_df.filter(
                        pl.col("same_initial_conditions")
                    )
                    type_dim_diff_df = type_dim_df.filter(
                        ~pl.col("same_initial_conditions")
                    )

                    type_dim_same = type_dim_same_df["distance"].to_list()
                    type_dim_diff = type_dim_diff_df["distance"].to_list()

                    if type_dim_same:
                        print(
                            f"    Same initial conditions: Mean={np.mean(type_dim_same):.4f}, Std={np.std(type_dim_same):.4f} (n={len(type_dim_same)})"
                        )
                    if type_dim_diff:
                        print(
                            f"    Diff initial conditions: Mean={np.mean(type_dim_diff):.4f}, Std={np.std(type_dim_diff):.4f} (n={len(type_dim_diff)})"
                        )

        # Step 4: Create visualization
        print("\n=== Step 4: Creating visualization ===")
        from panic_tda.datavis import plot_wasserstein_distribution

        # Filter to only homology dimension == 1
        filtered_wasserstein_df = wasserstein_df.filter(
            pl.col("homology_dimension") == 1
        )
        print(
            f"Filtered to homology dimension 1: {filtered_wasserstein_df.height} distances"
        )

        # Further filter to only include rows where initial_conditions_a and initial_conditions_b
        # match the first value in the initial_conditions_b column
        if filtered_wasserstein_df.height > 0:
            first_initial_condition = filtered_wasserstein_df["initial_conditions_b"][0]
            filtered_wasserstein_df = filtered_wasserstein_df.filter(
                (pl.col("initial_conditions_a") == first_initial_condition)
                & (pl.col("initial_conditions_b") == first_initial_condition)
            )
            print(
                f"Filtered to initial condition '{first_initial_condition}': {filtered_wasserstein_df.height} distances"
            )

        # Create the plot using the filtered DataFrame
        output_file = "output/vis/wasserstein_distribution.pdf"
        plot_wasserstein_distribution(filtered_wasserstein_df, output_file)
        print(f"Visualization saved to {output_file}")

    # #### Persistence Diagram Density Comparison
    # from panic_tda.datavis import plot_persistence_density

    # plot_persistence_density(pd_df, "output/vis/persistence-density.pdf")
