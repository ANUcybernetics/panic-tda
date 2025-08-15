"""WIP: charts and visualizations for (maybe) Cybernetics '26."""

import polars as pl
from sqlmodel import Session, select
import numpy as np
from typing import Dict, List, Tuple

from panic_tda.data_prep import (
    cache_dfs,
    load_pd_from_cache,
    calculate_wasserstein_distances,
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

    # Get persistence diagrams for these runs
    run_ids = [run.id for run in runs]
    statement = select(PersistenceDiagram).where(PersistenceDiagram.run_id.in_(run_ids))
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

    # Step 3: Compute pairwise Wasserstein distances
    print("\n=== Step 3: Computing pairwise Wasserstein distances ===")

    # We'll compute distances for text embeddings
    # Get unique embedding models in our PDs
    embedding_models = set()
    for pd in persistence_diagrams:
        if pd.embedding_model:
            embedding_models.add(pd.embedding_model)

    print(f"Found embedding models: {embedding_models}")

    # For each embedding model, compute pairwise distances
    all_distances = []
    labels = []  # Whether the pair is same prompt (True) or different (False)

    from panic_tda.tda import compute_wasserstein_distance

    # Debug: Check first few PDs to understand structure
    if persistence_diagrams:
        sample_pd = persistence_diagrams[0]
        if sample_pd.diagram_data and "dgms" in sample_pd.diagram_data:
            print("\nSample PD structure:")
            print(f"  Number of dimensions: {len(sample_pd.diagram_data['dgms'])}")
            for i, dgm in enumerate(sample_pd.diagram_data["dgms"]):
                if isinstance(dgm, np.ndarray):
                    print(f"  Dimension {i}: shape {dgm.shape}, size {dgm.size}")
                    if dgm.size > 0:
                        print(f"    Contains inf: {np.any(np.isinf(dgm))}")
                        print(
                            f"    Min/Max values: {np.min(dgm[~np.isinf(dgm)]):.3f} / {np.max(dgm[~np.isinf(dgm)]):.3f}"
                            if np.any(~np.isinf(dgm))
                            else "All inf"
                        )

    for embedding_model in embedding_models:
        print(f"\nProcessing embedding model: {embedding_model}")

        # Get PDs for this embedding model
        model_pds = [
            pd for pd in persistence_diagrams if pd.embedding_model == embedding_model
        ]
        print(f"  Found {len(model_pds)} PDs for this model")

        # Compute pairwise distances
        for i, pd1 in enumerate(model_pds):
            for j, pd2 in enumerate(model_pds):
                if i < j:  # Only compute upper triangle (symmetric matrix)
                    # Get the initial prompts for these PDs
                    run1 = next((r for r in runs if r.id == pd1.run_id), None)
                    run2 = next((r for r in runs if r.id == pd2.run_id), None)

                    if run1 and run2 and pd1.diagram_data and pd2.diagram_data:
                        if "dgms" in pd1.diagram_data and "dgms" in pd2.diagram_data:
                            # Compute distance for dimension 0 (connected components)
                            dgm1 = pd1.diagram_data["dgms"][0]
                            dgm2 = pd2.diagram_data["dgms"][0]

                            if isinstance(dgm1, np.ndarray) and isinstance(
                                dgm2, np.ndarray
                            ):
                                # Skip if either diagram is empty
                                if dgm1.size > 0 and dgm2.size > 0:
                                    # Filter out points with infinite death times
                                    finite_mask1 = ~np.isinf(dgm1).any(axis=1)
                                    finite_mask2 = ~np.isinf(dgm2).any(axis=1)
                                    dgm1_finite = dgm1[finite_mask1]
                                    dgm2_finite = dgm2[finite_mask2]

                                    # Only compute if we have finite points
                                    if dgm1_finite.size > 0 and dgm2_finite.size > 0:
                                        distance = compute_wasserstein_distance(
                                            dgm1_finite, dgm2_finite
                                        )
                                        # Only add if distance is not NaN
                                        if not np.isnan(distance):
                                            all_distances.append(distance)
                                            labels.append(
                                                run1.initial_prompt
                                                == run2.initial_prompt
                                            )

    print(f"\nComputed {len(all_distances)} pairwise distances")
    if all_distances:
        print("Distance statistics:")
        print(f"  Min: {np.min(all_distances):.4f}")
        print(f"  Max: {np.max(all_distances):.4f}")
        print(f"  Mean: {np.mean(all_distances):.4f}")
        print(f"  Std: {np.std(all_distances):.4f}")

        # Statistics by same/different prompt
        same_prompt_distances = [d for d, l in zip(all_distances, labels) if l]
        diff_prompt_distances = [d for d, l in zip(all_distances, labels) if not l]

        if same_prompt_distances:
            print(f"\nSame prompt pairs ({len(same_prompt_distances)} pairs):")
            print(f"  Mean: {np.mean(same_prompt_distances):.4f}")
            print(f"  Std: {np.std(same_prompt_distances):.4f}")

        if diff_prompt_distances:
            print(f"\nDifferent prompt pairs ({len(diff_prompt_distances)} pairs):")
            print(f"  Mean: {np.mean(diff_prompt_distances):.4f}")
            print(f"  Std: {np.std(diff_prompt_distances):.4f}")

        # Step 4: Create visualization
        print("\n=== Step 4: Creating visualization ===")
        from panic_tda.datavis import plot_wasserstein_distribution

        # Create the plot
        output_file = "output/vis/wasserstein_distribution.pdf"
        plot_wasserstein_distribution(all_distances, labels, output_file)
        print(f"Visualization saved to {output_file}")

    # #### Persistence Diagram Density Comparison
    # from panic_tda.datavis import plot_persistence_density

    # plot_persistence_density(pd_df, "output/vis/persistence-density.pdf")
