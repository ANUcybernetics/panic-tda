"""Charts and visualizations for IEEE SMC paper."""

import polars as pl
from sqlmodel import Session

from panic_tda.data_prep import cache_dfs, load_pd_from_cache
from panic_tda.local_modules.shared import cluster_counts, example_run_ids, run_counts


def ieee_smc_charts(session: Session) -> None:
    """
    Generate charts for paper publications.

    TODO this has actually been modified a bit (for exploring new options), so
    if we wanted to re-create the paper charts exactly you'd probably have to
    un-comment some lines and re-comment some others.
    """
    # pl.Config.set_tbl_rows(10)
    ### CACHING
    #
    cache_dfs(
        session,
        runs=False,
        embeddings=False,
        invocations=False,
        persistence_diagrams=True,
    )
    # cache_dfs(session, runs=True, embeddings=True, invocations=True)
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
    pd_df = load_pd_from_cache()
    print(pd_df.columns)
    print(pd_df.head())

    # #### Persistence Diagram Density Comparison
    # from panic_tda.datavis import plot_persistence_density

    # plot_persistence_density(pd_df, "output/vis/persistence-density.pdf")
