import os
import tempfile

import polars as pl
import polars.testing

from panic_tda.clustering_manager import cluster_all_data
from panic_tda.data_prep import (
    cache_dfs,
    load_clusters_df,
    load_embeddings_df,
    load_invocations_df,
    load_pd_df,
    load_runs_df,
)
from panic_tda.engine import perform_experiment
from panic_tda.schemas import ExperimentConfig


def test_end_to_end_with_caching(db_session):
    """Test end-to-end pipeline: experiment → clustering → caching → loading from cache."""
    
    # Use a temporary directory for cache to avoid touching real cache files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create cache directory
        cache_dir = os.path.join(temp_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # 1. Setup: Create config with dummy models
        config = ExperimentConfig(
            networks=[["DummyT2I", "DummyI2T"]],
            seeds=[-1],
            prompts=["Integration test with caching"],
            embedding_models=["Dummy", "Dummy2"],
            max_length=50,  # Reasonable size for testing
        )
        db_session.add(config)
        db_session.commit()
        db_session.refresh(config)
        
        # 2. Run the experiment
        db_url = str(db_session.get_bind().engine.url)
        perform_experiment(str(config.id), db_url)
        
        # 3. Run clustering with small downsample factor
        cluster_all_data(db_session, downsample=2)
        db_session.commit()
        
        # 4. Load all dataframes from database before caching
        runs_df_before = load_runs_df(db_session)
        invocations_df_before = load_invocations_df(db_session)
        embeddings_df_before = load_embeddings_df(db_session)
        # Note: cache_dfs adds semantic_drift to embeddings, so we need to do the same
        from panic_tda.data_prep import add_semantic_drift
        embeddings_df_before = add_semantic_drift(embeddings_df_before, db_session)
        clusters_df_before = load_clusters_df(db_session)
        pd_df_before = load_pd_df(db_session)
        
        # 5. Cache all dataframes to our temporary directory
        cache_dfs(db_session, clusters=True, cache_dir=cache_dir)
        
        # 6. Verify cache files exist in temporary directory
        assert os.path.exists(f"{cache_dir}/runs.parquet"), "Runs cache file not found"
        assert os.path.exists(f"{cache_dir}/invocations.parquet"), "Invocations cache file not found"
        assert os.path.exists(f"{cache_dir}/embeddings.parquet"), "Embeddings cache file not found"
        assert os.path.exists(f"{cache_dir}/clusters.parquet"), "Clusters cache file not found"
        assert os.path.exists(f"{cache_dir}/pd.parquet"), "Persistence diagram cache file not found"
        
        # 7. Load dataframes from cache
        runs_df_cached = pl.read_parquet(f"{cache_dir}/runs.parquet")
        invocations_df_cached = pl.read_parquet(f"{cache_dir}/invocations.parquet")
        embeddings_df_cached = pl.read_parquet(f"{cache_dir}/embeddings.parquet")
        clusters_df_cached = pl.read_parquet(f"{cache_dir}/clusters.parquet")
        pd_df_cached = pl.read_parquet(f"{cache_dir}/pd.parquet")
        
        # 8. Verify cached data matches original data
        # Sort dataframes for consistent comparison
        # Note: For runs_df, we can't use assert_frame_equal directly because network is a list column
        # Instead, we'll check shape and key columns
        assert runs_df_before.shape == runs_df_cached.shape, "Runs dataframe shapes don't match"
        assert set(runs_df_before.columns) == set(runs_df_cached.columns), "Runs dataframe columns don't match"
        
        # Check that the data is equivalent by comparing sorted values of non-list columns
        # We'll compare on run_id since that should be unique per row in the expanded dataframe
        before_sorted = runs_df_before.sort("run_id").select(["run_id", "experiment_id", "initial_prompt", "seed"])
        cached_sorted = runs_df_cached.sort("run_id").select(["run_id", "experiment_id", "initial_prompt", "seed"])
        pl.testing.assert_frame_equal(before_sorted, cached_sorted, check_column_order=False)
        
        pl.testing.assert_frame_equal(
            invocations_df_before.sort("id"),
            invocations_df_cached.sort("id"),
            check_column_order=False,
            check_row_order=False,
        )
        
        pl.testing.assert_frame_equal(
            embeddings_df_before.sort("id"),
            embeddings_df_cached.sort("id"),
            check_column_order=False,
            check_row_order=False,
        )
        
        pl.testing.assert_frame_equal(
            clusters_df_before.sort(["embedding_id", "embedding_model"]),
            clusters_df_cached.sort(["embedding_id", "embedding_model"]),
            check_column_order=False,
            check_row_order=False,
        )
        
        pl.testing.assert_frame_equal(
            pd_df_before.sort(["persistence_diagram_id", "homology_dimension", "birth"]),
            pd_df_cached.sort(["persistence_diagram_id", "homology_dimension", "birth"]),
            check_column_order=False,
            check_row_order=False,
        )
        
        # 9. Verify data integrity and invariants
        
        # Check runs dataframe
        assert runs_df_cached.height > 0, "No runs found in cache"
        assert runs_df_cached["experiment_id"].is_not_null().all(), "Found null experiment_id"
        assert runs_df_cached["network"].is_not_null().all(), "Found null network"
        assert runs_df_cached["entropy"].is_not_null().all(), "Found null entropy values"
        
        # Check invocations dataframe
        assert invocations_df_cached.height == 50, "Expected 50 invocations"
        assert invocations_df_cached["run_id"].is_not_null().all(), "Found null run_id"
        assert invocations_df_cached["model"].is_not_null().all(), "Found null model"
        assert invocations_df_cached["sequence_number"].min() == 0, "Sequence numbers should start at 0"
        assert invocations_df_cached["sequence_number"].max() == 49, "Sequence numbers should end at 49"
        
        # Check embeddings dataframe
        # With downsample=2, we should have ~25 embeddings per model
        assert embeddings_df_cached.height == 50, "Expected 50 embeddings (25 per model)"
        assert embeddings_df_cached["embedding_model"].unique().sort().to_list() == ["Dummy", "Dummy2"]
        assert embeddings_df_cached["text"].is_not_null().all(), "Found null text in embeddings"
        
        # Check clusters dataframe
        # With downsample=2, we should have ~25 embeddings per model clustered
        # So approximately 25-26 total cluster assignments (some embeddings may be filtered)
        assert 20 <= clusters_df_cached.height <= 30, f"Expected ~25 cluster assignments, got {clusters_df_cached.height}"
        assert clusters_df_cached["cluster_label"].is_not_null().all(), "Found null cluster labels"
        assert clusters_df_cached["algorithm"].unique().to_list() == ["hdbscan"]
        assert clusters_df_cached["epsilon"].unique().to_list() == [0.4]
        
        # Verify cluster labels are valid (either from output texts or "OUTLIER")
        unique_texts = embeddings_df_cached["text"].unique().to_list()
        unique_labels = clusters_df_cached["cluster_label"].unique().to_list()
        for label in unique_labels:
            assert label in unique_texts or label == "OUTLIER", f"Invalid cluster label: {label}"
        
        # Check persistence diagram dataframe
        assert pd_df_cached.height > 0, "No persistence diagram points found"
        assert pd_df_cached["persistence_diagram_id"].is_not_null().all()
        assert pd_df_cached["homology_dimension"].min() >= 0, "Negative homology dimension found"
        assert pd_df_cached["birth"].is_not_null().all(), "Found null birth values"
        assert pd_df_cached["death"].is_not_null().all(), "Found null death values"
        
        # Verify persistence = death - birth for finite values
        finite_pd = pd_df_cached.filter(pd_df_cached["persistence"].is_finite())
        if finite_pd.height > 0:
            calculated_persistence = finite_pd["death"] - finite_pd["birth"]
            assert (
                (finite_pd["persistence"] - calculated_persistence).abs() < 1e-6
            ).all(), "Persistence calculation mismatch"
        
        # Verify infinite persistence corresponds to infinite death
        infinite_pd = pd_df_cached.filter(pd_df_cached["persistence"].is_infinite())
        if infinite_pd.height > 0:
            assert infinite_pd["death"].is_infinite().all(), "Infinite persistence without infinite death"
        
        # 10. Cross-validate relationships between dataframes
        
        # Every invocation should belong to a run in runs_df
        invocation_run_ids = set(invocations_df_cached["run_id"].unique().to_list())
        runs_run_ids = set(runs_df_cached["run_id"].unique().to_list())
        assert invocation_run_ids.issubset(runs_run_ids), "Found invocations with invalid run_ids"
        
        # Every embedding should belong to an invocation
        embedding_inv_ids = set(embeddings_df_cached["invocation_id"].unique().to_list())
        invocation_ids = set(invocations_df_cached["id"].unique().to_list())
        assert embedding_inv_ids.issubset(invocation_ids), "Found embeddings with invalid invocation_ids"
        
        # Every cluster assignment should belong to an embedding
        # Note: With downsampling, not all embeddings will have cluster assignments
        cluster_emb_ids = set(clusters_df_cached["embedding_id"].unique().to_list())
        embedding_ids = set(embeddings_df_cached["id"].unique().to_list())
        assert cluster_emb_ids.issubset(embedding_ids), "Found cluster assignments for non-existent embeddings"
        
        # Every persistence diagram should belong to a run
        pd_run_ids = set(pd_df_cached["run_id"].unique().to_list())
        assert pd_run_ids.issubset(runs_run_ids), "Found persistence diagrams with invalid run_ids"