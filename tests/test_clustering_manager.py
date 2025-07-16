"""Tests for the clustering manager functionality."""

import pytest
from uuid import UUID
from sqlmodel import select, Session, create_engine
from panic_tda.clustering_manager import (
    cluster_all_data,
    get_clustering_status,
    get_cluster_details,
)
from panic_tda.schemas import ExperimentConfig, ClusteringResult, EmbeddingCluster
from panic_tda.engine import perform_experiment


def test_cluster_all_data_no_downsampling(db_session):
    """Test cluster_all_data with downsample=1 (no downsampling)."""
    # Create test experiments
    configs = []
    for i in range(2):
        config = ExperimentConfig(
            networks=[["DummyT2I", "DummyI2T"]],
            seeds=[-1],
            prompts=[f"test clustering {i}"],
            embedding_models=["Dummy", "Dummy2"],
            max_length=50,
        )
        db_session.add(config)
        db_session.commit()
        db_session.refresh(config)
        configs.append(config)
    
    # Run experiments to populate embeddings
    db_url = str(db_session.get_bind().engine.url)
    for config in configs:
        perform_experiment(str(config.id), db_url)
    
    # Cluster all data with no downsampling
    result = cluster_all_data(db_session, downsample=1, force=False)
    
    # Verify successful clustering
    assert result["status"] == "success"
    # With DummyT2I->DummyI2T network, only I2T outputs get embedded
    # 2 experiments * 50 invocations = 100 total embeddings
    assert result["total_embeddings"] == 100
    assert result["clustered_embeddings"] == 100  # All embeddings should be clustered
    assert result["embedding_models_count"] == 2  # Dummy and Dummy2
    assert result["total_clusters"] > 0  # Should have at least one cluster
    
    # Verify clustering results are stored in database
    clustering_results = db_session.exec(
        select(ClusteringResult)
    ).all()
    assert len(clustering_results) == 2  # One for each embedding model
    
    # Verify each clustering result has correct properties
    for cr in clustering_results:
        assert cr.embedding_model in ["Dummy", "Dummy2"]
        assert cr.algorithm == "hdbscan"
        assert cr.parameters == {
            "cluster_selection_epsilon": 0.6,
            "allow_single_cluster": True
        }
        assert len(cr.clusters) > 0  # Should have at least one cluster
        
        # Verify embedding assignments exist
        assignments = db_session.exec(
            select(EmbeddingCluster).where(EmbeddingCluster.clustering_result_id == cr.id)
        ).all()
        assert len(assignments) == 50  # 50 embeddings per model (100 total / 2 models)


def test_cluster_all_data_with_downsampling(db_session):
    """Test cluster_all_data with downsample=2."""
    # Create test experiment
    config = ExperimentConfig(
        networks=[["DummyT2I", "DummyI2T"]],
        seeds=[-1],
        prompts=["test clustering with downsampling"],
        embedding_models=["Dummy", "Dummy2"],
        max_length=100,
    )
    
    db_session.add(config)
    db_session.commit()
    db_session.refresh(config)
    
    # Run experiment to populate embeddings
    db_url = str(db_session.get_bind().engine.url)
    perform_experiment(str(config.id), db_url)
    
    # Cluster all data with downsampling
    result = cluster_all_data(db_session, downsample=2, force=False)
    
    # Verify successful clustering
    assert result["status"] == "success"
    assert result["total_embeddings"] == 100  # 100 invocations with I2T output
    assert result["clustered_embeddings"] == 50  # Half due to downsampling
    assert result["embedding_models_count"] == 2
    assert result["total_clusters"] > 0
    
    # Verify only half the embeddings are assigned clusters
    clustering_results = db_session.exec(
        select(ClusteringResult)
    ).all()
    
    for cr in clustering_results:
        assignments = db_session.exec(
            select(EmbeddingCluster).where(EmbeddingCluster.clustering_result_id == cr.id)
        ).all()
        assert len(assignments) == 25  # 25 embeddings per model (downsampled from 50)


def test_cluster_all_data_no_embeddings(db_session):
    """Test cluster_all_data when there are no embeddings."""
    # Create test experiment but don't run it
    config = ExperimentConfig(
        networks=[["DummyT2I", "DummyI2T"]],
        seeds=[-1],
        prompts=["test no embeddings"],
        embedding_models=["Dummy"],
        max_length=10,
    )
    
    db_session.add(config)
    db_session.commit()
    
    # Try to cluster - should handle gracefully
    result = cluster_all_data(db_session, downsample=1, force=False)
    
    # When there are no embeddings, the dataframe loading might fail
    assert result["status"] == "error"
    assert result["total_embeddings"] == 0
    assert result["clustered_embeddings"] == 0


def test_cluster_all_data_already_clustered(db_session):
    """Test cluster_all_data behavior when data is already clustered."""
    # Create test experiment
    config = ExperimentConfig(
        networks=[["DummyT2I", "DummyI2T"]],
        seeds=[-1],
        prompts=["test already clustered"],
        embedding_models=["Dummy"],
        max_length=20,
    )
    
    db_session.add(config)
    db_session.commit()
    db_session.refresh(config)
    
    # Run experiment to populate embeddings
    db_url = str(db_session.get_bind().engine.url)
    perform_experiment(str(config.id), db_url)
    
    # First clustering
    result1 = cluster_all_data(db_session, downsample=1, force=False)
    assert result1["status"] == "success"
    
    # Second clustering without force - should return already_clustered
    result2 = cluster_all_data(db_session, downsample=1, force=False)
    assert result2["status"] == "already_clustered"
    assert "already exist" in result2["message"]
    
    # Third clustering with force - should re-cluster
    result3 = cluster_all_data(db_session, downsample=1, force=True)
    assert result3["status"] == "success"


def test_get_clustering_status(db_session):
    """Test get_clustering_status function."""
    # Create multiple experiments
    configs = []
    for i in range(2):
        config = ExperimentConfig(
            networks=[["DummyT2I", "DummyI2T"]],
            seeds=[-1],
            prompts=[f"test status {i}"],
            embedding_models=["Dummy"],
            max_length=30,
        )
        db_session.add(config)
        db_session.commit()
        db_session.refresh(config)
        configs.append(config)
    
    # Run experiments
    db_url = str(db_session.get_bind().engine.url)
    for config in configs:
        perform_experiment(str(config.id), db_url)
    
    # Get clustering status before clustering
    status_before = get_clustering_status(db_session)
    
    # Verify status for both experiments before clustering
    for config in configs:
        assert config.id in status_before
        exp_status = status_before[config.id]
        assert exp_status["experiment_id"] == config.id
        assert exp_status["embedding_count"] == 15  # 15 I2T outputs per embedding model
        assert exp_status["clustered_count"] == 0  # Not clustered yet
        assert exp_status["is_fully_clustered"] is False
    
    # Cluster all data
    cluster_all_data(db_session, downsample=1, force=False)
    
    # Get clustering status after clustering
    status_after = get_clustering_status(db_session)
    
    # Verify status for both experiments after clustering
    for config in configs:
        assert config.id in status_after
        exp_status = status_after[config.id]
        assert exp_status["experiment_id"] == config.id
        assert exp_status["embedding_count"] == 15
        assert exp_status["clustered_count"] == 15  # All clustered now
        assert exp_status["is_fully_clustered"] is True


def test_get_cluster_details(db_session):
    """Test get_cluster_details function."""
    # Create test experiment
    config = ExperimentConfig(
        networks=[["DummyT2I", "DummyI2T"]],
        seeds=[-1],
        prompts=["test cluster details"],
        embedding_models=["Dummy", "Dummy2"],
        max_length=40,
    )
    
    db_session.add(config)
    db_session.commit()
    db_session.refresh(config)
    
    # Run experiment
    db_url = str(db_session.get_bind().engine.url)
    perform_experiment(str(config.id), db_url)
    
    # Cluster all data
    cluster_all_data(db_session, downsample=1, force=False)
    
    # Get cluster details for each embedding model
    for model in ["Dummy", "Dummy2"]:
        details = get_cluster_details(config.id, model, db_session)
        
        assert details is not None
        assert details["experiment_id"] == config.id
        assert details["embedding_model"] == model
        assert details["algorithm"] == "hdbscan"
        assert details["total_assignments"] == 20  # 40 outputs / 2 models = 20 per model
        assert details["total_clusters"] > 0
        
        # Verify cluster information
        assert len(details["clusters"]) > 0
        for cluster in details["clusters"]:
            assert "id" in cluster
            assert "medoid_text" in cluster
            assert "size" in cluster
            assert cluster["size"] > 0
        
        # Clusters should be sorted by size (descending)
        sizes = [c["size"] for c in details["clusters"]]
        assert sizes == sorted(sizes, reverse=True)
    
    # Test with non-existent model
    details_none = get_cluster_details(config.id, "NonExistentModel", db_session)
    assert details_none is None


def test_clustering_persistence_across_sessions(db_session):
    """Test that clustering results persist across database sessions."""
    # Create test experiment
    config = ExperimentConfig(
        networks=[["DummyT2I", "DummyI2T"]],
        seeds=[-1],
        prompts=["test persistence"],
        embedding_models=["Dummy"],
        max_length=25,
    )
    
    db_session.add(config)
    db_session.commit()
    db_session.refresh(config)
    experiment_id = config.id
    
    # Run experiment to populate embeddings
    db_url = str(db_session.get_bind().engine.url)
    perform_experiment(str(experiment_id), db_url)
    
    # Cluster all data
    result = cluster_all_data(db_session, downsample=1, force=False)
    assert result["status"] == "success"
    
    # Commit and close session
    db_session.commit()
    
    # Create new session and verify clustering results persist
    engine = create_engine(db_url)
    with Session(engine) as new_session:
        # Check clustering results exist
        clustering_results = new_session.exec(
            select(ClusteringResult)
        ).all()
        assert len(clustering_results) >= 1  # At least one clustering result should exist
        
        # Find the clustering result for our embedding model
        cr = next((cr for cr in clustering_results if cr.embedding_model == "Dummy"), None)
        assert cr is not None, "Should have clustering result for Dummy model"
        
        # Check embedding assignments exist
        assignments = new_session.exec(
            select(EmbeddingCluster).where(EmbeddingCluster.clustering_result_id == cr.id)
        ).all()
        # Since clustering is global, we should have at least some assignments
        assert len(assignments) > 0  # Some embeddings should be clustered
        
        # Check that embeddings from our experiment are included
        from panic_tda.schemas import Embedding, Invocation, Run
        our_embeddings = new_session.exec(
            select(Embedding.id)
            .join(Invocation)
            .join(Run)
            .where(Run.experiment_id == experiment_id)
        ).all()
        
        assigned_embedding_ids = {a.embedding_id for a in assignments}
        our_embedding_ids = set(our_embeddings)
        
        # At least some of our embeddings should be in the clustering
        assert len(our_embedding_ids.intersection(assigned_embedding_ids)) > 0
        
        # Verify attempting to cluster again returns already_clustered
        result2 = cluster_all_data(new_session, downsample=1, force=False)
        assert result2["status"] == "already_clustered"


def test_cluster_all_data_multiple_models(db_session):
    """Test clustering with multiple embedding models."""
    # Create test experiment with multiple embedding models
    config = ExperimentConfig(
        networks=[["DummyT2I", "DummyI2T"]],
        seeds=[-1],  # Single seed to avoid persistence diagram issues
        prompts=["test multiple models", "another prompt", "third prompt"],  # Multiple prompts for more data
        embedding_models=["Dummy", "Dummy2"],  # Only use existing models
        max_length=30,
    )
    
    db_session.add(config)
    db_session.commit()
    db_session.refresh(config)
    
    # Run experiment without persistence diagram stage
    db_url = str(db_session.get_bind().engine.url)
    from panic_tda.engine import perform_experiment
    
    # Run the full experiment - it will skip persistence diagrams if not enough data
    perform_experiment(str(config.id), db_url)
    
    # Cluster all data
    result = cluster_all_data(db_session, downsample=1, force=False)
    
    assert result["status"] == "success"
    # 1 seed * 3 prompts * 30 I2T outputs = 90 embeddings
    assert result["total_embeddings"] == 90
    assert result["clustered_embeddings"] == 90
    assert result["embedding_models_count"] == 2  # Two models
    # Should have clusters for each model
    assert result["total_clusters"] >= 2  # At least one cluster per model
    
    # Verify persistence
    clustering_results = db_session.exec(
        select(ClusteringResult)
    ).all()
    assert len(clustering_results) == 2  # One for each embedding model
    
    for cr in clustering_results:
        assert cr.embedding_model in ["Dummy", "Dummy2"]