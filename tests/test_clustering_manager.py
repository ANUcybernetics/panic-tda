"""Tests for the clustering manager functionality."""

from sqlmodel import Session, create_engine, select, func

from panic_tda.clustering_manager import (
    cluster_all_data,
    _save_clustering_results,
    delete_cluster_data,
    delete_single_cluster,
)
from panic_tda.engine import perform_experiment
from panic_tda.schemas import (
    ClusteringResult,
    Embedding,
    EmbeddingCluster,
    ExperimentConfig,
    Invocation,
    InvocationType,
    Run,
)
import numpy as np
from uuid import UUID


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
    result = cluster_all_data(db_session, downsample=1)

    # Verify successful clustering
    assert result["status"] == "success"
    # With DummyT2I->DummyI2T network, only I2T outputs get embedded
    # 2 experiments * 50 invocations = 100 total embeddings
    assert result["total_embeddings"] == 100
    assert result["clustered_embeddings"] == 100  # All embeddings should be clustered
    assert result["embedding_models_count"] == 2  # Dummy and Dummy2
    assert result["total_clusters"] > 0  # Should have at least one cluster

    # Verify clustering results are stored in database
    clustering_results = db_session.exec(select(ClusteringResult)).all()
    assert len(clustering_results) == 2  # One for each embedding model

    # Verify each clustering result has correct properties
    for cr in clustering_results:
        assert cr.embedding_model in ["Dummy", "Dummy2"]
        assert cr.algorithm == "hdbscan"
        assert cr.parameters == {
            "cluster_selection_epsilon": 0.4,
            "allow_single_cluster": True,
        }
        # Check that clustering assignments were created
        # Count unique clusters (distinct medoid_embedding_ids)
        unique_clusters = db_session.exec(
            select(func.count(func.distinct(EmbeddingCluster.medoid_embedding_id)))
            .where(EmbeddingCluster.clustering_result_id == cr.id)
        ).one()
        assert unique_clusters > 0  # Should have at least one cluster

        # Verify embedding assignments exist
        assignments = db_session.exec(
            select(EmbeddingCluster).where(
                EmbeddingCluster.clustering_result_id == cr.id
            )
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
    result = cluster_all_data(db_session, downsample=2)

    # Verify successful clustering
    assert result["status"] == "success"
    assert result["total_embeddings"] == 100  # 100 invocations with I2T output
    assert result["clustered_embeddings"] == 50  # Half due to downsampling
    assert result["embedding_models_count"] == 2
    assert result["total_clusters"] > 0

    # Verify only half the embeddings are assigned clusters
    clustering_results = db_session.exec(select(ClusteringResult)).all()

    for cr in clustering_results:
        assignments = db_session.exec(
            select(EmbeddingCluster).where(
                EmbeddingCluster.clustering_result_id == cr.id
            )
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
    result = cluster_all_data(db_session, downsample=1)

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
    result1 = cluster_all_data(db_session, downsample=1)
    assert result1["status"] == "success"

    # Second clustering - should create new clustering results (no longer returns already_clustered)
    result2 = cluster_all_data(db_session, downsample=1)
    assert result2["status"] == "success"

    # Verify we now have multiple clustering results
    clustering_results = db_session.exec(select(ClusteringResult)).all()
    assert len(clustering_results) >= 2  # Should have at least 2 clustering results

    # Delete existing clustering data
    from panic_tda.clustering_manager import delete_cluster_data

    delete_result = delete_cluster_data(db_session)
    assert delete_result["status"] == "success"

    # Third clustering after deletion - should re-cluster
    result3 = cluster_all_data(db_session, downsample=1)
    assert result3["status"] == "success"


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
    result = cluster_all_data(db_session, downsample=1)
    assert result["status"] == "success"

    # Commit and close session
    db_session.commit()

    # Create new session and verify clustering results persist
    engine = create_engine(db_url)
    with Session(engine) as new_session:
        # Check clustering results exist
        clustering_results = new_session.exec(select(ClusteringResult)).all()
        assert (
            len(clustering_results) >= 1
        )  # At least one clustering result should exist

        # Find the clustering result for our embedding model
        cr = next(
            (cr for cr in clustering_results if cr.embedding_model == "Dummy"), None
        )
        assert cr is not None, "Should have clustering result for Dummy model"

        # Check embedding assignments exist
        assignments = new_session.exec(
            select(EmbeddingCluster).where(
                EmbeddingCluster.clustering_result_id == cr.id
            )
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

        # Verify attempting to cluster again creates new clustering results
        result2 = cluster_all_data(new_session, downsample=1)
        assert result2["status"] == "success"

        # Should now have multiple clustering results
        clustering_results2 = new_session.exec(select(ClusteringResult)).all()
        assert len(clustering_results2) > len(clustering_results)


def test_medoid_embedding_id_stored(db_session):
    """Test that medoid_embedding_id is stored for each cluster."""
    # Use the same approach as test_cluster_all_data_multiple_models
    config = ExperimentConfig(
        networks=[["DummyT2I", "DummyI2T"], ["DummyT2I2", "DummyI2T2"]],
        seeds=[-1],
        prompts=["test multiple models"],
        embedding_models=["Dummy", "Dummy2"],
        max_length=15,
    )

    db_session.add(config)
    db_session.commit()
    db_session.refresh(config)

    # Run experiment to populate embeddings
    db_url = str(db_session.get_bind().engine.url)
    perform_experiment(str(config.id), db_url)

    # Cluster all data
    result = cluster_all_data(db_session, downsample=1)
    assert result["status"] == "success"
    assert result["total_embeddings"] > 0  # Make sure we have embeddings
    assert result["clustered_embeddings"] > 0  # Make sure they were clustered

    # Verify both models have clustering results with medoid_embedding_id
    for model in ["Dummy", "Dummy2"]:
        clustering_result = db_session.exec(
            select(ClusteringResult).where(ClusteringResult.embedding_model == model)
        ).first()

        assert clustering_result is not None
        
        # Count unique clusters for this result
        unique_clusters = db_session.exec(
            select(func.count(func.distinct(EmbeddingCluster.medoid_embedding_id)))
            .where(EmbeddingCluster.clustering_result_id == clustering_result.id)
        ).one()
        assert unique_clusters > 0
        
        # Check that medoid embeddings exist (except for outliers)
        medoid_ids = db_session.exec(
            select(func.distinct(EmbeddingCluster.medoid_embedding_id))
            .where(EmbeddingCluster.clustering_result_id == clustering_result.id)
            .where(EmbeddingCluster.medoid_embedding_id.is_not(None))
        ).all()
        
        for medoid_id in medoid_ids:
            # Verify the embedding exists
            # Convert to UUID if it's a string
            if isinstance(medoid_id, str):
                medoid_id = UUID(medoid_id)
            embedding = db_session.get(Embedding, medoid_id)
            assert embedding is not None


def test_cluster_all_data_multiple_models(db_session):
    """Test clustering with multiple embedding models."""
    # Create test experiment with multiple embedding models
    config = ExperimentConfig(
        networks=[["DummyT2I", "DummyI2T"]],
        seeds=[-1],  # Single seed to avoid persistence diagram issues
        prompts=[
            "test multiple models",
            "another prompt",
            "third prompt",
        ],  # Multiple prompts for more data
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
    result = cluster_all_data(db_session, downsample=1)

    assert result["status"] == "success"
    # 1 seed * 3 prompts * 30 I2T outputs = 90 embeddings
    assert result["total_embeddings"] == 90
    assert result["clustered_embeddings"] == 90
    assert result["embedding_models_count"] == 2  # Two models
    # Should have clusters for each model
    assert result["total_clusters"] >= 2  # At least one cluster per model

    # Verify persistence
    clustering_results = db_session.exec(select(ClusteringResult)).all()
    assert len(clustering_results) == 2  # One for each embedding model

    for cr in clustering_results:
        assert cr.embedding_model in ["Dummy", "Dummy2"]


def test_cluster_creation_with_known_medoids(db_session):
    """Test that cluster creation correctly handles medoid embedding IDs."""
    # Create a simple test setup with known embeddings
    experiment = ExperimentConfig(
        networks=[["DummyT2I", "DummyI2T"]],
        seeds=[-1],
        prompts=["test"],
        embedding_models=["Dummy"],
        max_length=10,
    )
    db_session.add(experiment)
    db_session.commit()

    run = Run(
        experiment_id=experiment.id,
        network=["DummyT2I", "DummyI2T"],
        initial_prompt="test",
        seed=-1,
        max_length=10,
    )
    db_session.add(run)
    db_session.commit()

    # Create some invocations with embeddings
    embedding_ids = []
    vectors = []
    texts = []

    for i in range(5):
        inv = Invocation(
            run_id=run.id,
            sequence_number=i,
            model="DummyI2T",  # Use 'model' not 'model_name'
            output_text=f"Text {i}",
            type=InvocationType.TEXT,
            seed=-1,
        )
        db_session.add(inv)
        db_session.commit()

        # Create embedding with known vector
        vector = np.random.randn(10)  # 10-dimensional embeddings
        emb = Embedding(
            invocation_id=inv.id,
            embedding_model="Dummy",
            vector=vector,
        )
        db_session.add(emb)
        db_session.commit()

        embedding_ids.append(emb.id)
        vectors.append(vector)
        texts.append(f"Text {i}")

    # Create mock clustering result with known cluster assignments
    # Simulate HDBSCAN output
    cluster_result = {
        "labels": np.array([0, 0, 1, 1, -1]),  # 2 clusters + 1 outlier
        "medoids": np.array([vectors[0], vectors[2]]),  # Medoids for clusters 0 and 1
        "medoid_indices": {
            0: 0,
            1: 2,
        },  # Cluster 0 medoid is at index 0, cluster 1 at index 2
    }

    # Test the save function
    from datetime import datetime
    start_time = datetime.utcnow()
    end_time = datetime.utcnow()
    
    result = _save_clustering_results(
        session=db_session,
        model_name="Dummy",
        embedding_ids=embedding_ids,
        cluster_result=cluster_result,
        texts=texts,
        downsample=1,
        epsilon=0.4,
        start_time=start_time,
        end_time=end_time,
    )

    assert result["status"] == "success"

    # Verify the clustering result was created
    clustering_result = db_session.exec(
        select(ClusteringResult).where(ClusteringResult.embedding_model == "Dummy")
    ).first()
    assert clustering_result is not None

    # Verify clusters were created correctly by checking unique medoid IDs
    unique_medoids = db_session.exec(
        select(func.distinct(EmbeddingCluster.medoid_embedding_id))
        .where(EmbeddingCluster.clustering_result_id == clustering_result.id)
    ).all()
    
    # Should have 3 unique medoid values: embedding_ids[0], embedding_ids[2], and None (outlier)
    assert len(unique_medoids) == 3
    assert None in unique_medoids  # Outlier cluster
    
    # Convert string UUIDs to UUID objects for comparison
    medoid_uuids = []
    for medoid in unique_medoids:
        if medoid is not None:
            medoid_uuids.append(UUID(medoid) if isinstance(medoid, str) else medoid)
    
    assert embedding_ids[0] in medoid_uuids  # Cluster 0 medoid
    assert embedding_ids[2] in medoid_uuids  # Cluster 1 medoid

    # Verify embedding cluster assignments
    assignments = db_session.exec(
        select(EmbeddingCluster).where(
            EmbeddingCluster.clustering_result_id == clustering_result.id
        )
    ).all()
    assert len(assignments) == 5  # All embeddings should be assigned

    for assignment in assignments:
        assert isinstance(assignment.embedding_id, UUID)
        assert isinstance(assignment.clustering_result_id, UUID)
        # medoid_embedding_id can be UUID or None (for outliers)
        assert assignment.medoid_embedding_id is None or isinstance(assignment.medoid_embedding_id, UUID)


def test_medoid_index_bounds_checking(db_session):
    """Test that medoid indices are properly bounds-checked."""
    # Create embeddings
    experiment = ExperimentConfig(
        networks=[["DummyT2I", "DummyI2T"]],
        seeds=[-1],
        prompts=["test"],
        embedding_models=["Dummy"],
        max_length=5,
    )
    db_session.add(experiment)
    db_session.commit()

    run = Run(
        experiment_id=experiment.id,
        network=["DummyT2I", "DummyI2T"],
        initial_prompt="test",
        seed=-1,
        max_length=5,
    )
    db_session.add(run)
    db_session.commit()

    embedding_ids = []
    vectors = []
    texts = []

    for i in range(3):
        inv = Invocation(
            run_id=run.id,
            sequence_number=i,
            model="DummyI2T",  # Use 'model' not 'model_name'
            output_text=f"Text {i}",
            type=InvocationType.TEXT,
            seed=-1,
        )
        db_session.add(inv)
        db_session.commit()

        emb = Embedding(
            invocation_id=inv.id,
            embedding_model="Dummy",
            vector=np.random.randn(10),
        )
        db_session.add(emb)
        db_session.commit()

        embedding_ids.append(emb.id)
        vectors.append(emb.vector)
        texts.append(f"Text {i}")

    # Test with invalid medoid index (out of bounds) and include an outlier
    cluster_result = {
        "labels": np.array([0, 0, -1]),  # Include an outlier
        "medoids": np.array([vectors[0]]),
        "medoid_indices": {0: 10},  # Invalid index (out of bounds)
    }

    # This should handle the error gracefully
    from datetime import datetime
    start_time = datetime.utcnow()
    end_time = datetime.utcnow()
    
    result = _save_clustering_results(
        session=db_session,
        model_name="Dummy",
        embedding_ids=embedding_ids,
        cluster_result=cluster_result,
        texts=texts,
        downsample=1,
        epsilon=0.4,
        start_time=start_time,
        end_time=end_time,
    )

    # The function should succeed but skip the invalid cluster
    assert result["status"] == "success"

    # Check that the outlier cluster was created but not the invalid cluster
    clustering_result = db_session.exec(
        select(ClusteringResult).where(ClusteringResult.embedding_model == "Dummy")
    ).first()

    # Count unique medoid IDs to verify clusters
    unique_medoids = db_session.exec(
        select(func.distinct(EmbeddingCluster.medoid_embedding_id))
        .where(EmbeddingCluster.clustering_result_id == clustering_result.id)
    ).all()

    # Should only have None (outlier) since the regular cluster had invalid medoid
    assert len(unique_medoids) == 1
    assert None in unique_medoids  # Only outlier cluster

    # Check that assignments still work for outliers
    assignments = db_session.exec(
        select(EmbeddingCluster).where(
            EmbeddingCluster.clustering_result_id == clustering_result.id
        )
    ).all()
    # Only the outlier should be assigned since the regular cluster was invalid
    assert len(assignments) == 1  # Only the outlier embedding


def test_uuid_type_validation(db_session):
    """Test that all UUID fields are properly validated during clustering."""
    # Create test data
    experiment = ExperimentConfig(
        networks=[["DummyT2I", "DummyI2T"]],
        seeds=[-1],
        prompts=["test uuid validation"],
        embedding_models=["Dummy"],
        max_length=20,
    )
    db_session.add(experiment)
    db_session.commit()
    db_session.refresh(experiment)

    # Run experiment
    db_url = str(db_session.get_bind().engine.url)
    perform_experiment(str(experiment.id), db_url)

    # Run clustering
    result = cluster_all_data(db_session, downsample=1)
    assert result["status"] == "success"

    # Verify all UUID fields are proper UUIDs
    clustering_results = db_session.exec(select(ClusteringResult)).all()

    for cr in clustering_results:
        assert isinstance(cr.id, UUID)

        # Check all medoid embeddings
        medoid_ids = db_session.exec(
            select(func.distinct(EmbeddingCluster.medoid_embedding_id))
            .where(EmbeddingCluster.clustering_result_id == cr.id)
            .where(EmbeddingCluster.medoid_embedding_id.is_not(None))
        ).all()
        
        for medoid_id in medoid_ids:
            # Convert to UUID if it's a string (SQLite quirk)
            if isinstance(medoid_id, str):
                medoid_id = UUID(medoid_id)
            assert isinstance(medoid_id, UUID)
            # Verify it references a real embedding
            embedding = db_session.get(Embedding, medoid_id)
            assert embedding is not None

        # Check all embedding assignments
        assignments = db_session.exec(
            select(EmbeddingCluster).where(
                EmbeddingCluster.clustering_result_id == cr.id
            )
        ).all()

        for assignment in assignments:
            assert isinstance(assignment.embedding_id, UUID)
            assert isinstance(assignment.clustering_result_id, UUID)
            # medoid_embedding_id can be UUID or None (for outliers)
            assert assignment.medoid_embedding_id is None or isinstance(assignment.medoid_embedding_id, UUID)

            # Verify references are valid
            embedding = db_session.get(Embedding, assignment.embedding_id)
            assert embedding is not None


def test_delete_cluster_data_all(db_session):
    """Test deleting all clustering data."""
    # Create and run a simple experiment
    config = ExperimentConfig(
        networks=[["DummyT2I", "DummyI2T"]],
        seeds=[-1],
        prompts=["test delete all"],
        embedding_models=["Dummy", "Dummy2"],
        max_length=10,
    )
    db_session.add(config)
    db_session.commit()
    db_session.refresh(config)

    # Run experiment
    db_url = str(db_session.get_bind().engine.url)
    perform_experiment(str(config.id), db_url)

    # Cluster the data
    cluster_result = cluster_all_data(db_session, downsample=1)
    assert cluster_result["status"] == "success"

    # Verify clustering data exists
    clustering_results = db_session.exec(select(ClusteringResult)).all()
    assert len(clustering_results) == 2  # One for each embedding model

    # Count unique clusters across all results
    total_clusters = 0
    for cr in clustering_results:
        unique_clusters = db_session.exec(
            select(func.count(func.distinct(EmbeddingCluster.medoid_embedding_id)))
            .where(EmbeddingCluster.clustering_result_id == cr.id)
        ).one()
        total_clusters += unique_clusters
    assert total_clusters > 0

    assignments = db_session.exec(select(EmbeddingCluster)).all()
    assert len(assignments) > 0

    # Delete all clustering data
    delete_result = delete_cluster_data(db_session, "all")
    assert delete_result["status"] == "success"
    assert delete_result["deleted_results"] == 2
    assert delete_result["deleted_assignments"] > 0

    # Verify all clustering data is deleted
    clustering_results = db_session.exec(select(ClusteringResult)).all()
    assert len(clustering_results) == 0

    # No more cluster table to check

    assignments = db_session.exec(select(EmbeddingCluster)).all()
    assert len(assignments) == 0


def test_delete_cluster_data_specific_model(db_session):
    """Test deleting clustering data for a specific embedding model."""
    # Create experiment with multiple embedding models
    config = ExperimentConfig(
        networks=[["DummyT2I", "DummyI2T"]],
        seeds=[-1],
        prompts=["test delete specific"],
        embedding_models=["Dummy", "Dummy2"],
        max_length=10,
    )
    db_session.add(config)
    db_session.commit()
    db_session.refresh(config)

    # Run experiment
    db_url = str(db_session.get_bind().engine.url)
    perform_experiment(str(config.id), db_url)

    # Cluster the data
    cluster_result = cluster_all_data(db_session, downsample=1)
    assert cluster_result["status"] == "success"

    # Verify clustering data exists for both models
    clustering_results = db_session.exec(select(ClusteringResult)).all()
    assert len(clustering_results) == 2

    dummy_result = next(
        (cr for cr in clustering_results if cr.embedding_model == "Dummy"), None
    )
    dummy2_result = next(
        (cr for cr in clustering_results if cr.embedding_model == "Dummy2"), None
    )
    assert dummy_result is not None
    assert dummy2_result is not None

    # Count assignments for each model
    dummy_assignments = db_session.exec(
        select(EmbeddingCluster).where(
            EmbeddingCluster.clustering_result_id == dummy_result.id
        )
    ).all()
    dummy2_assignments = db_session.exec(
        select(EmbeddingCluster).where(
            EmbeddingCluster.clustering_result_id == dummy2_result.id
        )
    ).all()
    initial_dummy_count = len(dummy_assignments)
    initial_dummy2_count = len(dummy2_assignments)

    # Delete only Dummy model's clustering data
    delete_result = delete_cluster_data(db_session, "Dummy")
    assert delete_result["status"] == "success"
    assert delete_result["deleted_results"] == 1
    assert delete_result["deleted_assignments"] == initial_dummy_count

    # Refresh the session to see committed changes
    db_session.expire_all()

    # Verify only Dummy model's data is deleted
    clustering_results = db_session.exec(select(ClusteringResult)).all()
    assert len(clustering_results) == 1
    assert clustering_results[0].embedding_model == "Dummy2"

    # Verify Dummy2's assignments still exist
    dummy2_assignments_after = db_session.exec(
        select(EmbeddingCluster).where(
            EmbeddingCluster.clustering_result_id == dummy2_result.id
        )
    ).all()
    assert len(dummy2_assignments_after) == initial_dummy2_count


def test_delete_cluster_data_not_found(db_session):
    """Test deleting clustering data when none exists."""
    # Try to delete when no clustering data exists
    delete_result = delete_cluster_data(db_session, "all")
    assert delete_result["status"] == "not_found"
    assert delete_result["message"] == "No clustering results found for all models"

    # Try to delete for a specific model that doesn't exist
    delete_result = delete_cluster_data(db_session, "NonExistentModel")
    assert delete_result["status"] == "not_found"
    assert (
        delete_result["message"]
        == "No clustering results found for model NonExistentModel"
    )


def test_delete_single_cluster(db_session):
    """Test deleting a single clustering result."""
    # Create a simple experiment
    config = ExperimentConfig(
        networks=[["DummyT2I", "DummyI2T"]],
        seeds=[-1],
        prompts=["test delete single"],
        embedding_models=["Dummy"],
        max_length=10,
    )
    db_session.add(config)
    db_session.commit()
    db_session.refresh(config)

    # Run experiment
    db_url = str(db_session.get_bind().engine.url)
    perform_experiment(str(config.id), db_url)

    # Cluster twice to create multiple clustering results
    cluster_all_data(db_session, downsample=1)
    cluster_all_data(db_session, downsample=1)

    # Verify we have multiple clustering results
    clustering_results = db_session.exec(select(ClusteringResult)).all()
    assert len(clustering_results) >= 2

    # Pick one to delete
    result_to_delete = clustering_results[0]
    result_to_keep = clustering_results[1]

    # Count assignments for the result to delete
    assignments_to_delete = db_session.exec(
        select(EmbeddingCluster).where(
            EmbeddingCluster.clustering_result_id == result_to_delete.id
        )
    ).all()
    assignments_to_keep = db_session.exec(
        select(EmbeddingCluster).where(
            EmbeddingCluster.clustering_result_id == result_to_keep.id
        )
    ).all()
    initial_delete_count = len(assignments_to_delete)
    initial_keep_count = len(assignments_to_keep)

    # Delete the single clustering result
    delete_result = delete_single_cluster(result_to_delete.id, db_session)

    assert delete_result["status"] == "success"
    assert delete_result["deleted_results"] == 1
    assert delete_result["deleted_assignments"] == initial_delete_count

    # Refresh the session to see committed changes
    db_session.expire_all()

    # Verify the specific result is deleted
    remaining_results = db_session.exec(select(ClusteringResult)).all()
    assert len(remaining_results) == len(clustering_results) - 1
    assert result_to_delete.id not in [r.id for r in remaining_results]
    assert result_to_keep.id in [r.id for r in remaining_results]

    # Verify assignments for the kept result still exist
    kept_assignments = db_session.exec(
        select(EmbeddingCluster).where(
            EmbeddingCluster.clustering_result_id == result_to_keep.id
        )
    ).all()
    assert len(kept_assignments) == initial_keep_count

    # Verify assignments for the deleted result are gone
    deleted_assignments = db_session.exec(
        select(EmbeddingCluster).where(
            EmbeddingCluster.clustering_result_id == result_to_delete.id
        )
    ).all()
    assert len(deleted_assignments) == 0


def test_delete_single_cluster_not_found(db_session):
    """Test deleting a single clustering result that doesn't exist."""
    non_existent_id = UUID("12345678-1234-5678-1234-567812345678")
    delete_result = delete_single_cluster(non_existent_id, db_session)
    assert delete_result["status"] == "not_found"
    assert (
        delete_result["message"]
        == f"Clustering result with ID {non_existent_id} not found"
    )
