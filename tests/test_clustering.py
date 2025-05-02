import numpy as np
import pytest
from uuid_v7.base import uuid7

from panic_tda.clustering import hdbscan
from panic_tda.embeddings import EMBEDDING_DIM
from panic_tda.schemas import Embedding


def test_hdbscan_clustering():
    # Create well-defined clusters using numpy
    embeddings_obj = []
    embeddings_arr = []

    # Generate three distinct clusters with clear separation
    # Cluster 1: centered around [0.1, 0.1, ..., 0.1]
    for i in range(6):
        # Create a vector with small random variations around the center
        vector = np.ones(EMBEDDING_DIM) * 0.1 + np.random.normal(0, 0.02, EMBEDDING_DIM)
        e = Embedding(
            id=uuid7(),
            invocation_id=uuid7(),
            embedding_model="TestModel",
            vector=vector.astype(np.float32),
        )
        embeddings_obj.append(e)
        embeddings_arr.append(vector.astype(np.float32))

    # Cluster 2: centered around [0.9, 0.9, ..., 0.9]
    for i in range(7):
        # Create a vector with small random variations around the center
        vector = np.ones(EMBEDDING_DIM) * 0.9 + np.random.normal(0, 0.02, EMBEDDING_DIM)
        e = Embedding(
            id=uuid7(),
            invocation_id=uuid7(),
            embedding_model="TestModel",
            vector=vector.astype(np.float32),
        )
        embeddings_obj.append(e)
        embeddings_arr.append(vector.astype(np.float32))

    # Cluster 3: centered around [0.5, 0.5, ..., 0.5]
    for i in range(5):
        # Create a vector with small random variations around the center
        vector = np.ones(EMBEDDING_DIM) * 0.5 + np.random.normal(0, 0.02, EMBEDDING_DIM)
        e = Embedding(
            id=uuid7(),
            invocation_id=uuid7(),
            embedding_model="TestModel",
            vector=vector.astype(np.float32),
        )
        embeddings_obj.append(e)
        embeddings_arr.append(vector.astype(np.float32))

    # Add some noise points that should not belong to any cluster
    for i in range(3):
        # Create random vectors in the embedding space
        vector = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        e = Embedding(
            id=uuid7(),
            invocation_id=uuid7(),
            embedding_model="TestModel",
            vector=vector,
        )
        embeddings_obj.append(e)
        embeddings_arr.append(vector)

    # Convert to numpy array
    embeddings_arr = np.array(embeddings_arr)

    # Test with default parameters
    labels = hdbscan(embeddings_arr)

    # Check that we get the expected result type
    assert isinstance(labels, list)
    assert len(labels) == len(embeddings_arr)
    assert all(isinstance(label, int) for label in labels)

    # Check that we have at least 2 clusters (might have some noise points)
    unique_labels = set(labels)
    # Remove noise points (labeled as -1) when counting clusters
    if -1 in unique_labels:
        unique_labels.remove(-1)
    assert len(unique_labels) >= 2

    # Verify that the first 6 points are in the same cluster
    first_cluster_label = labels[0]
    assert first_cluster_label != -1  # Should not be noise
    assert all(label == first_cluster_label for label in labels[1:6])

    # Verify that the next 7 points are in the same cluster
    second_cluster_label = labels[6]
    assert second_cluster_label != -1  # Should not be noise
    assert (
        second_cluster_label != first_cluster_label
    )  # Should be different from first cluster
    assert all(label == second_cluster_label for label in labels[7:13])

    # Check with custom parameters
    labels_custom = hdbscan(embeddings_arr, min_cluster_size=3, min_samples=2)
    assert len(labels_custom) == len(embeddings_arr)


@pytest.mark.skip(reason="This is flaky, and the whole approach needs to be rethought")
def test_hdbscan_outlier_detection():
    """Test that HDBSCAN correctly identifies outliers as noise points."""
    embeddings_obj = []
    embeddings_arr = []
    np.random.seed(42)  # Set seed for reproducibility

    # Create a single well-defined cluster with stronger cohesion
    cluster_center = np.ones(EMBEDDING_DIM) * 0.5

    # Create one well-defined, very tight cluster
    for i in range(10):
        # Use much smaller standard deviation to create tighter cluster
        vector = cluster_center + np.random.normal(0, 0.005, EMBEDDING_DIM)
        e = Embedding(
            id=uuid7(),
            invocation_id=uuid7(),
            embedding_model="TestModel",
            vector=vector.astype(np.float32),
        )
        embeddings_obj.append(e)
        embeddings_arr.append(vector.astype(np.float32))

    # Add several clear outliers at very distant points
    outlier_indices = []
    outlier_positions = [
        np.ones(EMBEDDING_DIM) * 0.9,  # Far away
        np.zeros(EMBEDDING_DIM),  # At origin
        np.ones(EMBEDDING_DIM) * (-0.9),  # Very negative space
    ]

    for i, pos in enumerate(outlier_positions):
        vector = pos + np.random.normal(0, 0.005, EMBEDDING_DIM)  # Smaller variation
        e = Embedding(
            id=uuid7(),
            invocation_id=uuid7(),
            embedding_model="TestModel",
            vector=vector.astype(np.float32),
        )
        embeddings_obj.append(e)
        embeddings_arr.append(vector.astype(np.float32))
        outlier_indices.append(10 + i)  # Calculate index in the array

    # Convert to numpy array
    embeddings_arr = np.array(embeddings_arr)

    # Run clustering with parameters better suited for test data
    labels = hdbscan(
        embeddings_arr, min_cluster_size=3, min_samples=2
    )  # More relaxed parameters

    # Verify that the outliers are labeled as noise (-1)
    for idx in outlier_indices:
        assert labels[idx] == -1, f"Outlier at index {idx} was not identified as noise"

    # Verify that the cluster is properly identified
    cluster_labels = [labels[i] for i in range(10)]
    # All cluster points should have the same non-noise label
    assert len(set(cluster_labels)) == 1, (
        f"Found multiple cluster labels: {set(cluster_labels)}"
    )
    assert set(cluster_labels).pop() != -1, (
        "Cluster points were incorrectly labeled as noise"
    )

    # Test with more lenient parameters - should still identify outliers
    lenient_labels = hdbscan(embeddings_arr, min_cluster_size=2, min_samples=1)
    for idx in outlier_indices:
        assert lenient_labels[idx] == -1, (
            "Outlier should be noise even with lenient parameters"
        )
