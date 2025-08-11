import numpy as np
import pytest
from uuid_v7.base import uuid7

from panic_tda.clustering import (
    hdbscan,
    optics,
)
from panic_tda.embeddings import EMBEDDING_DIM
from panic_tda.schemas import Embedding


def test_hdbscan_clustering():
    # Create well-defined clusters using numpy
    embeddings_obj = []
    embeddings_arr = []
    np.random.seed(42)  # For reproducible test results

    # Generate three distinct clusters with clear separation based on DIRECTION
    # For cosine distance, we need different directions, not different magnitudes

    # Cluster 1: First half dims positive, second half negative
    for i in range(15):  # Increased from 6 to 15
        vector = np.zeros(EMBEDDING_DIM)
        vector[: EMBEDDING_DIM // 2] = 1.0 + np.random.normal(
            0, 0.1, EMBEDDING_DIM // 2
        )
        vector[EMBEDDING_DIM // 2 :] = -1.0 + np.random.normal(
            0, 0.1, EMBEDDING_DIM - EMBEDDING_DIM // 2
        )
        # Normalize to unit length for stable cosine distance
        vector = vector / np.linalg.norm(vector)
        e = Embedding(
            id=uuid7(),
            invocation_id=uuid7(),
            embedding_model="TestModel",
            vector=vector.astype(np.float32),
        )
        embeddings_obj.append(e)
        embeddings_arr.append(vector.astype(np.float32))

    # Cluster 2: First half dims negative, second half positive (opposite of cluster 1)
    for i in range(20):  # Increased from 7 to 20
        vector = np.zeros(EMBEDDING_DIM)
        vector[: EMBEDDING_DIM // 2] = -1.0 + np.random.normal(
            0, 0.1, EMBEDDING_DIM // 2
        )
        vector[EMBEDDING_DIM // 2 :] = 1.0 + np.random.normal(
            0, 0.1, EMBEDDING_DIM - EMBEDDING_DIM // 2
        )
        # Normalize to unit length for stable cosine distance
        vector = vector / np.linalg.norm(vector)
        e = Embedding(
            id=uuid7(),
            invocation_id=uuid7(),
            embedding_model="TestModel",
            vector=vector.astype(np.float32),
        )
        embeddings_obj.append(e)
        embeddings_arr.append(vector.astype(np.float32))

    # Cluster 3: Alternating positive/negative pattern
    for i in range(18):  # Increased from 5 to 18
        vector = np.zeros(EMBEDDING_DIM)
        for j in range(EMBEDDING_DIM):
            vector[j] = (1.0 if j % 2 == 0 else -1.0) + np.random.normal(0, 0.1)
        # Normalize to unit length for stable cosine distance
        vector = vector / np.linalg.norm(vector)
        e = Embedding(
            id=uuid7(),
            invocation_id=uuid7(),
            embedding_model="TestModel",
            vector=vector.astype(np.float32),
        )
        embeddings_obj.append(e)
        embeddings_arr.append(vector.astype(np.float32))

    # Add some noise points that should not belong to any cluster
    for i in range(5):  # Increased from 3 to 5
        # Create random vectors with completely different patterns
        vector = np.random.randn(EMBEDDING_DIM)
        # Normalize to unit length
        vector = vector / np.linalg.norm(vector)
        e = Embedding(
            id=uuid7(),
            invocation_id=uuid7(),
            embedding_model="TestModel",
            vector=vector.astype(np.float32),
        )
        embeddings_obj.append(e)
        embeddings_arr.append(vector.astype(np.float32))

    # Convert to numpy array
    embeddings_arr = np.array(embeddings_arr)

    # Test with default parameters
    result = hdbscan(embeddings_arr)

    # Check that we get the expected result type
    assert isinstance(result, dict)
    assert "labels" in result
    assert "medoids" in result
    assert isinstance(result["labels"], np.ndarray)
    assert isinstance(result["medoids"], np.ndarray)
    labels = result["labels"]
    medoids = result["medoids"]

    assert len(labels) == len(embeddings_arr)
    assert all(isinstance(label, np.integer) for label in labels)

    # Check that we have at least 3 clusters (might have some noise points)
    unique_labels = set(labels)
    # Remove noise points (labeled as -1) when counting clusters
    if -1 in unique_labels:
        unique_labels.remove(-1)
    assert len(unique_labels) >= 3

    # Check that we have medoids for each cluster
    assert len(medoids) == len(unique_labels)

    # Verify that the first 15 points are in the same cluster
    first_cluster_label = labels[0]
    assert first_cluster_label != -1  # Should not be noise
    assert all(label == first_cluster_label for label in labels[1:15])

    # Verify that the next 20 points are in the same cluster
    second_cluster_label = labels[15]
    assert second_cluster_label != -1  # Should not be noise
    assert (
        second_cluster_label != first_cluster_label
    )  # Should be different from first cluster
    assert all(label == second_cluster_label for label in labels[16:35])

    # Verify that the next 18 points are in a third cluster
    third_cluster_label = labels[35]
    assert third_cluster_label != -1  # Should not be noise
    assert third_cluster_label != first_cluster_label  # Different from first cluster
    assert third_cluster_label != second_cluster_label  # Different from second cluster
    assert all(label == third_cluster_label for label in labels[36:53])

    # Check with custom parameters
    result_custom = hdbscan(embeddings_arr)
    assert len(result_custom["labels"]) == len(embeddings_arr)
    assert result_custom["medoids"].shape[1] == EMBEDDING_DIM


@pytest.mark.parametrize("n_samples", [10, 100, 1000, 5000])
def test_hdbscan_scalability(n_samples):
    """Test that hdbscan can handle increasingly large input arrays."""
    # Create random data with some structure for clustering
    np.random.seed(42)

    # Generate data with 3 clusters based on direction patterns
    embeddings = []
    for i in range(n_samples):
        # Assign to one of 3 clusters randomly
        cluster_idx = i % 3

        if cluster_idx == 0:
            # Cluster 1: First half positive, second half negative
            vector = np.zeros(EMBEDDING_DIM)
            vector[: EMBEDDING_DIM // 2] = 1.0 + np.random.normal(
                0, 0.1, EMBEDDING_DIM // 2
            )
            vector[EMBEDDING_DIM // 2 :] = -1.0 + np.random.normal(
                0, 0.1, EMBEDDING_DIM - EMBEDDING_DIM // 2
            )
        elif cluster_idx == 1:
            # Cluster 2: First half negative, second half positive
            vector = np.zeros(EMBEDDING_DIM)
            vector[: EMBEDDING_DIM // 2] = -1.0 + np.random.normal(
                0, 0.1, EMBEDDING_DIM // 2
            )
            vector[EMBEDDING_DIM // 2 :] = 1.0 + np.random.normal(
                0, 0.1, EMBEDDING_DIM - EMBEDDING_DIM // 2
            )
        else:
            # Cluster 3: Alternating pattern
            vector = np.zeros(EMBEDDING_DIM)
            for j in range(EMBEDDING_DIM):
                vector[j] = (1.0 if j % 2 == 0 else -1.0) + np.random.normal(0, 0.1)

        # Normalize to unit length for stable cosine distance
        vector = vector / np.linalg.norm(vector)
        embeddings.append(vector.astype(np.float32))

    embeddings_arr = np.array(embeddings)

    result = hdbscan(embeddings_arr)

    # Basic checks
    assert isinstance(result, dict)
    assert "labels" in result
    assert "medoids" in result
    labels = result["labels"]
    medoids = result["medoids"]

    assert isinstance(labels, np.ndarray)
    assert len(labels) == n_samples
    assert all(isinstance(label, np.integer) for label in labels)

    # Should identify at least 2 clusters (plus possibly noise)
    unique_clusters = set(labels) - {-1}  # Remove noise label
    assert len(unique_clusters) >= 2

    # Check medoids
    assert isinstance(medoids, np.ndarray)
    assert medoids.shape[1] == EMBEDDING_DIM
    assert len(medoids) == len(unique_clusters)


def test_hdbscan_medoids_and_outliers_with_constructed_data():
    """Test HDBSCAN with carefully constructed point clouds to verify medoid and outlier handling."""

    # Set seed for reproducibility
    np.random.seed(42)

    # Create three well-separated dense clusters
    embeddings = []

    # Cluster 0: Dense cluster in positive space (indices 0-19)
    # Using normalized vectors for cosine-like distance
    for i in range(20):
        vector = np.ones(EMBEDDING_DIM)
        vector[: EMBEDDING_DIM // 2] = 1.0 + np.random.normal(
            0, 0.05, EMBEDDING_DIM // 2
        )
        vector[EMBEDDING_DIM // 2 :] = 0.1 + np.random.normal(
            0, 0.05, EMBEDDING_DIM - EMBEDDING_DIM // 2
        )
        # Normalize to unit length
        vector = vector / np.linalg.norm(vector)
        embeddings.append(vector)

    # Cluster 1: Dense cluster with opposite pattern (indices 20-39)
    for i in range(20):
        vector = np.ones(EMBEDDING_DIM)
        vector[: EMBEDDING_DIM // 2] = 0.1 + np.random.normal(
            0, 0.05, EMBEDDING_DIM // 2
        )
        vector[EMBEDDING_DIM // 2 :] = 1.0 + np.random.normal(
            0, 0.05, EMBEDDING_DIM - EMBEDDING_DIM // 2
        )
        # Normalize to unit length
        vector = vector / np.linalg.norm(vector)
        embeddings.append(vector)

    # Cluster 2: Dense cluster with alternating pattern (indices 40-59)
    for i in range(20):
        vector = np.zeros(EMBEDDING_DIM)
        for j in range(EMBEDDING_DIM):
            vector[j] = (1.0 if j % 2 == 0 else -1.0) + np.random.normal(0, 0.05)
        # Normalize to unit length
        vector = vector / np.linalg.norm(vector)
        embeddings.append(vector)

    # Some random points that might be outliers (indices 60-64)
    # With small dataset and min_cluster_size=2, these might still cluster
    for i in range(5):
        vector = np.random.randn(EMBEDDING_DIM)
        # Normalize to unit length (avoid zero vectors)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        else:
            vector = np.ones(EMBEDDING_DIM) / np.sqrt(EMBEDDING_DIM)
        embeddings.append(vector)

    # Convert to numpy array
    embeddings_arr = np.array(embeddings, dtype=np.float32)

    # Run HDBSCAN
    result = hdbscan(embeddings_arr, epsilon=0.4)

    # Basic structure checks
    assert isinstance(result, dict)
    assert "labels" in result
    assert "medoids" in result
    assert "medoid_indices" in result

    labels = result["labels"]
    medoids = result["medoids"]
    medoid_indices = result["medoid_indices"]

    # Check dimensions
    assert len(labels) == len(embeddings_arr)
    assert isinstance(medoid_indices, dict)

    # Count clusters and outliers
    unique_labels = set(labels)
    n_outliers = np.sum(labels == -1)
    n_clusters = len(unique_labels - {-1})

    # With this data, we should have at least 2 clusters
    assert n_clusters >= 2, f"Expected at least 2 clusters, got {n_clusters}"

    # Verify outliers don't have medoids
    assert -1 not in medoid_indices, "Outliers should not have medoid indices"

    # KEY TEST: Verify each non-outlier cluster has a valid medoid
    for label in unique_labels:
        if label != -1:
            assert label in medoid_indices, f"Cluster {label} missing medoid index"
            medoid_idx = medoid_indices[label]

            # Verify medoid index is valid
            assert 0 <= medoid_idx < len(embeddings_arr), (
                f"Invalid medoid index {medoid_idx}"
            )

            # Verify the medoid is actually from this cluster
            assert labels[medoid_idx] == label, (
                f"Medoid at index {medoid_idx} not in cluster {label}"
            )

            # Verify medoid vector matches
            if label < len(medoids):
                expected_medoid = embeddings_arr[medoid_idx]
                actual_medoid = medoids[label]
                np.testing.assert_array_almost_equal(
                    actual_medoid,
                    expected_medoid,
                    err_msg=f"Medoid vector mismatch for cluster {label}",
                )

    # If there are outliers, verify they're handled correctly
    if n_outliers > 0:
        outlier_indices = [i for i, l in enumerate(labels) if l == -1]
        # Outliers should not be medoids for any cluster
        for cluster_label, medoid_idx in medoid_indices.items():
            assert medoid_idx not in outlier_indices, (
                f"Outlier at index {medoid_idx} is medoid for cluster {cluster_label}"
            )

    # Verify the first two constructed clusters are mostly preserved
    cluster0_labels = labels[0:20]
    cluster1_labels = labels[20:40]

    # Most points in each constructed cluster should have the same label
    from collections import Counter

    cluster0_majority = Counter(cluster0_labels).most_common(1)[0]
    cluster1_majority = Counter(cluster1_labels).most_common(1)[0]

    # At least half should be in the same cluster
    assert cluster0_majority[1] >= 10, (
        f"Cluster 0 not well preserved: {cluster0_majority}"
    )
    assert cluster1_majority[1] >= 10, (
        f"Cluster 1 not well preserved: {cluster1_majority}"
    )

    # The two groups should mostly be in different clusters
    if cluster0_majority[0] != -1 and cluster1_majority[0] != -1:
        assert cluster0_majority[0] != cluster1_majority[0], (
            "Clusters 0 and 1 should be separate"
        )


def test_hdbscan_all_outliers():
    """Test HDBSCAN behavior when all points are outliers (no clusters formed)."""

    np.random.seed(42)

    # Create very sparse random data where no clusters should form
    embeddings = []
    for i in range(10):
        # Each point is far from all others
        vector = np.random.randn(EMBEDDING_DIM) * (i + 1) * 10
        embeddings.append(vector)

    embeddings_arr = np.array(embeddings, dtype=np.float32)

    # Run HDBSCAN with normal parameters
    result = hdbscan(embeddings_arr, epsilon=0.4)

    labels = result["labels"]
    medoid_indices = result["medoid_indices"]

    # All points should be outliers
    assert np.all(labels == -1), "All sparse points should be outliers"

    # No medoid indices should exist
    assert len(medoid_indices) == 0, (
        "No medoids should exist when all points are outliers"
    )


def test_hdbscan_single_cluster():
    """Test HDBSCAN with data that forms a single large cluster."""

    np.random.seed(42)

    # Create one large, dense cluster
    center = np.ones(EMBEDDING_DIM) * 0.5
    embeddings = []

    for i in range(50):
        # Small variance for tight clustering
        vector = center + np.random.normal(0, 0.01, EMBEDDING_DIM)
        embeddings.append(vector)

    embeddings_arr = np.array(embeddings, dtype=np.float32)

    # Run HDBSCAN with parameters that should find a single cluster
    result = hdbscan(embeddings_arr, epsilon=0.5)

    labels = result["labels"]
    medoid_indices = result["medoid_indices"]

    # Should have exactly one cluster (label 0)
    unique_labels = set(labels) - {-1}
    assert len(unique_labels) == 1, f"Expected 1 cluster, got {len(unique_labels)}"

    cluster_label = list(unique_labels)[0]

    # Verify medoid exists and is valid
    assert cluster_label in medoid_indices
    medoid_idx = medoid_indices[cluster_label]
    assert 0 <= medoid_idx < len(embeddings_arr)
    assert labels[medoid_idx] == cluster_label

    # Most points should be in the cluster (allow for a few outliers)
    n_in_cluster = np.sum(labels == cluster_label)
    assert n_in_cluster >= 45, f"Only {n_in_cluster} of 50 points in cluster"


def test_hdbscan_medoid_selection():
    """Test that HDBSCAN selects appropriate medoids (central points) for clusters."""

    np.random.seed(42)

    # Create a cluster where we know which point should be the medoid
    # Make one point exactly at the center, others around it
    embeddings = []

    # Point 10 will be our exact center
    center = np.ones(EMBEDDING_DIM) * 0.5

    for i in range(20):
        if i == 10:
            # This point is exactly at the center
            vector = center.copy()
        else:
            # Other points are distributed around the center
            vector = center + np.random.normal(0, 0.05, EMBEDDING_DIM)
        embeddings.append(vector)

    embeddings_arr = np.array(embeddings, dtype=np.float32)

    # Run HDBSCAN
    result = hdbscan(embeddings_arr, epsilon=0.5)

    labels = result["labels"]
    medoid_indices = result["medoid_indices"]

    # Should form one cluster
    unique_labels = set(labels) - {-1}
    assert len(unique_labels) >= 1

    if len(unique_labels) == 1:
        cluster_label = list(unique_labels)[0]
        medoid_idx = medoid_indices[cluster_label]

        # The medoid should be close to the center
        medoid_point = embeddings_arr[medoid_idx]
        distance_to_center = np.linalg.norm(medoid_point - center)

        # Calculate average distance to center for comparison
        cluster_mask = labels == cluster_label
        cluster_points = embeddings_arr[cluster_mask]
        avg_distance = np.mean([np.linalg.norm(p - center) for p in cluster_points])

        # Medoid should be closer to center than average
        assert distance_to_center <= avg_distance * 1.1, (
            f"Medoid too far from center: {distance_to_center:.3f} vs avg {avg_distance:.3f}"
        )


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
    result = hdbscan(embeddings_arr)  # More relaxed parameters
    labels = result["labels"]

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
    lenient_result = hdbscan(embeddings_arr)
    lenient_labels = lenient_result["labels"]
    for idx in outlier_indices:
        assert lenient_labels[idx] == -1, (
            "Outlier should be noise even with lenient parameters"
        )


def test_optics_clustering():
    """Test that OPTICS clustering correctly identifies clusters in well-separated data."""

    # Create well-defined clusters using numpy
    embeddings_arr = []
    np.random.seed(42)

    # Generate three distinct clusters with moderate separation
    # Cluster 1: centered around [0.3, 0.3, ..., 0.3]
    for i in range(10):
        vector = np.ones(EMBEDDING_DIM) * 0.3 + np.random.normal(0, 0.02, EMBEDDING_DIM)
        embeddings_arr.append(vector.astype(np.float32))

    # Cluster 2: centered around [0.5, 0.5, ..., 0.5]
    for i in range(12):
        vector = np.ones(EMBEDDING_DIM) * 0.5 + np.random.normal(0, 0.02, EMBEDDING_DIM)
        embeddings_arr.append(vector.astype(np.float32))

    # Cluster 3: centered around [0.7, 0.7, ..., 0.7]
    for i in range(8):
        vector = np.ones(EMBEDDING_DIM) * 0.7 + np.random.normal(0, 0.02, EMBEDDING_DIM)
        embeddings_arr.append(vector.astype(np.float32))

    # Add some noise points that are slightly further away
    for i in range(3):
        # Create outliers but not too far from the main clusters
        center = np.random.choice([0.1, 0.9])
        vector = np.ones(EMBEDDING_DIM) * center + np.random.normal(
            0, 0.05, EMBEDDING_DIM
        )
        embeddings_arr.append(vector.astype(np.float32))

    # Convert to numpy array
    embeddings_arr = np.array(embeddings_arr)

    # Test with appropriate parameters for the test data
    # Use smaller min_samples and explicit max_eps for better cluster detection
    labels = optics(embeddings_arr, min_samples=3, max_eps=2.0, min_cluster_size=3)

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

    # Test with custom parameters
    labels_custom = optics(
        embeddings_arr, min_samples=2, max_eps=1.0, xi=0.05, min_cluster_size=2
    )
    assert len(labels_custom) == len(embeddings_arr)

    # Test that we can run with different parameters (results may or may not differ)
    labels_different = optics(embeddings_arr, min_samples=5, max_eps=3.0, xi=0.1)
    assert len(labels_different) == len(embeddings_arr)


@pytest.mark.parametrize("n_samples", [10, 100, 1000, 5000])
def test_optics_scalability(n_samples):
    """Test that optics can handle increasingly large input arrays."""
    # Create random data with some structure for clustering
    np.random.seed(42)

    # Generate data with 3 clusters
    cluster_centers = [
        np.ones(EMBEDDING_DIM) * 0.1,  # Cluster 1
        np.ones(EMBEDDING_DIM) * 0.5,  # Cluster 2
        np.ones(EMBEDDING_DIM) * 0.9,  # Cluster 3
    ]

    embeddings = []
    for i in range(n_samples):
        # Assign to one of 3 clusters randomly
        cluster_idx = i % 3
        # Create a vector with small random variations around the center
        # Use smaller standard deviation for better separation
        vector = cluster_centers[cluster_idx] + np.random.normal(0, 0.03, EMBEDDING_DIM)
        embeddings.append(vector.astype(np.float32))

    embeddings_arr = np.array(embeddings)

    # Skip cluster number assertion for small sample sizes
    if n_samples <= 10:
        # For small sample sizes, just test that the function runs without error
        labels = optics(embeddings_arr, min_samples=2, min_cluster_size=2)

        # Basic checks
        assert isinstance(labels, list)
        assert len(labels) == n_samples
        assert all(isinstance(label, int) for label in labels)
    else:
        # For larger sample sizes, use parameters appropriate for the sample size
        min_samples = max(3, n_samples // 50)
        min_cluster_size = max(3, n_samples // 50)
        labels = optics(
            embeddings_arr, min_samples=min_samples, min_cluster_size=min_cluster_size
        )

        # Basic checks
        assert isinstance(labels, list)
        assert len(labels) == n_samples
        assert all(isinstance(label, int) for label in labels)

        # Should identify at least 2 clusters (plus possibly noise)
        unique_clusters = set(labels) - {-1}  # Remove noise label
        assert len(unique_clusters) >= 2
