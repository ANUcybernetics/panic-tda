from typing import List

import numpy as np
from sklearn.cluster import HDBSCAN, OPTICS


def hdbscan(embeddings: np.ndarray) -> dict:
    """
    Perform HDBSCAN clustering on a list of embeddings.

    Args:
        embeddings: ndarray of shape (n_samples, n_features)
        min_cluster_size: The minimum size of clusters
        min_samples: The number of samples in a neighborhood for a point to be considered a core point
                    (defaults to the same value as min_cluster_size if None)

    Returns:
        Dictionary containing 'labels' (cluster labels for each embedding) and 'medoids' (cluster centers)
    """
    # Calculate min_cluster_size and min_samples based on dataset size
    # For large datasets (10s of thousands), use 0.1-0.5% of dataset size
    n_samples = embeddings.shape[0]
    min_cluster_size = max(2, int(n_samples * 0.001))  # 0.1% of dataset size
    min_samples = max(2, int(n_samples * 0.001))  # same as above

    # Normalize embeddings to unit length
    # For unit vectors, Euclidean distance = sqrt(2 - 2*cos(theta))
    # This is monotonic with cosine distance, so clustering results are equivalent
    embeddings_normalized = embeddings / np.linalg.norm(
        embeddings, axis=1, keepdims=True
    )

    # Euclidean distance between unit vectors ranges from 0 to 2
    # Same as cosine distance range, so we can use the same epsilon
    cluster_selection_epsilon = 0.3

    # Configure and run HDBSCAN with Euclidean metric on normalized vectors
    # This is equivalent to cosine distance but allows using store_centers
    hdbscan = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        allow_single_cluster=True,
        metric="euclidean",
        store_centers="medoid",  # Get medoids directly from HDBSCAN
        n_jobs=-1,
    )

    # Fit the model and return the cluster labels
    hdb = hdbscan.fit(embeddings_normalized)

    # Get medoids from HDBSCAN and create a mapping from cluster label to medoid vector
    medoids = {}
    if hasattr(hdb, "medoids_") and hdb.medoids_ is not None:
        # medoids_ contains indices of medoid points for each cluster
        unique_labels = np.unique(hdb.labels_)
        unique_labels = unique_labels[unique_labels != -1]  # Remove noise label

        for i, label in enumerate(unique_labels):
            if i < len(hdb.medoids_):
                medoid_idx = hdb.medoids_[i]
                # Use original (non-normalized) embeddings for the medoid
                medoids[label] = embeddings[medoid_idx]

    # Convert to list format expected by clustering manager
    medoids_list = (
        [
            medoids.get(i, np.zeros(embeddings.shape[1]))
            for i in range(max(medoids.keys()) + 1)
        ]
        if medoids
        else []
    )
    medoids_array = (
        np.array(medoids_list) if medoids_list else np.empty((0, embeddings.shape[1]))
    )

    return {"labels": hdb.labels_, "medoids": medoids_array}


def optics(
    embeddings: np.ndarray,
    min_samples: int = 5,
    max_eps: float = np.inf,
    xi: float = 0.05,
    min_cluster_size: int = None,
) -> List[int]:
    """
    Perform OPTICS clustering on a list of embeddings.

    Args:
        embeddings: ndarray of shape (n_samples, n_features)
        min_samples: The number of samples in a neighborhood for a point to be considered a core point
        max_eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other
                (defaults to infinity)
        xi: Determines the minimum steepness on the reachability plot that constitutes a cluster boundary
        min_cluster_size: The minimum size of clusters (defaults to min_samples if None)

    Returns:
        List of cluster labels (integers) corresponding to each embedding in the input list
    """

    # Set default min_cluster_size if not provided
    if min_cluster_size is None:
        min_cluster_size = min_samples

    # Configure and run OPTICS
    optics = OPTICS(
        min_samples=min_samples,
        max_eps=max_eps,
        xi=xi,
        min_cluster_size=min_cluster_size,
        n_jobs=-1,
    )

    # Fit the model and return the cluster labels
    labels = optics.fit_predict(embeddings)

    return labels.tolist()
