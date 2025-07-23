from typing import List

import numpy as np
from sklearn.cluster import HDBSCAN, OPTICS


def hdbscan(embeddings: np.ndarray, epsilon: float = 0.4) -> dict:
    """
    Perform HDBSCAN clustering on a list of embeddings.

    Args:
        embeddings: ndarray of shape (n_samples, n_features)
        epsilon: The epsilon value for cluster selection (default: 0.4)
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
    cluster_selection_epsilon = epsilon

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

    # HDBSCAN returns medoid vectors directly when store_centers='medoid'
    # We'll return both the medoid vectors and their indices for easier mapping
    medoids_array = np.empty((0, embeddings.shape[1]))
    medoid_indices = {}

    if hasattr(hdb, "medoids_") and hdb.medoids_ is not None and len(hdb.medoids_) > 0:
        # Get unique cluster labels (excluding noise)
        unique_labels = np.unique(hdb.labels_)
        unique_labels = unique_labels[unique_labels != -1]

        if len(unique_labels) > 0:
            # Create array indexed by cluster label
            max_label = max(unique_labels)
            medoids_array = np.zeros((max_label + 1, embeddings.shape[1]))

            # For each cluster, find the original embedding that corresponds to the normalized medoid
            for i, label in enumerate(sorted(unique_labels)):
                if i < len(hdb.medoids_):
                    normalized_medoid = hdb.medoids_[i]
                    cluster_mask = hdb.labels_ == label
                    cluster_indices = np.where(cluster_mask)[0]

                    # Find the original embedding closest to the normalized medoid
                    cluster_embeddings_norm = embeddings_normalized[cluster_mask]
                    distances = np.sum(
                        (cluster_embeddings_norm - normalized_medoid) ** 2, axis=1
                    )
                    best_idx = np.argmin(distances)
                    original_idx = cluster_indices[best_idx]

                    # Store the original (non-normalized) embedding and its index
                    medoids_array[label] = embeddings[original_idx]
                    medoid_indices[int(label)] = int(original_idx)

    return {"labels": hdb.labels_, "medoids": medoids_array, "medoid_indices": medoid_indices}


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
