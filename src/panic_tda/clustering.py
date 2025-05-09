from typing import List

import numpy as np
from sklearn.cluster import HDBSCAN, OPTICS


def hdbscan(embeddings: np.ndarray) -> List[int]:
    """
    Perform HDBSCAN clustering on a list of embeddings.

    Args:
        embeddings: ndarray of shape (n_samples, n_features)
        min_cluster_size: The minimum size of clusters
        min_samples: The number of samples in a neighborhood for a point to be considered a core point
                    (defaults to the same value as min_cluster_size if None)

    Returns:
        List of cluster labels (integers) corresponding to each embedding in the input list
    """
    # Calculate min_cluster_size and min_samples based on dataset size
    # For large datasets (10s of thousands), use 0.1-0.5% of dataset size
    n_samples = embeddings.shape[0]
    min_cluster_size = max(2, int(n_samples * 0.001))  # 0.1% of dataset size
    min_samples = max(2, int(n_samples * 0.0015))  # 0.15% of dataset size

    # Configure and run HDBSCAN
    hdbscan = HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples, n_jobs=-1
    )

    # Fit the model and return the cluster labels
    labels = hdbscan.fit_predict(embeddings)

    return labels.tolist()


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
