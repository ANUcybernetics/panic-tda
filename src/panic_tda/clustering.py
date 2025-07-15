from typing import List

import numpy as np
from sklearn.cluster import HDBSCAN, OPTICS
from sklearn.metrics.pairwise import cosine_distances


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
    # Cosine distance ranges from 0 to 2, so we need a smaller epsilon
    cluster_selection_epsilon = 0.3

    # Compute cosine distance matrix
    distance_matrix = cosine_distances(embeddings)

    # Configure and run HDBSCAN without store_centers since precomputed distances don't support it
    hdbscan = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        allow_single_cluster=True,
        metric="precomputed",
        n_jobs=-1,
    )

    # Fit the model and return the cluster labels
    hdb = hdbscan.fit(distance_matrix)
    
    # Compute medoids manually for each cluster using cosine distances
    unique_labels = np.unique(hdb.labels_)
    unique_labels = unique_labels[unique_labels != -1]  # Remove noise label
    
    medoids = []
    for label in unique_labels:
        # Get indices of points in this cluster
        cluster_indices = np.where(hdb.labels_ == label)[0]
        
        # Get the distance submatrix for this cluster
        cluster_distances = distance_matrix[np.ix_(cluster_indices, cluster_indices)]
        
        # Find the point with minimum sum of distances to all other points in the cluster
        medoid_idx_in_cluster = np.argmin(cluster_distances.sum(axis=1))
        medoid_idx = cluster_indices[medoid_idx_in_cluster]
        
        # Get the actual embedding vector for the medoid
        medoids.append(embeddings[medoid_idx])
    
    medoids = np.array(medoids) if medoids else np.empty((0, embeddings.shape[1]))

    return {"labels": hdb.labels_, "medoids": medoids}


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
