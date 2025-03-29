from typing import List

import numpy as np
from sklearn.cluster import HDBSCAN

from trajectory_tracer.schemas import Embedding

def hdbscan(embeddings: List[Embedding], min_cluster_size: int = 5, min_samples: int = None) -> List[int]:
    """
    Perform HDBSCAN clustering on a list of embeddings.

    Args:
        embeddings: List of Embedding objects to cluster
        min_cluster_size: The minimum size of clusters
        min_samples: The number of samples in a neighborhood for a point to be considered a core point
                    (defaults to the same value as min_cluster_size if None)

    Returns:
        List of cluster labels (integers) corresponding to each embedding in the input list
    """

    # Extract vectors from embeddings
    vectors = np.array([embedding.vector for embedding in embeddings])

    # Configure and run HDBSCAN
    hdbscan = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )

    # Fit the model and return the cluster labels
    labels = hdbscan.fit_predict(vectors)

    return labels.tolist()
