import json
from typing import Dict, List

import numpy as np
import polars as pl
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
    cluster_selection_epsilon = 0.75

    # Configure and run HDBSCAN
    hdbscan = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        allow_single_cluster=True,
        store_centers="medoid",
        n_jobs=-1,
    )

    # Fit the model and return the cluster labels
    hdb = hdbscan.fit(embeddings)

    return {"labels": hdb.labels_, "medoids": hdb.medoids_}


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


def create_label_map(
    cluster_labels: pl.Series,
    output_path: str = "cluster_label_map.json",
    cache: bool = True,
) -> Dict[str, int]:
    """
    Map string cluster labels to integers and optionally cache the mapping to a JSON file.

    Args:
        cluster_labels: A polars Series of string labels
        output_path: Path to save the JSON mapping file
        cache: Whether to write the mapping to a JSON file

    Returns:
        Dictionary mapping string labels to integers (-1 for "OUTLIER", 1+ for others)
    """
    # Get unique labels
    unique_labels = cluster_labels.unique().to_list()

    # Separate OUTLIER from other labels and sort the rest
    other_labels = sorted([
        label for label in unique_labels if label is not None and label != "OUTLIER"
    ])

    # Create mapping dictionary
    label_map = {}

    # Handle OUTLIER first if it exists
    if "OUTLIER" in unique_labels:
        label_map["OUTLIER"] = -1

    # Assign IDs to sorted labels
    next_id = 1
    for label in other_labels:
        label_map[label] = next_id
        next_id += 1

    # Write mapping to JSON file if cache is True
    if cache:
        with open(output_path, "w") as f:
            json.dump(label_map, f, indent=2)

    return label_map


def read_labels_from_cache(
    input_path: str = "cluster_label_map.json",
) -> Dict[str, int]:
    """
    Read the mapping of string cluster labels to integers from a JSON file.

    Args:
        input_path: Path to the JSON mapping file

    Returns:
        Dictionary mapping string labels to integers (-1 for "OUTLIER", 1+ for others)
    """
    # Read mapping from JSON file
    with open(input_path, "r") as f:
        label_map = json.load(f)

    return label_map
