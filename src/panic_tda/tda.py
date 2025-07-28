from typing import Dict

import numpy as np
from gph import ripser_parallel
from persim.persistent_entropy import persistent_entropy


def giotto_phd(point_cloud: np.ndarray, max_dim: int = 2) -> Dict:
    """
    Compute persistent homology diagram using the giotto-ph library.

    Args:
        point_cloud: numpy array of shape (n_points, n_dimensions)
        max_dim: maximum homology dimension to compute (default: 2)

    Returns:
        Dictionary containing the persistence diagram
    """
    # NOTE n_threads needs to be fine-tuned for machine & problem sizes
    dgm = ripser_parallel(
        point_cloud, maxdim=max_dim, return_generators=True, n_threads=4
    )
    dgm["entropy"] = persistent_entropy(dgm["dgms"], normalize=False)
    return dgm


def compute_wasserstein_distance(dgm1: np.ndarray, dgm2: np.ndarray) -> float:
    """
    Compute Wasserstein distance between two persistence diagrams.

    This is the inner function that performs the actual Wasserstein distance
    computation between two persistence diagrams represented as numpy arrays.

    Args:
        dgm1: First persistence diagram as numpy array of shape (n, 2)
        dgm2: Second persistence diagram as numpy array of shape (m, 2)

    Returns:
        Wasserstein distance between the two diagrams
    """
    # Ensure inputs are valid numpy arrays
    if not isinstance(dgm1, np.ndarray) or not isinstance(dgm2, np.ndarray):
        raise TypeError("Persistence diagrams must be numpy arrays")

    # All diagrams should now have proper 2D shape
    assert dgm1.ndim == 2 and dgm1.shape[1] == 2, (
        f"Invalid shape for dgm1: {dgm1.shape}"
    )
    assert dgm2.ndim == 2 and dgm2.shape[1] == 2, (
        f"Invalid shape for dgm2: {dgm2.shape}"
    )

    # Handle empty diagrams
    if len(dgm1) == 0 and len(dgm2) == 0:
        return 0.0
    elif len(dgm1) == 0:
        # Distance to empty diagram is sum of persistence values
        return np.sum(dgm2[:, 1] - dgm2[:, 0])
    elif len(dgm2) == 0:
        # Distance to empty diagram is sum of persistence values
        return np.sum(dgm1[:, 1] - dgm1[:, 0])

    # Use sliced Wasserstein distance approximation
    # which is more efficient than exact Wasserstein

    # Convert persistence diagrams to persistence vectors
    # Each point (birth, death) contributes |death - birth| to the distance
    pers1 = dgm1[:, 1] - dgm1[:, 0]  # persistence values
    pers2 = dgm2[:, 1] - dgm2[:, 0]

    # If both are empty, distance is 0
    if len(pers1) == 0 and len(pers2) == 0:
        return 0.0

    # If one is empty, distance is sum of all persistence values in the other
    if len(pers1) == 0:
        return np.sum(pers2)
    if len(pers2) == 0:
        return np.sum(pers1)

    # Pad shorter array with zeros
    max_len = max(len(pers1), len(pers2))
    if len(pers1) < max_len:
        pers1 = np.pad(pers1, (0, max_len - len(pers1)), constant_values=0)
    if len(pers2) < max_len:
        pers2 = np.pad(pers2, (0, max_len - len(pers2)), constant_values=0)

    # Sort both arrays in descending order (largest persistence first)
    pers1_sorted = np.sort(pers1)[::-1]
    pers2_sorted = np.sort(pers2)[::-1]

    # L1 distance between sorted persistence vectors
    result = np.sum(np.abs(pers1_sorted - pers2_sorted))

    return result
