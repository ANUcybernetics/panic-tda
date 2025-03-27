from typing import List

import numpy as np
from gph import ripser_parallel
from persim.persistent_entropy import persistent_entropy


def giotto_phd(point_cloud: np.ndarray, max_dim: int = 2) -> List[np.ndarray]:
    """
    Compute persistent homology diagram using the giotto-ph library.

    Args:
        point_cloud: numpy array of shape (n_points, n_dimensions)
        max_dim: maximum homology dimension to compute (default: 2)

    Returns:
        Dictionary containing the persistence diagram
    """
    dgm = ripser_parallel(point_cloud, maxdim=max_dim, n_threads=-1)
    _pe = persistent_entropy(dgm["dgms"], normalize=False)
    return dgm["dgms"]
