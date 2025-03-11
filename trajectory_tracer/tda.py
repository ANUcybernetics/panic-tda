from typing import Any, Dict

import numpy as np
from gph import ripser_parallel


def giotto_phd(point_cloud: np.ndarray, max_dim: int = 2) -> Dict[str, Any]:
    """
    Compute persistent homology diagram using the giotto-ph library.

    Args:
        point_cloud: numpy array of shape (n_points, n_dimensions)
        max_dim: maximum homology dimension to compute (default: 2)

    Returns:
        Dictionary containing the persistence diagram
    """
    return ripser_parallel(point_cloud, maxdim=max_dim, n_threads=-1)
