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
    # NOTE n_threads needs to be fine-tuned for machine & problem sizes
    dgm = ripser_parallel(
        point_cloud, maxdim=max_dim, return_generators=True, n_threads=4
    )
    dgm["entropy"] = persistent_entropy(dgm["dgms"], normalize=False)
    return dgm
