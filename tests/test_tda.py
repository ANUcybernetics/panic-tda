import numpy as np
import pytest

from trajectory_tracer.tda import giotto_phd


def test_giotto_phd():
    """Test the giotto_phd function for computing persistent homology diagrams."""
    # Create a simple point cloud (a circle-like shape)
    n_points = 30
    theta = np.linspace(0, 2*np.pi, n_points)
    point_cloud = np.column_stack((np.cos(theta), np.sin(theta)))

    # Compute the persistence diagram
    diagrams = giotto_phd(point_cloud, max_dim=1)

    # Check that the output is a list of ndarrays
    assert isinstance(diagrams, list)

    # Check that we have diagrams for dimensions 0 and 1
    assert len(diagrams) >= 2

    # Extract the 1-dimensional homology features (the circle)
    h1_features = diagrams[1]

    # There should be at least one persistent feature in H1 (the circle)
    assert len(h1_features) >= 1

    # The most persistent feature should have a reasonable lifespan
    # For a circle, there should be one feature with significant persistence
    persistence_values = h1_features[:, 1] - h1_features[:, 0]
    max_persistence = np.max(persistence_values)
    assert max_persistence > 0.5  # The circle should be clearly detected


@pytest.mark.slow
@pytest.mark.parametrize("n_points", range(1000, 11000, 1000))
def test_giotto_phd_large_point_cloud(benchmark, n_points):
    """Test the giotto_phd function on larger point clouds of increasing size."""
    n_dims = 768

    # Generate random points in 768-dimensional space
    point_cloud = np.random.normal(0, 1, (n_points, n_dims))

    # Benchmark the computation of persistence diagrams with increasing point cloud sizes
    diagrams = benchmark(lambda: giotto_phd(point_cloud, max_dim=2))

    # Check that the output is a list of ndarrays
    assert isinstance(diagrams, list)

    # Check that we have diagrams for dimensions 0, 1, and 2
    assert len(diagrams) >= 3

    # Extract the 1-dimensional and 2-dimensional homology features
    h1_features = diagrams[1]
    h2_features = diagrams[2]

    # Calculate persistence values
    h1_persistence = h1_features[:, 1] - h1_features[:, 0] if len(h1_features) > 0 else []
    h2_persistence = h2_features[:, 1] - h2_features[:, 0] if len(h2_features) > 0 else []

    # Log the size of the point cloud
    print(f"Processed point cloud with {n_points} points in {n_dims} dimensions")
