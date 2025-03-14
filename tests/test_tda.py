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
def test_giotto_phd_large_point_cloud():
    """Test the giotto_phd function on a larger point cloud."""
    # Create a larger point cloud (1000 points on a torus)
    n_points = 3000

    # Generate points on a torus
    R = 1.0  # large radius
    r = 0.3  # small radius

    # Angles
    theta = np.random.uniform(0, 2*np.pi, n_points)  # around the tube
    phi = np.random.uniform(0, 2*np.pi, n_points)    # around the centerline

    # Convert to Cartesian coordinates
    x = (R + r * np.cos(theta)) * np.cos(phi)
    y = (R + r * np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)

    point_cloud = np.column_stack((x, y, z))

    # Compute the persistence diagram
    diagrams = giotto_phd(point_cloud, max_dim=2)

    # Check that the output is a list of ndarrays
    assert isinstance(diagrams, list)

    # Check that we have diagrams for dimensions 0, 1, and 2
    assert len(diagrams) >= 3

    # Extract the 1-dimensional and 2-dimensional homology features
    h1_features = diagrams[1]
    h2_features = diagrams[2]

    # A torus should have two significant features in H1 and one in H2
    h1_persistence = h1_features[:, 1] - h1_features[:, 0] if len(h1_features) > 0 else []
    h2_persistence = h2_features[:, 1] - h2_features[:, 0] if len(h2_features) > 0 else []

    # Count significant features (with persistence > threshold)
    threshold = 0.1
    significant_h1 = np.sum(h1_persistence > threshold)
    significant_h2 = np.sum(h2_persistence > threshold)

    # Check for at least the expected topological features of a torus
    assert significant_h1 >= 2, "Should detect at least 2 significant H1 features for a torus"
    assert significant_h2 >= 1, "Should detect at least 1 significant H2 feature for a torus"
