import numpy as np

from trajectory_tracer.tda import giotto_phd


def test_giotto_phd():
    """Test the giotto_phd function for computing persistent homology diagrams."""
    # Create a simple point cloud (a circle-like shape)
    n_points = 30
    theta = np.linspace(0, 2*np.pi, n_points)
    point_cloud = np.column_stack((np.cos(theta), np.sin(theta)))

    # Compute the persistence diagram
    diagram = giotto_phd(point_cloud, max_dim=1)

    # Check that the output is a dictionary
    assert isinstance(diagram, dict)

    # Check that the diagram contains expected keys
    assert "dgms" in diagram

    # Check that we have diagrams for dimensions 0 and 1
    assert len(diagram["dgms"]) >= 2

    # Extract the 1-dimensional homology features (the circle)
    h1_features = diagram["dgms"][1]

    # There should be at least one persistent feature in H1 (the circle)
    assert len(h1_features) >= 1

    # The most persistent feature should have a reasonable lifespan
    # For a circle, there should be one feature with significant persistence
    persistence_values = h1_features[:, 1] - h1_features[:, 0]
    max_persistence = np.max(persistence_values)
    assert max_persistence > 0.5  # The circle should be clearly detected
