import numpy as np
import pytest
from gtda.plotting import plot_diagram

from trajectory_tracer.tda import giotto_phd


def test_giotto_phd():
    """Test the giotto_phd function for computing persistent homology diagrams."""
    # Create a simple point cloud (a circle-like shape)
    n_points = 100
    theta = np.linspace(0, 2 * np.pi, n_points)
    point_cloud = np.column_stack((np.cos(theta), np.sin(theta)))

    # Compute the persistence diagram
    result = giotto_phd(point_cloud, max_dim=2)
    print(result)

    # Check that the output is a dictionary
    assert isinstance(result, dict)

    # Check that the diagrams are in the "dgms" key
    assert "dgms" in result
    diagrams = result["dgms"]

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

    # Check that entropy was computed
    assert "entropy" in result


def test_plot_diagram():
    """Test plotting a persistence diagram."""
    # Create a simple point cloud (a circle-like shape)
    n_points = 100
    theta = np.linspace(0, 2 * np.pi, n_points)
    point_cloud = np.column_stack((np.cos(theta), np.sin(theta)))

    # Compute the persistence diagram
    result = giotto_phd(point_cloud, max_dim=2)
    diagrams = result["dgms"]

    # Import the plotting function

    # Plot the 1-dimensional homology features (the circle)
    h1_diagram = diagrams[1]

    # Add homology dimension as third column
    h1_diagram_with_dim = np.column_stack((h1_diagram, np.ones(h1_diagram.shape[0])))

    # Create the plot
    fig = plot_diagram(h1_diagram_with_dim)

    # Assert the figure is created
    assert fig is not None

    # Optional: customize the plot
    fig.update_layout(title="Persistence Diagram for a Circle")

    # Save the figure as HTML and/or image file
    # fig.write_html("circle_persistence_diagram.html")
    # fig.write_image("circle_persistence_diagram.png")


@pytest.mark.benchmark
@pytest.mark.parametrize("n_points", range(1000, 11000, 1000))
def test_giotto_phd_large_point_cloud(benchmark, n_points):
    """Test the giotto_phd function on larger point clouds of increasing size."""
    n_dims = 768

    # Generate random points in 768-dimensional space
    point_cloud = np.random.normal(0, 1, (n_points, n_dims))

    # Benchmark the computation of persistence diagrams with increasing point cloud sizes
    result = benchmark(lambda: giotto_phd(point_cloud, max_dim=2))

    # Check that the output is a dictionary
    assert isinstance(result, dict)

    # Check that the diagrams are in the "dgms" key
    assert "dgms" in result
    diagrams = result["dgms"]

    # Check that we have diagrams for dimensions 0, 1, and 2
    assert len(diagrams) >= 3

    # Check that entropy was computed
    assert "entropy" in result

    # Log the size of the point cloud
    print(f"Processed point cloud with {n_points} points in {n_dims} dimensions")
