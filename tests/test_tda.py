import os
import numpy as np
import pytest
from gtda.plotting import plot_diagram

from panic_tda.schemas import PersistenceDiagram, Run
from panic_tda.tda import giotto_phd


def test_giotto_phd():
    """Test the giotto_phd function for computing persistent homology diagrams."""
    # Ensure output directory exists
    os.makedirs("output/test", exist_ok=True)

    # Create a simple point cloud (a circle-like shape)
    n_points = 100
    theta = np.linspace(0, 2 * np.pi, n_points)
    point_cloud = np.column_stack((np.cos(theta), np.sin(theta)))

    # Compute the persistence diagram
    result = giotto_phd(point_cloud, max_dim=2)

    # Print each key and full values from the result to a file
    with open("output/test/persistence_diagram_keys.txt", "w") as f:
        for key, value in result.items():
            f.write(f"Key: {key}\n")
            f.write(f"Type: {type(value)}\n")
            f.write(
                f"Shape/Length: {len(value) if hasattr(value, '__len__') else 'N/A'}\n"
            )

            # Print full values for each key
            if key == "dgms":
                f.write("Persistence diagrams for each dimension:\n")
                for i, dgm in enumerate(value):
                    f.write(f"  Dimension {i}: array of shape {dgm.shape}\n")
                    f.write(f"  Full array values:\n{dgm}\n\n")
            elif key == "gens":
                f.write("Generators for each dimension:\n")
                for i, gen_list in enumerate(value):
                    f.write(f"  Dimension {i}: {len(gen_list)} generators\n")
                    for j, gen in enumerate(gen_list):
                        f.write(f"    Generator {j}: {gen}\n")
            elif key == "entropy":
                f.write(f"Entropy value: {value}\n")
            else:
                # For any other keys
                f.write(f"Value:\n{value}\n")

            f.write("\n" + "-" * 50 + "\n\n")

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
    # Ensure output directory exists
    os.makedirs("output/test", exist_ok=True)

    # Create a simple point cloud (a circle-like shape)
    n_points = 1000
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
    fig.write_html("output/test/giotto-pd.html")
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


def test_persistence_diagram_type_decorator(db_session):
    """Test that the PersistenceDiagramResultType can serialize and deserialize persistence diagram results."""
    # Create a simple point cloud
    n_points = 100
    theta = np.linspace(0, 2 * np.pi, n_points)
    point_cloud = np.column_stack((np.cos(theta), np.sin(theta)))

    # Compute the persistence diagram
    result = giotto_phd(point_cloud, max_dim=2)

    # Create a run to associate with the diagram

    # Create a sample run
    sample_run = Run(
        initial_prompt="test persistence diagram storage",
        network=["model1"],
        seed=42,
        max_length=3,
    )

    # Create a persistence diagram
    diagram = PersistenceDiagram(
        run_id=sample_run.id,
        embedding_model="test-embedding-model",
        diagram_data=result,
    )

    # Add to the database
    db_session.add(sample_run)
    db_session.add(diagram)
    db_session.commit()

    # Retrieve the diagram from the database
    # Replace select statement with session.get
    retrieved = db_session.get(PersistenceDiagram, diagram.id)

    # Verify the data is correctly serialized and deserialized
    assert retrieved.diagram_data is not None
    assert "dgms" in retrieved.diagram_data
    assert "entropy" in retrieved.diagram_data
    assert "gens" in retrieved.diagram_data

    # Check that the diagrams match
    original_dgms = result["dgms"]
    retrieved_dgms = retrieved.diagram_data["dgms"]

    assert len(original_dgms) == len(retrieved_dgms)

    # Check each dimension's diagram
    for dim in range(len(original_dgms)):
        assert np.array_equal(original_dgms[dim], retrieved_dgms[dim])

    # Verify entropy values match
    assert np.array_equal(result["entropy"], retrieved.diagram_data["entropy"])

    # Verify generators match
    original_gens = result["gens"]
    retrieved_gens = retrieved.diagram_data["gens"]

    assert len(original_gens) == len(retrieved_gens)

    # Check generators for each dimension
    for dim in range(len(original_gens)):
        # For non-empty generator lists
        if len(original_gens[dim]) > 0:
            assert len(original_gens[dim]) == len(retrieved_gens[dim])

            # Check each generator
            for i in range(len(original_gens[dim])):
                assert np.array_equal(original_gens[dim][i], retrieved_gens[dim][i])
