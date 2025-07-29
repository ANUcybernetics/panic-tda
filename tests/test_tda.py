import os
import numpy as np
import pytest
from gtda.plotting import plot_diagram

from panic_tda.schemas import PersistenceDiagram, Run
from panic_tda.tda import giotto_phd, compute_wasserstein_distance


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


def test_compute_wasserstein_distance():
    """Test the compute_wasserstein_distance function with known inputs and expected outputs."""

    # Test 1: Identical diagrams should have distance 0
    dgm1 = np.array([[0, 1], [0.5, 1.5], [1, 2]])
    dgm2 = np.array([[0, 1], [0.5, 1.5], [1, 2]])
    dist = compute_wasserstein_distance(dgm1, dgm2)
    assert dist == 0.0, f"Identical diagrams should have distance 0, got {dist}"

    # Test 2: Empty diagrams should have distance 0
    empty1 = np.array([]).reshape(0, 2)
    empty2 = np.array([]).reshape(0, 2)
    dist = compute_wasserstein_distance(empty1, empty2)
    assert dist == 0.0, f"Two empty diagrams should have distance 0, got {dist}"

    # Test 3: Empty diagram vs non-empty diagram
    dgm = np.array([[0, 1], [0.5, 2], [1, 3]])
    empty = np.array([]).reshape(0, 2)
    # Distance should be sum of persistence values: (1-0) + (2-0.5) + (3-1) = 1 + 1.5 + 2 = 4.5
    dist = compute_wasserstein_distance(dgm, empty)
    assert dist == 4.5, f"Expected distance 4.5, got {dist}"

    # Test 4: Same distance regardless of order
    dist_reverse = compute_wasserstein_distance(empty, dgm)
    assert dist_reverse == 4.5, f"Expected distance 4.5 (reversed), got {dist_reverse}"

    # Test 5: Simple known example
    dgm1 = np.array([[0, 1], [0, 2]])  # Persistence values: 1, 2
    dgm2 = np.array([[0, 1.5], [0, 2.5]])  # Persistence values: 1.5, 2.5
    # After sorting in descending order: [2, 1] vs [2.5, 1.5]
    # L1 distance: |2.5 - 2| + |1.5 - 1| = 0.5 + 0.5 = 1.0
    dist = compute_wasserstein_distance(dgm1, dgm2)
    assert dist == 1.0, f"Expected distance 1.0, got {dist}"

    # Test 6: Different sized diagrams
    dgm1 = np.array([[0, 1], [0, 2], [0, 3]])  # Persistence values: 1, 2, 3
    dgm2 = np.array([[0, 1.5], [0, 2.5]])  # Persistence values: 1.5, 2.5
    # After sorting in descending order: [3, 2, 1] vs [2.5, 1.5, 0] (padded)
    # L1 distance: |3 - 2.5| + |2 - 1.5| + |1 - 0| = 0.5 + 0.5 + 1 = 2.0
    dist = compute_wasserstein_distance(dgm1, dgm2)
    assert dist == 2.0, f"Expected distance 2.0, got {dist}"

    # Test 7: Test with persistence diagrams that have different birth times
    dgm1 = np.array([[1, 3], [2, 5]])  # Persistence values: 2, 3
    dgm2 = np.array([[0, 2], [3, 6]])  # Persistence values: 2, 3
    # Same persistence values, so after sorting: [3, 2] vs [3, 2]
    # L1 distance: |3 - 3| + |2 - 2| = 0
    dist = compute_wasserstein_distance(dgm1, dgm2)
    assert dist == 0.0, f"Expected distance 0.0 for same persistence values, got {dist}"

    # Test 8: Test error handling for invalid inputs
    with pytest.raises(TypeError):
        compute_wasserstein_distance("not an array", dgm1)

    with pytest.raises(TypeError):
        compute_wasserstein_distance(dgm1, "not an array")

    # Test 9: Test assertion for wrong shape
    wrong_shape = np.array([1, 2, 3])  # 1D array
    with pytest.raises(AssertionError):
        compute_wasserstein_distance(wrong_shape, dgm1)

    wrong_shape_2d = np.array([[1, 2, 3], [4, 5, 6]])  # Wrong number of columns
    with pytest.raises(AssertionError):
        compute_wasserstein_distance(dgm1, wrong_shape_2d)
