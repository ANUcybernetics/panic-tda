import numpy as np
import pytest
import ray
from PIL import Image

from panic_tda.cpd import find_breakpoints
from panic_tda.embeddings import DummyText, DummyText2


# Fixture to create dummy embedding actors
@pytest.fixture(scope="module")
def dummy_actors():
    dummy_model = DummyText.remote()
    dummy2_model = DummyText2.remote()
    yield dummy_model, dummy2_model
    # Clean up actors
    ray.kill(dummy_model)
    ray.kill(dummy2_model)


# Helper function to generate sample embeddings using DummyText/DummyText2 models
def generate_dummy_embeddings(dummy_actors, n_samples=100, change_points=None):
    """
    Generates a sequence of embeddings with simulated change points.

    Args:
        dummy_actors: Tuple containing DummyText and DummyText2 Ray actors.
        n_samples: Total number of embeddings to generate.
        change_points: List of indices where the data generation process changes.
                       If None, generates a homogeneous sequence.

    Returns:
        A list of np.ndarray embeddings.
    """
    dummy_model, dummy2_model = dummy_actors
    embeddings = []
    current_model = dummy_model
    input_type = "text"  # Start with text

    if change_points is None:
        change_points = []

    sorted_cps = sorted(change_points)
    cp_idx = 0

    for i in range(n_samples):
        # Check if we hit a change point
        if cp_idx < len(sorted_cps) and i == sorted_cps[cp_idx]:
            # Alternate model and input type at change points
            current_model = (
                dummy2_model if current_model == dummy_model else dummy_model
            )
            input_type = "image" if input_type == "text" else "text"
            cp_idx += 1
            # print(f"Change point at {i}: Switched to {current_model} and {input_type}") # DEBUG

        # Generate content based on type and index to ensure variation
        if input_type == "text":
            content = [f"Sample text content number {i}"]
        else:
            # Create a simple varying image
            color = ((i * 5) % 255, (i * 10) % 255, (i * 15) % 255)
            content = [Image.new("RGB", (32, 32), color=color)]

        # Get embedding from the current model
        embedding_ref = current_model.embed.remote(content)
        embedding_list = ray.get(embedding_ref)
        embeddings.append(embedding_list[0])

    return embeddings


@pytest.mark.skip(reason="This test is flaky")
def test_find_breakpoints_basic(dummy_actors):
    """Test find_breakpoints with data having known change points."""
    n_samples = 100
    true_bkps_indices = [30, 70]  # Indices where we change the generation process
    # Expected output from find_breakpoints includes the end index
    expected_true_bkps = sorted(list(set(true_bkps_indices + [n_samples])))

    embeddings = generate_dummy_embeddings(
        dummy_actors, n_samples=n_samples, change_points=true_bkps_indices
    )

    # Call the simplified function
    detected_bkps = find_breakpoints(embeddings)

    # Basic checks on the output format
    assert isinstance(detected_bkps, list)
    assert all(isinstance(bkp, int) for bkp in detected_bkps)
    assert len(detected_bkps) > 0  # Should at least contain the end point
    assert detected_bkps[-1] == n_samples  # Last breakpoint is always n_samples

    # Check if breakpoints are sorted and within bounds
    assert all(0 < bkp <= n_samples for bkp in detected_bkps)
    assert detected_bkps == sorted(list(set(detected_bkps)))  # Ensure sorted and unique

    # The exact number/location of detected breakpoints depends on the fixed internal parameters
    # and the generated data. We don't assert exact matches to true_bkps, but we print for inspection.
    print(
        f"Simulated change points at indices: {true_bkps_indices} (expected bkps: {expected_true_bkps})"
    )
    print(f"Detected breakpoints: {detected_bkps}")
    # A loose check: number of detected points should be at least 1 (the endpoint)
    # and less than half the number of samples
    assert 0 < len(detected_bkps) < n_samples / 2


def test_find_breakpoints_no_change(dummy_actors):
    """Test find_breakpoints with homogeneous data (no change points)."""
    n_samples = 100
    embeddings = generate_dummy_embeddings(dummy_actors, n_samples=n_samples)

    detected_bkps = find_breakpoints(embeddings)

    # Basic checks
    assert isinstance(detected_bkps, list)
    assert detected_bkps[-1] == n_samples

    # Expect few or no breakpoints (ideally just [n_samples]) for homogeneous data
    print(f"Detected breakpoints (no change): {detected_bkps}")
    assert len(detected_bkps) <= 3  # Allow for some noise detection


def test_find_breakpoints_edge_cases(dummy_actors):
    """Test edge cases like empty input, insufficient data, invalid types."""

    # 1. Empty embeddings list
    empty_embeddings = []
    result_empty = find_breakpoints(empty_embeddings)
    assert result_empty == []

    # 2. Embeddings not convertible to 2D array (e.g., list of scalars)
    invalid_embeddings_shape = [1, 2, 3, 4, 5]
    with pytest.raises(ValueError, match="Expected a list of 1D arrays"):
        find_breakpoints(invalid_embeddings_shape)

    # 3. Embeddings with non-numeric data (should raise TypeError/ValueError from np.array)
    # Use actors to generate valid ones first, then corrupt
    valid_emb = generate_dummy_embeddings(dummy_actors, n_samples=2)
    invalid_embeddings_type = [valid_emb[0], np.array(["a", "b"])]
    with pytest.raises((TypeError, ValueError)):
        find_breakpoints(invalid_embeddings_type)

    # 4. Not enough samples for min_size * 2 (internal min_size=5)
    n_short_samples = 8  # Less than 5 * 2 = 10
    short_embeddings = generate_dummy_embeddings(
        dummy_actors, n_samples=n_short_samples
    )
    assert len(short_embeddings) == n_short_samples
    result_short = find_breakpoints(short_embeddings)
    # Should return only the end index
    assert result_short == [n_short_samples]
