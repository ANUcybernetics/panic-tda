from unittest.mock import patch

import pytest
from PIL import Image

from src.db import get_in_memory_engine
from src.engine import MODEL_REGISTRY, invoke_and_yield_next, invoke_model, perform_run
from src.models import dummy_i2t, dummy_t2i
from src.schemas import Invocation, Network


@pytest.fixture(scope="function")
def in_memory_db():
    """Setup an in-memory database for testing."""
    # Get the engine instance for the in-memory database
    engine = get_in_memory_engine()

    # Patch the get_engine function to return our in-memory engine
    with patch("src.db.get_engine", return_value=engine):
        yield engine

    # No cleanup needed - in-memory DB disappears when the connection is closed

@pytest.fixture(scope="function")
def test_models():
    """Setup and teardown test models in the registry."""
    # Setup
    MODEL_REGISTRY["dummy_i2t"] = dummy_i2t
    MODEL_REGISTRY["dummy_t2i"] = dummy_t2i

    yield

    # Teardown
    if "dummy_i2t" in MODEL_REGISTRY:
        del MODEL_REGISTRY["dummy_i2t"]
    if "dummy_t2i" in MODEL_REGISTRY:
        del MODEL_REGISTRY["dummy_t2i"]

def test_invoke_model(test_models):
    """Test that models can be invoked correctly."""
    # Test text-to-image model
    image = invoke_model("dummy_t2i", "test prompt")
    assert isinstance(image, Image.Image)

    # Test image-to-text model
    test_image = Image.new('RGB', (512, 512), color='white')
    text = invoke_model("dummy_i2t", test_image)
    assert text == "dummy text caption"

def test_invoke_and_yield_next(in_memory_db, test_models):
    """Test that invoke_and_yield_next processes an invocation and creates the next one correctly."""
    # Create a test network
    network = Network(models=["dummy_t2i", "dummy_i2t"])

    # Create an initial invocation
    MODEL_SEED = 42
    RUN_ID = 1001
    initial_invocation = Invocation(
        model="dummy_t2i",
        input="test prompt",
        seed=MODEL_SEED,
        run_id=RUN_ID,
        network=network,
        sequence_number=0
    )

    # Process the invocation and get the next one
    next_invocation = invoke_and_yield_next(initial_invocation)

    # Verify first invocation was processed correctly
    assert isinstance(initial_invocation.output, Image.Image)

    # Verify next invocation is set up correctly
    assert next_invocation.model == "dummy_i2t"
    assert isinstance(next_invocation.input, Image.Image)
    assert next_invocation.sequence_number == 1

def test_perform_run(in_memory_db, test_models, monkeypatch):
    """Test that perform_run generates a sequence of invocations correctly."""
    # Create a test network with dummy models
    network = Network(models=["dummy_t2i", "dummy_i2t"])

    # Set test parameters
    MODEL_SEED = 42
    prompt = "test prompt"
    run_length = 3

    # Use monkeypatch to ensure a consistent run_id generation
    stable_hash_value = 12345
    monkeypatch.setattr("src.engine.hash", lambda x: stable_hash_value)

    # Perform the test run
    invocations = list(perform_run(network, prompt, MODEL_SEED, run_length))

    # Verify we got the expected number of invocations
    assert len(invocations) == run_length

    # Get the expected run_id from the engine's logic
    expected_run_id = abs(stable_hash_value % (10**10))

    # Verify all invocations have the same expected run_id
    for inv in invocations:
        assert inv.run_id == expected_run_id

    # Verify the sequence_number progression
    for i, invocation in enumerate(invocations):
        assert invocation.sequence_number == i

    # Verify the starting model
    assert invocations[0].model == "dummy_t2i"

    # Process invocations to check expected types
    for inv in invocations[:run_length-1]:
        next_inv = invoke_and_yield_next(inv)
        assert next_inv is not None

        # Verify output type based on model type
        if "t2i" in inv.model:
            assert isinstance(inv.output, Image.Image)
        elif "i2t" in inv.model:
            assert isinstance(inv.output, str)
