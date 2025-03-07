import pytest
from PIL import Image

from src.engine import MODEL_REGISTRY, invoke_and_yield_next, invoke_model, perform_run
from src.models import dummy_i2t, dummy_t2i
from src.schemas import Invocation, Network


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

def test_invoke_and_yield_next(test_db, test_models):
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

def test_perform_run(test_db, test_models):
    """Test that perform_run generates a sequence of invocations correctly."""
    # Create a test network with dummy models
    network = Network(models=["dummy_t2i", "dummy_i2t"])

    # Set test parameters
    MODEL_SEED = 42
    prompt = "test prompt"
    run_length = 100

    # Perform the test run
    invocations = list(perform_run(network, prompt, MODEL_SEED, run_length))

    # Verify we got the expected number of invocations
    assert len(invocations) == run_length

    # Get the run_id from the first invocation
    expected_run_id = invocations[0].run_id

    # Verify all invocations have the same run_id
    for inv in invocations:
        assert inv.run_id == expected_run_id

    # Verify the sequence_number progression
    for i, invocation in enumerate(invocations):
        assert invocation.sequence_number == i

    # Verify the starting model
    assert invocations[0].model == "dummy_t2i"

    # Verify output types based on model type without re-processing
    for inv in invocations:
        if "t2i" in inv.model:
            assert isinstance(inv.output, Image.Image)
        elif "i2t" in inv.model:
            assert isinstance(inv.output, str)
