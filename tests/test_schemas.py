import json
import tempfile
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest
from PIL import Image

from trajectory_tracer.schemas import (
    Embedding,
    ExperimentConfig,
    Invocation,
    InvocationType,
    Run,
)


def test_invocation_creation():
    """Test basic Invocation creation with required fields."""
    run_id = uuid4()
    invocation = Invocation(
        model="DummyI2T",
        type=InvocationType.TEXT,
        seed=42,
        run_id=run_id,
        output_text="Hi there!"
    )

    assert isinstance(invocation.id, UUID)
    assert invocation.started_at is None
    assert invocation.completed_at is None
    assert invocation.model == "DummyI2T"
    assert invocation.model == "DummyI2T"
    assert invocation.output == "Hi there!"
    assert invocation.seed == 42
    assert invocation.run_id == run_id
    assert invocation.sequence_number == 0


def test_invocation_output_property():
    """Test that output property correctly handles different types."""
    run_id = uuid4()

    # Test text output
    text_invocation = Invocation(
        model="DummyI2T",
        type=InvocationType.TEXT,
        seed=42,
        run_id=run_id,
        output_text="Text output"
    )
    assert text_invocation.output == "Text output"

    # Test image output
    test_image = Image.new('RGB', (100, 100), color='red')
    image_invocation = Invocation(
        model="DummyT2I",
        type=InvocationType.IMAGE,
        seed=42,
        run_id=run_id
    )
    image_invocation.output = test_image

    # Output should be a PIL Image
    assert isinstance(image_invocation.output, Image.Image)


def test_text_invocation_output_setter_validation():
    """Test that output setter validates input types for text invocations."""
    run_id = uuid4()

    # Create text invocation
    invocation = Invocation(
        model="DummyI2T",
        type=InvocationType.TEXT,
        seed=42,
        run_id=run_id
    )

    # Test setting to None
    invocation.output = None
    assert invocation.output is None

    # Test setting to text
    invocation.output = "Hello world"
    assert invocation.output == "Hello world"

    # Test error on invalid type
    with pytest.raises(TypeError, match="Expected str, Image, or None"):
        invocation.output = 12345


def test_image_invocation_output_setter_validation():
    """Test that output setter validates input types for image invocations."""
    run_id = uuid4()

    # Create image invocation
    invocation = Invocation(
        model="DummyT2I",
        type=InvocationType.IMAGE,
        seed=42,
        run_id=run_id
    )

    # Test setting to None
    invocation.output = None
    assert invocation.output is None

    # Test setting to image
    test_image = Image.new('RGB', (100, 100), color='blue')
    invocation.output = test_image
    assert isinstance(invocation.output, Image.Image)

    # Test error on invalid type
    with pytest.raises(TypeError, match="Expected str, Image, or None"):
        invocation.output = 12345


def test_run_validation_success():
    """Test Run validation with correctly ordered invocations."""
    run_id = uuid4()
    network = ["DummyI2T", "DummyT2I"]

    invocations = [
        Invocation(model="DummyI2T", type=InvocationType.TEXT, seed=1, run_id=run_id, sequence_number=0),
        Invocation(model="DummyT2I", type=InvocationType.IMAGE, seed=1, run_id=run_id, sequence_number=1),
        Invocation(model="DummyI2T", type=InvocationType.TEXT, seed=1, run_id=run_id, sequence_number=2),
    ]

    run = Run(id=run_id, seed=1, length=3, network=network, invocations=invocations)
    assert len(run.invocations) == 3


def test_embedding_creation():
    """Test Embedding creation and dimension property."""
    invocation_id = uuid4()
    vector = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    embedding = Embedding(
        invocation_id=invocation_id,
        embedding_model="text-embedding-ada-002",
    )
    embedding.vector = vector

    assert embedding.invocation_id == invocation_id
    assert embedding.embedding_model == "text-embedding-ada-002"
    np.testing.assert_array_equal(embedding.vector, vector)
    assert embedding.dimension == 4


def test_run_network_property():
    """Test the network property serializes and deserializes correctly."""
    network = ["DummyI2T", "DummyT2I"]
    run = Run(seed=1, length=3, network=network)

    # Test that we can get the network back as model classes
    retrieved_network = run.network
    assert len(retrieved_network) == 2
    assert retrieved_network[0] == "DummyI2T"
    assert retrieved_network[1] == "DummyT2I"


def test_experiment_config():
    """Test that ExperimentConfig can be properly loaded and validated."""

    config_data = {
        "networks": [["DummyI2T", "DummyT2I"], ["DummyT2I"]],
        "seeds": [42, 123],
        "prompts": ["First prompt", "Second prompt"],
        "embedders": ["embedder1", "embedder2"],
        "run_length": 5
    }

    # Create a temporary JSON file
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w+", delete=False) as f:
        json.dump(config_data, f)
        config_path = Path(f.name)

    try:
        # Load the config from the file
        with open(config_path, "r") as f:
            loaded_data = json.load(f)

        config = ExperimentConfig(**loaded_data)

        # Validate the config
        assert config.validate_equal_lengths() is True
        assert len(config.networks) == 2
        assert config.networks[0] == ["DummyI2T", "DummyT2I"]
        assert config.seeds == [42, 123]
        assert config.prompts == ["First prompt", "Second prompt"]
        assert config.embedders == ["embedder1", "embedder2"]
        assert config.run_length == 5

        # Test validation error with unequal list lengths
        invalid_data = config_data.copy()
        invalid_data["seeds"] = [42]  # Only one seed now

        with pytest.raises(ValueError):
            config = ExperimentConfig(**invalid_data)
            config.validate_equal_lengths()

    finally:
        # Clean up the temporary file
        config_path.unlink()


def test_experiment_config_invalid_values():
    """Test that ExperimentConfig raises errors for invalid values."""
    # Test empty list validation
    with pytest.raises(ValueError, match="List cannot be empty"):
        ExperimentConfig(
            networks=[],
            seeds=[42],
            prompts=["test"],
            embedders=["test"],
            run_length=5
        )

    # Test zero run_length validation
    with pytest.raises(ValueError, match="Run length must be greater than 0"):
        ExperimentConfig(
            networks=[["DummyI2T"]],
            seeds=[42],
            prompts=["test"],
            embedders=["test"],
            run_length=0
        )

    # Test negative run_length validation
    with pytest.raises(ValueError, match="Run length must be greater than 0"):
        ExperimentConfig(
            networks=[["DummyI2T"]],
            seeds=[42],
            prompts=["test"],
            embedders=["test"],
            run_length=-5
        )

    # Test missing required field
    with pytest.raises(ValueError):
        ExperimentConfig(
            networks=[["DummyI2T"]],
            seeds=[42],
            prompts=["test"],
            # Missing embedders
            run_length=5
        )
