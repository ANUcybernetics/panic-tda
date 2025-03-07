import json
import tempfile
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from PIL import Image

from src.schemas import (
    ContentType,
    Embedding,
    ExperimentConfig,
    Invocation,
    Network,
    Run,
)


def test_invocation_creation():
    """Test basic Invocation creation with required fields."""
    invocation = Invocation(
        model="gpt-4",
        input="Hello world",
        output="Hi there!",
        seed=42,
        run_id=1
    )

    assert isinstance(invocation.id, UUID)
    assert isinstance(invocation.timestamp, datetime)
    assert invocation.model == "gpt-4"
    assert invocation.input == "Hello world"
    assert invocation.output == "Hi there!"
    assert invocation.seed == 42
    assert invocation.run_id == 1
    assert invocation.sequence_number == 0
    assert isinstance(invocation.network, Network)


def test_invocation_type_detection():
    """Test that type() method correctly identifies content types."""
    invocation = Invocation(
        model="gpt-4",
        input="Hello world",
        output="Hi there!",
        seed=42,
        run_id=1
    )

    assert invocation.type("text content") == ContentType.TEXT
    assert invocation.type(Image.Image()) == ContentType.IMAGE


def test_run_validation_success():
    """Test Run validation with correctly ordered invocations."""
    invocations = [
        Invocation(model="gpt-4", input="First", output="First response", seed=1, run_id=1, sequence_number=0),
        Invocation(model="gpt-4", input="Second", output="Second response", seed=1, run_id=1, sequence_number=1),
        Invocation(model="gpt-4", input="Third", output="Third response", seed=1, run_id=1, sequence_number=2),
    ]

    run = Run(invocations=invocations)
    assert len(run.invocations) == 3


def test_run_validation_error():
    """Test Run validation fails with incorrectly ordered invocations."""
    invocations = [
        Invocation(model="gpt-4", input="First", output="First response", seed=1, run_id=1, sequence_number=0),
        Invocation(model="gpt-4", input="Second", output="Second response", seed=1, run_id=1, sequence_number=2),  # Wrong sequence
        Invocation(model="gpt-4", input="Third", output="Third response", seed=1, run_id=1, sequence_number=1),
    ]

    with pytest.raises(ValueError):
        Run(invocations=invocations)


def test_embedding_creation():
    """Test Embedding creation and dimension property."""
    invocation_id = uuid4()
    vector = [0.1, 0.2, 0.3, 0.4]

    embedding = Embedding(
        invocation_id=invocation_id,
        embedding_model="text-embedding-ada-002",
        vector=vector
    )

    assert embedding.invocation_id == invocation_id
    assert embedding.embedding_model == "text-embedding-ada-002"
    assert embedding.vector == vector
    assert embedding.dimension == 4

def test_experiment_config():
    """Test that ExperimentConfig can be properly loaded and validated."""

    config_data = {
        "networks": [["model1", "model2"], ["model3"]],
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
        assert config.networks[0] == ["model1", "model2"]
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
            networks=[["model1"]],
            seeds=[42],
            prompts=["test"],
            embedders=["test"],
            run_length=0
        )

    # Test negative run_length validation
    with pytest.raises(ValueError, match="Run length must be greater than 0"):
        ExperimentConfig(
            networks=[["model1"]],
            seeds=[42],
            prompts=["test"],
            embedders=["test"],
            run_length=-5
        )

    # Test missing required field
    with pytest.raises(ValueError):
        ExperimentConfig(
            networks=[["model1"]],
            seeds=[42],
            prompts=["test"],
            # Missing embedders
            run_length=5
        )
