from datetime import datetime
from uuid import UUID, uuid4

import pytest
from src.schemas import Embedding, Invocation, Network, Run


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

    assert invocation.type("text content") == "text"
    assert invocation.type(b"image bytes") == "image"


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
