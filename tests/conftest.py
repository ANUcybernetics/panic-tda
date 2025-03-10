from uuid import uuid4

import pytest
from sqlmodel import Session, SQLModel, create_engine

from trajectory_tracer.schemas import Embedding, Invocation, InvocationType, Run


@pytest.fixture
def in_memory_db():
    """Create a new in-memory database for tests."""
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    return engine

@pytest.fixture
def session(in_memory_db):
    """Create a new database session for a test."""
    with Session(in_memory_db) as session:
        yield session

@pytest.fixture
def sample_run():
    """Create a sample Run object."""
    return Run(
        id=uuid4(),
        network=["model1", "model2"],
        seed=42,
        length=5
    )

@pytest.fixture
def sample_text_invocation(sample_run):
    """Create a sample text Invocation object."""
    return Invocation(
        id=uuid4(),
        model="TextModel",
        type=InvocationType.TEXT,
        seed=42,
        run_id=sample_run.id,
        sequence_number=1,
        output_text="Sample output text"
    )

@pytest.fixture
def sample_embedding(sample_text_invocation):
    """Create a sample Embedding object."""
    import numpy as np
    vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    embedding = Embedding(
        id=uuid4(),
        invocation_id=sample_text_invocation.id,
        embedding_model="test-embedding-model"
    )
    embedding.vector = vector

    return embedding
