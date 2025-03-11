from uuid import UUID

import numpy as np
from PIL import Image
from sqlmodel import Session

from trajectory_tracer.schemas import Embedding, Invocation, InvocationType, Run


def test_run_creation(db_session: Session):
    """Test creating a Run object."""
    # Create a sample run
    sample_run = Run(
        initial_prompt="once upon a...",
        network=["model1", "model2"],
        seed=42,
        length=5
    )
    db_session.add(sample_run)
    db_session.commit()

    # Retrieve the run from the database
    retrieved_run = db_session.get(Run, sample_run.id)

    assert retrieved_run is not None
    assert retrieved_run.id == sample_run.id
    assert retrieved_run.network == ["model1", "model2"]
    assert retrieved_run.seed == 42
    assert retrieved_run.length == 5


def test_text_invocation(db_session: Session):
    """Test creating a text Invocation."""
    # Create a sample run
    sample_run = Run(
        initial_prompt="once upon a...",
        network=["model1", "model2"],
        seed=42,
        length=5
    )

    # Create a sample text invocation
    sample_text_invocation = Invocation(
        model="TextModel",
        type=InvocationType.TEXT,
        seed=42,
        run_id=sample_run.id,
        sequence_number=1,
        output_text="Sample output text"
    )

    db_session.add(sample_run)
    db_session.add(sample_text_invocation)
    db_session.commit()

    # Retrieve the invocation from the database
    retrieved = db_session.get(Invocation, sample_text_invocation.id)

    assert retrieved is not None
    assert retrieved.type == InvocationType.TEXT
    assert retrieved.output_text == "Sample output text"
    assert retrieved.output == "Sample output text"  # Test the property


def test_image_invocation(db_session: Session):
    """Test creating an image Invocation."""
    # Create a sample run
    sample_run = Run(
        initial_prompt="once upon a...",
        network=["model1", "model2"],
        seed=42,
        length=5
    )

    # Create a simple test image
    img = Image.new('RGB', (60, 30), color = 'red')

    invocation = Invocation(
        id=UUID('00000000-0000-0000-0000-000000000001'),
        model="ImageModel",
        type=InvocationType.IMAGE,
        seed=42,
        run_id=sample_run.id,
        sequence_number=2
    )
    invocation.output = img  # Test the output property setter for images

    db_session.add(sample_run)
    db_session.add(invocation)
    db_session.commit()

    # Retrieve the invocation from the database
    retrieved = db_session.get(Invocation, invocation.id)

    assert retrieved is not None
    assert retrieved.type == InvocationType.IMAGE
    assert retrieved.output_text is None
    assert retrieved.output_image_data is not None

    # Test that we can get the image back
    output_image = retrieved.output
    assert isinstance(output_image, Image.Image)
    assert output_image.width == 60
    assert output_image.height == 30

def test_embedding(db_session: Session):
    """Test creating and retrieving an Embedding."""
    # Create a sample text invocation
    sample_run = Run(
        initial_prompt="once upon a...",
        network=["model1", "model2"],
        seed=42,
        length=5
    )

    sample_text_invocation = Invocation(
        model="TextModel",
        type=InvocationType.TEXT,
        seed=42,
        run_id=sample_run.id,
        sequence_number=1,
        output_text="Sample output text"
    )

    # Create a sample embedding
    vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    sample_embedding = Embedding(
        invocation_id=sample_text_invocation.id,
        embedding_model="test-embedding-model"
    )
    sample_embedding.vector = vector

    db_session.add(sample_run)
    db_session.add(sample_text_invocation)
    db_session.add(sample_embedding)
    db_session.commit()

    # Retrieve the embedding from the database
    retrieved = db_session.get(Embedding, sample_embedding.id)

    assert retrieved is not None
    assert isinstance(retrieved.vector, np.ndarray)
    assert retrieved.vector.shape == (3,)
    assert np.allclose(retrieved.vector, np.array([0.1, 0.2, 0.3], dtype=np.float32))
