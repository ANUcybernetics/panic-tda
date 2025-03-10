from uuid import UUID

import numpy as np
from PIL import Image

from trajectory_tracer.schemas import Embedding, Invocation, InvocationType, Run


def test_run_creation(session, sample_run):
    """Test creating a Run object."""
    session.add(sample_run)
    session.commit()

    # Retrieve the run from the database
    retrieved_run = session.get(Run, sample_run.id)

    assert retrieved_run is not None
    assert retrieved_run.id == sample_run.id
    assert retrieved_run.network == ["model1", "model2"]
    assert retrieved_run.seed == 42
    assert retrieved_run.length == 5

def test_text_invocation(session, sample_run, sample_text_invocation):
    """Test creating a text Invocation."""
    session.add(sample_run)
    session.add(sample_text_invocation)
    session.commit()

    # Retrieve the invocation from the database
    retrieved = session.get(Invocation, sample_text_invocation.id)

    assert retrieved is not None
    assert retrieved.type == InvocationType.TEXT
    assert retrieved.output_text == "Sample output text"
    assert retrieved.output == "Sample output text"  # Test the property

def test_image_invocation(session, sample_run):
    """Test creating an image Invocation."""
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

    session.add(sample_run)
    session.add(invocation)
    session.commit()

    # Retrieve the invocation from the database
    retrieved = session.get(Invocation, invocation.id)

    assert retrieved is not None
    assert retrieved.type == InvocationType.IMAGE
    assert retrieved.output_text is None
    assert retrieved.output_image_data is not None

    # Test that we can get the image back
    output_image = retrieved.output
    assert isinstance(output_image, Image.Image)
    assert output_image.width == 60
    assert output_image.height == 30

def test_embedding(session, sample_text_invocation, sample_embedding):
    """Test creating and retrieving an Embedding."""
    session.add(sample_text_invocation)
    session.add(sample_embedding)
    session.commit()

    # Retrieve the embedding from the database
    retrieved = session.get(Embedding, sample_embedding.id)

    assert retrieved is not None
    assert isinstance(retrieved.vector, np.ndarray)
    assert retrieved.vector.shape == (3,)
    assert np.allclose(retrieved.vector, np.array([0.1, 0.2, 0.3], dtype=np.float32))
