import io

import numpy as np
import pytest
from PIL import Image
from uuid_v7.base import uuid7

from trajectory_tracer.embeddings import embed
from trajectory_tracer.schemas import Invocation, InvocationType


def test_dummy_embedding(db_session):
    """Test that the dummy embedding returns a random embedding vector."""
    # Create a sample invocation

    sample_text_invocation = Invocation(
        model="TextModel",
        type=InvocationType.TEXT,
        seed=42,
        output_text="Sample output text",
        run_id = uuid7()
    )
    db_session.add(sample_text_invocation)
    db_session.commit()

    # Get the embedding using the new approach
    embedding_vector = embed("Dummy", sample_text_invocation)

    # Check that the embedding has the correct properties
    assert len(embedding_vector) == 768  # Expected dimension

    # Verify that the vector contains random values between 0 and 1
    assert all(0 <= x <= 1 for x in embedding_vector)

    # Get another embedding and verify it's different (random)
    embedding2_vector = embed("Dummy", sample_text_invocation)
    # Can't directly compare numpy arrays with !=, use numpy's array_equal instead
    assert not np.array_equal(embedding_vector, embedding2_vector)


@pytest.mark.slow
def test_nomic_text_embedding(db_session):
    """Test that the nomic text embedding returns a valid embedding vector."""
    # Create a sample invocation
    sample_text_invocation = Invocation(
        model="TextModel",
        type=InvocationType.TEXT,
        seed=42,
        output_text="Sample output text",
        run_id=uuid7()
    )
    db_session.add(sample_text_invocation)
    db_session.commit()

    # Get the embedding using the actual model
    embedding = embed("NomicText", sample_text_invocation)

    # Check that the embedding has the correct properties
    assert embedding.invocation_id == sample_text_invocation.id
    assert embedding.embedder == "nomic-embed-text-v1.5"
    assert embedding.vector is not None
    assert len(embedding.vector) == 768  # Expected dimension

    # Verify it's a proper embedding vector
    assert embedding.vector.dtype == np.float32
    assert not np.all(embedding.vector == 0)  # Should not be all zeros


@pytest.mark.slow
def test_nomic_vision_embedding(db_session):
    """Test that the nomic vision embedding returns a valid embedding vector."""
    import io

    from PIL import Image

    # Create a sample image (red square)
    image = Image.new('RGB', (100, 100), color='red')
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    image_data = img_byte_arr.getvalue()

    # Create a sample image invocation
    sample_image_invocation = Invocation(
        model="ImageModel",
        type=InvocationType.IMAGE,
        seed=42,
        output_image_data=image_data,
        run_id=uuid7()
    )
    db_session.add(sample_image_invocation)
    db_session.commit()

    # Get the embedding using the actual model
    embedding = embed("NomicVision", sample_image_invocation)

    # Check that the embedding has the correct properties
    assert embedding.invocation_id == sample_image_invocation.id
    assert embedding.embedder == "nomic-embed-vision-v1.5"
    assert embedding.vector is not None
    assert len(embedding.vector) == 768

    # Verify it's a proper embedding vector
    assert embedding.vector.dtype == np.float32
    assert not np.all(embedding.vector == 0)  # Should not be all zeros


@pytest.mark.slow
def test_jina_clip_text_embedding(db_session):
    """Test that the JinaClip embedding returns a valid embedding vector for text."""
    # Create a sample invocation
    sample_text_invocation = Invocation(
        model="TextModel",
        type=InvocationType.TEXT,
        seed=42,
        output_text="Sample output text",
        run_id=uuid7()
    )
    db_session.add(sample_text_invocation)
    db_session.commit()

    # Get the embedding using the actual model
    embedding = embed("JinaClip", sample_text_invocation)

    # Check that the embedding has the correct properties
    assert embedding.invocation_id == sample_text_invocation.id
    assert embedding.embedder == "jina-clip-v2"
    assert embedding.vector is not None
    assert len(embedding.vector) == 768  # Expected dimension

    # Verify it's a proper embedding vector
    assert embedding.vector.dtype == np.float32
    assert not np.all(embedding.vector == 0)  # Should not be all zeros


@pytest.mark.slow
def test_jina_clip_image_embedding(db_session):
    """Test that the JinaClip embedding returns a valid embedding vector for images."""

    # Create a sample image (blue square)
    image = Image.new('RGB', (100, 100), color='blue')
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    image_data = img_byte_arr.getvalue()

    # Create a sample image invocation
    sample_image_invocation = Invocation(
        model="ImageModel",
        type=InvocationType.IMAGE,
        seed=42,
        output_image_data=image_data,
        run_id=uuid7()
    )
    db_session.add(sample_image_invocation)
    db_session.commit()

    # Get the embedding using the actual model
    embedding = embed("JinaClip", sample_image_invocation)

    # Check that the embedding has the correct properties
    assert embedding.invocation_id == sample_image_invocation.id
    assert embedding.embedder == "jina-clip-v2"
    assert embedding.vector is not None
    assert len(embedding.vector) == 768  # Expected dimension

    # Verify it's a proper embedding vector
    assert embedding.vector.dtype == np.float32
    assert not np.all(embedding.vector == 0)  # Should not be all zeros
