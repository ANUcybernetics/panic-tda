import numpy as np
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
    embedding = embed("Dummy", sample_text_invocation)

    # Check that the embedding has the correct properties
    assert embedding.invocation_id == sample_text_invocation.id
    assert embedding.embedding_model == "dummy-embedding"
    assert len(embedding.vector) == 768  # Expected dimension

    # Verify that the vector contains random values between 0 and 1
    assert all(0 <= x <= 1 for x in embedding.vector)

    # Get another embedding and verify it's different (random)
    embedding2 = embed("Dummy", sample_text_invocation)
    # Can't directly compare numpy arrays with !=, use numpy's array_equal instead
    assert not np.array_equal(embedding.vector, embedding2.vector)
