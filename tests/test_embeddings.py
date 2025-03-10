import numpy as np

from trajectory_tracer.embeddings import dummy_embedding


def test_dummy_embedding(sample_text_invocation):
    """Test that dummy_embedding returns a random embedding vector."""
    # Get the embedding
    embedding = dummy_embedding(sample_text_invocation)

    # Check that the embedding has the correct properties
    assert embedding.invocation_id == sample_text_invocation.id
    assert embedding.embedding_model == "dummy-embedding"
    assert len(embedding.vector) == 768  # Expected dimension

    # Verify that the vector contains random values between 0 and 1
    assert all(0 <= x <= 1 for x in embedding.vector)

    # Get another embedding and verify it's different (random)
    embedding2 = dummy_embedding(sample_text_invocation)
    # Can't directly compare numpy arrays with !=, use numpy's array_equal instead
    assert not np.array_equal(embedding.vector, embedding2.vector)
