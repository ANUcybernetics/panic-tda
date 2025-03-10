import uuid

from trajectory_tracer.embeddings import dummy_embedding
from trajectory_tracer.schemas import Invocation


def test_dummy_embedding():
    """Test that dummy_embedding returns a random embedding vector."""
    # Create a test invocation
    test_id = uuid.uuid4()
    invocation = Invocation(
        id=test_id,
        model="test_model",
        input="test input",
        output="test output",
        seed=42,
        run_id=1
    )

    # Get the embedding
    embedding = dummy_embedding(invocation)

    # Check that the embedding has the correct properties
    assert embedding.invocation_id == test_id
    assert embedding.embedding_model == "dummy-embedding"
    assert len(embedding.vector) == 768  # Expected dimension

    # Verify that the vector contains random values between 0 and 1
    assert all(0 <= x <= 1 for x in embedding.vector)

    # Get another embedding and verify it's different (random)
    embedding2 = dummy_embedding(invocation)
    assert embedding.vector != embedding2.vector
