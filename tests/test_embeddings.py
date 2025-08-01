import time

import numpy as np
import pytest
import ray
from PIL import Image

from panic_tda.embeddings import (
    Dummy,
    Dummy2,
    get_actor_class,
    list_models,
)
from panic_tda.schemas import Embedding, Invocation, InvocationType, Run


@pytest.fixture(scope="module")
def dummy_actors():
    """Module-scoped fixture for dummy embedding model actors."""
    actors = {
        "Dummy": Dummy.remote(),
        "Dummy2": Dummy2.remote(),
    }
    yield actors
    # Cleanup
    for actor in actors.values():
        ray.kill(actor)


@pytest.fixture(scope="module")
def embedding_model_actors():
    """Module-scoped fixture for all embedding model actors."""
    actors = {}
    for model_name in list_models():
        model_class = get_actor_class(model_name)
        actors[model_name] = model_class.remote()
    yield actors
    # Cleanup
    for actor in actors.values():
        ray.kill(actor)


def test_run_embeddings_by_model(db_session, dummy_actors):
    """Test the Run.embeddings method returns embeddings for a specific model."""
    # Create a run
    run = Run(
        network=["DummyT2I", "DummyI2T"],
        seed=42,
        max_length=2,
        initial_prompt="Test prompt",
    )
    db_session.add(run)
    db_session.commit()
    db_session.refresh(run)

    # Create invocations
    invocation1 = Invocation(
        model="DummyT2I",
        type=InvocationType.TEXT,
        seed=42,
        run_id=run.id,
        sequence_number=0,
        output_text="First invocation",
    )
    db_session.add(invocation1)

    invocation2 = Invocation(
        model="DummyI2T",
        type=InvocationType.TEXT,
        seed=42,
        run_id=run.id,
        sequence_number=1,
        output_text="Second invocation",
    )
    db_session.add(invocation2)
    db_session.commit()
    db_session.refresh(invocation1)
    db_session.refresh(invocation2)

    # Create embeddings with different models - now with Ray actors
    dummy_model = dummy_actors["Dummy"]
    embedding_vector1_ref = dummy_model.embed.remote([invocation1.output])
    embedding_vectors1 = ray.get(embedding_vector1_ref)
    embedding_vector1 = embedding_vectors1[0]  # Get the first embedding
    embedding1_1 = Embedding(
        invocation_id=invocation1.id,
        embedding_model="Dummy",
        vector=embedding_vector1,
    )
    db_session.add(embedding1_1)

    dummy2_model = dummy_actors["Dummy2"]
    embedding_vector2_ref = dummy2_model.embed.remote([invocation1.output])
    embedding_vectors2 = ray.get(embedding_vector2_ref)
    embedding_vector2 = embedding_vectors2[0]  # Get the first embedding
    embedding1_2 = Embedding(
        invocation_id=invocation1.id,
        embedding_model="Dummy2",
        vector=embedding_vector2,
    )
    db_session.add(embedding1_2)

    embedding_vector3_ref = dummy_model.embed.remote([invocation2.output])
    embedding_vectors3 = ray.get(embedding_vector3_ref)
    embedding_vector3 = embedding_vectors3[0]  # Get the first embedding
    embedding2_1 = Embedding(
        invocation_id=invocation2.id,
        embedding_model="Dummy",
        vector=embedding_vector3,
    )
    db_session.add(embedding2_1)

    db_session.commit()

    # Refresh the run object to ensure relationships are loaded
    db_session.refresh(run)

    # Test filtering by model name
    dummy_embeddings = run.embeddings["Dummy"]
    dummy2_embeddings = run.embeddings["Dummy2"]

    # Verify the filtering works correctly
    assert len(dummy_embeddings) == 2
    assert len(dummy2_embeddings) == 1
    assert all(e.embedding_model == "Dummy" for e in dummy_embeddings)
    assert all(e.embedding_model == "Dummy2" for e in dummy2_embeddings)

    # Verify embeddings are associated with the correct invocations
    assert any(e.invocation_id == invocation1.id for e in dummy_embeddings)
    assert any(e.invocation_id == invocation2.id for e in dummy_embeddings)
    assert all(e.invocation_id == invocation1.id for e in dummy2_embeddings)

    # Verify embeddings have valid vectors
    assert all(e.vector is not None and len(e.vector) > 0 for e in dummy_embeddings)
    assert all(e.vector is not None and len(e.vector) > 0 for e in dummy2_embeddings)


def test_invocation_embedding_property(db_session):
    """Test that the Invocation.embedding() method returns the correct embedding for a model."""
    # Create a run
    run = Run(
        network=["DummyT2I", "DummyI2T"],
        seed=42,
        max_length=2,
        initial_prompt="Test prompt",
    )
    db_session.add(run)
    db_session.commit()
    db_session.refresh(run)

    # Create an invocation
    invocation = Invocation(
        model="DummyT2I",
        type=InvocationType.TEXT,
        seed=42,
        run_id=run.id,
        sequence_number=0,
        output_text="Test output",
    )
    db_session.add(invocation)
    db_session.commit()
    db_session.refresh(invocation)

    # Create embeddings with different models
    try:
        dummy_model = Dummy.remote()
        dummy2_model = Dummy2.remote()

        # Get embeddings from models
        embedding_vector1_ref = dummy_model.embed.remote([invocation.output])
        embedding_vector2_ref = dummy2_model.embed.remote([invocation.output])

        embedding_vectors1 = ray.get(embedding_vector1_ref)
        embedding_vectors2 = ray.get(embedding_vector2_ref)

        embedding_vector1 = embedding_vectors1[0]
        embedding_vector2 = embedding_vectors2[0]

        # Create embedding objects
        embedding1 = Embedding(
            invocation_id=invocation.id,
            embedding_model="Dummy",
            vector=embedding_vector1,
        )
        db_session.add(embedding1)

        embedding2 = Embedding(
            invocation_id=invocation.id,
            embedding_model="Dummy2",
            vector=embedding_vector2,
        )
        db_session.add(embedding2)

        db_session.commit()
        db_session.refresh(invocation)

        # Test the embedding method
        dummy_embedding = invocation.embedding("Dummy")
        dummy2_embedding = invocation.embedding("Dummy2")
        nonexistent_embedding = invocation.embedding("NonexistentModel")

        # Verify the results
        assert dummy_embedding is not None
        assert dummy_embedding.embedding_model == "Dummy"
        assert np.array_equal(dummy_embedding.vector, embedding_vector1)

        assert dummy2_embedding is not None
        assert dummy2_embedding.embedding_model == "Dummy2"
        assert np.array_equal(dummy2_embedding.vector, embedding_vector2)

        # Should return None for a model that doesn't exist
        assert nonexistent_embedding is None
    finally:
        # Terminate the actors to clean up resources
        if "dummy_model" in locals():
            ray.kill(dummy_model)
        if "dummy2_model" in locals():
            ray.kill(dummy2_model)


def test_list_models():
    """Test that list_models returns a list of available embedding models."""

    # Call the list_models function
    available_models = list_models()

    # Check that it returns a list
    assert isinstance(available_models, list)

    # Check that the list is not empty
    assert len(available_models) > 0

    # Check that it contains the expected models we've tested
    expected_models = [
        "Dummy",
        "Dummy2",
        "Nomic",
        "NomicVision",
        "JinaClip",
        "JinaClipVision",
    ]
    for model in expected_models:
        assert model in available_models

    # Verify the base class is not included
    assert "EmbeddingModel" not in available_models


def test_get_actor_class():
    """Test that the get_model_class function returns the correct Ray actor class for a given model name."""

    # Test for a few models
    for model_name in list_models():
        model_class = get_actor_class(model_name)

        # Verify it's a Ray actor class
        assert isinstance(model_class, ray.actor.ActorClass)

        # Verify the class name matches our expectations
        assert type(model_class).__name__ == f"ActorClass({model_name})"

    # Test with nonexistent model
    with pytest.raises(ValueError):
        get_actor_class("NonexistentModel")


def test_run_missing_embeddings(db_session, dummy_actors):
    """Test that the Run.missing_embeddings method correctly identifies invocations without embeddings."""
    # Create a run
    run = Run(
        network=["DummyT2I", "DummyI2T"],
        seed=42,
        max_length=3,
        initial_prompt="Test prompt",
    )
    db_session.add(run)
    db_session.commit()
    db_session.refresh(run)

    # Create three invocations
    invocation1 = Invocation(
        model="DummyT2I",
        type=InvocationType.TEXT,
        seed=42,
        run_id=run.id,
        sequence_number=0,
        output_text="First invocation",
    )
    db_session.add(invocation1)

    invocation2 = Invocation(
        model="DummyI2T",
        type=InvocationType.TEXT,
        seed=42,
        run_id=run.id,
        sequence_number=1,
        output_text="Second invocation",
    )
    db_session.add(invocation2)

    invocation3 = Invocation(
        model="DummyT2I",
        type=InvocationType.TEXT,
        seed=42,
        run_id=run.id,
        sequence_number=2,
        output_text="Third invocation",
    )
    db_session.add(invocation3)
    db_session.commit()
    db_session.refresh(run)

    # Create a dummy model for embeddings
    dummy_model = dummy_actors["Dummy"]

    # Get embeddings for the first and third invocations only
    embedding_vector1_ref = dummy_model.embed.remote([invocation1.output])
    embedding_vectors1 = ray.get(embedding_vector1_ref)
    embedding_vector1 = embedding_vectors1[0]

    embedding_vector3_ref = dummy_model.embed.remote([invocation3.output])
    embedding_vectors3 = ray.get(embedding_vector3_ref)
    embedding_vector3 = embedding_vectors3[0]

    # Add embeddings for only the first and third invocations with "Dummy" model
    embedding1 = Embedding(
        invocation_id=invocation1.id,
        embedding_model="Dummy",
        vector=embedding_vector1,
    )
    db_session.add(embedding1)

    embedding3 = Embedding(
        invocation_id=invocation3.id,
        embedding_model="Dummy",
        vector=embedding_vector3,
    )
    db_session.add(embedding3)

    # Add an embedding for the first invocation with "Dummy2" model
    dummy2_model = dummy_actors["Dummy2"]
    embedding_vector1_2_ref = dummy2_model.embed.remote([invocation1.output])
    embedding_vectors1_2 = ray.get(embedding_vector1_2_ref)
    embedding_vector1_2 = embedding_vectors1_2[0]

    embedding1_2 = Embedding(
        invocation_id=invocation1.id,
        embedding_model="Dummy2",
        vector=embedding_vector1_2,
    )
    db_session.add(embedding1_2)

    # Add an embedding with null vector for the second invocation
    embedding2 = Embedding(
        invocation_id=invocation2.id,
        embedding_model="Dummy",
        vector=None,
    )
    db_session.add(embedding2)

    db_session.commit()
    db_session.refresh(run)

    # Test missing_embeddings for "Dummy" model
    missing_dummy = run.missing_embeddings("Dummy")

    # Should return invocation2 because it has a null vector
    assert len(missing_dummy) == 1
    assert missing_dummy[0].id == invocation2.id

    # Test missing_embeddings for "Dummy2" model
    missing_dummy2 = run.missing_embeddings("Dummy2")

    # Should return invocation2 and invocation3 (both missing "Dummy2" embeddings)
    assert len(missing_dummy2) == 2
    assert set(inv.id for inv in missing_dummy2) == {invocation2.id, invocation3.id}

    # Test missing_embeddings for a model that doesn't exist
    missing_nonexistent = run.missing_embeddings("NonexistentModel")

    # Should return all three invocations (all missing this embedding)
    assert len(missing_nonexistent) == 3
    assert set(inv.id for inv in missing_nonexistent) == {
        invocation1.id,
        invocation2.id,
        invocation3.id,
    }


@pytest.mark.slow
@pytest.mark.parametrize("model_name", list_models())
def test_embedding_model(model_name, embedding_model_actors):
    """Test that the embedding model returns valid vectors and is deterministic."""
    # Get the model actor from fixture
    model = embedding_model_actors[model_name]

    # Determine if this is an image model or text model
    is_image_model = "Vision" in model_name
    
    if is_image_model:
        # Create a sample image
        sample_input = [Image.new("RGB", (100, 100), color="blue")]
    else:
        # Create a sample text string
        sample_input = ["Sample output text"]

    # Test embedding
    embedding_ref = model.embed.remote(sample_input)
    embeddings = ray.get(embedding_ref)
    embedding = embeddings[0]  # Get the first embedding

    # Run it again to verify determinism
    embedding2_ref = model.embed.remote(sample_input)
    embeddings2 = ray.get(embedding2_ref)
    embedding2 = embeddings2[0]  # Get the first embedding

    # Check that the embedding has the correct properties
    assert embedding is not None
    assert len(embedding) == 768  # Expected dimension

    # Verify embedding is normalized (L2 norm close to 1.0)
    # vector_norm = np.linalg.norm(embedding)
    # assert 0.999 <= vector_norm <= 1.001, (
    #     f"Vector not normalized: norm = {vector_norm}"
    # )

    # Verify it's a proper embedding vector (except for dummy models which may not use float32)
    if not model_name.startswith("Dummy"):
        assert embedding.dtype == np.float32
        assert not np.all(embedding == 0)  # Should not be all zeros

    # Verify determinism
    assert np.array_equal(embedding, embedding2)


@pytest.mark.slow
@pytest.mark.parametrize("model_name", list_models())
@pytest.mark.parametrize("batch_size", [1, 8, 32, 64, 256])
def test_embedding_batch_performance(model_name, batch_size, embedding_model_actors):
    """Test the embedding models with increasingly larger batch sizes."""
    # Get the model actor from fixture
    model = embedding_model_actors[model_name]

    # Determine if this is an image model or text model
    is_image_model = "Vision" in model_name
    
    if is_image_model:
        # Create dummy images for the batch
        sample_inputs = [
            Image.new("RGB", (100, 100), color=(i % 255, (i * 2) % 255, (i * 3) % 255))
            for i in range(batch_size)
        ]
    else:
        # Create dummy text strings for the batch
        sample_inputs = [f"Sample text {i}" for i in range(batch_size)]

    # Measure time to process the batch
    start_time = time.time()

    # Get embeddings for the batch
    embedding_ref = model.embed.remote(sample_inputs)
    embeddings = ray.get(embedding_ref)

    elapsed_time = time.time() - start_time

    # Verify we got the correct number of embeddings
    assert len(embeddings) == batch_size

    # Check that all embeddings have the expected properties
    for embedding in embeddings:
        assert embedding is not None
        assert len(embedding) == 768  # Expected dimension
        assert embedding.dtype == np.float32
        assert not np.all(embedding == 0)  # Should not be all zeros

    # Log performance metrics
    print(
        f"{model_name} - Batch size {batch_size}: processed in {elapsed_time:.3f}s, "
        f"{elapsed_time / batch_size:.3f}s per item"
    )


@pytest.mark.slow
def test_nomic_embedding_actor_pool(embedding_model_actors):
    """Test that the Nomic embedding model works correctly with actor pooling."""
    # Create a batch of text inputs
    batch_size = 32
    num_batches = 4
    total_samples = batch_size * num_batches

    text_samples = [f"Sample text {i}" for i in range(total_samples)]
    batches = [
        text_samples[i : i + batch_size] for i in range(0, total_samples, batch_size)
    ]

    # Use the existing Nomic actor from the fixture for the main test
    nomic_actor = embedding_model_actors["Nomic"]

    # Create additional actors for the pool test (similar to engine.py implementation)
    model_name = "Nomic"
    model_class = get_actor_class(model_name)
    actor_count = 3  # Create 3 more for a total of 4
    additional_actors = [model_class.remote() for _ in range(actor_count)]
    actors = [nomic_actor] + additional_actors

    # Create an ActorPool
    pool = ray.util.ActorPool(actors)

    # Process batches in parallel using map_unordered
    # This returns an iterator that yields results directly
    batch_results = list(
        pool.map_unordered(lambda actor, batch: actor.embed.remote(batch), batches)
    )

    # Get results - batch_results already contains the actual results
    all_embeddings = []
    for batch_embeddings in batch_results:
        all_embeddings.extend(batch_embeddings)

    # Verify results
    assert len(all_embeddings) == total_samples

    # Check embedding properties
    for embedding in all_embeddings:
        assert embedding is not None
        assert len(embedding) == 768  # Expected dimension
        assert embedding.dtype == np.float32
        assert not np.all(embedding == 0)  # Should not be all zeros

    # Test for determinism by running the same input through different actors
    test_input = ["Test determinism across actors"]
    embeddings_from_actors = []

    for actor in actors:
        embedding_ref = actor.embed.remote(test_input)
        embedding = ray.get(embedding_ref)[0]
        embeddings_from_actors.append(embedding)

    # Verify all actors produce the same embedding for the same input
    for i in range(1, len(embeddings_from_actors)):
        assert np.array_equal(embeddings_from_actors[0], embeddings_from_actors[i])

    # Clean up only the additional actors we created
    for actor in additional_actors:
        ray.kill(actor)




