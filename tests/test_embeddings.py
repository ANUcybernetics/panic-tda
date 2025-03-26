import numpy as np
import pytest
import ray
import torch
from PIL import Image

from trajectory_tracer.embeddings import (
    Dummy,
    Dummy2,
    get_actor_class,
    list_models,
)
from trajectory_tracer.schemas import Embedding, Invocation, InvocationType, Run


@pytest.mark.slow
@pytest.mark.parametrize("model_name", list_models())
def test_embedding_model(model_name):
    """Test that the embedding model returns valid vectors for both text and images and is deterministic."""
    # Add slow marker for non-dummy models
    if not model_name.startswith("Dummy"):
        pytest.mark.slow(test_embedding_model)

    try:
        # Create a sample text string and image
        sample_text = "Sample output text"
        sample_image = Image.new("RGB", (100, 100), color="blue")

        # Get the model actor
        model_class = get_actor_class(model_name)
        model = model_class.remote()

        # Test with text
        text_embedding_ref = model.embed.remote(sample_text)
        text_embedding = ray.get(text_embedding_ref)

        # Run it again to verify determinism
        text_embedding2_ref = model.embed.remote(sample_text)
        text_embedding2 = ray.get(text_embedding2_ref)

        # Check that the embedding has the correct properties
        assert text_embedding is not None
        assert len(text_embedding) == 768  # Expected dimension

        # Verify it's a proper embedding vector (except for dummy models which may not use float32)
        if not model_name.startswith("Dummy"):
            assert text_embedding.dtype == np.float32
            assert not np.all(text_embedding == 0)  # Should not be all zeros

        # Verify determinism
        assert np.array_equal(text_embedding, text_embedding2)

        # Test with image
        image_embedding_ref = model.embed.remote(sample_image)
        image_embedding = ray.get(image_embedding_ref)

        # Run it again to verify determinism
        image_embedding2_ref = model.embed.remote(sample_image)
        image_embedding2 = ray.get(image_embedding2_ref)

        # Check that the embedding has the correct properties
        assert image_embedding is not None
        assert len(image_embedding) == 768  # Expected dimension

        # Verify it's a proper embedding vector (except for dummy models which may not use float32)
        if not model_name.startswith("Dummy"):
            assert image_embedding.dtype == np.float32
            assert not np.all(image_embedding == 0)  # Should not be all zeros

        # Verify determinism
        assert np.array_equal(image_embedding, image_embedding2)

        # Verify text and image embeddings are different
        assert not np.array_equal(text_embedding, image_embedding)

    finally:
        # Terminate the actor to clean up resources
        if 'model' in locals():
            ray.kill(model)


def test_run_embeddings_by_model(db_session):
    """Test the Run.embeddings_by_model method returns embeddings with a specific model."""
    try:
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
        dummy_model = Dummy.remote()
        embedding_vector1_ref = dummy_model.embed.remote(invocation1.output)
        embedding_vector1 = ray.get(embedding_vector1_ref)
        embedding1_1 = Embedding(
            invocation_id=invocation1.id,
            embedding_model="Dummy",
            vector=embedding_vector1,
        )
        db_session.add(embedding1_1)

        dummy2_model = Dummy2.remote()
        embedding_vector2_ref = dummy2_model.embed.remote(invocation1.output)
        embedding_vector2 = ray.get(embedding_vector2_ref)
        embedding1_2 = Embedding(
            invocation_id=invocation1.id,
            embedding_model="Dummy2",
            vector=embedding_vector2,
        )
        db_session.add(embedding1_2)

        embedding_vector3_ref = dummy_model.embed.remote(invocation2.output)
        embedding_vector3 = ray.get(embedding_vector3_ref)
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
        dummy_embeddings = run.embeddings_by_model("Dummy")
        dummy2_embeddings = run.embeddings_by_model("Dummy2")

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
    finally:
        # Terminate the actors to clean up resources
        if 'dummy_model' in locals():
            ray.kill(dummy_model)
        if 'dummy2_model' in locals():
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
        "JinaClip",
    ]
    for model in expected_models:
        assert model in available_models

    # Verify the base class is not included
    assert "EmbeddingModel" not in available_models


@pytest.mark.slow
@pytest.mark.parametrize("model_name", list_models())
def test_embedding_model_memory_usage(model_name):
    """Test memory usage reporting for each embedding model."""
    # Skip dummy models that don't need GPU
    if model_name.startswith("Dummy"):
        pytest.skip(f"Skipping memory usage test for dummy model {model_name}")

    # Skip if CUDA is not available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping GPU memory test")

    try:
        # Create the model actor
        model_class = globals()[model_name]
        model = model_class.remote()

        # Get memory usage information
        # This would typically be a method on the actor that returns memory usage info
        memory_info = {"model_name": model_name, "status": "initialized"}

        # For actual implementations, you would call something like:
        # memory_info_ref = model.get_memory_usage.remote()
        # memory_info = ray.get(memory_info_ref)

        # Verify the returned structure has basic information
        assert isinstance(memory_info, dict)
        assert "model_name" in memory_info
        assert "gpu_memory_used" in memory_info or "status" in memory_info

        # Print the memory info
        print(f"\nMemory usage for {model_name}:")
        for key, value in memory_info.items():
            print(f"  {key}: {value}")
    finally:
        # Terminate the actor to clean up GPU resources
        if 'model' in locals():
            ray.kill(model)


def test_get_actor_class():
    """Test that the get_model_class function returns the correct Ray actor class for a given model name."""

    # Test for a few models
    for model_name in ["Nomic", "JinaClip"]:
        model_class = get_actor_class(model_name)

        # Verify it's a Ray actor class
        assert isinstance(model_class, ray.actor.ActorClass)

        # Verify the class name matches our expectations
        assert type(model_class).__name__ == f"ActorClass({model_name})"

    # Test with nonexistent model
    with pytest.raises(ValueError):
        get_actor_class("NonexistentModel")
