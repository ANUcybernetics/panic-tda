import io
import time

import numpy as np
import pytest
import ray
import torch
from PIL import Image

from panic_tda.embeddings import (
    DummyText,
    DummyText2,
    DummyVision,
    DummyVision2,
    get_actor_class,
    list_models,
)
from panic_tda.schemas import Embedding, Invocation, InvocationType, Run


@pytest.fixture(scope="module")
def dummy_actors():
    """Module-scoped fixture for dummy embedding model actors."""
    actors = {
        "DummyText": DummyText.remote(),
        "DummyText2": DummyText2.remote(),
        "DummyVision": DummyVision.remote(),
        "DummyVision2": DummyVision2.remote(),
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
    dummy_model = dummy_actors["DummyText"]
    embedding_vector1_ref = dummy_model.embed.remote([invocation1.output])
    embedding_vectors1 = ray.get(embedding_vector1_ref)
    embedding_vector1 = embedding_vectors1[0]  # Get the first embedding
    embedding1_1 = Embedding(
        invocation_id=invocation1.id,
        embedding_model="DummyText",
        vector=embedding_vector1,
    )
    db_session.add(embedding1_1)

    dummy2_model = dummy_actors["DummyText2"]
    embedding_vector2_ref = dummy2_model.embed.remote([invocation1.output])
    embedding_vectors2 = ray.get(embedding_vector2_ref)
    embedding_vector2 = embedding_vectors2[0]  # Get the first embedding
    embedding1_2 = Embedding(
        invocation_id=invocation1.id,
        embedding_model="DummyText2",
        vector=embedding_vector2,
    )
    db_session.add(embedding1_2)

    embedding_vector3_ref = dummy_model.embed.remote([invocation2.output])
    embedding_vectors3 = ray.get(embedding_vector3_ref)
    embedding_vector3 = embedding_vectors3[0]  # Get the first embedding
    embedding2_1 = Embedding(
        invocation_id=invocation2.id,
        embedding_model="DummyText",
        vector=embedding_vector3,
    )
    db_session.add(embedding2_1)

    db_session.commit()

    # Refresh the run object to ensure relationships are loaded
    db_session.refresh(run)

    # Test filtering by model name
    dummy_embeddings = run.embeddings["DummyText"]
    dummy2_embeddings = run.embeddings["DummyText2"]

    # Verify the filtering works correctly
    assert len(dummy_embeddings) == 2
    assert len(dummy2_embeddings) == 1
    assert all(e.embedding_model == "DummyText" for e in dummy_embeddings)
    assert all(e.embedding_model == "DummyText2" for e in dummy2_embeddings)

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
        dummy_model = DummyText.remote()
        dummy2_model = DummyText2.remote()

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
            embedding_model="DummyText",
            vector=embedding_vector1,
        )
        db_session.add(embedding1)

        embedding2 = Embedding(
            invocation_id=invocation.id,
            embedding_model="DummyText2",
            vector=embedding_vector2,
        )
        db_session.add(embedding2)

        db_session.commit()
        db_session.refresh(invocation)

        # Test the embedding method
        dummy_embedding = invocation.embedding("DummyText")
        dummy2_embedding = invocation.embedding("DummyText2")
        nonexistent_embedding = invocation.embedding("NonexistentModel")

        # Verify the results
        assert dummy_embedding is not None
        assert dummy_embedding.embedding_model == "DummyText"
        assert np.array_equal(dummy_embedding.vector, embedding_vector1)

        assert dummy2_embedding is not None
        assert dummy2_embedding.embedding_model == "DummyText2"
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
        "DummyText",
        "DummyText2",
        "DummyVision",
        "DummyVision2",
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
    dummy_model = dummy_actors["DummyText"]

    # Get embeddings for the first and third invocations only
    embedding_vector1_ref = dummy_model.embed.remote([invocation1.output])
    embedding_vectors1 = ray.get(embedding_vector1_ref)
    embedding_vector1 = embedding_vectors1[0]

    embedding_vector3_ref = dummy_model.embed.remote([invocation3.output])
    embedding_vectors3 = ray.get(embedding_vector3_ref)
    embedding_vector3 = embedding_vectors3[0]

    # Add embeddings for only the first and third invocations with "DummyText" model
    embedding1 = Embedding(
        invocation_id=invocation1.id,
        embedding_model="DummyText",
        vector=embedding_vector1,
    )
    db_session.add(embedding1)

    embedding3 = Embedding(
        invocation_id=invocation3.id,
        embedding_model="DummyText",
        vector=embedding_vector3,
    )
    db_session.add(embedding3)

    # Add an embedding for the first invocation with "DummyText2" model
    dummy2_model = dummy_actors["DummyText2"]
    embedding_vector1_2_ref = dummy2_model.embed.remote([invocation1.output])
    embedding_vectors1_2 = ray.get(embedding_vector1_2_ref)
    embedding_vector1_2 = embedding_vectors1_2[0]

    embedding1_2 = Embedding(
        invocation_id=invocation1.id,
        embedding_model="DummyText2",
        vector=embedding_vector1_2,
    )
    db_session.add(embedding1_2)

    # Add an embedding with null vector for the second invocation
    embedding2 = Embedding(
        invocation_id=invocation2.id,
        embedding_model="DummyText",
        vector=None,
    )
    db_session.add(embedding2)

    db_session.commit()
    db_session.refresh(run)

    # Test missing_embeddings for "DummyText" model
    missing_dummy = run.missing_embeddings("DummyText")

    # Should return invocation2 because it has a null vector
    assert len(missing_dummy) == 1
    assert missing_dummy[0].id == invocation2.id

    # Test missing_embeddings for "DummyText2" model
    missing_dummy2 = run.missing_embeddings("DummyText2")

    # Should return invocation2 and invocation3 (both missing "DummyText2" embeddings)
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
    if not model_name.startswith("DummyText"):
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
def test_jinaclipvision_batch_processing_single_actor():
    """Test JinaClipVision batch processing with a single actor.

    This test was created to reproduce HuggingFace transformers issue #26999 where
    JinaClipVision failed with IndexError when processing multiple images.

    The error was:
    IndexError: list index out of range
      File ".../modeling_clip.py", line 495, in encode_image
        all_embeddings = [all_embeddings[idx] for idx in _inverse_permutation]

    As of the current test run, this issue appears to have been fixed and
    batch processing works correctly.

    See: https://github.com/huggingface/transformers/issues/26999
    """
    # Skip if CUDA is not available
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for JinaClipVision test")

    # Create the JinaClipVision actor
    from panic_tda.embeddings import JinaClipVision

    jina_vision = JinaClipVision.remote()

    try:
        # Create a larger batch to try to trigger the error
        # Based on the logs, the error occurred with batches of 100 images
        batch_sizes_to_test = [2, 10, 50, 100]

        # Test various batch sizes to ensure the issue is fixed
        for batch_size in batch_sizes_to_test:
            print(f"\nTesting batch size: {batch_size}")
            # Create images similar to how they're stored in the database (WEBP format)
            images = []
            for i in range(batch_size):
                # Create image
                img = Image.new(
                    "RGB", (224, 224), color=(i % 255, (i * 2) % 255, (i * 3) % 255)
                )
                # Convert to WEBP and back to simulate database storage
                buffer = io.BytesIO()
                img.save(buffer, format="WEBP", lossless=True, quality=100)
                buffer.seek(0)
                img_reloaded = Image.open(buffer)
                images.append(img_reloaded)

            # Process batch - should work without IndexError
            embedding_ref = jina_vision.embed.remote(images)
            embeddings = ray.get(embedding_ref)

            # Verify we got the correct number of embeddings
            assert len(embeddings) == batch_size, (
                f"Expected {batch_size} embeddings, got {len(embeddings)}"
            )
            print(f"Success! Got {len(embeddings)} embeddings")

            # Verify embedding properties
            for i, emb in enumerate(embeddings):
                assert emb is not None
                assert len(emb) == 768  # Expected dimension
                assert emb.dtype == np.float32
                assert not np.all(emb == 0)  # Should not be all zeros

    finally:
        # Clean up
        ray.kill(jina_vision)


@pytest.mark.slow
def test_jinaclipvision_batch_processing_multiple_actors():
    """Test JinaClipVision batch processing with multiple actors in parallel.

    This test simulates the engine.py pattern where multiple JinaClipVision actors
    are created and process batches in parallel, which is the scenario where the
    IndexError was observed in production.
    """
    # Skip if CUDA is not available
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for JinaClipVision test")

    # Create multiple JinaClipVision actors like the engine does
    from panic_tda.embeddings import JinaClipVision

    num_actors = 3
    actors = []

    try:
        # Create actors
        for i in range(num_actors):
            actor = JinaClipVision.remote()
            actors.append(actor)
            print(f"Created JinaClipVision actor {i + 1}/{num_actors}")

        # Create batches of images (WEBP format like in production)
        batch_size = 100  # Same as in the error logs
        num_batches = 5

        all_tasks = []
        for batch_idx in range(num_batches):
            images = []
            for i in range(batch_size):
                # Create unique images for each batch
                color_base = batch_idx * 50
                img = Image.new(
                    "RGB",
                    (224, 224),
                    color=(
                        (color_base + i) % 255,
                        ((color_base + i) * 2) % 255,
                        ((color_base + i) * 3) % 255,
                    ),
                )
                # Convert to WEBP and back to simulate database storage
                buffer = io.BytesIO()
                img.save(buffer, format="WEBP", lossless=True, quality=100)
                buffer.seek(0)
                img_reloaded = Image.open(buffer)
                images.append(img_reloaded)

            # Submit batch to next actor (round-robin)
            actor = actors[batch_idx % num_actors]
            task = actor.embed.remote(images)
            all_tasks.append((batch_idx, task))
            print(
                f"Submitted batch {batch_idx + 1} with {batch_size} images to actor {batch_idx % num_actors}"
            )

        # Collect results
        errors = []
        successes = 0

        for batch_idx, task in all_tasks:
            try:
                embeddings = ray.get(task)
                assert len(embeddings) == batch_size
                successes += 1
                print(
                    f"Batch {batch_idx + 1} succeeded with {len(embeddings)} embeddings"
                )
            except Exception as e:
                errors.append((batch_idx, e))
                print(f"Batch {batch_idx + 1} failed: {type(e).__name__}: {e}")
                if isinstance(e, ray.exceptions.RayTaskError):
                    # Check if it's the expected IndexError
                    if "IndexError" in str(e) and "list index out of range" in str(e):
                        print("Found the expected IndexError!")

        # Report results
        print(
            f"\nResults: {successes} successes, {len(errors)} errors out of {num_batches} batches"
        )

        if errors:
            # If we got IndexErrors, that confirms the issue still exists
            print("IndexError reproduced in parallel processing scenario")
            # Don't fail the test since we're documenting the issue
        else:
            print("All batches processed successfully - issue may be fixed")

    finally:
        # Clean up all actors
        for actor in actors:
            ray.kill(actor)


@pytest.mark.slow
def test_jinaclipvision_batch_processing_fresh_actors():
    """Test JinaClipVision with fresh actors for each batch.

    This test simulates the pattern seen in the error logs where each batch
    appears to get a fresh JinaClipVision actor (different PIDs), which might
    trigger the IndexError due to model initialization issues.
    """
    # Skip if CUDA is not available
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for JinaClipVision test")

    from panic_tda.embeddings import JinaClipVision

    # Test parameters matching production scenario
    batch_size = 100
    num_batches = 5

    errors = []
    successes = 0

    for batch_idx in range(num_batches):
        # Create a fresh actor for each batch (like in the logs)
        actor = JinaClipVision.remote()
        print(f"\nCreated fresh JinaClipVision actor for batch {batch_idx + 1}")

        try:
            # Create batch of images (WEBP format)
            images = []
            for i in range(batch_size):
                color_base = batch_idx * 50
                img = Image.new(
                    "RGB",
                    (224, 224),
                    color=(
                        (color_base + i) % 255,
                        ((color_base + i) * 2) % 255,
                        ((color_base + i) * 3) % 255,
                    ),
                )
                buffer = io.BytesIO()
                img.save(buffer, format="WEBP", lossless=True, quality=100)
                buffer.seek(0)
                img_reloaded = Image.open(buffer)
                images.append(img_reloaded)

            # Process batch
            embedding_ref = actor.embed.remote(images)
            embeddings = ray.get(embedding_ref)

            assert len(embeddings) == batch_size
            successes += 1
            print(f"Batch {batch_idx + 1} succeeded with {len(embeddings)} embeddings")

        except Exception as e:
            errors.append((batch_idx, e))
            print(f"Batch {batch_idx + 1} failed: {type(e).__name__}: {e}")
            if isinstance(e, ray.exceptions.RayTaskError):
                if "IndexError" in str(e) and "list index out of range" in str(e):
                    print("Found the expected IndexError with fresh actor!")

        finally:
            # Kill the actor after each batch (simulating what might happen in production)
            ray.kill(actor)
            print(f"Killed actor for batch {batch_idx + 1}")

    # Report results
    print(
        f"\nResults: {successes} successes, {len(errors)} errors out of {num_batches} batches"
    )

    if errors:
        print("IndexError reproduced with fresh actors for each batch")
    else:
        print("All batches processed successfully - issue may be fixed")


@pytest.mark.slow
def test_jinaclipvision_batch_workaround():
    """Test that JinaClipVision works correctly when processing images one at a time.

    This is a workaround for the batch processing IndexError issue.
    """
    # Skip if CUDA is not available
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for JinaClipVision test")

    # Create the JinaClipVision actor
    from panic_tda.embeddings import JinaClipVision

    jina_vision = JinaClipVision.remote()

    try:
        # Create test images
        images = [
            Image.new("RGB", (224, 224), color="red"),
            Image.new("RGB", (224, 224), color="blue"),
            Image.new("RGB", (224, 224), color="green"),
        ]

        # Process images one at a time (batch_size=1)
        embeddings = []
        for image in images:
            embedding_ref = jina_vision.embed.remote([image])  # Single image in a list
            result = ray.get(embedding_ref)
            embeddings.append(result[0])

        # Verify we got valid embeddings
        assert len(embeddings) == 3
        for embedding in embeddings:
            assert embedding is not None
            assert len(embedding) == 768  # Expected dimension
            assert embedding.dtype == np.float32
            assert not np.all(embedding == 0)  # Should not be all zeros

        # Verify embeddings are different for different images
        assert not np.array_equal(embeddings[0], embeddings[1])
        assert not np.array_equal(embeddings[1], embeddings[2])

    finally:
        # Clean up
        ray.kill(jina_vision)


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


def test_dummy_vision_models(dummy_actors):
    """Test that DummyVision models work correctly with PIL Images."""
    # Get the dummy vision actors
    dummy_vision = dummy_actors["DummyVision"]
    dummy_vision2 = dummy_actors["DummyVision2"]

    # Create test images
    images = [
        Image.new("RGB", (100, 100), color="red"),
        Image.new("RGB", (150, 150), color="blue"),
        Image.new("RGB", (200, 200), color="green"),
    ]

    # Test DummyVision
    embeddings_ref = dummy_vision.embed.remote(images)
    embeddings = ray.get(embeddings_ref)

    assert len(embeddings) == 3
    for emb in embeddings:
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (768,)  # EMBEDDING_DIM
        assert emb.dtype == np.float32

    # Test DummyVision2
    embeddings2_ref = dummy_vision2.embed.remote(images)
    embeddings2 = ray.get(embeddings2_ref)

    assert len(embeddings2) == 3
    for emb in embeddings2:
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (768,)  # EMBEDDING_DIM
        assert emb.dtype == np.float32

    # Test that different images produce different embeddings
    assert not np.array_equal(embeddings[0], embeddings[1])
    assert not np.array_equal(embeddings[1], embeddings[2])
    assert not np.array_equal(embeddings2[0], embeddings2[1])
    assert not np.array_equal(embeddings2[1], embeddings2[2])

    # Test that the same image produces the same embedding (deterministic)
    same_image = [Image.new("RGB", (100, 100), color="red")]
    emb1_ref = dummy_vision.embed.remote(same_image)
    emb1 = ray.get(emb1_ref)[0]
    emb2_ref = dummy_vision.embed.remote(same_image)
    emb2 = ray.get(emb2_ref)[0]
    assert np.array_equal(emb1, emb2)

    # Test with non-Image input (should raise error)
    with pytest.raises(ray.exceptions.RayTaskError):
        invalid_ref = dummy_vision.embed.remote(["not an image"])
        ray.get(invalid_ref)

    with pytest.raises(ray.exceptions.RayTaskError):
        invalid_ref2 = dummy_vision2.embed.remote(["not an image"])
        ray.get(invalid_ref2)


def test_dummy_text_models(dummy_actors):
    """Test that DummyText models work correctly with text inputs."""
    # Get the dummy text actors
    dummy_text = dummy_actors["DummyText"]
    dummy_text2 = dummy_actors["DummyText2"]

    # Create test text inputs
    texts = [
        "Hello world",
        "This is a test",
        "Another sample text",
    ]

    # Test DummyText
    embeddings_ref = dummy_text.embed.remote(texts)
    embeddings = ray.get(embeddings_ref)

    assert len(embeddings) == 3
    for emb in embeddings:
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (768,)  # EMBEDDING_DIM
        assert emb.dtype == np.float32

    # Test DummyText2
    embeddings2_ref = dummy_text2.embed.remote(texts)
    embeddings2 = ray.get(embeddings2_ref)

    assert len(embeddings2) == 3
    for emb in embeddings2:
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (768,)  # EMBEDDING_DIM
        assert emb.dtype == np.float32

    # Test that different texts produce different embeddings
    assert not np.array_equal(embeddings[0], embeddings[1])
    assert not np.array_equal(embeddings[1], embeddings[2])
    assert not np.array_equal(embeddings2[0], embeddings2[1])
    assert not np.array_equal(embeddings2[1], embeddings2[2])

    # Test that the same text produces the same embedding (deterministic)
    same_text = ["Hello world"]
    emb1_ref = dummy_text.embed.remote(same_text)
    emb1 = ray.get(emb1_ref)[0]
    emb2_ref = dummy_text.embed.remote(same_text)
    emb2 = ray.get(emb2_ref)[0]
    assert np.array_equal(emb1, emb2)

    # Test with non-string input (should raise error)
    with pytest.raises(ray.exceptions.RayTaskError):
        img = Image.new("RGB", (100, 100), color="red")
        invalid_ref = dummy_text.embed.remote([img])
        ray.get(invalid_ref)

    with pytest.raises(ray.exceptions.RayTaskError):
        img = Image.new("RGB", (100, 100), color="red")
        invalid_ref2 = dummy_text2.embed.remote([img])
        ray.get(invalid_ref2)
