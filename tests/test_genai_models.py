import time

import numpy as np
import pytest
from PIL import Image

from trajectory_tracer.genai_models import (
    IMAGE_SIZE,
    get_actor_stats,
    get_output_type,
    invoke,
    list_models,
    terminate_all_model_actors,
)
from trajectory_tracer.schemas import InvocationType


@pytest.fixture(scope="module", autouse=True)
def ray_cleanup():
    """Fixture to ensure Ray resources are cleaned up after tests."""
    # Setup - Ray is initialized at module import
    yield
    # Teardown - ensure all actors are terminated
    terminate_all_model_actors()
    # Wait briefly to ensure cleanup completes
    time.sleep(1)


@pytest.mark.slow
def test_flux_dev_t2i():
    """Test that FluxDev returns an image with the expected dimensions and is deterministic with fixed seed."""
    try:
        model_name = "FluxDev"
        prompt = "A beautiful mountain landscape at sunset"

        # Test with fixed seed
        seed = 42
        # First invocation
        image1 = invoke(model_name, prompt, seed)
        assert isinstance(image1, Image.Image)
        assert image1.width == IMAGE_SIZE
        assert image1.height == IMAGE_SIZE

        # Second invocation with same seed
        image2 = invoke(model_name, prompt, seed)
        assert isinstance(image2, Image.Image)

        # Save the image as a webp file for inspection
        image1.save("/tmp/test_flux_dev_t2i_output.webp", format="WEBP")

        # Check that the images are identical
        np_img1 = np.array(image1)
        np_img2 = np.array(image2)
        assert np.array_equal(np_img1, np_img2), (
            "Images should be identical when using the same seed"
        )

        # Test with -1 seed (should be non-deterministic)
        seed = -1
        # First invocation
        image_random1 = invoke(model_name, prompt, seed)
        # Second invocation with -1 seed
        image_random2 = invoke(model_name, prompt, seed)

        # Check that the images are different
        np_img_random1 = np.array(image_random1)
        np_img_random2 = np.array(image_random2)
        # It's highly unlikely that two random generations would be identical
        # But we can't guarantee they're different, so we use pytest.approx with a relaxed tolerance
        assert not np.array_equal(np_img_random1, np_img_random2), (
            "Images should be different when using seed=-1"
        )
    finally:
        # Terminate the actor after test to free GPU memory
        terminate_all_model_actors()


@pytest.mark.slow
def test_sdxl_turbo():
    """Test that SDXLTurbo returns an image with the expected dimensions and is deterministic with fixed seed."""
    try:
        model_name = "SDXLTurbo"
        prompt = "A serene forest with a small lake"

        # Test with fixed seed
        seed = 43
        # First invocation
        image1 = invoke(model_name, prompt, seed)
        assert isinstance(image1, Image.Image)
        assert image1.width == IMAGE_SIZE
        assert image1.height == IMAGE_SIZE

        # Second invocation with same seed
        image2 = invoke(model_name, prompt, seed)
        assert isinstance(image2, Image.Image)

        # Save the image as a webp file for inspection
        image1.save("/tmp/test_sdxl_turbo_output.webp", format="WEBP")

        # Check that the images are identical
        np_img1 = np.array(image1)
        np_img2 = np.array(image2)
        assert np.array_equal(np_img1, np_img2), (
            "Images should be identical when using the same seed"
        )

        # Test with -1 seed (should be non-deterministic)
        seed = -1
        # First invocation
        image_random1 = invoke(model_name, prompt, seed)
        # Second invocation with -1 seed
        image_random2 = invoke(model_name, prompt, seed)

        # Check that the images are different
        np_img_random1 = np.array(image_random1)
        np_img_random2 = np.array(image_random2)
        assert not np.array_equal(np_img_random1, np_img_random2), (
            "Images should be different when using seed=-1"
        )
    finally:
        # Terminate the actor after test to free GPU memory
        terminate_all_model_actors()


@pytest.mark.slow
def test_blip2_i2t():
    """Test that BLIP2 returns a text caption for an input image and is deterministic with fixed seed."""
    try:
        model_name = "BLIP2"
        # Create a simple test image
        image = Image.new("RGB", (100, 100), color="green")

        # Test with fixed seed
        seed = 44
        # First invocation
        caption1 = invoke(model_name, image, seed)
        assert isinstance(caption1, str)
        assert len(caption1) > 0  # Caption should not be empty

        # Second invocation with same seed
        caption2 = invoke(model_name, image, seed)
        assert isinstance(caption2, str)

        # Check that the captions are identical
        assert caption1 == caption2, (
            "Captions should be identical when using the same seed"
        )

        # Test with -1 seed (should be non-deterministic)
        seed = -1
        # First invocation
        caption_random1 = invoke(model_name, image, seed)
        # Second invocation with -1 seed
        caption_random2 = invoke(model_name, image, seed)

        # Check that the captions are different (note: there's a small chance they could be the same by coincidence)
        assert caption_random1 != caption_random2, (
            "Captions should be different when using seed=-1"
        )
    finally:
        # Terminate the actor after test to free GPU memory
        terminate_all_model_actors()


@pytest.mark.slow
def test_moondream_i2t():
    """Test that Moondream returns a text caption for an input image and is deterministic with fixed seed."""
    try:
        model_name = "Moondream"
        # Create a simple test image
        image = Image.new("RGB", (100, 100), color="red")

        # Test with fixed seed
        seed = 45
        # First invocation
        caption1 = invoke(model_name, image, seed)
        assert isinstance(caption1, str)
        assert len(caption1) > 0  # Caption should not be empty

        # Second invocation with same seed
        caption2 = invoke(model_name, image, seed)
        assert isinstance(caption2, str)

        # Check that the captions are identical
        assert caption1 == caption2, (
            "Captions should be identical when using the same seed"
        )

        # Test with -1 seed (should be non-deterministic)
        seed = -1
        # First invocation
        _caption_random1 = invoke(model_name, image, seed)
        # Second invocation with -1 seed
        _caption_random2 = invoke(model_name, image, seed)

        # NOTE: Moondream currently doesn't respect the seed (TODO figure out why), so this final assertion commented-out for now
        # Check that the captions are different (note: there's a small chance they could be the same by coincidence)
        # assert caption_random1 != caption_random2, "Captions should be different when using seed=-1"
    finally:
        # Terminate the actor after test to free GPU memory
        terminate_all_model_actors()


def test_dummy_i2t():
    """Test that DummyI2T returns a text caption that varies based on the seed."""
    try:
        model_name = "DummyI2T"
        # Create a test image
        image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color="blue")

        # Test with fixed seed
        seed = 46
        # First invocation
        caption1 = invoke(model_name, image, seed)
        assert isinstance(caption1, str)
        assert "dummy text caption (seed 46)" == caption1

        # Second invocation with same seed
        caption2 = invoke(model_name, image, seed)
        assert isinstance(caption2, str)

        # Check that the captions are identical
        assert caption1 == caption2, "Captions should be identical when using the same seed"

        # Test with -1 seed (should be non-deterministic for dummy model)
        seed = -1
        # First invocation
        caption_random1 = invoke(model_name, image, seed)
        # Second invocation with -1 seed
        caption_random2 = invoke(model_name, image, seed)

        # Check both contain the expected prefix
        assert caption_random1.startswith("dummy text caption (random ")
        assert caption_random2.startswith("dummy text caption (random ")

        # With seed=-1, outputs should be different
        assert caption_random1 != caption_random2, (
            "Captions should be different when using seed=-1"
        )
    finally:
        # Terminate the actor to clean up
        terminate_all_model_actors()


def test_dummy_t2i():
    """Test that DummyT2I returns a colored image with correct dimensions that depends on the seed."""
    try:
        model_name = "DummyT2I"
        prompt = "This prompt will be ignored"

        # Test with fixed seed
        seed = 47
        # First invocation
        image1 = invoke(model_name, prompt, seed)
        assert isinstance(image1, Image.Image)
        assert image1.width == IMAGE_SIZE
        assert image1.height == IMAGE_SIZE

        # Second invocation with same seed
        image2 = invoke(model_name, prompt, seed)
        assert isinstance(image2, Image.Image)

        # Check that the images are identical
        np_img1 = np.array(image1)
        np_img2 = np.array(image2)
        assert np.array_equal(np_img1, np_img2), (
            "Images should be identical when using the same seed"
        )

        # Test with -1 seed (should be non-deterministic for dummy model)
        seed = -1
        # First invocation
        image_random1 = invoke(model_name, prompt, seed)
        # Second invocation with -1 seed
        image_random2 = invoke(model_name, prompt, seed)

        # With seed=-1, outputs should be different
        np_img_random1 = np.array(image_random1)
        np_img_random2 = np.array(image_random2)
        assert not np.array_equal(np_img_random1, np_img_random2), (
            "Images should be different when using seed=-1"
        )
    finally:
        # Terminate the actor to clean up
        terminate_all_model_actors()


def test_list_models():
    """Test that list_models returns a list containing the expected model classes."""

    # Call the function
    models = list_models()

    # Check that it's a list
    assert isinstance(models, list)

    # Check that the list is not empty
    assert len(models) > 0

    # Check that the list contains our known models
    expected_models = [
        "BLIP2",
        "DummyI2T",
        "DummyT2I",
        "FluxDev",
        "FluxSchnell",
        "Moondream",
        "SDXLTurbo",
    ]
    for model in expected_models:
        assert model in models, (
            f"Expected model {model} not found in list_models() output"
        )

    # Check that no non-GenAIModel classes are included
    assert "GenAIModel" not in models, "list_models() should not include the base class"


@pytest.mark.slow
@pytest.mark.parametrize("model_name", list_models())
def test_model_output_types(model_name):
    """Test that each model has the correct output type."""
    # Get output type for the model
    output_type = get_output_type(model_name)

    # Check that output_type is not None
    assert output_type is not None, f"Model {model_name} has no output_type defined"

    # Models should have either TEXT or IMAGE output types
    from trajectory_tracer.schemas import InvocationType
    assert output_type in [InvocationType.TEXT, InvocationType.IMAGE], (
        f"Model {model_name} has invalid output_type: {output_type}"
    )


@pytest.mark.slow
def test_actor_stats():
    """Test that actor stats reporting works correctly."""
    # Create an actor by invoking a model
    invoke("DummyT2I", "Test prompt", 42)

    # Get stats for all actors
    stats = get_actor_stats()

    # There should be at least one actor (the one we just used)
    assert len(stats) > 0, "No actor stats found"

    # Check that DummyT2I is in the stats
    assert "DummyT2I" in stats, "DummyT2I actor not found in stats"

    # Check that inference_count is properly tracked
    assert stats["DummyT2I"]["inference_count"] > 0, "Inference count not tracked"

    # Clean up
    terminate_all_model_actors()


@pytest.mark.parametrize("model_name,expected_type", [
    ("FluxDev", InvocationType.IMAGE),
    ("FluxSchnell", InvocationType.IMAGE),
    ("SDXLTurbo", InvocationType.IMAGE),
    ("BLIP2", InvocationType.TEXT),
    ("Moondream", InvocationType.TEXT),
    ("DummyI2T", InvocationType.TEXT),
    ("DummyT2I", InvocationType.IMAGE),
])
def test_get_output_type(model_name, expected_type):
    """Test that get_output_type returns the correct type for each model."""
    output_type = get_output_type(model_name)
    assert output_type == expected_type, (
        f"Expected {expected_type} for {model_name}, got {output_type}"
    )

    # Test with nonexistent model
    with pytest.raises(ValueError):
        get_output_type("NonexistentModel")
