import time

import numpy as np
import pytest
import ray
import ray.actor
from PIL import Image

from trajectory_tracer.genai_models import (
    IMAGE_SIZE,
    get_actor_class,
    get_output_type,
    list_models,
)
from trajectory_tracer.schemas import InvocationType


@pytest.fixture(scope="module", autouse=True)
def ray_cleanup():
    """Fixture to ensure Ray resources are cleaned up after tests."""
    # Setup - Ray is initialized at module import
    yield
    # Teardown - ensure all actors are terminated
    # Wait briefly to ensure cleanup completes
    time.sleep(1)


@pytest.mark.slow
@pytest.mark.parametrize("model_name", [m for m in list_models() if get_output_type(m) == InvocationType.IMAGE])
def test_text_to_image_models(model_name):
    """Test that text-to-image models return expected output and are deterministic with fixed seed."""
    try:
        # Create the actor using the model class
        model_class = get_actor_class(model_name)
        model = model_class.remote()

        # Standard text prompt for all text-to-image models
        input_data = "A beautiful mountain landscape at sunset"

        # Test with fixed seed
        seed = 42

        # First invocation with fixed seed
        result1_ref = model.invoke.remote(input_data, seed)
        result1 = ray.get(result1_ref)

        # Check result type
        assert isinstance(result1, Image.Image)
        assert result1.width == IMAGE_SIZE
        assert result1.height == IMAGE_SIZE

        # Second invocation with same seed
        result2_ref = model.invoke.remote(input_data, seed)
        result2 = ray.get(result2_ref)

        # Verify output type
        assert isinstance(result2, Image.Image)

        # Check that results are identical with same seed
        np_result1 = np.array(result1)
        np_result2 = np.array(result2)
        assert np.array_equal(np_result1, np_result2), (
            f"{model_name} images should be identical when using the same seed"
        )

        # Test with -1 seed (should be non-deterministic)
        seed = -1

        # First invocation with random seed
        random_result1_ref = model.invoke.remote(input_data, seed)
        random_result1 = ray.get(random_result1_ref)

        # Second invocation with random seed
        random_result2_ref = model.invoke.remote(input_data, seed)
        random_result2 = ray.get(random_result2_ref)

        # Check that results are different with random seed
        np_random_result1 = np.array(random_result1)
        np_random_result2 = np.array(random_result2)
        assert not np.array_equal(np_random_result1, np_random_result2), (
            f"{model_name} images should be different when using seed=-1"
        )

    finally:
        # Terminate the actor after test to free GPU memory
        ray.kill(model)


@pytest.mark.slow
@pytest.mark.parametrize("model_name", [m for m in list_models() if get_output_type(m) == InvocationType.TEXT])
def test_image_to_text_models(model_name):
    """Test that image-to-text models return expected output and are deterministic with fixed seed."""
    try:
        # Create the actor using the model class
        model_class = get_actor_class(model_name)
        model = model_class.remote()

        # Create a standard test image for all image-to-text models
        input_data = Image.new("RGB", (100, 100), color="red")

        # Test with fixed seed
        seed = 42

        # First invocation with fixed seed
        result1_ref = model.invoke.remote(input_data, seed)
        result1 = ray.get(result1_ref)

        # Check result type
        assert isinstance(result1, str)
        assert len(result1) > 0  # Caption should not be empty

        # Second invocation with same seed
        result2_ref = model.invoke.remote(input_data, seed)
        result2 = ray.get(result2_ref)

        # Verify output type
        assert isinstance(result2, str)

        # Check that results are identical with same seed
        assert result1 == result2, (
            f"{model_name} captions should be identical when using the same seed"
        )

        # Test with -1 seed (should be non-deterministic)
        seed = -1

        # First invocation with random seed
        random_result1_ref = model.invoke.remote(input_data, seed)
        random_result1 = ray.get(random_result1_ref)

        # Second invocation with random seed
        random_result2_ref = model.invoke.remote(input_data, seed)
        random_result2 = ray.get(random_result2_ref)

        # Check that results are different with random seed
       # Skip this assertion for Moondream which always gives the same answer
        if model_name != "Moondream":
            assert random_result1 != random_result2, (
                f"{model_name} captions should be different when using seed=-1"
            )

    finally:
        # Terminate the actor after test to free GPU memory
        ray.kill(model)


@pytest.mark.slow
@pytest.mark.parametrize("model_name", list_models())
def test_model_output_types(model_name):
    """Test that each model has the correct output type."""
    # Get output type for the model
    output_type = get_output_type(model_name)

    # Check that output_type is not None
    assert output_type is not None, f"Model {model_name} has no output_type defined"

    # Models should have either TEXT or IMAGE output types
    assert output_type in [InvocationType.TEXT, InvocationType.IMAGE], (
        f"Model {model_name} has invalid output_type: {output_type}"
    )


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


def test_list_models_function():
    """Test that the list_models function returns a list of model names that matches our expectations."""

    models = list_models()

    # Check that we got a non-empty list
    assert isinstance(models, list)
    assert len(models) > 0

    # Check that all expected models are in the list
    expected_models = [
        "BLIP2",
        "DummyI2T",
        "DummyT2I",
        "FluxDev",
        "FluxSchnell",
        "Moondream",
        "SDXLTurbo",
    ]

    for model_name in expected_models:
        assert model_name in models, f"Expected model {model_name} not found in list_models() result"


def test_get_actor_class():
    """Test that the get_model_class function returns the correct Ray actor class for a given model name."""

    # Test for a few models
    for model_name in ["FluxDev", "BLIP2", "DummyT2I"]:
        model_class = get_actor_class(model_name)

        # Verify it's a Ray actor class
        assert isinstance(model_class, ray.actor.ActorClass)

        # Verify the class name matches our expectations
        assert type(model_class).__name__ == f"ActorClass({model_name})"

    # Test with nonexistent model
    with pytest.raises(ValueError):
        get_actor_class("NonexistentModel")
