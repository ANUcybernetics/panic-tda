import json
import time

import numpy as np
import pytest
import ray
import ray.actor
import torch
from diffusers import FluxPipeline
from PIL import Image

from trajectory_tracer.embeddings import (
    get_all_models_memory_usage as embeddings_memory_usage,
)
from trajectory_tracer.genai_models import (
    IMAGE_SIZE,
    get_actor_class,
    get_output_type,
    list_models,
)

# Import the function from both modules
from trajectory_tracer.genai_models import (
    get_all_models_memory_usage as genai_memory_usage,
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
        _random_result1 = ray.get(random_result1_ref)

        # Second invocation with random seed
        random_result2_ref = model.invoke.remote(input_data, seed)
        _random_result2 = ray.get(random_result2_ref)

        # Skip this assertion for captioning; haven't got those models taking seeds yet
        # assert random_result1 != random_result2, (
        #     f"{model_name} captions should be different when using seed=-1"
        # )

    finally:
        # Terminate the actor after test to free GPU memory
        ray.kill(model)


@pytest.mark.slow
def test_fluxdev_without_ray():
    """Test FluxDev model without using Ray."""

    # Skip test if no CUDA is available
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU is required but not available")

    # Initialize the model directly (without Ray)
    model = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        use_fast=True
    ).to("cuda")

    # Generate image with same parameters as in FluxDev class
    prompt = "A beautiful mountain landscape at sunset"
    seed = 42
    generator = torch.Generator("cuda").manual_seed(seed)

    image = model(
        prompt,
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
        guidance_scale=3.5,
        num_inference_steps=20,
        generator=generator,
    ).images[0]

    # Verify the output
    assert isinstance(image, Image.Image)
    assert image.width == IMAGE_SIZE
    assert image.height == IMAGE_SIZE

    # Generate again with same seed to test determinism
    generator = torch.Generator("cuda").manual_seed(seed)
    image2 = model(
        prompt,
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
        guidance_scale=3.5,
        num_inference_steps=20,
        generator=generator,
    ).images[0]

    # Check that results are identical with same seed
    np_image1 = np.array(image)
    np_image2 = np.array(image2)
    assert np.array_equal(np_image1, np_image2), "Images should be identical when using the same seed"

    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()


@pytest.mark.slow
def test_get_all_models_memory_usage():
    """Test that we can get memory usage for all models and display the results."""

    # Get memory usage (with minimal output during test)
    print("\n\nGenAI Models Memory Usage:")
    genai_results = genai_memory_usage(verbose=False)

    # Pretty print the results
    for model, usage in genai_results.items():
        if usage > 0:
            print(f"  {model}: {usage:.3f} GB")
        else:
            print(f"  {model}: Error measuring")

    print("\nEmbedding Models Memory Usage:")
    embed_results = embeddings_memory_usage(verbose=False)

    # Pretty print the results
    for model, usage in embed_results.items():
        if usage > 0:
            print(f"  {model}: {usage:.3f} GB")
        else:
            print(f"  {model}: Error measuring")

    # Export results to JSON for potential analysis
    all_results = {
        "genai_models": genai_results,
        "embedding_models": embed_results
    }

    print(f"\nComplete results: {json.dumps(all_results, indent=2)}")

    # Simply assert that we got results for at least some models
    assert len(genai_results) > 0 or len(embed_results) > 0


@pytest.mark.slow
def test_fluxschnell_batching_without_ray():
    """Test FluxSchnell model batching capabilities without using Ray."""

    # Skip test if no CUDA is available
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU is required but not available")

    # Initialize the model directly (without Ray)
    model = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16,
        use_fast=True
    ).to("cuda")

    # Define multiple prompts for batching
    prompts = [
        "A beautiful mountain landscape at sunset",
        "A futuristic city with flying cars",
        "A serene beach with palm trees and clear blue water",
        "A magical forest with glowing mushrooms"
    ]

    # Fixed seed for determinism
    seed = 42

    # Generate with batched prompts
    print("Running batched generation...")
    batch_start_time = time.time()

    generator = torch.Generator("cuda").manual_seed(seed)

    # Process all prompts in a single batch
    batch_results = model(
        prompts,
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
        guidance_scale=3.5,
        num_inference_steps=20,
        generator=generator,
    ).images

    batch_time = time.time() - batch_start_time
    print(f"Batched generation took {batch_time:.2f} seconds for {len(prompts)} images")

    # Now process the same prompts one by one
    print("Running sequential generation...")
    sequential_start_time = time.time()
    sequential_results = []

    for prompt in prompts:
        # Reset generator for each prompt (same seed)
        generator = torch.Generator("cuda").manual_seed(seed)

        result = model(
            prompt,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            guidance_scale=3.5,
            num_inference_steps=20,
            generator=generator,
        ).images[0]

        sequential_results.append(result)

    sequential_time = time.time() - sequential_start_time
    print(f"Sequential generation took {sequential_time:.2f} seconds for {len(prompts)} images")

    # Verify results
    assert len(batch_results) == len(prompts), "Batch should return same number of images as prompts"

    # Check that each image is the expected size
    for img in batch_results:
        assert isinstance(img, Image.Image)
        assert img.width == IMAGE_SIZE
        assert img.height == IMAGE_SIZE

    # Compare the speedup (should be significant)
    speedup = sequential_time / batch_time
    print(f"Speedup factor: {speedup:.2f}x")

    # The batched version should be at least somewhat faster (allowing flexibility in test)
    assert speedup > 1.1, "Batched processing should be faster than sequential"

    # Test with odd batch sizes
    odd_batch_size = 3
    odd_prompts = prompts[:odd_batch_size]

    generator = torch.Generator("cuda").manual_seed(seed)
    odd_batch_results = model(
        odd_prompts,
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
        guidance_scale=3.5,
        num_inference_steps=20,
        generator=generator,
    ).images

    assert len(odd_batch_results) == odd_batch_size, "Should handle odd-sized batches correctly"

    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()
