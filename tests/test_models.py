import numpy as np
import pytest
from PIL import Image

from trajectory_tracer.models import (
    BLIP2,
    IMAGE_SIZE,
    DummyI2T,
    DummyT2I,
    FluxDev,
    Moondream,
    SDXLTurbo,
)


@pytest.mark.slow
def test_flux_dev_t2i():
    """Test that flux_dev_t2i returns an image with the expected dimensions and is deterministic."""

    prompt = "A beautiful mountain landscape at sunset"
    seed = 42  # Add a specific seed for reproducibility

    # First invocation
    image1 = FluxDev.invoke(prompt, seed)
    assert isinstance(image1, Image.Image)
    assert image1.width == IMAGE_SIZE
    assert image1.height == IMAGE_SIZE

    # Second invocation with same seed
    image2 = FluxDev.invoke(prompt, seed)
    assert isinstance(image2, Image.Image)

    # Save the image as a webp file for inspection
    image1.save("/tmp/test_flux_dev_t2i_output.webp", format="WEBP")

    # Check that the images are identical
    np_img1 = np.array(image1)
    np_img2 = np.array(image2)
    assert np.array_equal(np_img1, np_img2), "Images should be identical when using the same seed"


@pytest.mark.slow
def test_sdxl_turbo():
    """Test that SDXLTurbo returns an image with the expected dimensions and is deterministic."""

    prompt = "A serene forest with a small lake"
    seed = 43  # Add a specific seed for reproducibility

    # First invocation
    image1 = SDXLTurbo.invoke(prompt, seed)
    assert isinstance(image1, Image.Image)
    assert image1.width == IMAGE_SIZE
    assert image1.height == IMAGE_SIZE

    # Second invocation with same seed
    image2 = SDXLTurbo.invoke(prompt, seed)
    assert isinstance(image2, Image.Image)

    # Save the image as a webp file for inspection
    image1.save("/tmp/test_sdxl_turbo_output.webp", format="WEBP")

    # Check that the images are identical
    np_img1 = np.array(image1)
    np_img2 = np.array(image2)
    assert np.array_equal(np_img1, np_img2), "Images should be identical when using the same seed"


@pytest.mark.slow
def test_blip2_i2t():
    """Test that BLIP2 returns a text caption for an input image and is deterministic."""

    # Create a simple test image
    image = Image.new('RGB', (100, 100), color='green')
    seed = 44  # Add a specific seed for reproducibility

    # First invocation
    caption1 = BLIP2.invoke(image, seed)
    assert isinstance(caption1, str)
    assert len(caption1) > 0  # Caption should not be empty

    # Second invocation with same seed
    caption2 = BLIP2.invoke(image, seed)
    assert isinstance(caption2, str)

    # Check that the captions are identical
    assert caption1 == caption2, "Captions should be identical when using the same seed"


@pytest.mark.slow
def test_moondream_i2t():
    """Test that moondream_i2t returns a text caption for an input image and is deterministic."""

    # Create a simple test image
    image = Image.new('RGB', (100, 100), color='red')
    seed = 45  # Add a specific seed for reproducibility

    # First invocation
    caption1 = Moondream.invoke(image, seed)
    assert isinstance(caption1, str)
    assert len(caption1) > 0  # Caption should not be empty

    # Second invocation with same seed
    caption2 = Moondream.invoke(image, seed)
    assert isinstance(caption2, str)

    # Check that the captions are identical
    assert caption1 == caption2, "Captions should be identical when using the same seed"


def test_dummy_i2t():
    """Test that dummy_i2t returns a fixed text caption and is deterministic."""

    # Create a test image
    image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color='blue')
    seed = 46  # Add a specific seed for reproducibility

    # First invocation
    caption1 = DummyI2T.invoke(image, seed)
    assert isinstance(caption1, str)
    assert caption1 == "dummy text caption"

    # Second invocation with same seed
    caption2 = DummyI2T.invoke(image, seed)
    assert isinstance(caption2, str)

    # Check that the captions are identical
    assert caption1 == caption2, "Captions should be identical when using the same seed"


def test_dummy_t2i():
    """Test that dummy_t2i returns a fixed blank image with correct dimensions and is deterministic."""

    prompt = "This prompt will be ignored"
    seed = 47  # Add a specific seed for reproducibility

    # First invocation
    image1 = DummyT2I.invoke(prompt, seed)
    assert isinstance(image1, Image.Image)
    assert image1.width == IMAGE_SIZE
    assert image1.height == IMAGE_SIZE

    # Check that the image is white (all pixels have RGB value of 255)
    pixels1 = list(image1.getdata())
    assert pixels1[0] == (255, 255, 255)  # Check top-left pixel
    assert pixels1[-1] == (255, 255, 255)  # Check bottom-right pixel

    # Second invocation with same seed
    image2 = DummyT2I.invoke(prompt, seed)
    assert isinstance(image2, Image.Image)

    # Check that the images are identical
    np_img1 = np.array(image1)
    np_img2 = np.array(image2)
    assert np.array_equal(np_img1, np_img2), "Images should be identical when using the same seed"
