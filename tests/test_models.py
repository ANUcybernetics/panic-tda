import pytest
from PIL import Image

from trajectory_tracer.models import IMAGE_SIZE, FluxDevT2I, SDXLTurbo


@pytest.mark.slow
def test_flux_dev_t2i():
    """Test that flux_dev_t2i returns an image with the expected dimensions."""

    prompt = "A beautiful mountain landscape at sunset"
    image = FluxDevT2I.invoke(prompt)

    assert isinstance(image, Image.Image)

    # Save the image as a webp file for inspection
    image.save("/tmp/test_flux_dev_t2i_output.webp", format="WEBP")
    assert image.width == IMAGE_SIZE
    assert image.height == IMAGE_SIZE


@pytest.mark.slow
def test_sdxl_turbo():
    """Test that SDXLTurbo returns an image with the expected dimensions."""

    prompt = "A serene forest with a small lake"
    image = SDXLTurbo.invoke(prompt)

    assert isinstance(image, Image.Image)

    # Save the image as a webp file for inspection
    image.save("/tmp/test_sdxl_turbo_output.webp", format="WEBP")
    assert image.width == IMAGE_SIZE
    assert image.height == IMAGE_SIZE


@pytest.mark.slow
def test_blip2_i2t():
    """Test that BLIP2I2T returns a text caption for an input image."""
    from trajectory_tracer.models import BLIP2I2T

    # Create a simple test image
    image = Image.new('RGB', (100, 100), color='green')

    caption = BLIP2I2T.invoke(image)

    assert isinstance(caption, str)
    assert len(caption) > 0  # Caption should not be empty


@pytest.mark.slow
def test_moondream_i2t():
    """Test that moondream_i2t returns a text caption for an input image."""
    from trajectory_tracer.models import MoondreamI2T

    # Create a simple test image
    image = Image.new('RGB', (100, 100), color='red')

    caption = MoondreamI2T.invoke(image)

    assert isinstance(caption, str)
    assert len(caption) > 0  # Caption should not be empty


def test_dummy_i2t():
    """Test that dummy_i2t returns a fixed text caption."""
    from trajectory_tracer.models import IMAGE_SIZE, DummyI2T

    # Create a test image
    image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color='blue')

    caption = DummyI2T.invoke(image)

    assert isinstance(caption, str)
    assert caption == "dummy text caption"


def test_dummy_t2i():
    """Test that dummy_t2i returns a fixed blank image with correct dimensions."""
    from trajectory_tracer.models import IMAGE_SIZE, DummyT2I

    prompt = "This prompt will be ignored"
    image = DummyT2I.invoke(prompt)

    assert isinstance(image, Image.Image)
    assert image.width == IMAGE_SIZE
    assert image.height == IMAGE_SIZE

    # Check that the image is white (all pixels have RGB value of 255)
    # Sample a few pixels to verify
    pixels = list(image.getdata())
    assert pixels[0] == (255, 255, 255)  # Check top-left pixel
    assert pixels[-1] == (255, 255, 255)  # Check bottom-right pixel
