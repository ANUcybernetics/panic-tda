import pytest
from PIL import Image


@pytest.mark.slow
def test_flux_dev_t2i():
    """Test that flux_dev_t2i returns an image with the expected dimensions."""
    from src.models import IMAGE_SIZE, flux_dev_t2i

    prompt = "A beautiful mountain landscape at sunset"
    image = flux_dev_t2i(prompt)

    assert isinstance(image, Image.Image)

    # Save the image as a webp file for inspection
    image.save("/tmp/test_flux_dev_t2i_output.webp", format="WEBP")
    assert image.width == IMAGE_SIZE
    assert image.height == IMAGE_SIZE

@pytest.mark.slow
def test_moondream_i2t():
    """Test that moondream_i2t returns a text caption for an input image."""
    from src.models import moondream_i2t

    # Create a simple test image
    image = Image.new('RGB', (100, 100), color='red')

    caption = moondream_i2t(image)

    assert isinstance(caption, str)
    assert len(caption) > 0  # Caption should not be empty

def test_dummy_i2t():
    """Test that dummy_i2t returns a fixed text caption."""
    from src.models import IMAGE_SIZE, dummy_i2t

    # Create a test image
    image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color='blue')

    caption = dummy_i2t(image)

    assert isinstance(caption, str)
    assert caption == "dummy text caption"


def test_dummy_t2i():
    """Test that dummy_t2i returns a fixed blank image with correct dimensions."""
    from src.models import IMAGE_SIZE, dummy_t2i

    prompt = "This prompt will be ignored"
    image = dummy_t2i(prompt)

    assert isinstance(image, Image.Image)
    assert image.width == IMAGE_SIZE
    assert image.height == IMAGE_SIZE

    # Check that the image is white (all pixels have RGB value of 255)
    # Sample a few pixels to verify
    pixels = list(image.getdata())
    assert pixels[0] == (255, 255, 255)  # Check top-left pixel
    assert pixels[-1] == (255, 255, 255)  # Check bottom-right pixel
