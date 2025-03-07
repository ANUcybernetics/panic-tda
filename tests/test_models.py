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
