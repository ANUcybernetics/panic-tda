from PIL import Image

def test_flux_dev_t2i():
    """Test that flux_dev_t2i returns an image with the expected dimensions."""
    from src.models import flux_dev_t2i

    prompt = "A beautiful mountain landscape at sunset"
    image = flux_dev_t2i(prompt)

    assert isinstance(image, Image.Image)
    assert image.width == 1024
    assert image.height == 1024
