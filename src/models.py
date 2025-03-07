from PIL import Image
from transformers import AutoModelForCausalLM

# Image size for all image operations
IMAGE_SIZE = 1024


def moondream_i2t(image: Image) -> str:
    """
    Generate a text caption for an input image using the Moondream model.

    Args:
        image: A PIL Image object to caption

    Returns:
        str: The generated caption
    """
    model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        revision="2025-01-09",
        trust_remote_code=True,
    )

    # Generate a normal-length caption for the provided image
    result = model.caption(image, length="normal")

    return result["caption"]


def flux_dev_t2i(prompt: str) -> Image:
    """
    Generate an image from a text prompt using the FLUX.1-dev model.

    Args:
        prompt: A text description of the image to generate

    Returns:
        Image.Image: The generated PIL Image
    """
    import torch
    from diffusers import FluxPipeline

    # Initialize the model with appropriate settings
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()  # Save VRAM by offloading to CPU

    # Generate the image with standard parameters
    image = pipe(
        prompt,
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
        guidance_scale=3.5,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]

    return image


def dummy_i2t(image: Image) -> str:
    """
    A dummy function that mimics the signature of moondream_i2t but returns a fixed text.

    Args:
        image: A PIL Image object (not used)

    Returns:
        str: A dummy text caption
    """
    return "dummy text caption"


def dummy_t2i(prompt: str) -> Image:
    """
    A dummy function that mimics the signature of flux_dev_t2i but returns a fixed image.

    Args:
        prompt: A text description (not used)

    Returns:
        Image.Image: A dummy blank image
    """
    # Create a blank white image
    dummy_image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color='white')
    return dummy_image
