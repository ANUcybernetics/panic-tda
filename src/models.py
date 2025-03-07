from PIL import Image
from transformers import AutoModelForCausalLM


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
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]

    return image
