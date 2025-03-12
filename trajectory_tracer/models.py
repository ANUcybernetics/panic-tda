import sys
from typing import Union


import torch
from diffusers import FluxPipeline, AutoPipelineForText2Image
from PIL import Image
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, Blip2Processor, Blip2ForConditionalGeneration

# Image size for all image operations
IMAGE_SIZE = 512

class AIModel(BaseModel):
    pass

# Text2Image models

class FluxDevT2I(AIModel):
    # name = "FLUX.1-dev"
    # url = "https://huggingface.co/black-forest-labs/FLUX.1-dev"

    @staticmethod
    def invoke(prompt: str) -> Image:
        """
        Generate an image from a text prompt using the FLUX.1-dev model.

        Args:
            prompt: A text description of the image to generate

        Returns:
            Image.Image: The generated PIL Image
        """
        # Check if CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required but not available")

        # Initialize the model with appropriate settings for GPU
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16
        )

        # Explicitly move to CUDA
        pipe = pipe.to("cuda")

        # Generate the image with standard parameters
        image = pipe(
            prompt,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            guidance_scale=3.5,
            num_inference_steps=50,
            max_sequence_length=512,
            generator=torch.Generator("cuda").manual_seed(0)  # Use CUDA for generation
        ).images[0]

        return image


class SDXLTurbo(AIModel):
    @staticmethod
    def invoke(prompt: str) -> Image:
        """
        Generate an image from a text prompt using the SDXL-Turbo model.

        Args:
            prompt: A text description of the image to generate

        Returns:
            Image.Image: The generated PIL Image
        """
        # Check if CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required but not available")

        # Import diffusers components here to avoid import errors when CUDA is not available

        # Initialize the model with appropriate settings for GPU
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16"
        )

        # Explicitly move to CUDA
        pipe = pipe.to("cuda")

        # Generate the image with SDXL-Turbo parameters
        # SDXL-Turbo is designed for fast inference with fewer steps
        image = pipe(
            prompt=prompt,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            num_inference_steps=4,  # SDXL-Turbo works with very few steps
            guidance_scale=0.0,     # Typically uses zero guidance scale
            generator=torch.Generator("cuda").manual_seed(0)
        ).images[0]

        return image


# Image2Text models

class MoondreamI2T(AIModel):
    # name = "Moondream 2"
    # url = "https://huggingface.co/vikhyatk/moondream2"

    @staticmethod
    def invoke(image: Image) -> str:
        """
        Generate a text caption for an input image using the Moondream model.

        Args:
            image: A PIL Image object to caption

        Returns:
            str: The generated caption
        """
        # Check if CUDA is available and use it
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            revision="2025-01-09",
            trust_remote_code=True
        ).to(device)

        # Generate a normal-length caption for the provided image
        result = model.caption(image, length="normal")

        return result["caption"]


class BLIP2I2T(AIModel):
    @staticmethod
    def invoke(image: Image) -> str:
        """
        Generate a text caption for an input image using the BLIP-2 model with OPT-2.7b.

        Args:
            image: A PIL Image object to caption

        Returns:
            str: The generated caption
        """
        # Check if CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required but not available")


        # Initialize the model with half-precision
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Process the input image
        inputs = processor(image, return_tensors="pt").to("cuda", torch.float16)

        # Generate the caption
        generated_ids = model.generate(**inputs, max_length=50)
        caption = processor.decode(generated_ids[0], skip_special_tokens=True)

        return caption

class DummyI2T(AIModel):
    # name = "dummy image2text"

    @staticmethod
    def invoke(image: Image) -> str:
        """
        A dummy function that mimics the signature of moondream_i2t but returns a fixed text.

        Args:
            image: A PIL Image object (not used)

        Returns:
            str: A dummy text caption
        """
        return "dummy text caption"

class DummyT2I(AIModel):
    # name = "dummy text2image"

    @staticmethod
    def invoke(prompt: str) -> Image:
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


def invoke(model_name: str, input: Union[str, Image]):
    """
    Dynamically dispatches to the specified model's invoke method.

    Args:
        model_name: The name of the model class to use
        input: Either a text prompt (str) or an image (PIL.Image)

    Returns:
        The result of the model's invoke method

    Raises:
        ValueError: If the model doesn't exist or input type is incompatible
    """
    current_module = sys.modules[__name__]

    # Try to find the model class in this module
    if not hasattr(current_module, model_name):
        raise ValueError(f"Model '{model_name}' not found. Available models: "
                         f"{[cls for cls in dir(current_module) if isinstance(getattr(current_module, cls), type) and issubclass(getattr(current_module, cls), AIModel)]}")

    # Get the model class
    model_class = getattr(current_module, model_name)

    # Check if it's a subclass of AIModel
    if not issubclass(model_class, AIModel):
        raise ValueError(f"'{model_name}' is not an AIModel subclass")

    # Check if CUDA is available before attempting to use non-dummy models
    if not model_name.startswith("Dummy") and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required but not available")

    # Call the invoke method
    return model_class.invoke(input)
