import random
import sys
from typing import ClassVar, Optional, Union

import torch
from diffusers import AutoPipelineForText2Image, FluxPipeline
from PIL import Image
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    Blip2ForConditionalGeneration,
    Blip2Processor,
)

from trajectory_tracer.schemas import InvocationType

# Image size for all image operations
IMAGE_SIZE = 512


class AIModel(BaseModel):
    output_type: ClassVar[InvocationType] = None

# Text2Image models

class FluxDev(AIModel):
    # name = "FLUX.1-dev"
    # url = "https://huggingface.co/black-forest-labs/FLUX.1-dev"
    output_type: ClassVar[InvocationType] = InvocationType.IMAGE

    @staticmethod
    def invoke(prompt: str, seed: int = -1) -> Image:
        """
        Generate an image from a text prompt using the FLUX.1-dev model.

        Args:
            prompt: A text description of the image to generate
            seed: Random seed for deterministic generation. If -1, random generation is used.

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

        # Set up generator with seed if specified, otherwise use None for random generation
        generator = None if seed == -1 else torch.Generator("cuda").manual_seed(seed)

        # Generate the image with standard parameters
        image = pipe(
            prompt,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            guidance_scale=3.5,
            num_inference_steps=20,
            generator=generator  # Use provided seed or random
        ).images[0]

        return image


class SDXLTurbo(AIModel):
    output_type: ClassVar[InvocationType] = InvocationType.IMAGE

    @staticmethod
    def invoke(prompt: str, seed: int = -1) -> Image:
        """
        Generate an image from a text prompt using the SDXL-Turbo model.

        Args:
            prompt: A text description of the image to generate
            seed: Random seed for deterministic generation. If -1, random generation is used.

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

        # Set up generator with seed if specified, otherwise use None for random generation
        generator = None if seed == -1 else torch.Generator("cuda").manual_seed(seed)

        # Generate the image with SDXL-Turbo parameters
        # SDXL-Turbo is designed for fast inference with fewer steps
        image = pipe(
            prompt=prompt,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            num_inference_steps=4,  # SDXL-Turbo works with very few steps
            guidance_scale=0.0,     # Typically uses zero guidance scale
            generator=generator  # Use provided seed or random
        ).images[0]

        return image


# Image2Text models

class Moondream(AIModel):
    # name = "Moondream 2"
    # url = "https://huggingface.co/vikhyatk/moondream2"
    output_type: ClassVar[InvocationType] = InvocationType.TEXT

    @staticmethod
    def invoke(image: Image, seed: int = -1) -> str:
        """
        Generate a text caption for an input image using the Moondream model.

        Args:
            image: A PIL Image object to caption
            seed: Random seed for deterministic generation. If -1, random generation is used.

        Returns:
            str: The generated caption
        """
        # Check if CUDA is available and use it
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize RNGs only if a specific seed is provided
        if seed != -1:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            random.seed(seed)

        model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            revision="2025-01-09",
            trust_remote_code=True
        ).to(device)

        # Generate a normal-length caption for the provided image
        result = model.caption(image, length="short")

        return result["caption"]


class BLIP2(AIModel):
    output_type: ClassVar[InvocationType] = InvocationType.TEXT

    @staticmethod
    def invoke(image: Image, seed: int = -1) -> str:
        """
        Generate a text caption for an input image using the BLIP-2 model with OPT-2.7b.

        Args:
            image: A PIL Image object to caption
            seed: Random seed for deterministic generation. If -1, random generation is used.

        Returns:
            str: The generated caption
        """
        # Check if CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required but not available")

        # Initialize RNGs only if a specific seed is provided
        if seed != -1:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            random.seed(seed)

        # Initialize the model with half-precision
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Process the input image
        inputs = processor(image, return_tensors="pt").to("cuda", torch.float16)

        # Generate the caption with the specified seed
        generated_ids = model.generate(
            **inputs,
            max_length=50,
            do_sample=True
        )
        caption = processor.decode(generated_ids[0], skip_special_tokens=True)

        return caption

class DummyI2T(AIModel):
    # name = "dummy image2text"
    output_type: ClassVar[InvocationType] = InvocationType.TEXT

    @staticmethod
    def invoke(image: Image, seed: int = -1) -> str:
        """
        A dummy function that mimics the signature of moondream_i2t but returns a fixed text.

        Args:
            image: A PIL Image object (not used)
            seed: Random seed for deterministic generation. If -1, random generation is used.

        Returns:
            str: A dummy text caption
        """
        base_text = "dummy text caption"

        if seed == -1:
            # Use system randomness for unpredictable result
            random_suffix = f" (random {random.randint(1000, 9999)})"
            return base_text + random_suffix
        else:
            # Deterministic output based on seed
            return f"{base_text} (seed {seed})"


class DummyT2I(AIModel):
    # name = "dummy text2image"
    output_type: ClassVar[InvocationType] = InvocationType.IMAGE

    @staticmethod
    def invoke(prompt: str, seed: int = -1) -> Image:
        """
        A dummy function that mimics the signature of flux_dev_t2i but returns a fixed image.

        Args:
            prompt: A text description (not used)
            seed: Random seed for deterministic generation. If -1, random generation is used.

        Returns:
            Image.Image: A dummy colored image that's consistent for the same seed
        """
        # Set the random seed if specified, otherwise use system randomness
        if seed != -1:
            random.seed(seed)

        # Generate random color that will be consistent for the same seed
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)

        # Create an image with the random color
        dummy_image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color=(r, g, b))
        return dummy_image


def invoke(model_name: str, input: Union[str, Image], seed: int = -1):
    """
    Dynamically dispatches to the specified model's invoke method.

    Args:
        model_name: The name of the model class to use
        input: Either a text prompt (str) or an image (PIL.Image)
        seed: Random seed for deterministic generation. If -1, random generation is used.

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

    # Call the invoke method with seed parameter
    return model_class.invoke(input, seed=seed)


def get_output_type(model_name: str) -> str:
    """
    Gets the output type of the specified model.

    Args:
        model_name: The name of the model class

    Returns:
        The output type (InvocationType.TEXT or InvocationType.IMAGE)

    Raises:
        ValueError: If the model doesn't exist or is not an AIModel subclass
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

    # Return the output_type
    return model_class.output_type
