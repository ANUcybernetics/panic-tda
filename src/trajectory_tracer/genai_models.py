import logging
import random
import sys
import warnings
from typing import Union

import diffusers
import ray
import torch
import transformers
from diffusers import AutoPipelineForText2Image, FluxPipeline
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    Blip2ForConditionalGeneration,
    Blip2Processor,
)

from trajectory_tracer.schemas import InvocationType

# Configure logging
logger = logging.getLogger(__name__)

# Suppress warnings and progress bars
warnings.filterwarnings("ignore", message=".*megablocks not available.*")
warnings.filterwarnings("ignore", message=".*Flash attention is not installed.*")
warnings.filterwarnings("ignore", message=".*xFormers is not installed.*")
warnings.filterwarnings("ignore", message=".*Using a slow image processor.*")
warnings.filterwarnings("ignore", category=UserWarning)

# You can also suppress the "All keys matched successfully" by setting
# the transformers logging level even more aggressively
transformers.logging.set_verbosity_error()

# Disable progress bars
original_tqdm = tqdm
tqdm.__init__ = lambda *args, **kwargs: None
tqdm.update = lambda *args, **kwargs: None
tqdm.close = lambda *args, **kwargs: None
tqdm.__iter__ = lambda _: iter([])

# Disable HuggingFace progress bars
transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()
diffusers.utils.logging.set_verbosity_error()
diffusers.utils.logging.disable_progress_bar()

# Image size for all image operations
IMAGE_SIZE = 512

# Model resource requirements (fraction of GPU needed)
MODEL_GPU_REQUIREMENTS = {
    "FluxDev": 0.8,
    "FluxSchnell": 0.8,
    "SDXLTurbo": 0.6,
    "Moondream": 0.3,
    "BLIP2": 0.3,
    "DummyI2T": 0.0,
    "DummyT2I": 0.0
}


class GenAIModel:
    """Base class for generative AI models."""

    def __init__(self):
        """Initialize the model and load to device."""
        raise NotImplementedError

    def invoke(self, input_data: Union[str, Image.Image], seed: int):
        """
        Invoke the model for inference.

        Args:
            input_data: Either a text prompt (str) or a PIL Image
            seed: Random seed for deterministic generation

        Returns:
            The result of inference (text or image)
        """
        raise NotImplementedError


# Text2Image models

@ray.remote(num_gpus=0.75)
class FluxDev(GenAIModel):
    def __init__(self):
        """Initialize the model and load to device."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required but not available")

        # Initialize the model with appropriate settings for GPU
        self._model = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
            use_fast=True
        ).to("cuda")

        # Try to compile the UNet's forward method
        try:
            if hasattr(self._model, "unet") and hasattr(self._model.unet, "forward"):
                original_forward = self._model.unet.forward
                self._model.unet.forward = torch.compile(
                    original_forward,
                    mode="reduce-overhead",
                    fullgraph=False,
                    dynamic=True,
                )
        except Exception as e:
            logger.warning(f"Could not compile FluxDev UNet forward method: {e}")

        logger.info(f"Model {self.__class__.__name__} loaded successfully")

    def invoke(self, prompt: str, seed: int) -> Image.Image:
        """Generate an image from a text prompt"""
        generator = None if seed == -1 else torch.Generator("cuda").manual_seed(seed)

        image = self._model(
            prompt,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            guidance_scale=3.5,
            num_inference_steps=20,
            generator=generator,
        ).images[0]

        return image


@ray.remote(num_gpus=0.75)
class FluxSchnell(GenAIModel):
    def __init__(self):
        """Initialize the model and load to device."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required but not available")

        # Initialize the model with appropriate settings for GPU
        self._model = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16,
            use_fast=True
        ).to("cuda")

        # Try to compile the UNet's forward method
        try:
            if hasattr(self._model, "unet") and hasattr(self._model.unet, "forward"):
                original_forward = self._model.unet.forward
                self._model.unet.forward = torch.compile(
                    original_forward,
                    mode="reduce-overhead",
                    fullgraph=False,
                    dynamic=True,
                )
        except Exception as e:
            logger.warning(f"Could not compile FluxSchnell UNet forward method: {e}")

        logger.info(f"Model {self.__class__.__name__} loaded successfully")

    def invoke(self, prompt: str, seed: int) -> Image.Image:
        """Generate an image from a text prompt"""
        generator = None if seed == -1 else torch.Generator("cuda").manual_seed(seed)

        image = self._model(
            prompt,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            guidance_scale=3.5,
            num_inference_steps=20,
            generator=generator,
        ).images[0]

        return image


@ray.remote(num_gpus=0.75)
class SDXLTurbo(GenAIModel):
    def __init__(self):
        """Initialize the model and load to device."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required but not available")

        # Initialize the model with appropriate settings for GPU
        self._model = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
            use_fast=True
        ).to("cuda")

        logger.info(f"Model {self.__class__.__name__} loaded successfully")

    def invoke(self, prompt: str, seed: int) -> Image.Image:
        """Generate an image from a text prompt"""
        generator = None if seed == -1 else torch.Generator("cuda").manual_seed(seed)

        image = self._model(
            prompt=prompt,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            num_inference_steps=4,
            guidance_scale=0.0,
            generator=generator,
        ).images[0]

        return image


# Image2Text models

@ray.remote(num_gpus=0.2)
class Moondream(GenAIModel):
    def __init__(self):
        """Initialize the model and load to device."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required but not available")

        # Initialize the model and move to GPU
        self._model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            revision="2025-01-09",
            trust_remote_code=True
        ).to("cuda")

        # Try to compile the model
        try:
            if hasattr(self._model, "model"):
                self._model.model = torch.compile(self._model.model)
        except Exception as e:
            logger.warning(f"Could not compile Moondream model: {e}")

        logger.info(f"Model {self.__class__.__name__} loaded successfully")

    def invoke(self, image: Image.Image, seed: int) -> str:
        """Generate a text caption for an input image"""
        if seed != -1:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            random.seed(seed)

        result = self._model.caption(image, length="short")
        return result["caption"]


@ray.remote(num_gpus=0.2)
class BLIP2(GenAIModel):
    def __init__(self):
        """Initialize the model and load to device."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required but not available")

        # Initialize the model with half-precision
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", use_fast=True)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=torch.float16,
            device_map="auto"
        )

        logger.info(f"Model {self.__class__.__name__} loaded successfully")

    def invoke(self, image: Image.Image, seed: int) -> str:
        """Generate a text caption for an input image"""
        if seed != -1:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            random.seed(seed)

        # Process the input image
        inputs = self.processor(image, return_tensors="pt").to("cuda", torch.float16)

        # Generate the caption
        generated_ids = self.model.generate(**inputs, max_length=50, do_sample=True)
        caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)

        return caption


@ray.remote(num_gpus=0)
class DummyI2T(GenAIModel):
    def __init__(self):
        """Initialize the dummy model."""
        logger.info(f"Model {self.__class__.__name__} loaded successfully")

    def invoke(self, image: Image.Image, seed: int) -> str:
        """Return a dummy text caption"""
        base_text = "dummy text caption"

        if seed == -1:
            # Use system randomness for unpredictable result
            random_suffix = f" (random {random.randint(1000, 9999)})"
            return base_text + random_suffix
        else:
            # Deterministic output based on seed
            return f"{base_text} (seed {seed})"


@ray.remote(num_gpus=0)
class DummyT2I(GenAIModel):
    def __init__(self):
        """Initialize the dummy model."""
        logger.info(f"Model {self.__class__.__name__} loaded successfully")

    def invoke(self, prompt: str, seed: int) -> Image.Image:
        """Return a dummy colored image"""
        # Set the random seed if specified
        if seed != -1:
            random.seed(seed)

        # Generate random color
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)

        # Create an image with the random color
        return Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(r, g, b))


def get_output_type(model_name: str) -> InvocationType:
    """
    Gets the output type of the specified model.

    TODO maybe make this part of the model class itself

    Args:
        model_name: The name of the model class

    Returns:
        InvocationType: The output type (TEXT or IMAGE)
    """
    # Map of model names to their output types
    output_types = {
        "FluxDev": InvocationType.IMAGE,
        "FluxSchnell": InvocationType.IMAGE,
        "SDXLTurbo": InvocationType.IMAGE,
        "Moondream": InvocationType.TEXT,
        "BLIP2": InvocationType.TEXT,
        "DummyI2T": InvocationType.TEXT,
        "DummyT2I": InvocationType.IMAGE
    }

    if model_name not in output_types:
        raise ValueError(f"Model '{model_name}' not found")

    return output_types[model_name]


def list_models():
    """
    Returns a list of all available model names (GenAIModel subclasses).

    Returns:
        list: Names of all available models
    """
    current_module = sys.modules[__name__]

    models = []
    # Find all classes with type name ActorClass(class_name)
    for name, cls in vars(current_module).items():
        if type(cls).__name__ == f"ActorClass({name})":
            models.append(name)

    # Find all Ray actor classes derived from EmbeddingModel

    return models


def get_actor_class(model_name: str) -> ray.actor.ActorClass:
    """
    Get the Ray actor class for a specific model name.

    Args:
        model_name: Name of the model to get the class for

    Returns:
        Ray actor class for the specified model

    Raises:
        ValueError: If the model name is not found
    """
    current_module = sys.modules[__name__]

    # Find the class in the current module
    model_class = getattr(current_module, model_name, None)

    if model_class is None or type(model_class).__name__ != f"ActorClass({model_name})":
        raise ValueError(f"Model '{model_name}' not found or is not a Ray actor class")

    return model_class
