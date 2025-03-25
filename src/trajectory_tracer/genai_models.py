import base64
import logging
import random
import sys
import warnings
from io import BytesIO
from typing import ClassVar, Union

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


class GenAIModel:
    """Base class for generative AI models."""
    output_type: ClassVar[InvocationType] = None
    _model = None
    inference_count: int = 0

    def __init__(self):
        """Initialize the model class."""
        self.inference_count = 0
        self._model = None

    def load_model(self):
        """Load the model to the appropriate device (typically GPU)."""
        if self._model is None:
            logger.debug(f"Loading model {self.__class__.__name__}")
            self._model = self.load_to_device()
            logger.info(f"Model {self.__class__.__name__} loaded successfully")
        return self._model

    def load_to_device(self):
        """Load the model to the appropriate device (typically GPU)."""
        raise NotImplementedError

    def get_stats(self):
        """Get statistics for this model instance."""
        return {
            "model_name": self.__class__.__name__,
            "inference_count": self.inference_count,
            "gpu_memory_used": self._get_gpu_memory_used(),
        }

    def _get_gpu_memory_used(self):
        """Get the amount of GPU memory used by this model in GB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024 * 1024)
        return 0

    def invoke(self, input_data: Union[str, Image.Image], seed: int):
        """
        Invoke the model for inference.

        Args:
            input_data: Either a text prompt (str) or a PIL Image
            seed: Random seed for deterministic generation

        Returns:
            The result of inference (text or serialized image)
        """
        # Initialize model if needed
        self.load_model()
        self.inference_count += 1

        # Handle different input types
        if isinstance(input_data, dict) and "image_data" in input_data:
            # Deserialize the base64 image
            image_bytes = base64.b64decode(input_data["image_data"])
            image = Image.open(BytesIO(image_bytes))

            # Process image input based on output type
            if self.output_type == InvocationType.TEXT:
                return self.process_image(image, seed)
            else:
                raise ValueError(f"Model {self.__class__.__name__} cannot process image input")
        elif isinstance(input_data, Image.Image):
            # Process raw image input based on output type
            if self.output_type == InvocationType.TEXT:
                return self.process_image(input_data, seed)
            else:
                raise ValueError(f"Model {self.__class__.__name__} cannot process image input")
        elif isinstance(input_data, str):
            # Process text input based on output type
            if self.output_type == InvocationType.IMAGE:
                image = self.process_text(input_data, seed)

                # Serialize image for network transport
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                return {"image_data": img_str}
            elif self.output_type == InvocationType.TEXT:
                return self.process_text(input_data, seed)
            else:
                raise ValueError(f"Model {self.__class__.__name__} has an invalid output type")
        else:
            raise ValueError(f"Input type {type(input_data)} not supported. Use str or Image.Image.")

    def process_text(self, prompt: str, seed: int = -1) -> Image.Image:
        """Process text input to generate an image."""
        raise NotImplementedError

    def process_image(self, image: Image.Image, seed: int = -1) -> str:
        """Process image input to generate text."""
        raise NotImplementedError


# Text2Image models

@ray.remote(num_gpus=1)
class FluxDev(GenAIModel):
    output_type: ClassVar[InvocationType] = InvocationType.IMAGE

    def load_to_device(self):
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

        # Use a more targeted approach to compilation for complex models
        try:
            # Only compile the UNet's forward method, not the entire model
            if hasattr(pipe, "unet") and hasattr(pipe.unet, "forward"):
                original_forward = pipe.unet.forward
                pipe.unet.forward = torch.compile(
                    original_forward,
                    mode="reduce-overhead",
                    fullgraph=False,  # Important for complex models
                    dynamic=True,  # Handle variable input sizes
                )
        except Exception as e:
            logger.warning(f"Could not compile FluxDev UNet forward method: {e}")
            # Continue without compilation

        return pipe

    def process_text(self, prompt: str, seed: int = -1) -> Image.Image:
        """
        Generate an image from a text prompt using the FLUX.1-dev model.

        Args:
            prompt: A text description of the image to generate
            seed: Random seed for deterministic generation. If -1, random generation is used.

        Returns:
            Image.Image: The generated PIL Image
        """
        # Set up generator with seed if specified, otherwise use None for random generation
        generator = None if seed == -1 else torch.Generator("cuda").manual_seed(seed)

        # Generate the image with standard parameters
        image = self._model(
            prompt,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            guidance_scale=3.5,
            num_inference_steps=20,
            generator=generator,  # Use provided seed or random
        ).images[0]

        return image


@ray.remote(num_gpus=1)
class FluxSchnell(GenAIModel):
    output_type: ClassVar[InvocationType] = InvocationType.IMAGE

    def load_to_device(self):
        # Check if CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required but not available")

        # Initialize the model with appropriate settings for GPU
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16
        )

        # Explicitly move to CUDA
        pipe = pipe.to("cuda")

        # Use a more targeted approach to compilation for complex models
        try:
            # Only compile the UNet's forward method, not the entire model
            if hasattr(pipe, "unet") and hasattr(pipe.unet, "forward"):
                original_forward = pipe.unet.forward
                pipe.unet.forward = torch.compile(
                    original_forward,
                    mode="reduce-overhead",
                    fullgraph=False,  # Important for complex models
                    dynamic=True,  # Handle variable input sizes
                )
        except Exception as e:
            logger.warning(f"Could not compile FluxSchnell UNet forward method: {e}")
            # Continue without compilation

        return pipe

    def process_text(self, prompt: str, seed: int = -1) -> Image.Image:
        """
        Generate an image from a text prompt using the FLUX.1-schnell model.

        Args:
            prompt: A text description of the image to generate
            seed: Random seed for deterministic generation. If -1, random generation is used.

        Returns:
            Image.Image: The generated PIL Image
        """
        # Set up generator with seed if specified, otherwise use None for random generation
        generator = None if seed == -1 else torch.Generator("cuda").manual_seed(seed)

        # Generate the image with standard parameters
        image = self._model(
            prompt,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            guidance_scale=3.5,
            num_inference_steps=20,
            generator=generator,  # Use provided seed or random
        ).images[0]

        return image


@ray.remote(num_gpus=1)
class SDXLTurbo(GenAIModel):
    output_type: ClassVar[InvocationType] = InvocationType.IMAGE

    def load_to_device(self):
        # Check if CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required but not available")

        # Initialize the model with appropriate settings for GPU
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16"
        )

        # Explicitly move to CUDA
        return pipe.to("cuda")

    def process_text(self, prompt: str, seed: int = -1) -> Image.Image:
        """
        Generate an image from a text prompt using the SDXL-Turbo model.

        Args:
            prompt: A text description of the image to generate
            seed: Random seed for deterministic generation. If -1, random generation is used.

        Returns:
            Image.Image: The generated PIL Image
        """
        # Set up generator with seed if specified, otherwise use None for random generation
        generator = None if seed == -1 else torch.Generator("cuda").manual_seed(seed)

        # Generate the image with SDXL-Turbo parameters
        # SDXL-Turbo is designed for fast inference with fewer steps
        image = self._model(
            prompt=prompt,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            num_inference_steps=4,  # SDXL-Turbo works with very few steps
            guidance_scale=0.0,  # Typically uses zero guidance scale
            generator=generator,  # Use provided seed or random
        ).images[0]

        return image


# Image2Text models

@ray.remote(num_gpus=1)
class Moondream(GenAIModel):
    output_type: ClassVar[InvocationType] = InvocationType.TEXT

    def load_to_device(self):
        # Check if CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required but not available")

        # Initialize the model and move to GPU
        model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            revision="2025-01-09",
            trust_remote_code=True
        ).to("cuda")

        # Compile the model if it has a forward method
        try:
            if hasattr(model, "model"):
                model.model = torch.compile(model.model)
        except Exception as e:
            logger.warning(f"Could not compile Moondream model: {e}")

        return model

    def process_image(self, image: Image.Image, seed: int = -1) -> str:
        """
        Generate a text caption for an input image using the Moondream model.

        Args:
            image: A PIL Image object to caption
            seed: Random seed for deterministic generation. If -1, random generation is used.

        Returns:
            str: The generated caption
        """
        # Initialize RNGs only if a specific seed is provided
        if seed != -1:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            random.seed(seed)

        # Generate a normal-length caption for the provided image
        result = self._model.caption(image, length="short")

        return result["caption"]


@ray.remote(num_gpus=1)
class BLIP2(GenAIModel):
    output_type: ClassVar[InvocationType] = InvocationType.TEXT

    def load_to_device(self):
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

        return {"processor": processor, "model": model}

    def process_image(self, image: Image.Image, seed: int = -1) -> str:
        """
        Generate a text caption for an input image using the BLIP-2 model with OPT-2.7b.

        Args:
            image: A PIL Image object to caption
            seed: Random seed for deterministic generation. If -1, random generation is used.

        Returns:
            str: The generated caption
        """
        processor = self._model["processor"]
        model = self._model["model"]

        # Initialize RNGs only if a specific seed is provided
        if seed != -1:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            random.seed(seed)

        # Process the input image
        inputs = processor(image, return_tensors="pt").to("cuda", torch.float16)

        # Generate the caption with the specified seed
        generated_ids = model.generate(**inputs, max_length=50, do_sample=True)
        caption = processor.decode(generated_ids[0], skip_special_tokens=True)

        return caption


@ray.remote
class DummyI2T(GenAIModel):
    output_type: ClassVar[InvocationType] = InvocationType.TEXT

    def load_to_device(self):
        # No model to load for dummy class
        return None

    def process_image(self, image: Image.Image, seed: int = -1) -> str:
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


@ray.remote
class DummyT2I(GenAIModel):
    output_type: ClassVar[InvocationType] = InvocationType.IMAGE

    def load_to_device(self):
        # No model to load for dummy class
        return None

    def process_text(self, prompt: str, seed: int = -1) -> Image.Image:
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
        dummy_image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(r, g, b))
        return dummy_image


# Functions for model invocation using Named Actors

def get_model_actor(model_name: str):
    """
    Gets or creates a named actor for the specified model.

    Uses Ray's named actors for automatic caching.

    Args:
        model_name: Name of the model class

    Returns:
        Handle to the named actor
    """
    # Validate the model exists
    current_module = sys.modules[__name__]
    if not hasattr(current_module, model_name):
        raise ValueError(
            f"Model '{model_name}' not found. Available models: {list_models()}"
        )

    actor_name = f"{model_name}_actor"
    model_class = getattr(current_module, model_name)

    # Define a consistent namespace for all actors
    actor_namespace = "trajectory_tracer"

    try:
        # Try to get the actor if it exists
        actor = ray.get_actor(actor_name, namespace=actor_namespace)
        logger.debug(f"Retrieved existing actor for model {model_name}")
        return actor
    except ValueError:
        # Actor doesn't exist, create a new one
        logger.info(f"Creating new actor for model {model_name}")
        # Ensure the class is properly ray-remote decorated
        if not hasattr(model_class, "remote"):
            raise ValueError(f"Model class {model_name} is not properly decorated with @ray.remote")

        # Create the actor with proper configuration
        actor = model_class.options(
            name=actor_name,
            namespace=actor_namespace,  # Set consistent namespace
            lifetime="detached",  # Keep alive after parent process exits
            max_restarts=3,       # Auto-restart on failures, up to 3 times
        ).remote()

        # Verify the actor was created successfully
        try:
            # Ping the actor with a simple remote method call
            ray.get(actor.get_stats.remote(), timeout=10)
            logger.info(f"Actor for model {model_name} created and verified")
        except Exception as e:
            logger.error(f"Failed to create actor for model {model_name}: {e}")
            # Attempt to clean up the failed actor
            try:
                ray.kill(actor)
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up actor for model {model_name}: {cleanup_error}")
            raise RuntimeError(f"Failed to initialize actor for model {model_name}: {e}")

        return actor


def invoke(model_name: str, input_data: Union[str, Image.Image], seed: int = -1):
    """
    Invoke a model through its Ray named actor.

    Args:
        model_name: Name of the model class
        input_data: Either a text prompt (str) or a PIL Image
        seed: Random seed for deterministic generation

    Returns:
        The result of model inference (text or image)
    """
    # Get or create the named actor
    actor = get_model_actor(model_name)

    # Prepare input data for network transmission
    if isinstance(input_data, Image.Image):
        # Serialize the image
        buffered = BytesIO()
        input_data.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        serialized_input = {"image_data": img_str}
    else:
        # Text input
        serialized_input = input_data

    # Invoke the model
    result = ray.get(actor.invoke.remote(serialized_input, seed))

    # Process the result
    if isinstance(result, dict) and "image_data" in result:
        # Deserialize the image
        image_bytes = base64.b64decode(result["image_data"])
        return Image.open(BytesIO(image_bytes))
    else:
        # Text result
        return result


def get_output_type(model_name: str) -> InvocationType:
    """
    Gets the output type of the specified model.

    Args:
        model_name: The name of the model class

    Returns:
        InvocationType: The output type (TEXT or IMAGE)
    """
    current_module = sys.modules[__name__]

    # Validate the model exists and is a GenAIModel subclass
    if not is_genai_model_subclass(model_name):
        raise ValueError(
            f"Model '{model_name}' not found or is not a GenAIModel subclass. Available models: {list_models()}"
        )

    # Get the model class
    model_class = getattr(current_module, model_name)

    return model_class.output_type


def terminate_model_actor(model_name: str):
    """
    Terminate the named actor for a specific model.

    Args:
        model_name: Name of the model class
    """
    actor_name = f"{model_name}_actor"
    actor_namespace = "trajectory_tracer"

    try:
        # Get a reference to the actor
        actor = ray.get_actor(actor_name, namespace=actor_namespace)

        # Kill the actor
        ray.kill(actor)
        logger.info(f"Actor for model {model_name} terminated")
    except ValueError:
        # Actor doesn't exist
        logger.warning(f"No actor found for model {model_name}")


def terminate_all_model_actors():
    """Terminate all model actors."""
    for model_name in list_models():
        terminate_model_actor(model_name)

    logger.info("All model actors terminated")


def get_actor_stats():
    """
    Get statistics for all active model actors.

    Returns:
        dict: Statistics for each actor
    """
    stats = {}
    actor_namespace = "trajectory_tracer"

    for model_name in list_models():
        actor_name = f"{model_name}_actor"
        try:
            # Try to get the actor
            actor = ray.get_actor(actor_name, namespace=actor_namespace)

            # Get its stats
            actor_stats = ray.get(actor.get_stats.remote())
            stats[model_name] = actor_stats
        except ValueError:
            # Actor doesn't exist
            pass
        except ray.exceptions.RayActorError:
            # Actor has died
            stats[model_name] = {"status": "dead"}

    return stats


def list_models():
    """
    Returns a list of all available model names (GenAIModel subclasses).

    Returns:
        list: Names of all available models
    """
    current_module = sys.modules[__name__]
    model_classes = []

    # Find all model classes using is_genai_model_subclass helper
    for name in dir(current_module):
        # Skip private attributes
        if name.startswith('_'):
            continue

        # Use the helper function to check if it's a GenAIModel subclass
        if is_genai_model_subclass(name):
            model_classes.append(name)
            logger.debug(f"Found model class: {name}")

    logger.info(f"Found {len(model_classes)} model classes: {model_classes}")
    return model_classes


def is_genai_model_subclass(model_name: str) -> bool:
    """
    Check if a model name represents a Ray actor class created from GenAIModel subclass.

    There may well be a nicer way to do this with Ray, but I'm not sure what it is.

    Args:
        model_name: Name of the potential model class

    Returns:
        bool: True if the model appears to be a GenAIModel subclass wrapped in a Ray actor
    """
    try:
        current_module = sys.modules[__name__]
        attr = getattr(current_module, model_name, None)

        if attr is None:
            return False

        attr_type_name = type(attr).__name__
        expected_type_name = f"ActorClass({model_name})"

        return attr_type_name == expected_type_name
    except (TypeError, AttributeError):
        return False
