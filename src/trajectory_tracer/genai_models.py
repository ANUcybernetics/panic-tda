import random
import sys

# Suppress warnings and progress bars
import warnings
from typing import ClassVar, Union

import diffusers
import torch
import transformers
from diffusers import AutoPipelineForText2Image, FluxPipeline
from PIL import Image
from pydantic import BaseModel
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    Blip2ForConditionalGeneration,
    Blip2Processor,
)

from trajectory_tracer.schemas import InvocationType

warnings.filterwarnings("ignore", message=".*megablocks not available.*")

# Disable tqdm progress bars

# Replace tqdm with a no-op version
original_tqdm = tqdm
tqdm.__init__ = lambda *args, **kwargs: None
tqdm.update = lambda *args, **kwargs: None
tqdm.close = lambda *args, **kwargs: None
tqdm.__iter__ = lambda _: iter([])
# tqdm.pandas = lambda *args, **kwargs: None

# Disable HuggingFace progress bars

transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()


diffusers.utils.logging.set_verbosity_error()
diffusers.utils.logging.disable_progress_bar()

# Image size for all image operations
IMAGE_SIZE = 512

# Cache to store loaded models
_MODEL_CACHE = {}


class GenAIModel(BaseModel):
    output_type: ClassVar[InvocationType] = None

    @classmethod
    def get_model(cls):
        """Get or create the model instance."""
        model_name = cls.__name__
        if model_name not in _MODEL_CACHE:
            _MODEL_CACHE[model_name] = cls.load_to_device()
        return _MODEL_CACHE[model_name]

    @classmethod
    def load_to_device(cls):
        """Load the model to the appropriate device (typically GPU)."""
        raise NotImplementedError

    @classmethod
    def report_memory_usage(cls):
        """
        Load model to device, report VRAM usage, and unload.

        Returns:
            dict: Memory usage information in GB
        """
        # Ensure CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required but not available")

        # Get initial GPU memory usage
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        initial_memory = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)  # GB

        # Load the model
        print(f"Loading {cls.__name__} to measure memory usage...")
        model = cls.load_to_device()

        # Measure peak memory after loading
        torch.cuda.synchronize()
        loaded_memory = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)  # GB
        peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)  # GB

        # Calculate memory usage
        model_size = loaded_memory - initial_memory

        # Report memory usage
        memory_info = {
            "model_name": cls.__name__,
            "model_size_gb": round(model_size, 2),
            "peak_memory_gb": round(peak_memory, 2),
            "current_memory_gb": round(loaded_memory, 2)
        }

        print(f"Memory usage for {cls.__name__}:")
        print(f"  Model size: {memory_info['model_size_gb']} GB")
        print(f"  Peak memory: {memory_info['peak_memory_gb']} GB")

        # Unload model
        if hasattr(model, "to") and callable(model.to):
            model.to("cpu")
        elif isinstance(model, dict):
            for component_name, component in model.items():
                if hasattr(component, "to") and callable(component.to):
                    component.to("cpu")

        # Force garbage collection
        del model
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        final_memory = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)  # GB
        memory_info["residual_memory_gb"] = round(final_memory - initial_memory, 2)

        print(f"  Residual memory after unloading: {memory_info['residual_memory_gb']} GB")

        return memory_info


# Text2Image models


class FluxDev(GenAIModel):
    # name = "FLUX.1-dev"
    # url = "https://huggingface.co/black-forest-labs/FLUX.1-dev"
    output_type: ClassVar[InvocationType] = InvocationType.IMAGE

    @classmethod
    def load_to_device(cls):
        # Check if CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required but not available")

        # Initialize the model with appropriate settings for GPU
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
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
            print(f"Warning: Could not compile FluxDev UNet forward method: {e}")
            # Continue without compilation

        return pipe

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
        # Get the cached model
        pipe = FluxDev.get_model()

        # Set up generator with seed if specified, otherwise use None for random generation
        generator = None if seed == -1 else torch.Generator("cuda").manual_seed(seed)

        # Generate the image with standard parameters
        image = pipe(
            prompt,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            guidance_scale=3.5,
            num_inference_steps=20,
            generator=generator,  # Use provided seed or random
        ).images[0]

        return image


class FluxSchnell(GenAIModel):
    output_type: ClassVar[InvocationType] = InvocationType.IMAGE

    @classmethod
    def load_to_device(cls):
        # Check if CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required but not available")

        # Initialize the model with appropriate settings for GPU
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16
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
            print(f"Warning: Could not compile FluxSchnell UNet forward method: {e}")
            # Continue without compilation

        return pipe

    @staticmethod
    def invoke(prompt: str, seed: int = -1) -> Image:
        """
        Generate an image from a text prompt using the FLUX.1-schnell model.

        Args:
            prompt: A text description of the image to generate
            seed: Random seed for deterministic generation. If -1, random generation is used.

        Returns:
            Image.Image: The generated PIL Image
        """
        # Get the cached model
        pipe = FluxSchnell.get_model()

        # Set up generator with seed if specified, otherwise use None for random generation
        generator = None if seed == -1 else torch.Generator("cuda").manual_seed(seed)

        # Generate the image with standard parameters
        image = pipe(
            prompt,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            guidance_scale=3.5,
            num_inference_steps=20,
            generator=generator,  # Use provided seed or random
        ).images[0]

        return image


class SDXLTurbo(GenAIModel):
    output_type: ClassVar[InvocationType] = InvocationType.IMAGE

    @classmethod
    def load_to_device(cls):
        # Check if CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required but not available")

        # Initialize the model with appropriate settings for GPU
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
        )

        # Explicitly move to CUDA
        return pipe.to("cuda")

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
        # Get the cached model
        pipe = SDXLTurbo.get_model()

        # Set up generator with seed if specified, otherwise use None for random generation
        generator = None if seed == -1 else torch.Generator("cuda").manual_seed(seed)

        # Generate the image with SDXL-Turbo parameters
        # SDXL-Turbo is designed for fast inference with fewer steps
        image = pipe(
            prompt=prompt,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            num_inference_steps=4,  # SDXL-Turbo works with very few steps
            guidance_scale=0.0,  # Typically uses zero guidance scale
            generator=generator,  # Use provided seed or random
        ).images[0]

        return image


# Image2Text models


class Moondream(GenAIModel):
    # name = "Moondream 2"
    # url = "https://huggingface.co/vikhyatk/moondream2"
    output_type: ClassVar[InvocationType] = InvocationType.TEXT

    @classmethod
    def load_to_device(cls):
        # Check if CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required but not available")

        # Initialize the model and move to GPU
        model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2", revision="2025-01-09", trust_remote_code=True
        ).to("cuda")

        # Compile the model if it has a forward method
        # Note: since Moondream uses trust_remote_code, need to be careful with compiling
        # and only compile the model's transformer blocks if possible
        try:
            if hasattr(model, "model"):
                model.model = torch.compile(model.model)
        except Exception as e:
            print(f"Warning: Could not compile Moondream model: {e}")

        return model

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
        # Get the cached model
        model = Moondream.get_model()

        # Initialize RNGs only if a specific seed is provided
        if seed != -1:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            random.seed(seed)

        # Generate a normal-length caption for the provided image
        result = model.caption(image, length="short")

        return result["caption"]


class BLIP2(GenAIModel):
    output_type: ClassVar[InvocationType] = InvocationType.TEXT

    @classmethod
    def load_to_device(cls):
        # Check if CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required but not available")

        # Initialize the model with half-precision
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16, device_map="auto"
        )

        return {"processor": processor, "model": model}

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
        # Get cached model components
        components = BLIP2.get_model()
        processor = components["processor"]
        model = components["model"]

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


class DummyI2T(GenAIModel):
    # name = "dummy image2text"
    output_type: ClassVar[InvocationType] = InvocationType.TEXT

    @classmethod
    def load_to_device(cls):
        # No model to load for dummy class
        return None

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


class DummyT2I(GenAIModel):
    # name = "dummy text2image"
    output_type: ClassVar[InvocationType] = InvocationType.IMAGE

    @classmethod
    def load_to_device(cls):
        # No model to load for dummy class
        return None

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
        dummy_image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(r, g, b))
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
        raise ValueError(
            f"Model '{model_name}' not found. Available models: "
            f"{[cls for cls in dir(current_module) if isinstance(getattr(current_module, cls), type) and issubclass(getattr(current_module, cls), GenAIModel)]}"
        )

    # Get the model class
    model_class = getattr(current_module, model_name)

    # Check if it's a subclass of GenAIModel
    if not issubclass(model_class, GenAIModel):
        raise ValueError(f"'{model_name}' is not an GenAIModel subclass")

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
        ValueError: If the model doesn't exist or is not an GenAIModel subclass
    """
    current_module = sys.modules[__name__]

    # Try to find the model class in this module
    if not hasattr(current_module, model_name):
        raise ValueError(
            f"Model '{model_name}' not found. Available models: "
            f"{[cls for cls in dir(current_module) if isinstance(getattr(current_module, cls), type) and issubclass(getattr(current_module, cls), GenAIModel)]}"
        )

    # Get the model class
    model_class = getattr(current_module, model_name)

    # Check if it's a subclass of GenAIModel
    if not issubclass(model_class, GenAIModel):
        raise ValueError(f"'{model_name}' is not an GenAIModel subclass")

    # Return the output_type
    return model_class.output_type


def unload_all_models():
    """Unload all models from the cache and free GPU memory."""
    global _MODEL_CACHE

    # Attempt to properly unload each model
    for model_name, model in list(_MODEL_CACHE.items()):
        # Handle different model types differently
        try:
            # For models with an explicit to() method (like diffusers pipelines)
            if hasattr(model, "to") and callable(model.to):
                model.to("cpu")  # Move to CPU first
            # For dictionary of components (like BLIP2)
            elif isinstance(model, dict):
                for component_name, component in model.items():
                    if hasattr(component, "to") and callable(component.to):
                        component.to("cpu")
        except Exception as e:
            print(f"Warning: Error unloading model {model_name}: {e}")

    # Clear the cache
    _MODEL_CACHE.clear()

    # Force CUDA garbage collection
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Make sure CUDA operations are completed

    print("All models unloaded from GPU.")


def list_models():
    """
    Returns a list of all available model names (GenAIModel subclasses).

    Returns:
        list: Names of all available models
    """
    current_module = sys.modules[__name__]

    # Find all GenAIModel subclasses in this module
    models = [
        cls
        for cls in dir(current_module)
        if isinstance(getattr(current_module, cls), type)
        and issubclass(getattr(current_module, cls), GenAIModel)
        and cls != "GenAIModel"  # Exclude the base class
    ]

    return models
