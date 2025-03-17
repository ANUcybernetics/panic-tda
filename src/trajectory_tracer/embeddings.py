import sys
from typing import Any, Dict, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoImageProcessor, AutoModel

# other potential multimodal embedding models:
# - https://huggingface.co/nvidia/MM-Embed
# - https://huggingface.co/jinaai/jina-clip-v2
# - https://huggingface.co/nielsr/imagebind-huge

# Model cache to avoid reloading models
_MODEL_CACHE: Dict[str, Any] = {}


def unload_all_models():
    """Unload all models from the cache and free memory."""
    global _MODEL_CACHE

    # Attempt to properly unload each model
    for model_name, model in list(_MODEL_CACHE.items()):
        # Handle different model types differently
        try:
            # For models with an explicit to() method
            if hasattr(model, "to") and callable(model.to):
                model.to("cpu")  # Move to CPU first
            # For dictionary of components
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

    print("All embedding models unloaded.")


class EmbeddingModel(BaseModel):
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

    @staticmethod
    def embed(content: Union[str, Image.Image]) -> np.ndarray:
        """Compute the actual embedding vector."""
        raise NotImplementedError


class NomicText(EmbeddingModel):
    @classmethod
    def load_to_device(cls):
        return SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1", trust_remote_code=True
        )

    @staticmethod
    def embed(text: str) -> np.ndarray:
        """
        Calculate the embedding vector for the given text.

        Args:
            text: The text to embed

        Returns:
            The calculated embedding vector
        """
        model = NomicText.get_model()
        sentences = [f"clustering: {text}"]
        return model.encode(sentences)[
            0
        ]  # Get first element to flatten (1, 768) to (768,)


class NomicVision(EmbeddingModel):
    @classmethod
    def load_to_device(cls):
        processor = AutoImageProcessor.from_pretrained(
            "nomic-ai/nomic-embed-vision-v1.5"
        )
        vision_model = AutoModel.from_pretrained(
            "nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True
        )
        vision_model.eval()
        return {"processor": processor, "model": vision_model}

    @staticmethod
    def embed(image: Image.Image) -> np.ndarray:
        """
        Calculate the embedding vector for the given image.

        Args:
            image: The image to embed

        Returns:
            The calculated embedding vector
        """
        # Get cached model components
        cache = NomicVision.get_model()
        processor = cache["processor"]
        vision_model = cache["model"]

        # Process the image
        inputs = processor(image, return_tensors="pt")

        # Calculate embeddings
        with torch.no_grad():
            img_emb = vision_model(**inputs).last_hidden_state
            img_embeddings = F.normalize(img_emb[:, 0], p=2, dim=1)

        # Convert to numpy array for storage
        return img_embeddings[0].numpy()


class Nomic(EmbeddingModel):
    @classmethod
    def load_to_device(cls):
        # This class uses other models, no need to cache anything specific
        return None

    @staticmethod
    def embed(content: Union[str, Image.Image]) -> np.ndarray:
        """
        Compute the embedding vector based on content type.

        Args:
            content: Either text or image to embed

        Returns:
            The calculated embedding vector
        """
        if isinstance(content, str):
            return NomicText.embed(content)
        elif isinstance(content, Image.Image):
            return NomicVision.embed(content)
        else:
            raise ValueError(f"Unsupported content type for Nomic: {type(content)}")


class JinaClip(EmbeddingModel):
    @classmethod
    def load_to_device(cls):
        # Choose the standard embedding dimension
        truncate_dim = 768
        return SentenceTransformer(
            "jinaai/jina-clip-v2", trust_remote_code=True, truncate_dim=truncate_dim
        )

    @staticmethod
    def embed(content: Union[str, Image.Image]) -> np.ndarray:
        """
        Calculate the embedding vector for the given content.

        Args:
            content: Either text or image to embed

        Returns:
            The calculated embedding vector
        """
        # Get cached model
        model = JinaClip.get_model()

        # Get the embedding
        return model.encode(content, normalize_embeddings=True)


## from here these ones used for testing


class Dummy(EmbeddingModel):
    @classmethod
    def load_to_device(cls):
        # Dummy model doesn't need to create or cache anything
        return None

    @staticmethod
    def embed(content: Union[str, Image.Image]) -> np.ndarray:
        """
        Generate a random embedding vector for testing purposes.

        Args:
            content: Either text or image (ignored in this dummy implementation)

        Returns:
            A random vector of dimension 768
        """
        # Generate a random vector of dimension 768
        return np.random.rand(768).astype(np.float32)


class Dummy2(EmbeddingModel):
    @classmethod
    def load_to_device(cls):
        # Dummy model doesn't need to create or cache anything
        return None

    @staticmethod
    def embed(content: Union[str, Image.Image]) -> np.ndarray:
        """
        Generate a(nother) random embedding vector for testing purposes.

        Args:
            content: Either text or image (ignored in this dummy implementation)

        Returns:
            A random vector of dimension 768
        """
        # Generate a random vector of dimension 768
        return np.random.rand(768).astype(np.float32)


def embed(embedding_model_name: str, content: Union[str, Image.Image]) -> np.ndarray:
    """
    Dynamically dispatches to the specified embedding_model's embed method.

    Args:
        embedding_model_name: Name of the embedding_model to use
        content: Either text or image to embed

    Returns:
        The calculated embedding vector

    Raises:
        ValueError: If the embedding_model doesn't exist
    """
    current_module = sys.modules[__name__]

    # Try to find the model class in this module
    if not hasattr(current_module, embedding_model_name):
        raise ValueError(f"embedding_model '{embedding_model_name}' not found.")

    # Get the model class
    model_class = getattr(current_module, embedding_model_name)

    # Check if it's a subclass of EmbeddingModel
    if not issubclass(model_class, EmbeddingModel):
        raise ValueError(f"'{embedding_model_name}' is not an EmbeddingModel subclass")

    # Call the embed method with the content
    return model_class.embed(content)


def list_models():
    """
    Returns a list of all available embedding model names (EmbeddingModel subclasses).

    Returns:
        list: Names of all available embedding models
    """
    current_module = sys.modules[__name__]

    # Find all EmbeddingModel subclasses in this module
    models = [
        cls
        for cls in dir(current_module)
        if isinstance(getattr(current_module, cls), type)
        and issubclass(getattr(current_module, cls), EmbeddingModel)
        and cls != "EmbeddingModel"  # Exclude the base class
    ]

    return models
