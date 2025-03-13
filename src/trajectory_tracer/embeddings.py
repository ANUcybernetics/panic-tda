import sys
from typing import Union

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

class EmbeddingModel(BaseModel):
    @staticmethod
    def embed(content: Union[str, Image.Image]) -> np.ndarray:
        """Compute the actual embedding vector."""
        raise NotImplementedError


class NomicText(EmbeddingModel):
    @staticmethod
    def embed(text: str) -> np.ndarray:
        """
        Calculate the embedding vector for the given text.

        Args:
            text: The text to embed

        Returns:
            The calculated embedding vector
        """
        model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
        sentences = [f"clustering: {text}"]
        return model.encode(sentences)[0]  # Get first element to flatten (1, 768) to (768,)


class NomicVision(EmbeddingModel):
    @staticmethod
    def embed(image: Image.Image) -> np.ndarray:
        """
        Calculate the embedding vector for the given image.

        Args:
            image: The image to embed

        Returns:
            The calculated embedding vector
        """
        # Load the model and processor
        processor = AutoImageProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1.5")
        vision_model = AutoModel.from_pretrained("nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True)
        vision_model.eval()

        # Process the image
        inputs = processor(image, return_tensors="pt")

        # Calculate embeddings
        with torch.no_grad():
            img_emb = vision_model(**inputs).last_hidden_state
            img_embeddings = F.normalize(img_emb[:, 0], p=2, dim=1)

        # Convert to numpy array for storage
        return img_embeddings[0].numpy()


class Nomic(EmbeddingModel):
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
    @staticmethod
    def embed(content: Union[str, Image.Image]) -> np.ndarray:
        """
        Calculate the embedding vector for the given content.

        Args:
            content: Either text or image to embed

        Returns:
            The calculated embedding vector
        """
        # Choose the standard embedding dimension
        truncate_dim = 768

        # Initialize the model
        model = SentenceTransformer('jinaai/jina-clip-v2', trust_remote_code=True, truncate_dim=truncate_dim)

        # Get the embedding
        return model.encode(content, normalize_embeddings=True)


## from here these ones used for testing

class Dummy(EmbeddingModel):
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
