import base64
import logging
import sys
from io import BytesIO
from typing import Union

import numpy as np
import ray
import torch
import torch.nn.functional as F
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoImageProcessor, AutoModel

# Configure logging
logger = logging.getLogger(__name__)

# Fixed embedding dimension
EMBEDDING_DIM = 768

class EmbeddingModel:
    """Base class for embedding models."""

    def __init__(self):
        """Initialize the model and load to device."""
        raise NotImplementedError

    def embed(self, content_data):
        """Process the content and return an embedding."""
        # Check if we need to deserialize an image
        if isinstance(content_data, dict) and "image_data" in content_data:
            # Deserialize the base64 image
            image_bytes = base64.b64decode(content_data["image_data"])
            content = Image.open(BytesIO(image_bytes))
        else:
            # Use content as is (text)
            content = content_data

        # Process content and return embedding
        return self.process_content(content)

    def process_content(self, content: Union[str, Image.Image]) -> np.ndarray:
        """Process the content and return an embedding vector."""
        raise NotImplementedError


@ray.remote(num_gpus=0.5)
class Nomic(EmbeddingModel):
    def __init__(self):
        """Initialize the model and load to device."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required but not available")

        # Load text model
        self.text_model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1", trust_remote_code=True
        ).to("cuda")

        # Load vision model components
        self.processor = AutoImageProcessor.from_pretrained(
            "nomic-ai/nomic-embed-vision-v1.5"
        )
        self.vision_model = AutoModel.from_pretrained(
            "nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True
        ).to("cuda").eval()

        logger.info(f"Model {self.__class__.__name__} loaded successfully")

    def process_content(self, content: Union[str, Image.Image]) -> np.ndarray:
        """Calculate the embedding vector for either text or image."""
        if isinstance(content, str):
            # Text embedding
            sentences = [f"clustering: {content}"]
            return self.text_model.encode(sentences)[0]  # Flatten (1, 768) to (768,)

        elif isinstance(content, Image.Image):
            # Image embedding
            # Process the image
            inputs = self.processor(content, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            # Calculate embeddings
            with torch.no_grad():
                img_emb = self.vision_model(**inputs).last_hidden_state
                img_embeddings = F.normalize(img_emb[:, 0], p=2, dim=1)

            # Convert to numpy array
            return img_embeddings[0].cpu().numpy()
        else:
            raise ValueError(f"Unsupported content type: {type(content)}. Expected str or PIL.Image.")


@ray.remote(num_gpus=0.5)
class JinaClip(EmbeddingModel):
    def __init__(self):
        """Initialize the model and load to device."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required but not available")

        self.model = SentenceTransformer(
            "jinaai/jina-clip-v2",
            trust_remote_code=True,
            truncate_dim=EMBEDDING_DIM
        ).to("cuda")

        logger.info(f"Model {self.__class__.__name__} loaded successfully")

    def process_content(self, content: Union[str, Image.Image]) -> np.ndarray:
        """Calculate the embedding vector for text or image."""
        if not isinstance(content, (str, Image.Image)):
            raise ValueError(f"Expected string or PIL.Image input, got {type(content)}")

        # JINA CLIP handles both text and images
        return self.model.encode(content, normalize_embeddings=True)


@ray.remote
class Dummy(EmbeddingModel):
    def __init__(self):
        """Initialize the dummy model."""
        logger.info(f"Model {self.__class__.__name__} loaded successfully")

    def process_content(self, content: Union[str, Image.Image]) -> np.ndarray:
        """Generate a random embedding vector for testing."""
        # Generate a random vector of dimension 768
        return np.random.rand(EMBEDDING_DIM).astype(np.float32)


@ray.remote
class Dummy2(EmbeddingModel):
    def __init__(self):
        """Initialize the dummy model."""
        logger.info(f"Model {self.__class__.__name__} loaded successfully")

    def process_content(self, content: Union[str, Image.Image]) -> np.ndarray:
        """Generate a(nother) random embedding vector for testing."""
        # Generate a random vector of dimension 768
        return np.random.rand(EMBEDDING_DIM).astype(np.float32)


def list_models():
    """
    Returns a list of all available embedding model names (EmbeddingModel subclasses).

    Returns:
        list: Names of all available embedding models
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
