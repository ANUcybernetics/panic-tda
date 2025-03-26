import logging
import sys
from typing import Union

import numpy as np
import ray
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

# Configure logging
logger = logging.getLogger(__name__)

# Fixed embedding dimension
EMBEDDING_DIM = 768

class EmbeddingModel:
    """Base class for embedding models."""

    def __init__(self):
        """Initialize the model and load to device."""
        raise NotImplementedError

    def embed(self, content: Union[str, Image.Image]):
        """Process the content and return an embedding."""
        raise NotImplementedError


@ray.remote(num_gpus=0.2)
class Nomic(EmbeddingModel):
    def __init__(self):
        """Initialize the model and load to device."""
        # Load text model components using transformers directly
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.text_model = AutoModel.from_pretrained(
            "nomic-ai/nomic-embed-text-v1.5",
            trust_remote_code=True,
            safe_serialization=True
        ).to("cuda").eval()

        # Load vision model components
        self.processor = AutoImageProcessor.from_pretrained(
            "nomic-ai/nomic-embed-vision-v1.5"
        )
        self.vision_model = AutoModel.from_pretrained(
            "nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True
        ).to("cuda").eval()

        logger.info(f"Model {self.__class__.__name__} loaded successfully")

    def mean_pooling(self, model_output, attention_mask):
        """Perform mean pooling on token embeddings."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def embed(self, content: Union[str, Image.Image]):
        """Process the content and return an embedding."""
        if isinstance(content, str):
            # Text embedding using transformers directly
            sentences = [f"clustering: {content.strip()}"]

            # Tokenize input
            encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

            # Move to GPU
            encoded_input = {k: v.to("cuda") for k, v in encoded_input.items()}

            # Get embeddings
            with torch.no_grad():
                model_output = self.text_model(**encoded_input)

            # Process embeddings
            embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
            embeddings = F.normalize(embeddings, p=2, dim=1)

            # Return numpy array
            return embeddings[0].cpu().numpy()

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


@ray.remote(num_gpus=0.2)
class JinaClip(EmbeddingModel):
    def __init__(self):
        """Initialize the model and load to device."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required but not available")

        # Load model using transformers directly
        self.model = AutoModel.from_pretrained(
            'jinaai/jina-clip-v2',
            trust_remote_code=True
        ).to("cuda").eval()

        logger.info(f"Model {self.__class__.__name__} loaded successfully")

    def embed(self, content: Union[str, Image.Image]):
        """Process the content and return an embedding."""
        if not isinstance(content, (str, Image.Image)):
            raise ValueError(f"Expected string or PIL.Image input, got {type(content)}")

        with torch.no_grad():
            if isinstance(content, str):
                # Text embedding
                text_embeddings = self.model.encode_text(
                    [content],
                    truncate_dim=EMBEDDING_DIM,
                    task='retrieval.query'
                )
                return text_embeddings[0]
            else:
                # Image embedding
                image_embeddings = self.model.encode_image(
                    [content],  # Wrap in list to match encode_text format
                    truncate_dim=EMBEDDING_DIM
                )
                return image_embeddings[0]  # Return the first (and only) embedding


@ray.remote
class Dummy(EmbeddingModel):
    def __init__(self):
        """Initialize the dummy model."""
        logger.info(f"Model {self.__class__.__name__} loaded successfully")

    def embed(self, content: Union[str, Image.Image]):
        """Process the content and return an embedding."""
        if isinstance(content, str):
            # For text, use the hash of the string to seed a deterministic vector
            seed = sum(ord(c) for c in content)
            np.random.seed(seed)
        elif isinstance(content, Image.Image):
            # For images, use basic image properties to create a deterministic seed
            img_array = np.array(content)
            # Use the sum of pixel values as a seed
            seed = int(np.sum(img_array) % 10000)
            np.random.seed(seed)
        else:
            # Fall back to a fixed seed for unknown types
            np.random.seed(42)

        # Generate a deterministic vector using the seeded random number generator
        vector = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        # Reset the random seed to avoid affecting other code
        np.random.seed(None)
        return vector


@ray.remote
class Dummy2(EmbeddingModel):
    def __init__(self):
        """Initialize the dummy model."""
        logger.info(f"Model {self.__class__.__name__} loaded successfully")

    def embed(self, content: Union[str, Image.Image]):
        """Process the content and return an embedding."""
        if isinstance(content, str):
            # For text, create deterministic values based on character positions
            chars = [ord(c) for c in (content[:100] if len(content) > 100 else content)]
            # Pad or truncate to ensure we have enough values
            chars = (chars + [0] * EMBEDDING_DIM)[:EMBEDDING_DIM]
            # Normalize to 0-1 range
            vector = np.array(chars) / 255.0
        elif isinstance(content, Image.Image):
            # Resize image to a small fixed size and use pixel values
            small_img = content.resize((16, 16)).convert('L')  # Convert to grayscale
            pixels = np.array(small_img).flatten()
            # Repeat or truncate to match embedding dimension
            pixels = np.tile(pixels, EMBEDDING_DIM // len(pixels) + 1)[:EMBEDDING_DIM]
            # Normalize to 0-1 range
            vector = pixels / 255.0
        else:
            # Create a fixed pattern for unknown types
            vector = np.linspace(0, 1, EMBEDDING_DIM)

        return vector.astype(np.float32)


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
