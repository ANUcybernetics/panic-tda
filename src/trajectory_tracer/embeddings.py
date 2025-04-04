import gc
import logging
import sys
import warnings
from typing import List

import numpy as np
import ray
import torch
import torch.nn.functional as F
import transformers
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

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

# Fixed embedding dimension
EMBEDDING_DIM = 768


class EmbeddingModel:
    """Base class for embedding models."""

    def __init__(self):
        """Initialize the model and load to device."""
        raise NotImplementedError

    def embed(self, contents: List[str]) -> List[np.ndarray]:
        """Process a batch of text items and return embeddings."""
        raise NotImplementedError

    @classmethod
    def get_memory_usage(cls, verbose=False):
        """
        Estimate GPU memory usage in GB for this model.

        Args:
            verbose: If True, print detailed memory information

        Returns:
            float: Estimated GPU memory usage in GB
        """

        # Free up existing memory
        gc.collect()
        torch.cuda.empty_cache()

        # Measure memory before loading
        torch.cuda.synchronize()
        start_mem = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)

        if verbose:
            print(f"Starting memory usage: {start_mem:.2f} GB")

        # Initialize the model
        try:
            model_instance = cls()

            # Force CUDA synchronization to ensure all memory operations are complete
            torch.cuda.synchronize()

            # Measure after loading
            end_mem = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)

            if verbose:
                print(
                    f"Peak memory usage: {torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024):.2f} GB"
                )
                print(f"Final memory usage: {end_mem:.2f} GB")

            # Clean up
            del model_instance
            gc.collect()
            torch.cuda.empty_cache()

            # Return the difference
            return end_mem - start_mem

        except Exception as e:
            if verbose:
                print(f"Error measuring memory: {e}")
            return -1  # Indicate error


@ray.remote(num_gpus=0.02)
class Nomic(EmbeddingModel):
    def __init__(self):
        """Initialize the model and load to device."""
        # Load model using SentenceTransformer
        self.model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v2-moe",
            trust_remote_code=True
        )

        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        self.model.eval()

        logger.info(f"Model {self.__class__.__name__} loaded successfully")

    def embed(self, contents: List[str]) -> List[np.ndarray]:
        """Process a batch of text items and return embeddings."""
        if not contents:
            return []

        # Check that all items are text
        if not all(isinstance(item, str) for item in contents):
            raise ValueError("All items must be strings for embedding")

        # Get embeddings using SentenceTransformer with the appropriate prompt prefix
        with torch.no_grad():
            embeddings = self.model.encode(
                contents,
                convert_to_tensor=True,
                normalize_embeddings=True,
                prompt_name="passage"  # This adds the "search_document:" prefix automatically
            )

            # Convert to list of numpy arrays
            if torch.is_tensor(embeddings):
                embeddings = embeddings.cpu().numpy()
                return [emb for emb in embeddings]
            return embeddings


@ray.remote(num_gpus=0.04)
class JinaClip(EmbeddingModel):
    def __init__(self):
        """Initialize the model and load to device."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required but not available")

        # Load model using transformers directly
        self.model = (
            AutoModel.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True)
            .to("cuda")
            .eval()
        )

        logger.info(f"Model {self.__class__.__name__} loaded successfully")

    def embed(self, contents: List[str]) -> List[np.ndarray]:
        """Process a batch of text items and return embeddings."""
        if not contents:
            return []

        with torch.no_grad():
            # Text embedding
            text_embeddings = self.model.encode_text(
                contents, truncate_dim=EMBEDDING_DIM, task="retrieval.query"
            )
            return [emb for emb in text_embeddings]


@ray.remote(num_gpus=0.03)
class STSBMpnet(EmbeddingModel):
    def __init__(self):
        """Initialize the model and load to device."""
        self.model = SentenceTransformer("sentence-transformers/stsb-mpnet-base-v2")
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        self.model.eval()
        logger.info(f"Model {self.__class__.__name__} loaded successfully")

    def embed(self, contents: List[str]) -> List[np.ndarray]:
        """Process a batch of content items and return embeddings."""
        if not contents:
            return []

        # Check that all items are text (this model only supports text)
        if not all(isinstance(item, str) for item in contents):
            raise ValueError(
                "This model only supports text inputs. All items must be strings."
            )

        # Get embeddings
        with torch.no_grad():
            embeddings = self.model.encode(
                contents, convert_to_tensor=True, normalize_embeddings=True
            )

            # Convert to list of numpy arrays
            if torch.is_tensor(embeddings):
                embeddings = embeddings.cpu().numpy()
                return [emb for emb in embeddings]
            return embeddings


@ray.remote(num_gpus=0.03)
class STSBRoberta(EmbeddingModel):
    def __init__(self):
        """Initialize the model and load to device."""
        self.model = SentenceTransformer("sentence-transformers/stsb-roberta-base-v2")
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        self.model.eval()
        logger.info(f"Model {self.__class__.__name__} loaded successfully")

    def embed(self, contents: List[str]) -> List[np.ndarray]:
        """Process a batch of content items and return embeddings."""
        if not contents:
            return []

        # Check that all items are text (this model only supports text)
        if not all(isinstance(item, str) for item in contents):
            raise ValueError(
                "This model only supports text inputs. All items must be strings."
            )

        # Get embeddings
        with torch.no_grad():
            embeddings = self.model.encode(
                contents, convert_to_tensor=True, normalize_embeddings=True
            )

            # Convert to list of numpy arrays
            if torch.is_tensor(embeddings):
                embeddings = embeddings.cpu().numpy()
                return [emb for emb in embeddings]
            return embeddings


@ray.remote(num_gpus=0.02)
class STSBDistilRoberta(EmbeddingModel):
    def __init__(self):
        """Initialize the model and load to device."""
        self.model = SentenceTransformer(
            "sentence-transformers/stsb-distilroberta-base-v2"
        )
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        self.model.eval()
        logger.info(f"Model {self.__class__.__name__} loaded successfully")

    def embed(self, contents: List[str]) -> List[np.ndarray]:
        """Process a batch of content items and return embeddings."""
        if not contents:
            return []

        # Check that all items are text (this model only supports text)
        if not all(isinstance(item, str) for item in contents):
            raise ValueError(
                "This model only supports text inputs. All items must be strings."
            )

        # Get embeddings
        with torch.no_grad():
            embeddings = self.model.encode(
                contents, convert_to_tensor=True, normalize_embeddings=True
            )

            # Convert to list of numpy arrays
            if torch.is_tensor(embeddings):
                embeddings = embeddings.cpu().numpy()
                return [emb for emb in embeddings]
            return embeddings


@ray.remote
class Dummy(EmbeddingModel):
    def __init__(self):
        """Initialize the dummy model."""
        logger.info(f"Model {self.__class__.__name__} loaded successfully")

    def embed(self, contents: List[str]) -> List[np.ndarray]:
        """Process a batch of text items and return embeddings."""
        embeddings = []

        # Check that all items are strings
        if not all(isinstance(item, str) for item in contents):
            raise ValueError("All items must be strings for embedding")

        for content in contents:
            # For text, use the hash of the string to seed a deterministic vector
            seed = sum(ord(c) for c in content)
            np.random.seed(seed)

            # Generate a deterministic vector using the seeded random number generator
            vector = np.random.rand(EMBEDDING_DIM).astype(np.float32)
            embeddings.append(vector)

        # Reset the random seed to avoid affecting other code
        np.random.seed(None)
        return embeddings


@ray.remote
class Dummy2(EmbeddingModel):
    def __init__(self):
        """Initialize the dummy model."""
        logger.info(f"Model {self.__class__.__name__} loaded successfully")

    def embed(self, contents: List[str]) -> List[np.ndarray]:
        """Process a batch of text items and return embeddings."""
        embeddings = []

        # Check that all items are strings
        if not all(isinstance(item, str) for item in contents):
            raise ValueError("All items must be strings for embedding")

        for content in contents:
            # For text, create deterministic values based on character positions
            chars = [ord(c) for c in (content[:100] if len(content) > 100 else content)]
            # Pad or truncate to ensure we have enough values
            chars = (chars + [0] * EMBEDDING_DIM)[:EMBEDDING_DIM]
            # Normalize to 0-1 range
            vector = np.array(chars) / 255.0
            embeddings.append(vector.astype(np.float32))

        return embeddings


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


def get_all_models_memory_usage(verbose=False):
    """
    Returns a dictionary with the memory usage of all models.

    Args:
        verbose: If True, print detailed memory information

    Returns:
        dict: Mapping of model names to their memory usage in GB
    """
    memory_usage = {}

    for model_name in list_models():
        print(f"Measuring memory usage for {model_name}...")

        # Extract the actual class from the ActorClass wrapper
        actual_class = getattr(sys.modules[__name__], model_name)

        if hasattr(actual_class, "get_memory_usage"):
            usage = actual_class.get_memory_usage(verbose=verbose)
            memory_usage[model_name] = usage
        else:
            memory_usage[model_name] = -1  # No method available

    return memory_usage
