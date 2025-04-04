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
from sentence_transformers.util import batch_to_device
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

# Create a custom SentenceTransformer that doesn't sort by length to avoid index issues
class NoSortingSentenceTransformer(SentenceTransformer):
    """SentenceTransformer that doesn't sort by length to avoid index issues."""

    def encode(self, sentences, batch_size=32, show_progress_bar=None, output_value="sentence_embedding",
               convert_to_numpy=True, convert_to_tensor=False, device=None, normalize_embeddings=False,
               precision="float32", **kwargs):
        """Remove sorting to avoid index issues."""
        self.eval()

        # Handle single item inputs
        input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, "__len__"):
            sentences = [sentences]
            input_was_string = True

        if device is None:
            device = self.device

        all_embeddings = []

        # Process items one by one (to avoid any sorting issues)
        for sentence in sentences:
            with torch.no_grad():
                features = self.tokenize([sentence])
                features = batch_to_device(features, device)

                out_features = self.forward(features)
                embedding = out_features['sentence_embedding']

                if normalize_embeddings:
                    embedding = F.normalize(embedding, p=2, dim=1)

                # Handle conversion options
                if convert_to_numpy:
                    embedding = embedding.cpu().numpy()[0]  # Get the first (only) item
                elif convert_to_tensor:
                    # Keep as tensor
                    embedding = embedding[0]  # Get the first (only) item
                else:
                    embedding = embedding[0]  # Get the first (only) item

                all_embeddings.append(embedding)

        # Handle return format
        if convert_to_tensor and not convert_to_numpy and len(all_embeddings) > 0:
            all_embeddings = torch.stack(all_embeddings)

        # If input was a single string, return just the embedding (not in a list)
        if input_was_string:
            return all_embeddings[0]

        return all_embeddings


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
        # Load model using NoSortingSentenceTransformer instead
        self.model = NoSortingSentenceTransformer(
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

        # Get embeddings using NoSortingSentenceTransformer with the appropriate prompt prefix
        with torch.no_grad():
            embeddings = self.model.encode(
                contents,
                convert_to_numpy=True,
                normalize_embeddings=True,
                prompt_name="passage"  # This adds the "search_document:" prefix automatically
            )

            # Return list of numpy arrays
            return [emb for emb in embeddings] if isinstance(embeddings, np.ndarray) and embeddings.ndim > 1 else embeddings


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


@ray.remote(num_gpus=1)
class STSBMpnet(EmbeddingModel):
    def __init__(self):
        """Initialize the model and load to device."""
        # Use the custom NoSortingSentenceTransformer
        self.model = NoSortingSentenceTransformer("sentence-transformers/stsb-mpnet-base-v2")
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
                contents, convert_to_numpy=True, normalize_embeddings=True
            )

            # Return list of numpy arrays
            return [emb for emb in embeddings] if isinstance(embeddings, np.ndarray) and embeddings.ndim > 1 else embeddings


@ray.remote(num_gpus=0.03)
class STSBRoberta(EmbeddingModel):
    def __init__(self):
        """Initialize the model and load to device."""
        # Use the custom NoSortingSentenceTransformer
        self.model = NoSortingSentenceTransformer("sentence-transformers/stsb-roberta-base-v2")
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
                contents, convert_to_numpy=True, normalize_embeddings=True
            )

            # Return list of numpy arrays
            return [emb for emb in embeddings] if isinstance(embeddings, np.ndarray) and embeddings.ndim > 1 else embeddings


@ray.remote(num_gpus=0.02)
class STSBDistilRoberta(EmbeddingModel):
    def __init__(self):
        """Initialize the model and load to device."""
        # Use the custom NoSortingSentenceTransformer
        self.model = NoSortingSentenceTransformer(
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
                contents, convert_to_numpy=True, normalize_embeddings=True
            )

            # Return list of numpy arrays
            return [emb for emb in embeddings] if isinstance(embeddings, np.ndarray) and embeddings.ndim > 1 else embeddings


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
