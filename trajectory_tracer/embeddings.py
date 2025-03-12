import sys

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel
from transformers import AutoImageProcessor, AutoModel

from trajectory_tracer.schemas import Embedding, Invocation, InvocationType

# other potential multimodal embedding models:
# - https://huggingface.co/nvidia/MM-Embed
# - https://huggingface.co/jinaai/jina-clip-v2
# - https://huggingface.co/nielsr/imagebind-huge

class EmbeddingModel(BaseModel):
    pass


class NomicText(EmbeddingModel):
    @staticmethod
    def embed(invocation: Invocation) -> Embedding:
        """
        Calculate an embedding for the output text of an invocation using the nomic-embed-text model.

        Args:
            invocation: The Invocation object containing the text output to embed

        Returns:
            An Embedding object with the calculated vector
        """
        # Only works with text outputs
        if invocation.type != InvocationType.TEXT or not invocation.output_text:
            raise ValueError("Cannot embed non-text output with nomic-embed-text")

        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
        sentences = [f"clustering: {invocation.output}"]
        vector = model.encode(sentences)

        # Return the embedding
        return Embedding(
            invocation_id=invocation.id,
            embedding_model="nomic-embed-text-v1.5",
            vector=vector
        )


class NomicVision(EmbeddingModel):
    @staticmethod
    def embed(invocation: Invocation) -> Embedding:
        """
        Calculate an embedding for an image output of an invocation using the nomic-embed-vision model.

        Args:
            invocation: The Invocation object containing the image output to embed

        Returns:
            An Embedding object with the calculated vector
        """
        # Only works with image outputs
        if invocation.type != InvocationType.IMAGE or not invocation.output_image_data:
            raise ValueError("Cannot embed non-image output with nomic-embed-vision")

        # Load the model and processor
        processor = AutoImageProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1.5")
        vision_model = AutoModel.from_pretrained("nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True)
        vision_model.eval()

        # Get the image from output property
        image = invocation.output

        # Process the image
        inputs = processor(image, return_tensors="pt")

        # Calculate embeddings
        with torch.no_grad():
            img_emb = vision_model(**inputs).last_hidden_state
            img_embeddings = F.normalize(img_emb[:, 0], p=2, dim=1)

        # Convert to numpy array for storage
        vector = img_embeddings[0].numpy()

        # Return the embedding
        return Embedding(
            invocation_id=invocation.id,
            embedding_model="nomic-embed-vision-v1.5",
            vector=vector
        )


class Nomic(EmbeddingModel):
    @staticmethod
    def embed(invocation: Invocation) -> Embedding:
        """
        Dispatch to the appropriate nomic embedding function based on the invocation type.

        Args:
            invocation: The Invocation object to embed

        Returns:
            An Embedding object with the calculated vector

        Raises:
            ValueError: If the invocation type is not supported
        """
        if invocation.type == InvocationType.TEXT:
            return NomicText.embed(invocation)
        elif invocation.type == InvocationType.IMAGE:
            return NomicVision.embed(invocation)
        else:
            raise ValueError(f"Unsupported invocation type for nomic embedding: {invocation.type}")


## from here these ones used for testing

class Dummy(EmbeddingModel):
    @staticmethod
    def embed(invocation: Invocation) -> Embedding:
        """
        Generate a random embedding vector for testing purposes.

        Args:
            invocation: The Invocation object to generate an embedding for

        Returns:
            An Embedding object with a random vector of dimension 768
        """
        # Generate a random vector of dimension 768
        vector = np.random.rand(768).astype(np.float32)

        # Return the embedding
        return Embedding(
            invocation_id=invocation.id,
            embedding_model="dummy-embedding",
            vector=vector
        )


class Dummy2(EmbeddingModel):
    @staticmethod
    def embed(invocation: Invocation) -> Embedding:
        """
        Generate a(nother) random embedding vector for testing purposes.

        Args:
            invocation: The Invocation object to generate an embedding for

        Returns:
            An Embedding object with a random vector of dimension 768
        """
        # Generate a random vector of dimension 768
        vector = np.random.rand(768).astype(np.float32)

        # Return the embedding
        return Embedding(
            invocation_id=invocation.id,
            embedding_model="dummy2-embedding",
            vector=vector
        )


def embed(embedding_model: str, invocation: Invocation) -> Embedding:
    """
    Dynamically dispatches to the specified embedding model's embed method.

    Args:
        embedding_model: The name of the embedding model class to use
        invocation: The Invocation object to embed

    Returns:
        The result of the model's embed method

    Raises:
        ValueError: If the embedding model doesn't exist
    """
    current_module = sys.modules[__name__]

    # Try to find the model class in this module
    if not hasattr(current_module, embedding_model):
        raise ValueError(f"Embedding model '{embedding_model}' not found. Available models: "
                         f"{[cls for cls in dir(current_module) if isinstance(getattr(current_module, cls), type) and issubclass(getattr(current_module, cls), EmbeddingModel)]}")

    # Get the model class
    model_class = getattr(current_module, embedding_model)

    # Check if it's a subclass of EmbeddingModel
    if not issubclass(model_class, EmbeddingModel):
        raise ValueError(f"'{embedding_model}' is not an EmbeddingModel subclass")

    # Call the embed method
    return model_class.embed(invocation)
