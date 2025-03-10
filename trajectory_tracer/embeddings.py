import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

from trajectory_tracer.schemas import Embedding, Invocation


def nomic_embed_text(invocation: Invocation) -> Embedding:
    """
    Calculate an embedding for the input text of an invocation using the nomic-embed-text model.

    Args:
        invocation: The Invocation object containing the input to embed

    Returns:
        An Embedding object with the calculated vector
    """
    # Only works with text inputs
    if isinstance(invocation.input, Image.Image):
        raise ValueError("Cannot embed image input with nomic-embed-text")

    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # Prepare input text with the required prefix for RAG
    text_with_prefix = f"search_query: {invocation.input}"

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('nomic-ai/nomic-embed-text-v1.5')
    model = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
    model.eval()

    # Tokenize the input
    encoded_input = tokenizer([text_with_prefix], padding=True, truncation=True, return_tensors='pt')

    # Calculate embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Process the embeddings
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # Convert to list for storage
    vector = embeddings[0].tolist()

    # Return the embedding
    return Embedding(
        invocation_id=invocation.id,
        embedding_model="nomic-embed-text-v1.5",
        vector=vector
    )


def nomic_embed_vision(invocation: Invocation) -> Embedding:
    """
    Calculate an embedding for an image input of an invocation using the nomic-embed-vision model.

    Args:
        invocation: The Invocation object containing the image input to embed

    Returns:
        An Embedding object with the calculated vector
    """
    # Only works with image inputs
    if isinstance(invocation.input, str):
        raise ValueError("Cannot embed text input with nomic-embed-vision")


    # Load the model and processor
    processor = AutoImageProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1.5")
    vision_model = AutoModel.from_pretrained("nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True)
    vision_model.eval()

    # The input is already a PIL Image, so we can use it directly
    image = invocation.input

    # Process the image
    inputs = processor(image, return_tensors="pt")

    # Calculate embeddings
    with torch.no_grad():
        img_emb = vision_model(**inputs).last_hidden_state
        img_embeddings = F.normalize(img_emb[:, 0], p=2, dim=1)

    # Convert to list for storage
    vector = img_embeddings[0].tolist()

    # Return the embedding
    return Embedding(
        invocation_id=invocation.id,
        embedding_model="nomic-embed-vision-v1.5",
        vector=vector
    )

# other potential multimodal embedding models:
# - https://huggingface.co/nvidia/MM-Embed
# - https://huggingface.co/jinaai/jina-clip-v2
# - https://huggingface.co/nielsr/imagebind-huge

def dummy_embedding(invocation: Invocation) -> Embedding:
    """
    Generate a random embedding vector for testing purposes.

    Args:
        invocation: The Invocation object to generate an embedding for

    Returns:
        An Embedding object with a random vector of dimension 768
    """
    import numpy as np

    # Generate a random vector of dimension 768
    vector = np.random.rand(768).tolist()

    # Return the embedding
    return Embedding(
        invocation_id=invocation.id,
        embedding_model="dummy-embedding",
        vector=vector
    )
