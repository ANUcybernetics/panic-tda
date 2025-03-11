import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

from trajectory_tracer.schemas import Embedding, Invocation, InvocationType

# other potential multimodal embedding models:
# - https://huggingface.co/nvidia/MM-Embed
# - https://huggingface.co/jinaai/jina-clip-v2
# - https://huggingface.co/nielsr/imagebind-huge

def nomic_text(invocation: Invocation) -> Embedding:
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

    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # Prepare output text with the required prefix for RAG
    text_with_prefix = f"search_query: {invocation.output_text}"

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('nomic-ai/nomic-embed-text-v1.5')
    model = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
    model.eval()

    # Tokenize the output
    encoded_input = tokenizer([text_with_prefix], padding=True, truncation=True, return_tensors='pt')

    # Calculate embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Process the embeddings
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # Convert to numpy array for storage
    vector = embeddings[0].numpy()

    # Return the embedding
    return Embedding(
        invocation_id=invocation.id,
        embedding_model="nomic-embed-text-v1.5",
        vector=vector
    )


def nomic_vision(invocation: Invocation) -> Embedding:
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


def nomic(invocation: Invocation) -> Embedding:
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
        return nomic_text(invocation)
    elif invocation.type == InvocationType.IMAGE:
        return nomic_vision(invocation)
    else:
        raise ValueError(f"Unsupported invocation type for nomic embedding: {invocation.type}")


## from here these ones used for testing

def dummy(invocation: Invocation) -> Embedding:
    """
    Generate a random embedding vector for testing purposes.

    Args:
        invocation: The Invocation object to generate an embedding for

    Returns:
        An Embedding object with a random vector of dimension 768
    """
    import numpy as np

    # Generate a random vector of dimension 768
    vector = np.random.rand(768).astype(np.float32)

    # Return the embedding
    return Embedding(
        invocation_id=invocation.id,
        embedding_model="dummy-embedding",
        vector=vector
    )


def dummy2(invocation: Invocation) -> Embedding:
    """
    Generate a(nother) random embedding vector for testing purposes.

    Args:
        invocation: The Invocation object to generate an embedding for

    Returns:
        An Embedding object with a random vector of dimension 768
    """
    import numpy as np

    # Generate a random vector of dimension 768
    vector = np.random.rand(768).astype(np.float32)

    # Return the embedding
    return Embedding(
        invocation_id=invocation.id,
        embedding_model="dummy2-embedding",
        vector=vector
    )
