from pathlib import Path
from typing import Union

import torch
from schemas import Invocation, InvocationType
from transformers import Pipeline, pipeline

# Global cache for models to avoid reloading
_MODEL_CACHE = {}

def get_pipeline(model_name: str, task_type: InvocationType) -> Pipeline:
    """
    Get a HuggingFace pipeline for the specified model and task type.
    Caches models to avoid reloading.

    Args:
        model_name: The HuggingFace model name
        task_type: The type of task (text or image)

    Returns:
        A HuggingFace pipeline
    """
    cache_key = f"{model_name}_{task_type.value}"

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if task_type == InvocationType.TEXT:
        # Configure text generation pipeline
        pipe = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    elif task_type == InvocationType.IMAGE:
        # Configure image generation pipeline
        pipe = pipeline(
            "image-to-image" if "image-to-image" in model_name else "text-to-image",
            model=model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    else:
        raise ValueError(f"Unsupported invocation type: {task_type}")

    _MODEL_CACHE[cache_key] = pipe
    return pipe

def save_output(output: Union[str, torch.Tensor], invocation: Invocation) -> Union[str, Path]:
    """
    Save the model output to a file if needed or return as string.

    Args:
        output: The model output (string or image tensor)
        invocation: The invocation object

    Returns:
        Either the string output or path to saved file
    """
    if invocation.type == InvocationType.TEXT:
        return output
    elif invocation.type == InvocationType.IMAGE:
        # For images, save to a file
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        # Create a unique filename based on invocation ID
        filename = output_dir / f"{invocation.id}.png"

        # Save the image
        output.save(filename)
        return filename
    else:
        raise ValueError(f"Unsupported invocation type: {invocation.type}")

def invoke(invocation: Invocation) -> Invocation:
    """
    Execute a model invocation and update the invocation with the output.

    Args:
        invocation: An Invocation object with model, input, and parameters

    Returns:
        The updated Invocation with output field filled
    """
    # Get the appropriate pipeline
    pipe = get_pipeline(invocation.model, invocation.type)

    # Prepare input
    input_content = invocation.input
    if isinstance(input_content, Path) and input_content.exists():
        if invocation.type == InvocationType.IMAGE:
            # Load image from file
            from PIL import Image
            input_content = Image.open(input_content)
        else:
            # Load text from file
            with open(input_content, 'r') as f:
                input_content = f.read()

    # Set generation parameters
    gen_kwargs = {"seed": invocation.seed}

    if invocation.type == InvocationType.TEXT:
        gen_kwargs.update({
            "max_new_tokens": 256,
            "do_sample": True,
            "temperature": 0.7,
        })

    # Run the model
    with torch.random.fork_rng():
        torch.manual_seed(invocation.seed)
        if invocation.type == InvocationType.TEXT:
            result = pipe(input_content, **gen_kwargs)
            output = result[0]["generated_text"]
            # Often we want just the new content, not the input repeated
            if output.startswith(input_content):
                output = output[len(input_content):]
        else:
            result = pipe(input_content, **gen_kwargs)
            output = result[0]  # Usually a PIL Image for image pipelines

    # Save output and update invocation
    output_path_or_str = save_output(output, invocation)
    invocation.output = output_path_or_str

    return invocation

def generate_caption_with_moondream(image_path: Union[str, Path]) -> str:
    """
    Generate a caption for an image using the Moondream 2 model.

    Args:
        image_path: Path to the image file

    Returns:
        Generated caption as a string
    """
    # Import here to avoid circular imports
    from PIL import Image
    from transformers import AutoModelForCausalLM

    # Load image
    if isinstance(image_path, str):
        image_path = Path(image_path)

    image = Image.open(image_path)

    # Use cache if available
    cache_key = "moondream2_captioning"
    if cache_key not in _MODEL_CACHE:
        model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            revision="2025-01-09",
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
        _MODEL_CACHE[cache_key] = model

    model = _MODEL_CACHE[cache_key]

    # Generate caption
    caption = model.caption(image, length="normal")["caption"]

    return caption
