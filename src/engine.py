import logging
from typing import Iterator, Optional, Union
from uuid import uuid4

import numpy as np
import torch
from PIL import Image

from src.db import save_invocation
from src.models import flux_dev_t2i, moondream_i2t
from src.schemas import Invocation, Network

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dictionary mapping model names to functions
MODEL_REGISTRY = {
    "moondream_i2t": moondream_i2t,
    "flux_dev_t2i": flux_dev_t2i,
}

def invoke_model(model_name: str, content: Union[str, Image.Image]) -> Union[str, Image.Image]:
    """
    Invoke a model with the given content and return the result.

    Args:
        model_name: Name of the model to invoke
        content: Input content (text or image)

    Returns:
        The model's output (text or image)
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")

    model_func = MODEL_REGISTRY[model_name]

    try:
        return model_func(content)
    except Exception as e:
        logger.error(f"Error invoking model {model_name}: {e}")
        raise

def invoke_and_yield_next(invocation: Invocation) -> Optional[Invocation]:
    """
    Process an invocation by running its model on the input,
    saving the result, and creating the next invocation.

    Args:
        invocation: The invocation to process

    Returns:
        The next invocation in the sequence with:
        - Input set to the output of the current invocation
        - Model set to the next model in the network
        - Sequence number incremented
        Or None if the invocation output is None
    """
    # Invoke the model
    try:
        output = invoke_model(invocation.model, invocation.input)
        invocation.output = output
    except Exception as e:
        logger.error(f"Failed to invoke model {invocation.model}: {e}")
        # In case of failure, set output to a descriptive error
        invocation.output = f"ERROR: {str(e)}"

    # Save the completed invocation
    save_invocation(invocation)
    logger.info(f"Completed invocation {invocation.sequence_number} with model {invocation.model}")

    # If output is None, can't proceed
    if invocation.output is None:
        logger.warning(f"No output from invocation {invocation.sequence_number}, cannot create next invocation")
        return None

    # Get the next model from the network (cycling through the list)
    next_model_idx = (invocation.network.models.index(invocation.model) + 1) % len(invocation.network.models)
    next_model = invocation.network.models[next_model_idx]

    # Create the next invocation
    next_invocation = Invocation(
        model=next_model,
        input=invocation.output,  # Output of current becomes input of next
        output=None,  # Will be set when this invocation is processed
        seed=invocation.seed,
        run_id=invocation.run_id,
        network=invocation.network,
        sequence_number=invocation.sequence_number + 1
    )

    return next_invocation

def perform_run(network: Network, prompt: str, seed: int, run_length: int) -> Iterator[Invocation]:
    """
    Perform a run through a network starting with a text prompt.

    Args:
        network: The network containing models to cycle through
        prompt: The initial text prompt
        seed: Random seed for reproducibility
        run_length: Number of invocations to perform

    Returns:
        Iterator yielding each invocation in sequence
    """
    if not network.models:
        raise ValueError("Network must contain at least one model")

    # Set random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Generate a UUID4 for the first invocation and use it for both id and run_id
    invocation_id = uuid4()

    # Create the initial invocation with the first model
    initial_invocation = Invocation(
        id=invocation_id,  # Set the UUID explicitly
        model=network.models[0],
        input=prompt,
        output=None,  # Will be set during processing
        seed=seed,
        run_id=int(str(invocation_id.int)[:10]),  # Convert UUID to integer and use first 10 digits
        network=network,
    )
    logger.info(f"Starting run {invocation_id} with seed {seed} and {len(network.models)} models")

    current_invocation = initial_invocation

    # Process each invocation in sequence up to run_length
    for i in range(run_length):
        # Skip if the current invocation is None (which would happen if the previous invocation failed)
        if current_invocation is None:
            logger.warning(f"Run {invocation_id} ended early after {i} invocations due to None invocation")
            break

        # Yield the current invocation before processing
        yield current_invocation

        # Process the current invocation and get the next one
        # Note: invoke_and_yield_next already saves the invocation to the database
        current_invocation = invoke_and_yield_next(current_invocation)

    completed_count = run_length if current_invocation is not None else run_length - 1
    logger.info(f"Completed run {invocation_id} with {completed_count} invocations")
