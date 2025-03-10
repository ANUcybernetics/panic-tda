import logging
from typing import Callable, List, Optional, Union

from PIL import Image

from trajectory_tracer.db import db
from trajectory_tracer.models import invoke
from trajectory_tracer.schemas import Embedding, Invocation, InvocationType, Run

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_invocation(
    model: str,
    input: Union[str, Image.Image],
    run_id: str,
    sequence_number: int,
    input_invocation_id: Optional[str] = None,
    seed: int = 42
) -> Invocation:
    """
    Create an invocation object based on model type and input.

    Args:
        model: Name of the model to use
        input: Either a text prompt or an image
        run_id: The associated run ID
        sequence_number: Order in the run sequence
        input_invocation_id: Optional ID of the input invocation
        seed: Random seed for reproducibility

    Returns:
        A new Invocation object
    """
    # Determine invocation type based on input data
    invocation_type = InvocationType.TEXT if isinstance(input, str) else InvocationType.IMAGE

    # Create invocation without output (will be set later)
    invocation = Invocation(
        model=model,
        type=invocation_type,
        run_id=run_id,
        sequence_number=sequence_number,
        input_invocation_id=input_invocation_id,
        seed=seed
    )

    return invocation

def perform_invocation(invocation: Invocation, input: Union[str, Image.Image]) -> Invocation:
    """
    Perform the actual model invocation and update the invocation object with the result.

    Args:
        invocation: The invocation object to update
        input: The input data for the model

    Returns:
        The updated invocation with output
    """
    try:
        logger.info(f"Invoking model {invocation.model} with {type(input).__name__} input")
        result = invoke(invocation.model, input)
        invocation.output = result
        return invocation
    except Exception as e:
        logger.error(f"Error invoking model {invocation.model}: {e}")
        raise

def perform_run(run: Run) -> Run:
    """
    Execute a complete run based on the specified network of models.

    Args:
        run: The Run object containing configuration

    Returns:
        The updated Run with completed invocations
    """
    try:
        logger.info(f"Starting run {run.id} with seed {run.seed}")

        # Create a database session
        with db.create_session() as session:
            # Initialize with the first prompt
            current_input = run.initial_prompt
            previous_invocation_id = None

            # Process each model in the network sequence
            for sequence_number, model_name in enumerate(run.network):
                # Create invocation
                invocation = create_invocation(
                    model=model_name,
                    input=current_input,
                    run_id=run.id,
                    sequence_number=sequence_number,
                    input_invocation_id=previous_invocation_id,
                    seed=run.seed
                )

                # Execute the invocation
                invocation = perform_invocation(invocation, current_input)

                # Save to database
                db.create(session, invocation)

                # Set up for next invocation
                current_input = invocation.output
                previous_invocation_id = invocation.id

                logger.info(f"Completed invocation {sequence_number+1}/{len(run.network)}: {model_name}")

            # Refresh run to include all invocations
            run = db.get_by_id(session, Run, run.id)

        logger.info(f"Run {run.id} completed successfully")
        return run

    except Exception as e:
        logger.error(f"Error performing run {run.id}: {e}")
        raise

def embed_invocation(invocation: Invocation, embedding_fn: Callable) -> Embedding:
    """
    Generate an embedding for an invocation using the specified embedding function.

    Args:
        invocation: The invocation to embed
        embedding_fn: Function that takes an invocation and returns an Embedding

    Returns:
        The created Embedding object
    """
    try:
        logger.info(f"Creating embedding for invocation {invocation.id} with {embedding_fn.__name__}")
        embedding = embedding_fn(invocation)
        return embedding
    except Exception as e:
        logger.error(f"Error creating embedding for invocation {invocation.id}: {e}")
        raise


def create_run(network: List[str], initial_prompt: str, seed: int = 42) -> Run:
    """
    Create a new run with the specified parameters.

    Args:
        network: List of model names to use in sequence
        initial_prompt: The text prompt to start the run with
        seed: Random seed for reproducibility

    Returns:
        The created Run object
    """
    try:
        logger.info(f"Creating new run with network: {network}")

        # Create run object
        run = Run(
            network=network,
            initial_prompt=initial_prompt,
            seed=seed,
            length=len(network)
        )

        # Save to database
        with db.create_session() as session:
            db.create(session, run)

        logger.info(f"Created run with ID: {run.id}")
        return run

    except Exception as e:
        logger.error(f"Error creating run: {e}")
        raise
