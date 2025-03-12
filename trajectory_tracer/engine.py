import itertools
import logging
from datetime import datetime
from typing import List, Optional, Union

from PIL import Image
from sqlmodel import Session

from trajectory_tracer.embeddings import embed
from trajectory_tracer.models import invoke
from trajectory_tracer.schemas import (
    Embedding,
    ExperimentConfig,
    Invocation,
    InvocationType,
    Run,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_invocation(
    model: str,
    input: Union[str, Image.Image],
    run_id: str,
    sequence_number: int,
    session: Session,
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
        session: SQLModel Session for database operations
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

    # Save to database
    session.add(invocation)
    session.commit()
    session.refresh(invocation)

    logger.info(f"Created invocation with ID: {invocation.id}")
    return invocation

def perform_invocation(invocation: Invocation, input: Union[str, Image.Image], session: Session) -> Invocation:
    """
    Perform the actual model invocation and update the invocation object with the result.

    Args:
        invocation: The invocation object to update
        input: The input data for the model
        session: SQLModel Session for database operations

    Returns:
        The updated invocation with output
    """
    try:
        logger.info(f"Invoking model {invocation.model} with {type(input).__name__} input")
        invocation.started_at = datetime.now()
        result = invoke(invocation.model, input)
        invocation.completed_at = datetime.now()
        invocation.output = result

        # Save changes to database
        session.add(invocation)
        session.commit()
        session.refresh(invocation)

        return invocation
    except Exception as e:
        logger.error(f"Error invoking model {invocation.model}: {e}")
        raise


def create_run(network: List[str], initial_prompt: str, session: Session, seed: int = 42, run_length: int = None) -> Run:
    """
    Create a new run with the specified parameters.

    Args:
        network: List of model names to use in sequence
        initial_prompt: The text prompt to start the run with
        seed: Random seed for reproducibility
        session: SQLModel Session for database operations
        run_length: Length of the run

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
            length=run_length
        )

        # Save to database
        session.add(run)
        session.commit()
        session.refresh(run)

        logger.info(f"Created run with ID: {run.id}")
        return run

    except Exception as e:
        logger.error(f"Error creating run: {e}")
        raise


def perform_run(run: Run, session: Session) -> Run:
    """
    Execute a complete run based on the specified network of models.

    Args:
        run: The Run object containing configuration
        session: SQLModel Session for database operations

    Returns:
        The updated Run with completed invocations
    """
    try:
        logger.info(f"Starting run {run.id} with seed {run.seed}")

        # Initialize with the first prompt
        current_input = run.initial_prompt
        previous_invocation_id = None
        network_length = len(run.network)

        # Process each invocation in the run
        for sequence_number in range(run.length):
            # Get the next model in the network (cycling if necessary)
            model_index = sequence_number % network_length
            model_name = run.network[model_index]

            # Create invocation
            invocation = create_invocation(
                model=model_name,
                input=current_input,
                run_id=run.id,
                sequence_number=sequence_number,
                input_invocation_id=previous_invocation_id,
                seed=run.seed,
                session=session
            )

            # Execute the invocation
            invocation = perform_invocation(invocation, current_input, session=session)

            # Set up for next invocation
            current_input = invocation.output
            previous_invocation_id = invocation.id

            logger.info(f"Completed invocation {sequence_number}/{run.length}: {model_name}")

        # Refresh the run to include the invocations
        session.refresh(run)

        logger.info(f"Run {run.id} completed successfully")
        return run

    except Exception as e:
        logger.error(f"Error performing run {run.id}: {e}")
        raise


def embed_invocation(invocation: Invocation, embedding_model: str, session: Session) -> Embedding:
    """
    Generate an embedding for an invocation using the specified embedding model.

    Args:
        invocation: The invocation to embed
        embedding_model: Name of the embedding model to use
        session: SQLModel Session for database operations

    Returns:
        The created Embedding object
    """
    try:
        logger.info(f"Creating embedding for invocation {invocation.id} with {embedding_model}")
        started_at_ts = datetime.now()
        embedding = embed(embedding_model, invocation)
        embedding.completed_at = datetime.now()
        embedding.started_at = started_at_ts

        # Save to database
        session.add(embedding)
        session.commit()
        session.refresh(embedding)

        return embedding
    except Exception as e:
        logger.error(f"Error creating embedding for invocation {invocation.id}: {e}")
        raise


def embed_run(run: Run, embedding_model: str, session: Session) -> List[Embedding]:
    """
    Generate embeddings for all invocations in a run using the specified embedding model.

    Args:
        run: The Run object containing invocations to embed
        embedding_model: Name of the embedding model to use
        session: SQLModel Session for database operations

    Returns:
        List of created Embedding objects
    """
    try:
        logger.info(f"Embedding all invocations for run {run.id}")
        embeddings = []

        # Ensure invocations are loaded
        if not run.invocations:
            session.refresh(run)

        # Generate embeddings for each invocation
        for invocation in run.invocations:
            embedding = embed_invocation(invocation, embedding_model, session)
            embeddings.append(embedding)

        logger.info(f"Successfully embedded {len(embeddings)} invocations for run {run.id}")
        return embeddings

    except Exception as e:
        logger.error(f"Error embedding run {run.id}: {e}")
        raise


def perform_experiment(config: ExperimentConfig, session: Session) -> None:
    """
    Create and execute runs for all combinations defined in the ExperimentConfig.

    Args:
        config: The experiment configuration containing network definitions,
               seeds, prompts, and embedding model specifications
        session: SQLModel Session for database operations
    """
    # Generate cartesian product of all parameters
    combinations = list(itertools.product(
        config.networks,
        config.seeds,
        config.prompts,
        config.embedders
    ))

    total_combinations = len(combinations)
    logger.info(f"Starting experiment with {total_combinations} total run configurations")

    # Process each combination
    successful_runs = 0
    for i, (network, seed, prompt, embedding_model) in enumerate(combinations):
        try:
            # Create the run
            logger.info(f"Creating run {i+1}/{total_combinations}: {network} with seed {seed}")
            run = create_run(
                network=network,
                initial_prompt=prompt,
                seed=seed,
                session=session,
                run_length=config.run_length
            )

            # Execute the run
            run = perform_run(run, session)

            # Generate embeddings for the run
            embeddings = embed_run(run, embedding_model, session)
            logger.info(f"Generated {len(embeddings)} embeddings with model {embedding_model}")

            successful_runs += 1
            logger.info(f"Completed run {i+1}/{total_combinations}")

        except Exception as e:
            logger.error(f"Error processing run {i+1}/{total_combinations}: {e}")
            # Continue with next run even if this one fails

    logger.info(f"Experiment completed with {successful_runs} successful runs")
