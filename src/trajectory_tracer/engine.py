import itertools
import logging
import sys
from datetime import datetime
from typing import List, Optional, Union

from PIL import Image
from sqlmodel import Session

from trajectory_tracer.db import incomplete_embeddings
from trajectory_tracer.models import get_output_type, invoke
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
    invocation_type = get_output_type(model)

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


def create_embedding(embedder: str, invocation: Invocation, session: Session = None) -> Embedding:
    """
    Create an empty embedding object for the specified embedder and persist it to the database.

    Args:
        embedder: The name of the embedder class to use
        invocation: The Invocation object to embed
        session: SQLModel Session for database operations

    Returns:
        An Embedding object without the vector calculated yet

    Raises:
        ValueError: If the embedder doesn't exist or if the invocation type is incompatible
    """
    # Validate that the embedder class exists
    current_module = sys.modules["trajectory_tracer.embeddings"]
    if not hasattr(current_module, embedder):
        raise ValueError(f"Embedder '{embedder}' not found. Available embedders: "
                         f"{[cls for cls in dir(current_module) if isinstance(getattr(current_module, cls), type) and issubclass(getattr(current_module, cls), getattr(current_module, 'EmbeddingModel'))]}")

    # Get the model class and verify it's a valid EmbeddingModel
    model_class = getattr(current_module, embedder)
    if not issubclass(model_class, getattr(current_module, 'EmbeddingModel')):
        raise ValueError(f"'{embedder}' is not an EmbeddingModel subclass")

    # First check if output exists - no embedding can work without output
    if invocation.output is None:
        raise ValueError("Cannot embed an invocation with no output")

    # Handle special case for Nomic which dispatches to specific models
    if embedder == "Nomic":
        if invocation.type == InvocationType.TEXT:
            return create_embedding("NomicText", invocation, session)
        elif invocation.type == InvocationType.IMAGE:
            return create_embedding("NomicVision", invocation, session)
        else:
            raise ValueError(f"Unsupported invocation type for nomic embedding: {invocation.type}")

    # Create the embedding object
    embedding = Embedding(
        invocation_id=invocation.id,
        embedder=embedder,
        vector=None
    )

    # Save to database if session is provided
    if session:
        logger.info(f"Persisting empty embedding for invocation {invocation.id} with embedder {embedder}")
        session.add(embedding)
        session.commit()
        session.refresh(embedding)

    return embedding


def perform_embedding(embedding: Embedding, session: Session) -> Embedding:
    """
    Perform the embedding calculation for an existing Embedding object
    and update it with the computed vector and timestamps.

    Args:
        embedding: The Embedding object to compute the vector for
        session: SQLModel Session for database operations

    Returns:
        The updated Embedding object with vector calculated

    Raises:
        ValueError: If the embedding doesn't have an associated invocation
    """
    try:
        logger.info(f"Computing vector for embedding {embedding.id} with {embedding.embedder}")

        # Make sure we have the invocation data
        if not embedding.invocation:
            raise ValueError(f"Embedding {embedding.id} has no associated invocation")

        # Set the start timestamp
        embedding.started_at = datetime.now()

        # Get the content to embed from the invocation output
        content = embedding.invocation.output

        # Use the embed function from embeddings to calculate the vector
        from trajectory_tracer.embeddings import embed
        embedding.vector = embed(embedding.embedder, content)

        # Set the completion timestamp
        embedding.completed_at = datetime.now()

        # Save to database
        session.add(embedding)
        session.commit()
        session.refresh(embedding)

        logger.info(f"Successfully computed vector for embedding {embedding.id}")
        return embedding

    except Exception as e:
        logger.error(f"Error performing embedding {embedding.id}: {e}")
        raise


def compute_missing_embeds(session: Session) -> int:
    """
    Find all the Embedding objects in the database without a vector,
    perform the embedding calculation, and update the database.

    Args:
        session: SQLModel Session for database operations

    Returns:
        Number of embeddings that were processed
    """
    try:
        logger.info("Finding embedding objects without vectors...")

        # Query for embeddings where vector is None
        incomplete = incomplete_embeddings(session)
        total_to_process = len(incomplete)
        logger.info(f"Found {total_to_process} embeddings without vectors")

        if total_to_process == 0:
            return 0

        processed_count = 0
        for embedding in incomplete:
            try:
                # Generate the embedding
                logger.info(f"Processing embedding {embedding.id} for invocation {embedding.invocation_id} with model {embedding.embedder}")

                # Perform the embedding calculation
                perform_embedding(embedding, session)

                processed_count += 1
                logger.info(f"Successfully processed embedding {processed_count}/{total_to_process}")

            except Exception as e:
                logger.error(f"Error processing embedding {embedding.id}: {e}")
                session.rollback()

        logger.info(f"Completed processing {processed_count}/{total_to_process} missing embeddings")
        return processed_count

    except Exception as e:
        logger.error(f"Error in perform_missing_embeds: {e}")
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
    ))

    total_combinations = len(combinations)
    logger.info(f"Starting experiment with {total_combinations} total run configurations")

    # Process each combination
    successful_runs = 0
    for i, (network, seed, prompt) in enumerate(combinations):
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

            # create "empty" embedding objects - will fill in the vectors later
            for embedder in config.embedders:
                for invocation in run.invocations:
                    create_embedding(embedder, invocation, session)

            successful_runs += 1
            logger.info(f"Completed run {i+1}/{total_combinations}")

        except Exception as e:
            logger.error(f"Error processing run {i+1}/{total_combinations}: {e}")
            # Continue with next run even if this one fails

    # finally, run the embedders for all embeddings that are missing vectors
    compute_missing_embeds(session)

    logger.info(f"Experiment completed with {successful_runs} successful runs")
