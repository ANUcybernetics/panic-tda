import itertools
import logging
import sys
from datetime import datetime
from typing import List, Optional, Union
from uuid import UUID

import numpy as np
from PIL import Image
from sqlmodel import Session

from trajectory_tracer.db import incomplete_embeddings, incomplete_persistence_diagrams
from trajectory_tracer.embeddings import embed
from trajectory_tracer.genai_models import get_output_type, invoke, unload_all_models
from trajectory_tracer.schemas import (
    Embedding,
    ExperimentConfig,
    Invocation,
    InvocationType,
    Run,
)
from trajectory_tracer.tda import giotto_phd

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_invocation(
    model: str,
    input: Union[str, Image.Image],
    run_id: str,
    sequence_number: int,
    session: Session,
    input_invocation_id: Optional[str] = None,
    seed: int = 42,
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
        seed=seed,
    )

    # Save to database
    session.add(invocation)
    session.commit()
    session.refresh(invocation)

    logger.debug(f"Created invocation with ID: {invocation.id}")
    return invocation


def perform_invocation(
    invocation: Invocation, input: Union[str, Image.Image], session: Session
) -> Invocation:
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
        logger.debug(
            f"Invoking model {invocation.model} with {type(input).__name__} input"
        )
        invocation.started_at = datetime.now()
        result = invoke(invocation.model, input, invocation.seed)
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


def create_run(
    network: List[str],
    initial_prompt: str,
    session: Session,
    seed: int = 42,
    run_length: int = None,
) -> Run:
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
        logger.debug(f"Creating new run with network: {network}")

        # Create run object
        run = Run(
            network=network, initial_prompt=initial_prompt, seed=seed, length=run_length
        )

        # Save to database
        session.add(run)
        session.commit()
        session.refresh(run)

        logger.debug(f"Created run with ID: {run.id}")
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

        # Track outputs to detect loops (only if seed is not -1)
        seen_outputs = set()
        track_duplicates = run.seed != -1

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
                session=session,
            )

            # Execute the invocation
            invocation = perform_invocation(invocation, current_input, session=session)

            # Check for duplicate outputs if tracking is enabled
            if track_duplicates:
                # Convert output to a hashable representation based on type
                hashable_output = None
                if isinstance(invocation.output, str):
                    hashable_output = invocation.output
                elif isinstance(invocation.output, Image.Image):
                    # Convert image to a bytes representation for hashing
                    import io

                    buffer = io.BytesIO()
                    # TODO if the duplicate-tracking for images is too aggressive, bump up the quality
                    invocation.output.save(buffer, format="JPEG", quality=30)
                    hashable_output = buffer.getvalue()
                else:
                    # Handle any other types that might occur
                    hashable_output = str(invocation.output)

                if hashable_output in seen_outputs:
                    logger.info(
                        f"Detected duplicate output in run {run.id} at sequence {sequence_number}. Stopping run."
                    )
                    break
                seen_outputs.add(hashable_output)

            # Set up for next invocation
            current_input = invocation.output
            previous_invocation_id = invocation.id

            logger.debug(
                f"Completed invocation {sequence_number}/{run.length}: {model_name}"
            )

        # Refresh the run to include the invocations
        session.refresh(run)

        logger.info(f"Run {run.id} completed successfully")
        return run

    except Exception as e:
        logger.error(f"Error performing run {run.id}: {e}")
        raise


def create_embedding(
    embedding_model: str, invocation: Invocation, session: Session = None
) -> Embedding:
    """
    Create an empty embedding object for the specified embedding model and persist it to the database.

    Args:
        embedding_model: The name of the embedding model class to use
        invocation: The Invocation object to embed
        session: SQLModel Session for database operations

    Returns:
        An Embedding object without the vector calculated yet

    Raises:
        ValueError: If the embedding model doesn't exist or if the invocation type is incompatible
    """
    # Validate that the embedding_model class exists
    embeddings_module = sys.modules["trajectory_tracer.embeddings"]
    if not hasattr(embeddings_module, embedding_model):
        raise ValueError(
            f"Embedding model '{embedding_model}' not found. Available embedding models: "
            f"{[cls for cls in dir(embeddings_module) if isinstance(getattr(embeddings_module, cls), type) and issubclass(getattr(embeddings_module, cls), getattr(embeddings_module, 'EmbeddingModel'))]}"
        )

    # Get the model class and verify it's a valid EmbeddingModel
    model_class = getattr(embeddings_module, embedding_model)
    if not issubclass(model_class, getattr(embeddings_module, "EmbeddingModel")):
        raise ValueError(f"'{embedding_model}' is not an EmbeddingModel subclass")

    # First check if output exists - no embedding can work without output
    if invocation.output is None:
        raise ValueError("Cannot embed an invocation with no output")

    # Handle special case for Nomic which dispatches to specific models
    if embedding_model == "Nomic":
        if invocation.type == InvocationType.TEXT:
            return create_embedding("NomicText", invocation, session)
        elif invocation.type == InvocationType.IMAGE:
            return create_embedding("NomicVision", invocation, session)
        else:
            raise ValueError(
                f"Unsupported invocation type for nomic embedding: {invocation.type}"
            )

    # Create the embedding object
    embedding = Embedding(
        invocation_id=invocation.id, embedding_model=embedding_model, vector=None
    )

    # Save to database if session is provided
    if session:
        logger.debug(
            f"Persisting empty embedding for invocation {invocation.id} with embedding model {embedding_model}"
        )
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
        logger.debug(
            f"Computing vector for embedding {embedding.id} with {embedding.embedding_model}"
        )

        # Make sure we have the invocation data
        if not embedding.invocation:
            raise ValueError(f"Embedding {embedding.id} has no associated invocation")

        # Set the start timestamp
        embedding.started_at = datetime.now()

        # Get the content to embed from the invocation output
        content = embedding.invocation.output

        # Use the embed function from embeddings to calculate the vector
        embedding.vector = embed(embedding.embedding_model, content)

        # Set the completion timestamp
        embedding.completed_at = datetime.now()

        # Save to database
        session.add(embedding)
        session.commit()
        session.refresh(embedding)

        logger.debug(f"Successfully computed vector for embedding {embedding.id}")
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
        logger.debug("Finding embedding objects without vectors...")

        # Query for embeddings where vector is None
        incomplete = incomplete_embeddings(session)
        total_to_process = len(incomplete)
        logger.debug(f"Found {total_to_process} embeddings without vectors")

        if total_to_process == 0:
            return 0

        processed_count = 0
        for embedding in incomplete:
            try:
                # Generate the embedding
                logger.debug(
                    f"Processing embedding {embedding.id} for invocation {embedding.invocation_id} with model {embedding.embedding_model}"
                )

                # Perform the embedding calculation
                perform_embedding(embedding, session)

                processed_count += 1
                logger.debug(
                    f"Successfully processed embedding {processed_count}/{total_to_process}"
                )

            except Exception as e:
                logger.error(f"Error processing embedding {embedding.id}: {e}")
                session.rollback()

        logger.debug(
            f"Completed processing {processed_count}/{total_to_process} missing embeddings"
        )
        return processed_count

    except Exception as e:
        logger.error(f"Error in perform_missing_embeds: {e}")
        raise


def create_persistence_diagram(run_id: UUID, embedding_model: str, session: Session):
    """
    Create a persistence diagram object for a run.

    Args:
        run_id: The ID of the run
        session: SQLModel Session for database operations

    Returns:
        The created PersistenceDiagram object
    """
    from trajectory_tracer.schemas import PersistenceDiagram

    try:
        logger.debug(
            f"Creating persistence diagram for run {run_id} and embedding model {embedding_model}"
        )

        # Create persistence diagram object with empty generators
        pd = PersistenceDiagram(
            run_id=run_id, embedding_model=embedding_model, generators=[]
        )

        # Save to database
        session.add(pd)
        session.commit()
        session.refresh(pd)

        logger.debug(f"Created persistence diagram with ID: {pd.id}")
        return pd

    except Exception as e:
        logger.error(f"Error creating persistence diagram for run {run_id}: {e}")
        raise


def perform_persistence_diagram(persistence_diagram, session: Session):
    """
    Calculate and populate the generators for a persistence diagram.

    Args:
        persistence_diagram: The PersistenceDiagram object to update
        session: SQLModel Session for database operations

    Returns:
        The updated PersistenceDiagram object with generators
    """
    try:
        # Get the run associated with this diagram
        run = persistence_diagram.run

        logger.debug(f"Computing persistence diagram for run {run.id}")

        # Check if run is complete using is_complete property
        if not run.is_complete:
            raise ValueError(f"Run {run.id} is not complete")

        # Get embeddings for the specific embedding model
        embeddings = run.embeddings_by_model(persistence_diagram.embedding_model)

        # Check that we have one embedding for each sequence number
        sequence_numbers = set(emb.invocation.sequence_number for emb in embeddings)
        if len(sequence_numbers) != run.length:
            missing = set(range(run.length)) - sequence_numbers
            raise ValueError(
                f"Run {run.id} is missing embeddings for sequence numbers: {missing}"
            )

        # Check that all embeddings have vectors
        missing_vectors = [emb.id for emb in embeddings if emb.vector is None]
        if missing_vectors:
            raise ValueError(
                f"Run {run.id} has embeddings without vectors: {missing_vectors}"
            )

        # Sort embeddings by sequence number
        sorted_embeddings = sorted(
            embeddings, key=lambda e: e.invocation.sequence_number
        )

        # Create point cloud from embedding vectors
        point_cloud = np.array([emb.vector for emb in sorted_embeddings])

        # Set start timestamp - only timing the persistence diagram computation
        persistence_diagram.started_at = datetime.now()

        # Compute persistence diagram
        persistence_diagram.generators = giotto_phd(point_cloud)

        # Set completion timestamp
        persistence_diagram.completed_at = datetime.now()

        # Save to database
        session.add(persistence_diagram)
        session.commit()
        session.refresh(persistence_diagram)

        logger.debug(f"Successfully computed persistence diagram for run {run.id}")
        return persistence_diagram

    except Exception as e:
        logger.error(f"Error computing persistence diagram: {e}")
        raise


def compute_missing_persistence_diagrams(session: Session) -> int:
    """
    Find all the PersistenceDiagram objects in the database without generators,
    perform the persistence diagram calculation, and update the database.

    Args:
        session: SQLModel Session for database operations

    Returns:
        Number of persistence diagrams that were processed
    """
    try:
        logger.debug("Finding persistence diagram objects without generators...")

        # Query for persistence diagrams with empty generators
        incomplete_pds = incomplete_persistence_diagrams(session)
        total_to_process = len(incomplete_pds)
        logger.debug(f"Found {total_to_process} persistence diagrams to compute")

        if total_to_process == 0:
            return 0

        processed_count = 0
        for i, pd in enumerate(incomplete_pds):
            try:
                logger.debug(
                    f"Computing persistence diagram {i + 1}/{total_to_process} (ID: {pd.id}) for run {pd.run_id}"
                )
                perform_persistence_diagram(pd, session)
                processed_count += 1
                logger.debug(
                    f"Completed persistence diagram {i + 1}/{total_to_process} (ID: {pd.id})"
                )
            except Exception as e:
                logger.error(
                    f"Error computing persistence diagram {i + 1}/{total_to_process} (ID: {pd.id}) for run {pd.run_id} with embedding model {pd.embedding_model}: {e}"
                )

        logger.debug(
            f"Completed processing {processed_count}/{total_to_process} missing persistence diagrams"
        )
        return processed_count

    except Exception as e:
        logger.error(f"Error in compute_missing_persistence_diagrams: {e}")
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
    combinations = list(
        itertools.product(
            config.networks,
            config.seeds,
            config.prompts,
        )
    )

    total_combinations = len(combinations)
    logger.debug(
        f"Starting experiment with {total_combinations} total run configurations"
    )

    # Process each combination
    successful_runs = 0
    for i, (network, seed, prompt) in enumerate(combinations):
        try:
            # Create the run
            logger.debug(
                f"Creating run {i + 1}/{total_combinations}: {network} with seed {seed}"
            )
            run = create_run(
                network=network,
                initial_prompt=prompt,
                seed=seed,
                session=session,
                run_length=config.run_length,
            )

            # Execute the run
            logger.debug(f"Executing run {i + 1}/{total_combinations} (ID: {run.id})")
            run = perform_run(run, session)
            logger.debug(
                f"Model invocations complete for run {i + 1}/{total_combinations} (ID: {run.id})"
            )

            # create "empty" embedding objects - will fill in the vectors later
            for embedding_model in config.embedding_models:
                logger.debug(
                    f"Creating empty embeddings for model {embedding_model} in run {run.id}"
                )
                for j, invocation in enumerate(run.invocations):
                    embedding = create_embedding(embedding_model, invocation, session)
                    logger.debug(
                        f"Created empty embedding {embedding.id} for invocation {invocation.id} ({j + 1}/{len(run.invocations)})"
                    )

                # Create empty persistence diagram for this run and embedding model
                pd = create_persistence_diagram(run.id, embedding_model, session)
                logger.debug(
                    f"Created empty persistence diagram {pd.id} for run {run.id} with model {embedding_model}"
                )

            successful_runs += 1
            logger.info(f"Completed run {i + 1}/{total_combinations} (ID: {run.id})")
            # Unload all AI models from memory after each run to conserve GPU resources
            unload_all_models()

        except Exception as e:
            logger.error(f"Error processing run {i + 1}/{total_combinations}: {e}")
            unload_all_models()
            # Continue with next run even if this one fails

    # Compute embeddings for all runs
    logger.debug(
        f"Starting computation of embeddings for all {successful_runs} successful runs"
    )
    processed_embeddings = compute_missing_embeds(session)
    logger.info(
        f"Completed computation of {processed_embeddings} embeddings across all runs"
    )

    # After all embeddings are computed, calculate persistence diagrams
    logger.debug("Starting computation of persistence diagrams for all runs")
    processed_pds = compute_missing_persistence_diagrams(session)
    logger.info(
        f"Completed computation of {processed_pds} persistence diagrams across all runs"
    )

    logger.info(f"Experiment completed with {successful_runs} successful runs")
