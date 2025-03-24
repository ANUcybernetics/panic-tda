import hashlib
import io
import itertools
import logging
from datetime import datetime
from uuid import UUID

import numpy as np
import ray
from PIL import Image

from trajectory_tracer.db import get_session_from_connection_string
from trajectory_tracer.embeddings import embed
from trajectory_tracer.genai_models import get_output_type, invoke, unload_all_models
from trajectory_tracer.schemas import (
    Embedding,
    ExperimentConfig,
    Invocation,
    PersistenceDiagram,
    Run,
)
from trajectory_tracer.tda import giotto_phd

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Initialize Ray (will be a no-op if already initialized)
# ray.init(ignore_reinit_error=True)


def get_output_hash(output):
    """
    Convert model output to a hashable representation.

    Args:
        output: The model output (string, image, or other)

    Returns:
        A hashable representation of the output
    """

    if isinstance(output, str):
        return hashlib.sha256(output.encode()).hexdigest()
    elif isinstance(output, Image.Image):
        # Convert image to a bytes representation for hashing
        buffer = io.BytesIO()
        output.save(buffer, format="JPEG", quality=30)
        return hashlib.sha256(buffer.getvalue()).hexdigest()
    else:
        # Convert other types to string first
        return hashlib.sha256(str(output).encode()).hexdigest()


@ray.remote(num_returns="dynamic")
def run_generator(run_id: str, db_str: str):
    """
    Generate invocations for a run in sequence.

    Args:
        run_id: UUID string of the run
        db_str: Database connection string

    Yields:
        Invocation IDs as strings
    """
    with get_session_from_connection_string(db_str) as session:
        # Load the run
        run_uuid = UUID(run_id)
        run = session.get(Run, run_uuid)
        if not run:
            raise ValueError(f"Run {run_id} not found")

        logger.debug(f"Starting run generator for run {run_id} with seed {run.seed}")

        # Initialize with the first prompt
        current_input = run.initial_prompt
        previous_invocation_uuid = None
        network_length = len(run.network)

        # Track outputs to detect loops (only if seed is not -1)
        seen_outputs = set()
        track_duplicates = run.seed != -1

        # Process each invocation in the run
        for sequence_number in range(run.max_length):
            # Get the next model in the network (cycling if necessary)
            model_index = sequence_number % network_length
            model_name = run.network[model_index]

            # Create invocation
            invocation = Invocation(
                model=model_name,
                type=get_output_type(model_name),
                run_id=run_uuid,
                sequence_number=sequence_number,
                input_invocation_id=previous_invocation_uuid,
                seed=run.seed,
            )

            # Save to database
            session.add(invocation)
            session.commit()
            session.refresh(invocation)

            invocation_id = str(invocation.id)
            logger.debug(f"Created invocation with ID: {invocation_id}")

            # Execute the invocation
            logger.debug(
                f"Invoking model {invocation.model} with {type(current_input).__name__} input"
            )
            invocation.started_at = datetime.now()
            result = invoke(invocation.model, current_input, invocation.seed)
            invocation.completed_at = datetime.now()
            invocation.output = result

            # Save changes to database
            session.add(invocation)
            session.commit()

            # Check for duplicate outputs if tracking is enabled
            if track_duplicates:
                output_hash = get_output_hash(invocation.output)

                if output_hash in seen_outputs:
                    logger.info(
                        f"Detected duplicate output in run {run_id} at sequence {sequence_number}. Stopping run."
                    )
                    # Yield the ID of the final invocation
                    yield invocation_id
                    break
                seen_outputs.add(output_hash)

            # Yield the invocation ID
            yield invocation_id

            # Set up for next invocation
            current_input = invocation.output
            previous_invocation_uuid = UUID(invocation_id)

            logger.debug(
                f"Completed invocation {sequence_number}/{run.max_length}: {model_name}"
            )

            logger.debug(f"Run generator for {run_id} completed")


@ray.remote
def compute_embedding(invocation_id: str, embedding_model: str, db_str: str) -> str:
    """
    Compute and store embedding for an invocation.

    Args:
        invocation_id: UUID string of the invocation
        embedding_model: Name of the embedding model to use
        db_str: Database connection string

    Returns:
        Embedding ID as string
    """
    with get_session_from_connection_string(db_str) as session:
        # Load the invocation
        invocation_uuid = UUID(invocation_id)
        invocation = session.get(Invocation, invocation_uuid)
        if not invocation:
            raise ValueError(f"Invocation {invocation_id} not found")

        # Create the embedding object
        embedding = Embedding(
            invocation_id=invocation_uuid, embedding_model=embedding_model, vector=None
        )

        # Save to database
        session.add(embedding)
        session.commit()
        session.refresh(embedding)

        embedding_id = str(embedding.id)
        logger.debug(
            f"Created empty embedding {embedding_id} for invocation {invocation_id}"
        )

        # Set the start timestamp
        embedding.started_at = datetime.now()

        # Get the content to embed from the invocation output
        content = invocation.output
        # Use the embed function from embeddings to calculate the vector
        embedding.vector = embed(embedding.embedding_model, content)

        # Set the completion timestamp
        embedding.completed_at = datetime.now()

        # Save to database
        session.add(embedding)
        session.commit()

        logger.debug(f"Successfully computed vector for embedding {embedding_id}")
        return embedding_id


@ray.remote
def compute_persistence_diagram(run_id: str, embedding_model: str, db_str: str) -> str:
    """
    Compute and store persistence diagram for a run.

    Args:
        run_id: UUID string of the run
        embedding_model: Name of the embedding model to use
        db_str: Database connection string

    Returns:
        PersistenceDiagram ID as string
    """
    with get_session_from_connection_string(db_str) as session:
        # Create persistence diagram object with empty generators
        run_uuid = UUID(run_id)
        pd = PersistenceDiagram(
            run_id=run_uuid, embedding_model=embedding_model, generators=None
        )

        # Save to database
        session.add(pd)
        session.commit()
        session.refresh(pd)

        pd_id = str(pd.id)
        logger.debug(f"Created empty persistence diagram {pd_id} for run {run_id}")

        # Load the run and its embeddings
        run = session.get(Run, run_uuid)
        if not run:
            raise ValueError(f"Run {run_id} not found")

        # Get embeddings for the specific embedding model
        embeddings = run.embeddings_by_model(embedding_model)

        # Check if there are enough embeddings to compute a persistence diagram
        if len(embeddings) < 2:
            raise ValueError(
                f"Not enough embeddings ({len(embeddings)}) to compute a persistence diagram. Need at least 2 points."
            )

        # Check that all embeddings have vectors
        missing_vectors = [emb.id for emb in embeddings if emb.vector is None]
        if missing_vectors:
            raise ValueError(
                f"Run {run_id} has embeddings without vectors: {missing_vectors}"
            )

        # Sort embeddings by sequence number
        sorted_embeddings = sorted(
            embeddings, key=lambda e: e.invocation.sequence_number
        )

        # Create point cloud from embedding vectors
        point_cloud = np.array([emb.vector for emb in sorted_embeddings])

        # Set start timestamp
        pd.started_at = datetime.now()

        # Compute persistence diagram
        pd.generators = giotto_phd(point_cloud)

        # Set completion timestamp
        pd.completed_at = datetime.now()

        # Save to database
        session.add(pd)
        session.commit()

        logger.debug(f"Successfully computed persistence diagram for run {run_id}")
        return pd_id


def process_run_generators(run_ids, db_str):
    """
    Process multiple run generators in parallel, respecting dependencies within each run.

    Args:
        run_ids: List of run UUIDs as strings
        db_str: Database connection string

    Returns:
        List of all invocation IDs from all generators
    """
    # Create refs for all run generators
    generator_refs = [run_generator.remote(run_id, db_str) for run_id in run_ids]

    all_invocation_ids = []

    # Get all results from all generators
    for gen_ref in generator_refs:
        try:
            # Get the iterator object from the generator reference
            iter_ref = ray.get(gen_ref)

            # Convert to list to get all elements
            invocation_ids = list(ray.get(item) for item in iter_ref)
            all_invocation_ids.extend(invocation_ids)
        except Exception as e:
            logger.error(f"Error processing generator: {e}")

    return all_invocation_ids


def perform_experiment(config: ExperimentConfig, db_str: str) -> None:
    """
    Create and execute runs for all combinations defined in the ExperimentConfig using Ray.

    Args:
        config: The experiment configuration
        db_str: Database connection string
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

    # Create runs for all combinations
    run_ids = []
    with get_session_from_connection_string(db_str) as session:
        for i, (network, seed, prompt) in enumerate(combinations):
            try:
                # Create the run
                logger.debug(
                    f"Creating run {i + 1}/{total_combinations}: {network} with seed {seed}"
                )
                run = Run(
                    network=network,
                    initial_prompt=prompt,
                    seed=seed,
                    max_length=config.max_length,
                )

                # Save to database
                session.add(run)
                session.commit()
                session.refresh(run)

                run_ids.append(str(run.id))
                logger.debug(f"Created run with ID: {run.id}")

            except Exception as e:
                logger.error(f"Error creating run {i + 1}/{total_combinations}: {e}")

    # Process all run generators in parallel while respecting dependencies
    invocation_ids = process_run_generators(run_ids, db_str)

    # Unload all genai_models to conserve memory for the embedding models
    unload_all_models()

    logger.info(
        f"Generated {len(invocation_ids)} invocations across {len(run_ids)} runs"
    )

    # Compute embeddings for all invocations (for each embedding model)
    embedding_tasks = []
    for invocation_id in invocation_ids:
        for embedding_model in config.embedding_models:
            embedding_tasks.append(
                compute_embedding.remote(invocation_id, embedding_model, db_str)
            )

    # Wait for all embedding computations to complete
    embedding_ids = ray.get(embedding_tasks)
    logger.info(f"Computed {len(embedding_ids)} embeddings")

    # Compute persistence diagrams for all runs (for each embedding model)
    # Use limited parallelism for persistence diagram computation as it uses all CPU cores
    pd_tasks = []
    for run_id in run_ids:
        for embedding_model in config.embedding_models:
            # Option to limit concurrency by processing in batches
            pd_tasks.append(
                compute_persistence_diagram.options(num_cpus=0.5).remote(
                    run_id, embedding_model, db_str
                )
            )

    # Wait for all persistence diagram computations to complete
    pd_ids = ray.get(pd_tasks)
    logger.info(f"Computed {len(pd_ids)} persistence diagrams")

    logger.info(f"Experiment completed with {len(run_ids)} successful runs")


def perform_experiment_with_ray(config: ExperimentConfig, db_str: str) -> None:
    """
    Entry point for running an experiment with Ray parallelization.

    Args:
        config: The experiment configuration
        db_str: Database connection string
    """
    try:
        logger.info(f"Starting experiment with Ray, using {db_str}")
        perform_experiment(config, db_str)
    except Exception as e:
        logger.error(f"Error in Ray experiment: {e}")
        raise
    finally:
        # Make sure to clean up Ray resources
        unload_all_models()
