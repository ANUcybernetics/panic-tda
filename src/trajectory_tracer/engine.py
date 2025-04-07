import hashlib
import io
import itertools
import logging
from datetime import datetime
from uuid import UUID

import numpy as np
import ray
import torch
from PIL import Image
from ray.util import ActorPool

from trajectory_tracer.db import get_session_from_connection_string
from trajectory_tracer.embeddings import get_actor_class as get_embedding_actor_class
from trajectory_tracer.genai_models import get_actor_class as get_genai_actor_class
from trajectory_tracer.genai_models import get_output_type
from trajectory_tracer.schemas import (
    Embedding,
    ExperimentConfig,
    Invocation,
    InvocationType,
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


@ray.remote(num_cpus=1, num_gpus=0, num_returns="dynamic")
def run_generator(run_id: str, db_str: str, model_actors: dict):
    """
    Generate invocations for a run in sequence.

    Args:
        run_id: UUID string of the run
        db_str: Database connection string
        model_actors: Dictionary mapping model names to Ray actor handles

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

            # Execute the invocation using the appropriate actor from model_actors
            logger.debug(
                f"Invoking model {invocation.model} with {type(current_input).__name__} input"
            )
            invocation.started_at = datetime.now()

            # Get the actor for this model
            actor = model_actors.get(model_name)
            if not actor:
                raise ValueError(f"No actor found for model {model_name}")

            # Call the invoke method on the actor
            result = ray.get(actor.invoke.remote(current_input, invocation.seed))

            invocation.completed_at = datetime.now()
            invocation.output = result

            # Save changes to database
            session.add(invocation)
            session.commit()

            # Check for duplicate outputs if tracking is enabled
            if track_duplicates:
                output_hash = get_output_hash(invocation.output)

                if output_hash in seen_outputs:
                    logger.debug(
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
def compute_embeddings(actor, invocation_ids, embedding_model, db_str):
    """Process a batch of embedding calculation tasks"""
    with get_session_from_connection_string(db_str) as session:
        # Load all invocations
        invocations = {}
        missing_invocations = []
        existing_embedding_ids = []

        # First load all invocations from database
        for invocation_id in invocation_ids:
            invocation_uuid = UUID(invocation_id)
            invocation = session.get(Invocation, invocation_uuid)
            if invocation:
                invocations[invocation_id] = invocation
            else:
                missing_invocations.append(invocation_id)

        # Raise an error if any invocations couldn't be found
        if missing_invocations:
            raise ValueError(f"Invocations not found: {missing_invocations}")

        # Create embedding objects for invocations that don't already have them
        embeddings = []
        embedding_mapping = {}  # Maps invocation_id to corresponding embedding
        contents = []  # Contents to be embedded
        invocations_to_process = {}  # Only invocations that need new embeddings

        for invocation_id, invocation in invocations.items():
            # Check if this invocation already has an embedding for this model
            existing_embedding = invocation.embedding(embedding_model)
            if existing_embedding:
                # If embedding already exists, add its ID to the results and skip processing
                existing_embedding_ids.append(str(existing_embedding.id))
            else:
                # Only process invocations that don't have embeddings yet
                embedding = Embedding(
                    invocation_id=UUID(invocation_id),
                    embedding_model=embedding_model,
                    vector=None,
                    started_at=datetime.now(),
                )
                embeddings.append(embedding)
                embedding_mapping[invocation_id] = embedding
                contents.append(invocation.output)
                invocations_to_process[invocation_id] = invocation

        # If all invocations already have embeddings, return their IDs
        if not embeddings:
            logger.debug(
                f"All {len(existing_embedding_ids)} embeddings already exist, skipping computation"
            )
            return existing_embedding_ids

        # Save empty embeddings
        session.add_all(embeddings)
        session.commit()
        for embedding in embeddings:
            session.refresh(embedding)

        # Compute embeddings in batch if there's content to process
        if contents:
            vectors = ray.get(actor.embed.remote(contents))

            # Match vectors back to embeddings
            for i, invocation_id in enumerate(invocations_to_process.keys()):
                embedding = embedding_mapping[invocation_id]
                embedding.vector = vectors[i]
                embedding.completed_at = datetime.now()

            # Save updated embeddings
            session.add_all(embeddings)
            session.commit()

        # Return list of embedding IDs (both new and existing)
        new_embedding_ids = [str(embedding.id) for embedding in embeddings]
        all_embedding_ids = existing_embedding_ids + new_embedding_ids
        logger.debug(
            f"Successfully computed {len(new_embedding_ids)} vectors in batch (plus {len(existing_embedding_ids)} existing)"
        )
        return all_embedding_ids


@ray.remote(num_cpus=8)
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
        # Check if a persistence diagram already exists for this run and embedding model
        run_uuid = UUID(run_id)
        existing_pd = (
            session.query(PersistenceDiagram)
            .filter(
                PersistenceDiagram.run_id == run_uuid,
                PersistenceDiagram.embedding_model == embedding_model,
            )
            .first()
        )

        if existing_pd:
            logger.debug(
                f"Persistence diagram already exists for run {run_id} with model {embedding_model}"
            )
            return str(existing_pd.id)

        # Create persistence diagram object with empty diagram data
        pd = PersistenceDiagram(
            run_id=run_uuid, embedding_model=embedding_model, diagram_data=None
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
        embeddings = run.embeddings[embedding_model]

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

        # Compute persistence diagram - store the entire result
        pd.diagram_data = giotto_phd(point_cloud)

        # Set completion timestamp
        pd.completed_at = datetime.now()

        # Save to database
        session.add(pd)
        session.commit()

        logger.debug(f"Successfully computed persistence diagram for run {run_id}")
        return pd_id


def init_runs(experiment_id, db_str):
    """
    Initialize runs for an experiment and group them by network.

    Args:
        experiment_id: UUID of the experiment
        db_str: Database connection string

    Returns:
        List of lists of run IDs, where each inner list contains runs with the same network
    """
    # Dictionary to group run_ids by network
    network_to_runs = {}

    with get_session_from_connection_string(db_str) as session:
        # Get the experiment config
        config = session.get(ExperimentConfig, experiment_id)
        if not config:
            raise ValueError(f"Experiment config {experiment_id} not found")

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
            f"Creating {total_combinations} runs for experiment {experiment_id}"
        )

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
                    experiment_id=experiment_id,
                )

                # Save to database
                session.add(run)
                session.commit()
                session.refresh(run)

                # Add run ID to the appropriate network group
                network_key = tuple(network)  # Convert list to tuple for dict key
                if network_key not in network_to_runs:
                    network_to_runs[network_key] = []
                network_to_runs[network_key].append(str(run.id))

                logger.debug(f"Created run with ID: {run.id}")

            except Exception as e:
                logger.error(f"Error creating run {i + 1}/{total_combinations}: {e}")

    # Convert dictionary to list of lists, preserving network grouping
    return list(network_to_runs.values())


def perform_runs_stage(run_ids, db_str):
    """
    Process multiple run generators in parallel, respecting dependencies within each run.
    Creates model actors and dispatches tasks appropriately.

    Args:
        run_ids: List of run UUIDs as strings (all sharing the same network)
        db_str: Database connection string

    Returns:
        List of all invocation IDs from all generators
    """
    if not run_ids:
        return []

    # Create a dictionary to hold actors for each required model
    model_actors = {}
    used_models = set()

    # First collect all models needed for these runs
    # Since all runs share the same network, we only need to check one run
    with get_session_from_connection_string(db_str) as session:
        sample_run = session.get(Run, UUID(run_ids[0]))
        if sample_run:
            for model_name in sample_run.network:
                used_models.add(model_name)

    # Create actors for each unique model
    for model_name in used_models:
        model_actor_class = get_genai_actor_class(model_name)
        model_actors[model_name] = model_actor_class.remote()
        logger.debug(f"Created actor for model {model_name}")

    # Create refs for all run generators
    generator_refs = [
        run_generator.remote(run_id, db_str, model_actors) for run_id in run_ids
    ]

    all_invocation_ids = []

    # Get all results from all generators
    for gen_ref in generator_refs:
        # Get the iterator object from the generator reference
        iter_ref = ray.get(gen_ref)

        # Convert to list to get all elements
        invocation_ids = list(ray.get(item) for item in iter_ref)
        all_invocation_ids.extend(invocation_ids)

    # Clean up model actors
    for model_name, actor in model_actors.items():
        ray.kill(actor)
        logger.debug(f"Terminated actor for model {model_name}")
    # free up GPU resources
    torch.cuda.empty_cache()

    return all_invocation_ids


def perform_embeddings_stage(
    invocation_ids, embedding_models, db_str, num_actors=8, batch_size=32
):
    """
    Process embeddings for multiple invocations using ActorPool for load balancing.
    Note: Only text invocations are supported for embedding.

    Args:
        invocation_ids: List of invocation IDs to process
        embedding_models: List of embedding model names to use
        db_str: Database connection string
        num_actors: Number of actors to create per model (default: 8)
        batch_size: Size of batches to process at once (default: 32)

    Returns:
        List of embedding IDs that were successfully created
    """
    all_embedding_ids = []

    for embedding_model in embedding_models:
        logger.info(f"Processing text embeddings with model {embedding_model}")

        # Get the actor class for this embedding model
        embedding_actor_class = get_embedding_actor_class(embedding_model)

        # Filter to only include text invocations
        with get_session_from_connection_string(db_str) as session:
            text_invocations = []
            for invocation_id in invocation_ids:
                invocation = session.get(Invocation, UUID(invocation_id))
                if invocation.type == InvocationType.TEXT:
                    text_invocations.append(invocation_id)

        if not text_invocations:
            logger.info(f"No text invocations found to embed with {embedding_model}")
            continue

        logger.info(f"Found {len(text_invocations)} text invocations to embed")

        # Limit the number of actors based on the number of batches
        num_batches = (len(text_invocations) + batch_size - 1) // batch_size
        actor_count = min(num_actors, num_batches)

        # Create actor instances
        actors = [embedding_actor_class.remote() for _ in range(actor_count)]

        # Create an ActorPool
        pool = ActorPool(actors)

        logger.info(
            f"Created actor pool with {actor_count} actors for model {embedding_model}"
        )

        # Create batches of text invocation IDs
        text_batches = [
            text_invocations[i : i + batch_size]
            for i in range(0, len(text_invocations), batch_size)
        ]

        logger.info(
            f"Processing {len(text_batches)} batches with batch size {batch_size}"
        )

        # Process embedding batches in parallel using the actor pool
        # The map_unordered function returns an iterator of actual results, not object references
        batch_embedding_ids = list(
            pool.map_unordered(
                lambda actor, batch: compute_embeddings.remote(
                    actor, batch, embedding_model, db_str
                ),
                text_batches,
            )
        )

        # Flatten the list of lists of embedding IDs
        for batch_ids in batch_embedding_ids:
            all_embedding_ids.extend(batch_ids)

        logger.info(
            f"Computed {len(all_embedding_ids)} text embeddings with model {embedding_model}"
        )

        # Clean up actors
        for actor in actors:
            ray.kill(actor)

    # free up GPU resources
    torch.cuda.empty_cache()

    return all_embedding_ids


def perform_pd_stage(run_ids, embedding_models, db_str):
    """
    Compute persistence diagrams for all runs using specified embedding models.

    Args:
        run_ids: List of run UUIDs as strings
        embedding_models: List of embedding model names to use
        db_str: Database connection string

    Returns:
        List of persistence diagram IDs
    """
    pd_tasks = []
    for run_id in run_ids:
        for embedding_model in embedding_models:
            pd_tasks.append(
                compute_persistence_diagram.remote(run_id, embedding_model, db_str)
            )

    # Wait for all persistence diagram computations to complete
    pd_ids = ray.get(pd_tasks)
    logger.info(f"Computed {len(pd_ids)} persistence diagrams")

    return pd_ids


def perform_experiment(experiment_config_id: str, db_str: str) -> None:
    """
    Create and execute runs for all combinations defined in the ExperimentConfig using Ray.

    Args:
        experiment_config_id: UUID string of the experiment configuration
        db_str: Database connection string
    """
    try:
        # Load the experiment config from the database
        with get_session_from_connection_string(db_str) as session:
            experiment_id = UUID(experiment_config_id)
            config = session.get(ExperimentConfig, experiment_id)
            if not config:
                raise ValueError(f"Experiment config {experiment_config_id} not found")

            # Set started_at timestamp for experiment
            config.started_at = datetime.now()
            session.add(config)
            session.commit()
            logger.info(f"Started experiment with ID: {experiment_id}")
            logger.info(
                f"To check on the status, run `trajectory-tracer experiment-status {experiment_id}`"
            )

        # Initialize runs and group them by network (combinations are calculated inside init_runs)
        run_groups = init_runs(experiment_id, db_str)

        # Process each group of runs (sharing the same network) separately
        all_run_ids = []
        all_invocation_ids = []

        for i, run_group in enumerate(run_groups):
            logger.info(
                f"Processing run group {i + 1}/{len(run_groups)} with {len(run_group)} runs"
            )

            # Add to the master list of all run IDs
            all_run_ids.extend(run_group)

            # Process this group of runs and collect invocation IDs
            group_invocation_ids = perform_runs_stage(run_group, db_str)
            all_invocation_ids.extend(group_invocation_ids)

            logger.info(
                f"Completed {len(group_invocation_ids)} invocations for run group {i + 1}"
            )

        logger.info(
            f"Generated {len(all_invocation_ids)} invocations across {len(all_run_ids)} runs"
        )

        # Reload config to get embedding models
        with get_session_from_connection_string(db_str) as session:
            config = session.get(ExperimentConfig, experiment_id)
            embedding_models = config.embedding_models

        # Compute embeddings for all invocations
        embedding_ids = perform_embeddings_stage(
            all_invocation_ids, embedding_models, db_str
        )
        logger.info(f"Computed {len(embedding_ids)} embeddings")

        # Compute persistence diagrams for all runs
        perform_pd_stage(all_run_ids, embedding_models, db_str)
        logger.info(f"Experiment completed with {len(all_run_ids)} successful runs")

        # Set completed_at timestamp for experiment
        with get_session_from_connection_string(db_str) as session:
            experiment = session.get(ExperimentConfig, experiment_id)
            experiment.completed_at = datetime.now()
            session.add(experiment)
            session.commit()
            logger.info(f"Experiment {experiment_id} marked as completed")

    except Exception as e:
        logger.error(f"Error performing experiment: {e}")
        raise
