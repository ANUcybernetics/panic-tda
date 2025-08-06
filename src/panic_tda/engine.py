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
from sqlmodel import func, select

from panic_tda.db import get_session_from_connection_string
from panic_tda.embeddings import get_actor_class as get_embedding_actor_class
from panic_tda.genai_models import get_actor_class as get_genai_actor_class
from panic_tda.genai_models import get_output_type
from panic_tda.schemas import (
    Embedding,
    ExperimentConfig,
    Invocation,
    InvocationType,
    PersistenceDiagram,
    Run,
)
from panic_tda.tda import giotto_phd

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
    """Generate invocations for a run in sequence."""
    with get_session_from_connection_string(db_str) as session:
        # Load the run
        run_uuid = UUID(run_id)
        run = session.get(Run, run_uuid)

        if not run:
            raise ValueError(f"Run {run_id} not found")

        network_length = len(run.network)
        logger.debug(f"Starting run generator for run {run_id} with seed {run.seed}")

        # Find the existing invocation with max sequence number (if any)
        current_invocation = session.exec(
            select(Invocation)
            .where(Invocation.run_id == run_uuid)
            .order_by(Invocation.sequence_number.desc())
            .limit(1)
        ).first()

        # Track outputs to detect loops (only if seed is not -1)
        track_duplicates = run.seed != -1
        seen_outputs = set()

        if current_invocation:
            # Found an existing invocation, set up state based on it
            sequence_number = current_invocation.sequence_number

            # Get previous invocation for input
            if sequence_number > 0:
                previous_invocation = session.exec(
                    select(Invocation).where(
                        Invocation.run_id == run_uuid,
                        Invocation.sequence_number == sequence_number - 1,
                    )
                ).first()
                current_input = (
                    previous_invocation.output
                    if previous_invocation
                    else run.initial_prompt
                )
                previous_invocation_uuid = (
                    previous_invocation.id if previous_invocation else None
                )
            else:
                current_input = run.initial_prompt
                previous_invocation_uuid = None

            # Collect hashes of all previous outputs to detect duplicates
            if track_duplicates:
                previous_invocations = session.exec(
                    select(Invocation).where(
                        Invocation.run_id == run_uuid,
                        Invocation.sequence_number < sequence_number,
                    )
                ).all()

                for inv in previous_invocations:
                    if inv.output is not None:
                        seen_outputs.add(get_output_hash(inv.output))
        else:
            # No existing invocations, start fresh
            sequence_number = 0
            current_input = run.initial_prompt
            previous_invocation_uuid = None

            # Create the first invocation
            model_index = 0
            model_name = run.network[model_index]

            current_invocation = Invocation(
                model=model_name,
                type=get_output_type(model_name),
                run_id=run_uuid,
                sequence_number=sequence_number,
                input_invocation_id=previous_invocation_uuid,
                seed=run.seed,
            )

            session.add(current_invocation)
            session.commit()
            session.refresh(current_invocation)

        # Process each invocation in the run
        while sequence_number < run.max_length:
            # Safety check for null input
            if current_input is None:
                logger.warning(
                    f"Null input detected at sequence {sequence_number}. Using default prompt."
                )
                current_input = "Continue from the previous point."

            # Get the model for this sequence number
            model_index = sequence_number % network_length
            model_name = run.network[model_index]

            invocation = current_invocation
            invocation_id = str(invocation.id)
            logger.debug(f"Working with invocation ID: {invocation_id}")

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
            previous_invocation_uuid = invocation.id
            sequence_number += 1

            if sequence_number < run.max_length:
                # Create next invocation
                next_model_index = sequence_number % network_length
                next_model_name = run.network[next_model_index]

                current_invocation = Invocation(
                    model=next_model_name,
                    type=get_output_type(next_model_name),
                    run_id=run_uuid,
                    sequence_number=sequence_number,
                    input_invocation_id=previous_invocation_uuid,
                    seed=run.seed,
                )

                session.add(current_invocation)
                session.commit()
                session.refresh(current_invocation)

            logger.debug(
                f"Completed invocation {sequence_number - 1}/{run.max_length}: {model_name}"
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
        run_uuid = UUID(run_id)
        run = session.get(Run, run_uuid)
        if not run:
            raise ValueError(f"Run {run_id} not found")

        # Check if a persistence diagram already exists for this run and embedding model
        existing_pd = None
        for pd in run.persistence_diagrams:
            if pd.embedding_model == embedding_model:
                existing_pd = pd
                break

        if existing_pd and existing_pd.diagram_data is not None:
            logger.debug(
                f"Persistence diagram already exists for run {run_id} with model {embedding_model}"
            )
            return str(existing_pd.id)
        elif existing_pd:
            # If diagram_data is None or invalid, delete the existing persistence diagram
            logger.debug(
                f"Found persistence diagram with invalid data for run {run_id}. Deleting and recreating."
            )
            session.delete(existing_pd)
            session.commit()

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
        try:
            pd.diagram_data = giotto_phd(point_cloud)
        except MemoryError as e:
            logger.error(
                f"Memory allocation (bad_alloc) error during persistence diagram computation for run {run_id}: {e}"
            )
            pd.diagram_data = None
        except Exception as e:
            logger.error(f"Error computing persistence diagram for run {run_id}: {e}")
            pd.diagram_data = None

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
                if invocation and invocation.type == InvocationType.TEXT:
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
    task_count = len(run_ids) * len(embedding_models)
    logger.info(f"Computing {task_count} persistence diagrams for {len(run_ids)} runs")

    # Create compute tasks for all run/model combinations
    for run_id in run_ids:
        for embedding_model in embedding_models:
            pd_tasks.append(
                compute_persistence_diagram.remote(run_id, embedding_model, db_str)
            )

    # Get results from all tasks
    pd_ids = ray.get(pd_tasks)
    logger.info(f"Completed {len(pd_ids)} persistence diagrams")

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
                f"To check on the status, run `panic-tda experiment-status {experiment_id}`"
            )

            if not config.runs:
                # Initialize runs and group them by network (combinations are calculated inside init_runs)
                run_groups = init_runs(experiment_id, db_str)
            else:
                # Group runs by network for restart
                network_to_runs = {}
                for run in config.runs:
                    network_key = tuple(run.network)
                    if network_key not in network_to_runs:
                        network_to_runs[network_key] = []
                    network_to_runs[network_key].append(str(run.id))
                run_groups = list(network_to_runs.values())

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

            # Get all invocation IDs for all runs
            # pull from the DB in case it was a "resumed" run
            all_invocation_ids = []
            for run_id in all_run_ids:
                invocations = session.exec(
                    select(Invocation.id).where(Invocation.run_id == UUID(run_id))
                ).all()
                all_invocation_ids.extend([str(inv_id) for inv_id in invocations])

        logger.info(f"Found {len(all_invocation_ids)} total invocations to process")

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
            session.refresh(experiment)
            logger.info(f"Experiment {experiment_id} marked as completed")
            return experiment

    except Exception as e:
        logger.error(f"Error performing experiment: {e}")
        raise


def check_run_invocations(experiment, session):
    """Check that all runs have the expected invocations."""
    issues = []

    # Get all runs for this experiment
    runs = session.exec(select(Run).where(Run.experiment_id == experiment.id)).all()

    for run in runs:
        # Check invocation count
        invocation_count = session.exec(
            select(func.count())
            .select_from(Invocation)
            .where(Invocation.run_id == run.id)
        ).one()

        # Check for sequence_number 0
        has_first = (
            session.exec(
                select(Invocation).where(
                    Invocation.run_id == run.id, Invocation.sequence_number == 0
                )
            ).first()
            is not None
        )

        # Check for sequence_number (max_length - 1)
        has_last = (
            session.exec(
                select(Invocation).where(
                    Invocation.run_id == run.id,
                    Invocation.sequence_number == run.max_length - 1,
                )
            ).first()
            is not None
        )

        if not has_first or not has_last or invocation_count != run.max_length:
            issues.append({
                "run_id": run.id,
                "missing_first": not has_first,
                "missing_last": not has_last,
                "expected_count": run.max_length,
                "actual_count": invocation_count,
            })

    if issues:
        logger.info(f"Found {len(issues)} runs with invocation issues")
        for issue in issues:
            logger.info(
                f"Run {issue['run_id']}: first={not issue['missing_first']}, "
                f"last={not issue['missing_last']}, "
                f"count={issue['actual_count']}/{issue['expected_count']}"
            )
    else:
        logger.info("All runs have correct invocation sequences")

    return issues


def check_embeddings(experiment, session):
    """Check that all invocations have proper embeddings."""
    issues = []

    # Get all text invocations for all runs in this experiment
    invocations = session.exec(
        select(Invocation)
        .join(Run, Invocation.run_id == Run.id)
        .where(
            Run.experiment_id == experiment.id, Invocation.type == InvocationType.TEXT
        )
    ).all()

    for invocation in invocations:
        for embedding_model in experiment.embedding_models:
            # Count embeddings for this invocation and model
            embedding_count = session.exec(
                select(func.count())
                .select_from(Embedding)
                .where(
                    Embedding.invocation_id == invocation.id,
                    Embedding.embedding_model == embedding_model,
                )
            ).one()

            # Check for null vectors
            null_vectors = session.exec(
                select(func.count())
                .select_from(Embedding)
                .where(
                    Embedding.invocation_id == invocation.id,
                    Embedding.embedding_model == embedding_model,
                    Embedding.vector.is_(None),
                )
            ).one()

            if embedding_count != 1 or null_vectors > 0:
                issues.append({
                    "invocation_id": invocation.id,
                    "embedding_model": embedding_model,
                    "embedding_count": embedding_count,
                    "has_null_vector": null_vectors > 0,
                })

    if issues:
        logger.info(f"Found {len(issues)} invocations with embedding issues")
        for issue in issues:
            logger.info(
                f"Invocation {issue['invocation_id']}: "
                f"model={issue['embedding_model']}, "
                f"count={issue['embedding_count']}, "
                f"null_vector={issue['has_null_vector']}"
            )
    else:
        logger.info("All invocations have correct embeddings")

    return issues


def check_persistence_diagrams(experiment, session):
    """Check that all runs have proper persistence diagrams."""
    issues = []

    # Tracking dictionaries for different types of issues by embedding model
    missing_by_model = {}
    duplicate_by_model = {}
    invalid_models = set()  # Track models that aren't in experiment.embedding_models

    # Initialize counters for each embedding model
    for model in experiment.embedding_models:
        missing_by_model[model] = 0
        duplicate_by_model[model] = 0

    # Get all runs for this experiment
    runs = session.exec(select(Run).where(Run.experiment_id == experiment.id)).all()
    run_ids = [run.id for run in runs]

    # First check for PDs with invalid embedding models
    if run_ids:
        # Get all PDs for this experiment's runs
        all_pds = session.exec(
            select(PersistenceDiagram).where(PersistenceDiagram.run_id.in_(run_ids))
        ).all()

        # Check for invalid embedding models
        for pd in all_pds:
            if pd.embedding_model not in experiment.embedding_models:
                invalid_models.add(pd.embedding_model)
                issues.append({
                    "run_id": pd.run_id,
                    "embedding_model": pd.embedding_model,
                    "issue_type": "invalid_model",
                    "pd_count": 1,
                    "has_null_data": pd.diagram_data is None,
                })

    # Now check for missing or duplicate PDs for valid models
    for run in runs:
        for embedding_model in experiment.embedding_models:
            # Count persistence diagrams for this run and model
            pd_count = session.exec(
                select(func.count())
                .select_from(PersistenceDiagram)
                .where(
                    PersistenceDiagram.run_id == run.id,
                    PersistenceDiagram.embedding_model == embedding_model,
                )
            ).one()

            # Check for null diagram data
            null_data = session.exec(
                select(func.count())
                .select_from(PersistenceDiagram)
                .where(
                    PersistenceDiagram.run_id == run.id,
                    PersistenceDiagram.embedding_model == embedding_model,
                    PersistenceDiagram.diagram_data.is_(None),
                )
            ).one()

            # Track issues by type
            if pd_count == 0 or null_data == pd_count:
                missing_by_model[embedding_model] += 1
                issues.append({
                    "run_id": run.id,
                    "embedding_model": embedding_model,
                    "issue_type": "missing",
                    "pd_count": pd_count,
                    "has_null_data": null_data > 0,
                })
            elif pd_count > 1:
                duplicate_by_model[embedding_model] += 1
                issues.append({
                    "run_id": run.id,
                    "embedding_model": embedding_model,
                    "issue_type": "duplicate",
                    "pd_count": pd_count,
                    "has_null_data": null_data > 0,
                })

    if issues:
        logger.info(f"Found {len(issues)} runs with persistence diagram issues")

        # Print summary of missing PDs by model
        missing_models = [
            model for model, count in missing_by_model.items() if count > 0
        ]
        if missing_models:
            logger.info("Missing persistence diagrams by embedding model:")
            for model in missing_models:
                logger.info(f"  - {model}: {missing_by_model[model]} missing PDs")

        # Print summary of duplicate PDs by model
        duplicate_models = [
            model for model, count in duplicate_by_model.items() if count > 0
        ]
        if duplicate_models:
            logger.info("Duplicate persistence diagrams by embedding model:")
            for model in duplicate_models:
                logger.info(
                    f"  - {model}: {duplicate_by_model[model]} runs with duplicate PDs"
                )

        # Print summary of invalid models
        if invalid_models:
            logger.info("Invalid embedding models (not in experiment configuration):")
            for model in invalid_models:
                count = sum(
                    1
                    for issue in issues
                    if issue.get("issue_type") == "invalid_model"
                    and issue["embedding_model"] == model
                )
                logger.info(f"  - {model}: {count} diagrams to be removed")

        # Print details for each issue
        for issue in issues:
            issue_type = issue.get("issue_type", "unknown")
            logger.info(
                f"Run {issue['run_id']}: "
                f"model={issue['embedding_model']}, "
                f"type={issue_type}, "
                f"count={issue['pd_count']}, "
                f"null_data={issue['has_null_data']}"
            )
    else:
        logger.info("All runs have correct persistence diagrams")

    return issues


def fix_run_invocations(issues, experiment, db_str):
    """Fix missing or extra invocations in runs."""
    logger.info("Fixing run invocation issues...")

    # Group issues by network to use perform_runs_stage efficiently
    network_to_runs = {}

    with get_session_from_connection_string(db_str) as session:
        for issue in issues:
            run = session.get(Run, issue["run_id"])
            if run:
                network_key = tuple(run.network)
                if network_key not in network_to_runs:
                    network_to_runs[network_key] = []
                network_to_runs[network_key].append(str(run.id))

                # Delete existing invocations for this run
                invocations = session.exec(
                    select(Invocation).where(Invocation.run_id == run.id)
                ).all()
                for invocation in invocations:
                    session.delete(invocation)

        session.commit()

    # For each network, regenerate invocations using perform_runs_stage
    for network, run_ids in network_to_runs.items():
        logger.info(
            f"Regenerating invocations for {len(run_ids)} runs with network {list(network)}"
        )
        perform_runs_stage(run_ids, db_str)

    logger.info("Fixed run invocation issues")


def fix_embeddings(issues, experiment, db_str):
    """Fix missing or invalid embeddings."""
    logger.info("Fixing embedding issues...")

    # Group by embedding model to process efficiently
    model_to_invocations = {}

    # Remove existing bad embeddings
    with get_session_from_connection_string(db_str) as session:
        for issue in issues:
            if issue["embedding_model"] not in model_to_invocations:
                model_to_invocations[issue["embedding_model"]] = []

            # Only add each invocation once per model
            invocation_id = str(issue["invocation_id"])
            if invocation_id not in model_to_invocations[issue["embedding_model"]]:
                model_to_invocations[issue["embedding_model"]].append(invocation_id)

            # Delete existing embeddings for this invocation and model
            embeddings = session.exec(
                select(Embedding).where(
                    Embedding.invocation_id == issue["invocation_id"],
                    Embedding.embedding_model == issue["embedding_model"],
                )
            ).all()
            for embedding in embeddings:
                session.delete(embedding)

        session.commit()

    # Process each model's invocations
    for model, invocation_ids in model_to_invocations.items():
        logger.info(
            f"Generating embeddings for {len(invocation_ids)} invocations with model {model}"
        )
        perform_embeddings_stage(invocation_ids, [model], db_str)

    logger.info("Fixed embedding issues")


def fix_persistence_diagrams(issues, experiment, db_str):
    """Fix missing or invalid persistence diagrams."""
    logger.info("Fixing persistence diagram issues...")

    # Group by embedding model to process efficiently
    model_to_missing_runs = {}  # For runs with no PDs (count == 0)
    model_to_null_runs = {}  # For runs with PDs but null data
    missing_count = 0
    null_data_count = 0
    duplicate_count = 0
    invalid_model_count = 0

    # Process persistence diagrams, keeping one valid diagram per run/model pair
    with get_session_from_connection_string(db_str) as session:
        for issue in issues:
            # Handle invalid embedding models (not in experiment configuration)
            if issue.get("issue_type") == "invalid_model":
                invalid_model_count += 1
                run_uuid = issue["run_id"]
                embedding_model = issue["embedding_model"]

                # Delete all persistence diagrams for this run/invalid model combination
                diagrams = session.exec(
                    select(PersistenceDiagram).where(
                        PersistenceDiagram.run_id == run_uuid,
                        PersistenceDiagram.embedding_model == embedding_model,
                    )
                ).all()

                for diagram in diagrams:
                    session.delete(diagram)
                continue

            # Skip models not in experiment configuration
            if issue["embedding_model"] not in experiment.embedding_models:
                continue

            # Initialize dictionaries for this model if needed
            embedding_model = issue["embedding_model"]
            if embedding_model not in model_to_missing_runs:
                model_to_missing_runs[embedding_model] = []
            if embedding_model not in model_to_null_runs:
                model_to_null_runs[embedding_model] = []

            # Only add each run once per model
            run_id = str(issue["run_id"])
            run_uuid = issue["run_id"]

            # Separate completely missing PDs from PDs with null data
            if (
                run_id not in model_to_missing_runs[embedding_model]
                and run_id not in model_to_null_runs[embedding_model]
            ):
                if issue["pd_count"] == 0:
                    # Completely missing PD
                    missing_count += 1
                    model_to_missing_runs[embedding_model].append(run_id)
                elif issue["has_null_data"] and issue["pd_count"] == 1:
                    # Has PD but with null data
                    null_data_count += 1
                    model_to_null_runs[embedding_model].append(run_id)
                elif issue["pd_count"] > 1:
                    duplicate_count += 1

                    # Find all persistence diagrams for this run/model
                    diagrams = session.exec(
                        select(PersistenceDiagram).where(
                            PersistenceDiagram.run_id == run_uuid,
                            PersistenceDiagram.embedding_model == embedding_model,
                        )
                    ).all()

                    # Keep the first valid diagram (with non-null data)
                    valid_diagram = None
                    for diagram in diagrams:
                        if diagram.diagram_data is not None:
                            valid_diagram = diagram
                            break

                    # Delete all diagrams except the valid one
                    for diagram in diagrams:
                        if valid_diagram is None or diagram.id != valid_diagram.id:
                            session.delete(diagram)

                    # Only regenerate if we didn't find a valid diagram
                    if valid_diagram is None:
                        model_to_null_runs[embedding_model].append(run_id)
                        null_data_count += 1

        session.commit()

    # First process each model's runs that are completely missing PDs
    for model, run_ids in model_to_missing_runs.items():
        if run_ids:
            logger.info(
                f"Generating completely missing persistence diagrams for {len(run_ids)} runs with model {model}"
            )
            perform_pd_stage(run_ids, [model], db_str)

    # Then process each model's runs that have PDs with null data
    for model, run_ids in model_to_null_runs.items():
        if run_ids:
            logger.info(
                f"Regenerating persistence diagrams with null data for {len(run_ids)} runs with model {model}"
            )
            perform_pd_stage(run_ids, [model], db_str)

    logger.info(
        f"Fixed persistence diagram issues ({missing_count} missing, {null_data_count} null data, "
        f"{duplicate_count} duplicate, {invalid_model_count} invalid model)"
    )


def experiment_doctor(experiment_id: str, db_str: str, fix: bool):
    """
    Diagnose and fix issues with an experiment's data.

    Performs several checks:
    1. Ensures all runs have the correct invocation sequence
    2. Ensures all invocations have proper embeddings
    3. Ensures all runs have proper persistence diagrams

    Args:
        experiment_id: UUID string of the experiment to check
        db_str: Database connection string
        fix: If True, automatically fix any issues found; if False, only report issues
    """
    with get_session_from_connection_string(db_str) as session:
        # Load the experiment
        experiment_uuid = UUID(experiment_id)
        experiment = session.get(ExperimentConfig, experiment_uuid)
        if not experiment:
            logger.error(f"Experiment {experiment_id} not found")
            return

        logger.info(f"Checking experiment {experiment_id}")

        # Check runs and invocations
        run_issues = check_run_invocations(experiment, session)
        if run_issues:
            if fix:
                fix_run_invocations(run_issues, experiment, db_str)
            else:
                logger.info("Use --fix flag to repair missing or extra invocations")

        # Check embeddings
        embedding_issues = check_embeddings(experiment, session)
        if embedding_issues:
            if fix:
                fix_embeddings(embedding_issues, experiment, db_str)
            else:
                logger.info("Use --fix flag to repair missing embeddings")

        # Check persistence diagrams
        pd_issues = check_persistence_diagrams(experiment, session)
        if pd_issues:
            if fix:
                fix_persistence_diagrams(pd_issues, experiment, db_str)
            else:
                logger.info("Use --fix flag to repair missing persistence diagrams")
