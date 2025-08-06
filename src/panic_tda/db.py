from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import List
from uuid import UUID

import numpy as np
import sqlalchemy
from humanize.time import naturaldelta
from sqlalchemy.pool import QueuePool
from sqlmodel import Session, SQLModel, create_engine, func, select

from panic_tda.genai_models import estimated_time
from panic_tda.schemas import (
    Embedding,
    ExperimentConfig,
    Invocation,
    PersistenceDiagram,
    Run,
)

# helper functions for working with db_str: str values


def get_engine_from_connection_string(db_str, timeout=30):
    """Create an engine with configurable timeout.

    Args:
        db_str: Database connection string
        timeout: SQLite timeout in seconds (default: 30)
    """
    engine = create_engine(
        db_str,
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=10,
        connect_args={"timeout": timeout},
    )
    
    # Set connection-specific pragmas for performance (these don't persist)
    if db_str.startswith("sqlite"):
        @sqlalchemy.event.listens_for(engine, "connect")
        def set_connection_pragmas(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            # These pragmas are connection-specific and improve performance
            cursor.execute("PRAGMA synchronous=NORMAL")  # Less durability, more speed
            cursor.execute("PRAGMA cache_size=10000")  # Larger cache
            cursor.close()

    return engine


@contextmanager
def get_session_from_connection_string(db_str, timeout=30):
    """Get a session from the connection string with pooling.

    Args:
        db_str: Database connection string
        timeout: SQLite timeout in seconds (default: 30)
    """
    engine = get_engine_from_connection_string(db_str, timeout=timeout)
    session = Session(engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def set_sqlite_pragmas(db_str):
    """Set SQLite WAL mode for better concurrency.
    
    This should be called once when creating/initializing the database.
    WAL mode persists across connections, unlike other pragmas.
    
    Args:
        db_str: Database connection string
    """
    if not db_str.startswith("sqlite"):
        return  # Only apply to SQLite databases
    
    # Create a temporary connection just to set WAL mode
    engine = create_engine(db_str)
    try:
        with engine.connect() as conn:
            # Check current journal mode first
            result = conn.execute(sqlalchemy.text("PRAGMA journal_mode"))
            current_mode = result.scalar()
            
            # Only set WAL if not already set (to avoid locking issues)
            if current_mode != "wal":
                conn.execute(sqlalchemy.text("PRAGMA journal_mode=WAL"))  # Write-Ahead Logging
                conn.commit()
    except Exception as e:
        # Log but don't fail - WAL is an optimization, not a requirement
        import logging
        logging.debug(f"Could not set SQLite WAL mode: {e}")
    finally:
        engine.dispose()


def create_db_and_tables(db_str):
    """
    Creates all database tables if they don't exist.

    Args:
        db_str: Database connection string

    Returns:
        The SQLAlchemy engine
    """
    engine = get_engine_from_connection_string(db_str)

    # Import SQLModel for creating tables

    # Create all tables defined in SQLModel classes
    SQLModel.metadata.create_all(engine)
    
    # Set SQLite pragmas once during database creation
    set_sqlite_pragmas(db_str)

    return engine


## some helper functions


def read_invocation(invocation_id: UUID, session: Session):
    """
    Fetches a single invocation by its UUID.

    Args:
        invocation_id: UUID string of the invocation to fetch
        session: The database session

    Returns:
        An Invocation object or None if not found
    """
    return session.get(Invocation, invocation_id)


def delete_invocation(invocation_id: UUID, session: Session):
    """
    Deletes an invocation and all its related embeddings.

    Args:
        invocation_id: UUID of the invocation to fetch
        session: The database session

    Returns:
        True if the invocation was deleted, False if not found
    """
    invocation = session.get(Invocation, invocation_id)
    if not invocation:
        return False

    # This will automatically delete all related embeddings due to cascade configuration
    session.delete(invocation)
    session.commit()
    return True


def delete_experiment(experiment_id: UUID, session: Session):
    """
    Deletes an experiment configuration and all its related runs, invocations, and embeddings.

    Args:
        experiment_id: UUID of the experiment to delete
        session: The database session

    Returns:
        True if the experiment was deleted, False if not found
    """
    experiment = session.get(ExperimentConfig, experiment_id)
    if not experiment:
        return False

    # This will cascade delete all related entities due to cascade configuration
    session.delete(experiment)
    session.commit()
    return True


def read_experiment_config(experiment_id: UUID, session: Session):
    """
    Fetches a single experiment configuration by its UUID.

    Args:
        experiment_id: UUID string of the experiment config to fetch
        session: The database session

    Returns:
        An ExperimentConfig object or None if not found
    """
    return session.get(ExperimentConfig, experiment_id)


def read_run(run_id: UUID, session: Session):
    """
    Fetches a single run by its UUID.

    Args:
        run_id: UUID string of the run to fetch
        session: The database session

    Returns:
        A Run object or None if not found
    """
    return session.get(Run, run_id)


def read_embedding(embedding_id: UUID, session: Session):
    """
    Fetches a single Embedding by its UUID.

    Args:
        embedding_id: UUID string of the embedding to fetch
        session: The database session

    Returns:
        An Embedding object or None if not found
    """
    return session.get(Embedding, embedding_id)


def find_embedding_for_vector(vector: np.ndarray, session: Session) -> Embedding:
    """
    Finds the first embedding with a vector that exactly matches the given vector.

    Args:
        vector: Numpy ndarray to match against
        session: The database session

    Returns:
        An Embedding object with a matching vector

    Raises:
        ValueError: If no matching embedding is found
    """
    # Query the database directly with the numpy array - NumpyArrayType will handle serialization
    statement = select(Embedding).where(
        Embedding.vector.is_not(None), Embedding.vector == vector
    )
    embedding = session.exec(statement).first()

    if embedding is None:
        raise ValueError("No embedding found with the given vector")

    return embedding


def list_invocations(session: Session):
    """
    Returns all invocations.

    Args:
        session: The database session

    Returns:
        A list of Invocation objects
    """

    statement = select(Invocation)

    return session.exec(statement).all()


def list_runs(session: Session):
    """
    Returns all runs.

    Args:
        session: The database session

    Returns:
        A list of Run objects
    """

    statement = select(Run)
    return session.exec(statement).all()


def latest_experiment(session: Session):
    """
    Returns the most recent experiment configuration ordered by started_at time.

    Args:
        session: The database session

    Returns:
        The most recent ExperimentConfig object or None if no experiments exist
    """
    statement = (
        select(ExperimentConfig).order_by(ExperimentConfig.started_at.desc()).limit(1)
    )
    return session.exec(statement).first()


def list_experiments(session: Session):
    """
    Returns all experiment configurations.

    Args:
        session: The database session

    Returns:
        A list of ExperimentConfig objects
    """
    statement = select(ExperimentConfig)
    return session.exec(statement).all()


def list_embeddings(session: Session):
    """
    Returns all embeddings.

    Args:
        session: The database session

    Returns:
        A list of Embedding objects
    """
    statement = select(Embedding)
    return session.exec(statement).all()


def list_persistence_diagrams(session: Session):
    """
    Returns all persistence diagrams.

    Args:
        session: The database session

    Returns:
        A list of PersistenceDiagram objects
    """
    statement = select(PersistenceDiagram)
    return session.exec(statement).all()


def incomplete_embeddings(session: Session):
    """
    Returns all Embedding objects without vector data, ordered by embedding model.

    Args:
        session: The database session

    Returns:
        A list of Embedding objects that have null vector values
    """

    statement = (
        select(Embedding)
        .where(Embedding.vector.is_(None))
        .order_by(Embedding.embedding_model)
    )
    return session.exec(statement).all()


def count_invocations(session: Session) -> int:
    """
    Returns the count of invocations in the database.

    Args:
        session: The database session

    Returns:
        The number of Invocation records
    """
    statement = select(func.count()).select_from(Invocation)
    return session.exec(statement).one()


def print_run_info(run_id: UUID, session: Session):
    """
    Prints detailed information about a run and its related entities.

    Args:
        run_id: UUID of the run to inspect
        session: The database session

    Returns:
        None - information is printed to stdout
    """
    run = read_run(run_id, session)
    if not run:
        print(f"Run {run_id} not found")
        return

    # Print basic run information
    print(f"Run {run.id}")
    print(f"  Network: {' -> '.join(run.network)} -> ...")
    print(f"  Seed: {run.seed}")
    print(f"  Max Length: {run.max_length}")
    print(f"  Initial Prompt: {run.initial_prompt}")
    print(f"  Number of Invocations: {len(run.invocations)}")
    print(f"  Stop Reason: {run.stop_reason}")

    # Count embeddings by model
    embedding_counts = {}
    for emb in run.embeddings:
        model = emb.embedding_model
        if model not in embedding_counts:
            embedding_counts[model] = 0
        embedding_counts[model] += 1

    print("  Embeddings:")
    for model, count in embedding_counts.items():
        print(f"    {model}: {count}")

    # Print information about persistence diagrams
    print("  Persistence Diagrams:")
    for pd in run.persistence_diagrams:
        generators = pd.get_generators_as_arrays()
        print(f"    {pd.embedding_model}: {len(generators)} generators")

        if len(generators) > 0:
            # Print details about each generator
            for i, gen in enumerate(generators[:3]):  # Show just first 3 for brevity
                print(f"      Generator {i}: Shape {gen.shape}")

            if len(generators) > 3:
                print(f"      ... and {len(generators) - 3} more generators")


def print_experiment_info(
    experiment_config: ExperimentConfig, session: Session
) -> None:
    """
    Get a status report of the experiment's progress.

    Reports:
    - Invocation progress: Percentage of completed invocations out of total expected
    - Embedding progress: Percentage of completed embeddings broken down by model
    - Persistence diagram progress: Percentage of runs with completed diagrams
    """
    # Import here to avoid circular imports

    runs = list(experiment_config.runs)  # Ensure runs are loaded into a list
    if not runs:
        print("No runs found in experiment")
        return

    # Calculate expected total invocations for the experiment
    total_expected_invocations = (
        len(experiment_config.networks)
        * len(experiment_config.seeds)
        * len(experiment_config.prompts)
        * experiment_config.max_length
    )

    # Single query to get max sequence number for each run, grouped by run_id
    run_ids = [run.id for run in runs]

    if not run_ids:
        total_actual_invocations = 0
    else:
        query = (
            select(
                Invocation.run_id,
                func.max(Invocation.sequence_number).label("max_sequence"),
            )
            .where(Invocation.run_id.in_(run_ids))
            .group_by(Invocation.run_id)
        )

        results = session.exec(query).all()

        # Create dictionary of run_id to max_sequence
        max_sequences = {run_id: max_seq for run_id, max_seq in results}

        # Calculate total invocations
        total_actual_invocations = 0
        for run in runs:
            # Get max sequence (or None if run not in results)
            max_sequence = max_sequences.get(run.id)
            # Add (max_sequence or -1) + 1 to match original logic
            total_actual_invocations += (max_sequence or -1) + 1

    # Calculate invocation progress
    invocation_percent = (total_actual_invocations / total_expected_invocations) * 100

    # Calculate the ETA for all runs
    estimated_time_remaining = 0
    for run in runs:
        # Calculate average invocation time for this run's models
        estimated_time_remaining += (
            (run.max_length - max_sequences.get(run.id, 0))
            * sum(estimated_time(model) for model in run.network)
            / len(run.network)
        )

    if experiment_config.started_at:
        # Create time string for invocations
        invocation_time_str = (
            f" (est. {naturaldelta(timedelta(seconds=estimated_time_remaining))} remaining)"
            if invocation_percent < 100.0
            else " (completed)"
        )
    else:
        invocation_time_str = "(not yet started)"

    # Embedding progress - overall and per model (but divided by two because only text invocations get embedded)
    expected_embeddings_total = (
        total_actual_invocations * len(experiment_config.embedding_models) / 2
    )

    # Get the total count of embeddings for this experiment using a database query
    # Group by embedding_model to get counts per model
    embedding_counts_by_model = session.exec(
        select(Embedding.embedding_model, func.count(Embedding.id).label("count"))
        .join(Invocation, Embedding.invocation_id == Invocation.id)
        .join(Run, Invocation.run_id == Run.id)
        .where(Run.experiment_id == experiment_config.id)
        .group_by(Embedding.embedding_model)
    ).all()

    # Calculate the total by summing up all model counts
    actual_embeddings_total = sum(count for _, count in embedding_counts_by_model)

    embedding_percent_total = (
        (actual_embeddings_total / expected_embeddings_total) * 100
        if expected_embeddings_total > 0
        else 0
    )

    # Per-model embedding statistics
    model_stats = {}
    # Create a dictionary of model names to counts from the query results
    model_count_dict = {model: count for model, count in embedding_counts_by_model}

    for model in experiment_config.embedding_models:
        # Get the actual count from query results (or 0 if not found)
        actual_for_model = model_count_dict.get(model, 0)
        # Expected embeddings for this model (each text invocation needs an embedding)
        expected_for_model = int(
            total_actual_invocations / 2
        )  # Only text invocations get embedded
        percent_for_model = (
            (actual_for_model / expected_for_model) * 100
            if expected_for_model > 0
            else 0
        )
        model_stats[model] = (
            actual_for_model,
            expected_for_model,
            percent_for_model,
        )

    embedding_time_str = (
        " (in progress)" if embedding_percent_total < 100.0 else " (completed)"
    )

    # Get missing persistence diagrams using the ExperimentConfig method
    missing_diagrams = experiment_config.missing_persistence_diagrams()

    # Count total expected persistence diagrams
    total_expected_diagrams = len(runs) * len(experiment_config.embedding_models)

    # Calculate completed diagrams and percentage
    completed_diagrams = total_expected_diagrams - len(missing_diagrams)
    diagram_percent = (
        (completed_diagrams / total_expected_diagrams * 100)
        if total_expected_diagrams > 0
        else 0
    )

    # Calculate diagram time string consistent with other time strings
    if experiment_config.started_at:
        diagram_time_str = (
            " (in progress)" if diagram_percent < 100.0 else " (completed)"
        )
    else:
        diagram_time_str = "(not yet started)"

    # Calculate counts per embedding model
    missing_by_model = {}
    for run, embedding_model in missing_diagrams:
        if embedding_model not in missing_by_model:
            missing_by_model[embedding_model] = 0
        missing_by_model[embedding_model] += 1

    # Prepare model-specific statistics dictionary
    diagram_model_stats = {}
    for model in experiment_config.embedding_models:
        missing_for_model = missing_by_model.get(model, 0)
        expected_for_model = len(runs)
        completed_for_model = expected_for_model - missing_for_model
        percent_for_model = (
            (completed_for_model / expected_for_model * 100)
            if expected_for_model > 0
            else 0
        )
        diagram_model_stats[model] = (
            completed_for_model,
            expected_for_model,
            percent_for_model,
        )

    # Calculate elapsed time from start
    elapsed_seconds = (
        (datetime.now() - experiment_config.started_at).total_seconds()
        if experiment_config.started_at
        else 0
    )

    # Summarize configuration information
    network_summary = f"{len(experiment_config.networks)} networks"
    embedding_model_summary = (
        f"{len(experiment_config.embedding_models)} embedding models"
    )
    seed_summary = f"{len(experiment_config.seeds)} seeds"
    prompt_summary = f"{len(experiment_config.prompts)} prompts"

    status_report = (
        f"Experiment Configuration:\n"
        f"  ID: {experiment_config.id}\n"
        f"  Total Runs: {len(runs)}\n"
        f"  Networks: {network_summary} {experiment_config.networks}\n"
        f"  Embedding Models: {embedding_model_summary} {experiment_config.embedding_models}\n"
        f"  Seeds: {seed_summary} {experiment_config.seeds}\n"
        f"  Prompts: {prompt_summary} {[p[:50] + '...' if len(p) > 50 else p for p in experiment_config.prompts]}\n"
        f"  Max Length: {experiment_config.max_length}\n\n"
        f"Experiment Status:\n"
        f"  Invocation Progress: {invocation_percent:.1f}% ({total_actual_invocations}/{total_expected_invocations}){invocation_time_str}\n"
        f"  Embedding Progress (Overall): {embedding_percent_total:.1f}% ({actual_embeddings_total}/{expected_embeddings_total}){embedding_time_str}\n"
    )

    # Add per-model embedding statistics
    for model, (actual, expected, percent) in model_stats.items():
        status_report += f"    - {model}: {percent:.1f}% ({actual}/{expected})\n"

    status_report += f"  Persistence Diagrams (Overall): {diagram_percent:.1f}% ({completed_diagrams}/{total_expected_diagrams}){diagram_time_str}\n"

    # Add per-model persistence diagram statistics
    for model, (completed, expected, percent) in diagram_model_stats.items():
        status_report += f"    - {model}: {percent:.1f}% ({completed}/{expected})\n"

    status_report += (
        f"  Elapsed Time: {naturaldelta(timedelta(seconds=elapsed_seconds))}"
    )

    print(status_report)


def export_experiments(
    source_db_str: str,
    target_db_str: str,
    experiment_ids: List[str],
    timeout: int = 30,
) -> None:
    """
    Export a subset of experiments from source database to target database.

    This function copies all data related to the specified experiments while
    maintaining referential integrity. It creates the target database and
    tables if they don't exist.

    Args:
        source_db_str: Connection string for source database
        target_db_str: Connection string for target database
        experiment_ids: List of experiment UUIDs to export
        timeout: SQLite timeout in seconds (default: 30)

    Raises:
        ValueError: If any experiment_ids are not found in source database
    """
    # Create target database and tables if needed
    create_db_and_tables(target_db_str)

    # Convert string IDs to UUID objects
    experiment_uuids = [UUID(exp_id) for exp_id in experiment_ids]

    with get_session_from_connection_string(
        source_db_str, timeout=timeout
    ) as source_session:
        with get_session_from_connection_string(
            target_db_str, timeout=timeout
        ) as target_session:
            # Verify all experiments exist
            for exp_uuid in experiment_uuids:
                exp_config = source_session.get(ExperimentConfig, exp_uuid)
                if not exp_config:
                    raise ValueError(
                        f"Experiment {exp_uuid} not found in source database"
                    )

            # 1. Copy ExperimentConfig records
            for exp_uuid in experiment_uuids:
                exp_config = source_session.get(ExperimentConfig, exp_uuid)
                # Create new instance to avoid session conflicts
                new_config = ExperimentConfig(
                    id=exp_config.id,
                    networks=exp_config.networks,
                    seeds=exp_config.seeds,
                    prompts=exp_config.prompts,
                    max_length=exp_config.max_length,
                    embedding_models=exp_config.embedding_models,
                    started_at=exp_config.started_at,
                    completed_at=exp_config.completed_at,
                )
                target_session.add(new_config)

            # Commit experiments first to satisfy FK constraints
            target_session.commit()

            # 2. Get all runs for these experiments
            runs_query = select(Run).where(Run.experiment_id.in_(experiment_uuids))
            runs = source_session.exec(runs_query).all()
            run_ids = [run.id for run in runs]

            # Copy Run records
            for run in runs:
                new_run = Run(
                    id=run.id,
                    experiment_id=run.experiment_id,
                    network=run.network,
                    seed=run.seed,
                    initial_prompt=run.initial_prompt,
                    max_length=run.max_length,
                    stop_reason=run.stop_reason,
                )
                target_session.add(new_run)

            # Commit runs before dependent tables
            target_session.commit()

            # 3. Get all invocations for these runs
            if run_ids:
                invocations_query = select(Invocation).where(
                    Invocation.run_id.in_(run_ids)
                )
                invocations = source_session.exec(invocations_query).all()
                invocation_ids = [inv.id for inv in invocations]

                # Copy Invocation records
                for inv in invocations:
                    new_inv = Invocation(
                        id=inv.id,
                        run_id=inv.run_id,
                        sequence_number=inv.sequence_number,
                        type=inv.type,
                        model=inv.model,
                        seed=inv.seed,
                        started_at=inv.started_at,
                        completed_at=inv.completed_at,
                        input_invocation_id=inv.input_invocation_id,
                        output_text=inv.output_text,
                        output_image_data=inv.output_image_data,
                    )
                    target_session.add(new_inv)

                # Commit invocations before embeddings
                target_session.commit()

                # 4. Get all embeddings for these invocations
                if invocation_ids:
                    embeddings_query = select(Embedding).where(
                        Embedding.invocation_id.in_(invocation_ids)
                    )
                    embeddings = source_session.exec(embeddings_query).all()

                    # Copy Embedding records
                    for emb in embeddings:
                        new_emb = Embedding(
                            id=emb.id,
                            invocation_id=emb.invocation_id,
                            embedding_model=emb.embedding_model,
                            vector=emb.vector,
                        )
                        target_session.add(new_emb)

                    # Commit embeddings
                    target_session.commit()

                # 5. Get all persistence diagrams for these runs
                diagrams_query = select(PersistenceDiagram).where(
                    PersistenceDiagram.run_id.in_(run_ids)
                )
                diagrams = source_session.exec(diagrams_query).all()

                # Copy PersistenceDiagram records
                for diag in diagrams:
                    new_diag = PersistenceDiagram(
                        id=diag.id,
                        run_id=diag.run_id,
                        embedding_model=diag.embedding_model,
                        started_at=diag.started_at,
                        completed_at=diag.completed_at,
                        diagram_data=diag.diagram_data,
                    )
                    target_session.add(new_diag)

                # Final commit for persistence diagrams
                target_session.commit()
