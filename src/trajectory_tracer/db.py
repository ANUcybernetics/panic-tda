from contextlib import contextmanager
from uuid import UUID

import sqlalchemy
from sqlalchemy.pool import QueuePool
from sqlmodel import Session, SQLModel, create_engine, func, select

from trajectory_tracer.schemas import (
    Embedding,
    ExperimentConfig,
    Invocation,
    PersistenceDiagram,
    Run,
)

# helper functions for working with db_str: str values

def get_engine_from_connection_string(db_str):
    engine = create_engine(
        db_str,
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=10,
        connect_args={"timeout": 30}
    )

    # Configure SQLite for better concurrency
    @sqlalchemy.event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
        cursor.execute("PRAGMA synchronous=NORMAL")  # Less durability, more speed
        cursor.execute("PRAGMA cache_size=10000")  # Larger cache
        cursor.close()

    return engine


@contextmanager
def get_session_from_connection_string(db_str):
    """Get a session from the connection string with pooling"""
    engine = get_engine_from_connection_string(db_str)
    session = Session(engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


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
    statement = select(ExperimentConfig).order_by(ExperimentConfig.started_at.desc()).limit(1)
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
