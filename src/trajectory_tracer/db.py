from contextlib import contextmanager
from uuid import UUID

import sqlalchemy
from sqlalchemy.pool import QueuePool
from sqlmodel import Session, create_engine, func, select

from trajectory_tracer.schemas import Embedding, Invocation, PersistenceDiagram, Run

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


def incomplete_persistence_diagrams(session: Session):
    """
    Returns all PersistenceDiagram objects without generator data.

    Args:
        session: The database session

    Returns:
        A list of PersistenceDiagram objects that have empty generators
    """

    statement = select(PersistenceDiagram).where(
        # Check for empty generators list
        PersistenceDiagram.generators == []
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
