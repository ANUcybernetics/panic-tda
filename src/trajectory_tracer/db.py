from contextlib import contextmanager

from sqlmodel import Session, SQLModel, create_engine, select

from trajectory_tracer.schemas import Embedding, Invocation, PersistenceDiagram, Run


class Database:
    _instance = None

    def __init__(self, connection_string: str):
        """Initialize the database with a connection string."""
        self.engine = create_engine(connection_string)
        SQLModel.metadata.create_all(self.engine)

    def create_session(self) -> Session:
        """Create and return a new database session."""
        return Session(self.engine)

    @contextmanager
    def get_session(self):
        """Provide a transactional scope around a series of operations."""
        session = self.create_session()
        try:
            yield session
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()


def get_database(connection_string: str = "sqlite:///trajectory_tracer.sqlite") -> Database:
    """Get or create a Database instance with the specified connection string."""
    return Database(connection_string)


## some helper functions

def incomplete_embeddings(session: Session):
    """
    Returns all Embedding objects without vector data, ordered by embedding model.

    Args:
        session: The database session

    Returns:
        A list of Embedding objects that have null vector values
    """

    statement = select(Embedding).where(Embedding.vector.is_(None)).order_by(Embedding.embedding_model)
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
