from contextlib import contextmanager

from sqlmodel import Session, SQLModel, create_engine


class Database:
    def __init__(self, connection_string: str = "sqlite:///trajectory_tracer.db"):
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

# Create a global instance for convenience
db = Database()
