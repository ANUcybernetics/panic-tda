from contextlib import contextmanager
from typing import List, Optional, Type, TypeVar
from uuid import UUID

from sqlmodel import Session, SQLModel, create_engine, select

# Define a generic type for SQLModel classes
T = TypeVar('T', bound=SQLModel)

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

    def create(self, session: Session, model: SQLModel) -> None:
        """Create a new record in the database."""
        session.add(model)
        session.commit()
        session.refresh(model)

    def get_by_id(self, session: Session, model_class: Type[T], id: UUID) -> Optional[T]:
        """Get a record by its ID."""
        return session.get(model_class, id)

    def list(self, session: Session, model_class: Type[T], **filters) -> List[T]:
        """List all records of a specific model class with optional filters."""
        query = select(model_class)

        # Add filters if provided
        for attr, value in filters.items():
            if hasattr(model_class, attr):
                query = query.where(getattr(model_class, attr) == value)

        return session.exec(query).all()

    def update(self, session: Session, model: SQLModel, **values) -> None:
        """Update an existing record with provided values."""
        for key, value in values.items():
            if hasattr(model, key):
                setattr(model, key, value)

        session.add(model)
        session.commit()
        session.refresh(model)

    def delete(self, session: Session, model: SQLModel) -> None:
        """Delete a record from the database."""
        session.delete(model)
        session.commit()

# Create a global instance for convenience
db = Database()
