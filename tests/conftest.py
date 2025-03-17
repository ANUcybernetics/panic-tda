import pytest
from sqlmodel import Session, SQLModel, create_engine


@pytest.fixture
def db_session():
    """Create a new in-memory database and session for a test."""
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        yield session

    # Database is automatically closed when the connection is closed


@pytest.fixture
def session():
    """Alias for db_session for backward compatibility"""
    with db_session() as session:
        yield session
