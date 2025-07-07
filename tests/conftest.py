import os
import tempfile

import pytest
import ray
from sqlmodel import Session, SQLModel, create_engine


@pytest.fixture
def db_session():
    """Create a new temporary file-based database and session for a test."""
    # Create a temporary file
    db_file_handle, db_file_path = tempfile.mkstemp(suffix=".sqlite")

    # Close the file handle (SQLite will manage the file)
    os.close(db_file_handle)

    # Create the database connection
    db_url = f"sqlite:///{db_file_path}"
    engine = create_engine(db_url)
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        # Store the database URL and file path as session attributes
        session.db_url = db_url
        session.db_file_path = db_file_path
        yield session

    # Clean up: remove the temporary file
    if os.path.exists(db_file_path):
        try:
            os.unlink(db_file_path)
        except PermissionError:
            # If we can't delete immediately (e.g., Windows might keep a lock)
            # we can ignore - temp files will be cleaned up eventually
            pass


@pytest.fixture
def session():
    """Alias for db_session for backward compatibility"""
    with db_session() as session:
        yield session


@pytest.fixture(scope="session", autouse=True)
def ray_session():
    """Initialize Ray once for the entire test session."""
    # Initialize Ray with appropriate resources
    # Don't specify num_gpus to let Ray auto-detect
    ray.init(ignore_reinit_error=True)
    yield
    # Shutdown Ray after all tests
    ray.shutdown()
