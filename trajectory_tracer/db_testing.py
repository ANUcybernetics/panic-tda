import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from trajectory_tracer.db import DB_PATH, _engine, init_db


@contextmanager
def db_sandbox():
    """Create an isolated SQLite database for testing."""
    global _engine

    # Save original connection and DB path
    original_engine = _engine
    original_db_path = DB_PATH

    # Create temporary file
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.sqlite')
    temp_db.close()
    db_path = Path(temp_db.name)

    try:
        # Close existing connection
        if _engine:
            _engine = None

        # Initialize test database
        init_db(db_path)

        yield db_path
    finally:
        # Close test connection
        if _engine:
            _engine = None

        # Restore original connection and DB path
        _engine = original_engine
        init_db(original_db_path)

        # Delete test database
        if db_path.exists():
            os.unlink(db_path)
