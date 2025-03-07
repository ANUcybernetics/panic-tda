import sys
from pathlib import Path

import pytest

from src.db_testing import db_sandbox

# Get the absolute path to the trajectory_tracer directory
project_root = Path(__file__).parent.parent.absolute()

# Add to path
sys.path.insert(0, str(project_root))

@pytest.fixture
def test_db():
    """Provide a temporary isolated database for each test."""
    with db_sandbox() as db_path:
        yield db_path
