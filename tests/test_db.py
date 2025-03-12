from uuid import UUID

import numpy as np
import pytest
from PIL import Image
from sqlalchemy import text
from sqlmodel import Session, select
from uuid_v7.base import uuid7

from trajectory_tracer.db import Database
from trajectory_tracer.schemas import Embedding, Invocation, InvocationType, Run


def test_run_creation(db_session: Session):
    """Test creating a Run object."""
    # Create a sample run
    sample_run = Run(
        initial_prompt="once upon a...",
        network=["model1", "model2"],
        seed=42,
        length=5
    )
    db_session.add(sample_run)
    db_session.commit()

    # Retrieve the run from the database
    retrieved_run = db_session.get(Run, sample_run.id)

    assert retrieved_run is not None
    assert retrieved_run.id == sample_run.id
    assert retrieved_run.network == ["model1", "model2"]
    assert retrieved_run.seed == 42
    assert retrieved_run.length == 5


def test_text_invocation(db_session: Session):
    """Test creating a text Invocation."""
    # Create a sample run
    sample_run = Run(
        initial_prompt="once upon a...",
        network=["model1", "model2"],
        seed=42,
        length=5
    )

    # Create a sample text invocation
    sample_text_invocation = Invocation(
        model="TextModel",
        type=InvocationType.TEXT,
        seed=42,
        run_id=sample_run.id,
        sequence_number=1,
        output_text="Sample output text"
    )

    db_session.add(sample_run)
    db_session.add(sample_text_invocation)
    db_session.commit()

    # Retrieve the invocation from the database
    retrieved = db_session.get(Invocation, sample_text_invocation.id)

    assert retrieved is not None
    assert retrieved.type == InvocationType.TEXT
    assert retrieved.output_text == "Sample output text"
    assert retrieved.output == "Sample output text"  # Test the property


def test_image_invocation(db_session: Session):
    """Test creating an image Invocation."""
    # Create a sample run
    sample_run = Run(
        initial_prompt="once upon a...",
        network=["model1", "model2"],
        seed=42,
        length=5
    )

    # Create a simple test image
    img = Image.new('RGB', (60, 30), color = 'red')

    invocation = Invocation(
        id=UUID('00000000-0000-0000-0000-000000000001'),
        model="ImageModel",
        type=InvocationType.IMAGE,
        seed=42,
        run_id=sample_run.id,
        sequence_number=2
    )
    invocation.output = img  # Test the output property setter for images

    db_session.add(sample_run)
    db_session.add(invocation)
    db_session.commit()

    # Retrieve the invocation from the database
    retrieved = db_session.get(Invocation, invocation.id)

    assert retrieved is not None
    assert retrieved.type == InvocationType.IMAGE
    assert retrieved.output_text is None
    assert retrieved.output_image_data is not None

    # Test that we can get the image back
    output_image = retrieved.output
    assert isinstance(output_image, Image.Image)
    assert output_image.width == 60
    assert output_image.height == 30

def test_embedding(db_session: Session):
    """Test creating and retrieving an Embedding."""
    # Create a sample text invocation
    sample_run = Run(
        initial_prompt="once upon a...",
        network=["model1", "model2"],
        seed=42,
        length=5
    )

    sample_text_invocation = Invocation(
        model="TextModel",
        type=InvocationType.TEXT,
        seed=42,
        run_id=sample_run.id,
        sequence_number=1,
        output_text="Sample output text"
    )

    # Create a sample embedding
    vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    sample_embedding = Embedding(
        invocation_id=sample_text_invocation.id,
        embedding_model="test-embedding-model"
    )
    sample_embedding.vector = vector

    db_session.add(sample_run)
    db_session.add(sample_text_invocation)
    db_session.add(sample_embedding)
    db_session.commit()

    # Retrieve the embedding from the database
    retrieved = db_session.get(Embedding, sample_embedding.id)

    assert retrieved is not None
    assert isinstance(retrieved.vector, np.ndarray)
    assert retrieved.vector.shape == (3,)
    assert np.allclose(retrieved.vector, np.array([0.1, 0.2, 0.3], dtype=np.float32))


def test_persistence_diagram_storage(db_session: Session):
    """Test storing and retrieving PersistenceDiagram objects."""
    from trajectory_tracer.schemas import PersistenceDiagram

    # Create arrays of different shapes/dimensions
    array1 = np.array([[0.1, 0.5], [0.2, 0.7], [0.4, 0.9]])
    array2 = np.array([[0.3, 0.8]])
    array3 = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    # Create a persistence diagram with these arrays
    diagram = PersistenceDiagram(
        run_id=uuid7(),
        generators=[array1, array2, array3]
    )

    db_session.add(diagram)
    db_session.commit()

    # Retrieve the diagram from the database
    retrieved = db_session.get(PersistenceDiagram, diagram.id)

    assert retrieved is not None

    # Get the generators as arrays
    retrieved_arrays = retrieved.get_generators_as_arrays()

    # Verify we have the right number of arrays
    assert len(retrieved_arrays) == 3

    # Verify the arrays have correct shapes
    assert retrieved_arrays[0].shape == (3, 2)
    assert retrieved_arrays[1].shape == (1, 2)
    assert retrieved_arrays[2].shape == (2, 3)

    # Verify the array contents
    assert np.allclose(retrieved_arrays[0], array1)
    assert np.allclose(retrieved_arrays[1], array2)
    assert np.allclose(retrieved_arrays[2], array3)


def test_database_initialization():
    """Test the Database class initialization."""
    # Test with default connection string
    db = Database()
    assert db.engine is not None

    # Test with custom connection string
    custom_db = Database(connection_string="sqlite:///test.db")
    assert custom_db.engine is not None
    assert str(custom_db.engine.url) == "sqlite:///test.db"


def test_database_create_session():
    """Test creating a new database session."""
    db = Database(connection_string="sqlite:///:memory:")
    session = db.create_session()
    assert session is not None

    # Test that we can perform operations with the session
    try:
        session.execute(text("SELECT 1"))
        session.commit()
    except Exception as e:
        pytest.fail(f"Session failed to execute query: {e}")
    finally:
        session.close()


def test_database_context_manager():
    """Test the context manager functionality of get_session."""
    db = Database(connection_string="sqlite:///:memory:")

    # Test normal operation
    with db.get_session() as session:
        # Create a sample run
        sample_run = Run(
            initial_prompt="testing context manager",
            network=["test"],
            seed=123,
            length=1
        )
        session.add(sample_run)

    # Verify the run was committed
    with db.get_session() as session:
        statement = select(Run).where(Run.initial_prompt == "testing context manager")
        runs = session.exec(statement).all()
        assert len(runs) == 1
        assert runs[0].seed == 123

    # Test rollback on exception
    try:
        with db.get_session() as session:
            # Create another sample run
            sample_run = Run(
                initial_prompt="should be rolled back",
                network=["test"],
                seed=456,
                length=1
            )
            session.add(sample_run)
            raise ValueError("Test exception to trigger rollback")
    except ValueError:
        pass

    # Verify the run was rolled back
    with db.get_session() as session:
        statement = select(Run).where(Run.initial_prompt == "should be rolled back")
        runs = session.exec(statement).all()
        assert len(runs) == 0
