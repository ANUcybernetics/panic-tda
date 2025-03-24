import tempfile
from uuid import UUID

import numpy as np
import pytest
from PIL import Image
from sqlalchemy import text
from sqlmodel import Session, select
from uuid_v7.base import uuid7

from trajectory_tracer.db import (
    Database,
    get_session_from_connection_string,
    incomplete_embeddings,
    list_embeddings,
    list_invocations,
    read_invocation,
    read_run,
)
from trajectory_tracer.schemas import (
    Embedding,
    Invocation,
    InvocationType,
    PersistenceDiagram,
    Run,
)


def test_read_invocation(db_session: Session):
    """Test the read_invocation function."""
    # Create a sample run
    sample_run = Run(
        initial_prompt="test read_invocation", network=["model1"], seed=42, max_length=3
    )

    # Create a sample invocation
    sample_invocation = Invocation(
        model="TextModel",
        type=InvocationType.TEXT,
        seed=42,
        run_id=sample_run.id,
        sequence_number=1,
        output_text="Sample output text",
    )

    # Add to the session
    db_session.add(sample_run)
    db_session.add(sample_invocation)
    db_session.commit()

    # Test the read_invocation function

    # Test with valid ID
    invocation = read_invocation(sample_invocation.id, db_session)
    assert invocation is not None
    assert invocation.id == sample_invocation.id
    assert invocation.model == "TextModel"
    assert invocation.output_text == "Sample output text"

    # Test with invalid ID
    nonexistent_id = uuid7()
    invocation = read_invocation(nonexistent_id, db_session)
    assert invocation is None


def test_read_run(db_session: Session):
    """Test the read_run function."""
    # Create a sample run
    sample_run = Run(
        initial_prompt="test read_run",
        network=["model1", "model2"],
        seed=42,
        max_length=5,
    )

    # Add to the session
    db_session.add(sample_run)
    db_session.commit()

    # Test with valid ID
    run = read_run(sample_run.id, db_session)
    assert run is not None
    assert run.id == sample_run.id
    assert run.initial_prompt == "test read_run"
    assert run.network == ["model1", "model2"]
    assert run.seed == 42

    # Test with invalid ID
    nonexistent_id = uuid7()
    run = read_run(nonexistent_id, db_session)
    assert run is None


def test_run_creation(db_session: Session):
    """Test creating a Run object."""
    # Create a sample run
    sample_run = Run(
        initial_prompt="once upon a...",
        network=["model1", "model2"],
        seed=42,
        max_length=5,
    )
    db_session.add(sample_run)
    db_session.commit()

    # Retrieve the run from the database
    retrieved_run = db_session.get(Run, sample_run.id)

    assert retrieved_run is not None
    assert retrieved_run.id == sample_run.id
    assert retrieved_run.network == ["model1", "model2"]
    assert retrieved_run.seed == 42
    assert retrieved_run.max_length == 5


def test_text_invocation(db_session: Session):
    """Test creating a text Invocation."""
    # Create a sample run
    sample_run = Run(
        initial_prompt="once upon a...",
        network=["model1", "model2"],
        seed=42,
        max_length=5,
    )

    # Create a sample text invocation
    sample_text_invocation = Invocation(
        model="TextModel",
        type=InvocationType.TEXT,
        seed=42,
        run_id=sample_run.id,
        sequence_number=1,
        output_text="Sample output text",
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
        max_length=5,
    )

    # Create a simple test image
    img = Image.new("RGB", (60, 30), color="red")

    invocation = Invocation(
        id=UUID("00000000-0000-0000-0000-000000000001"),
        model="ImageModel",
        type=InvocationType.IMAGE,
        seed=42,
        run_id=sample_run.id,
        sequence_number=2,
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
        max_length=5,
    )

    sample_text_invocation = Invocation(
        model="TextModel",
        type=InvocationType.TEXT,
        seed=42,
        run_id=sample_run.id,
        sequence_number=1,
        output_text="Sample output text",
    )

    # Create a sample embedding
    vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    sample_embedding = Embedding(
        invocation_id=sample_text_invocation.id, embedding_model="test-embedding-model"
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

    # Create arrays of different shapes/dimensions
    array1 = np.array([[0.1, 0.5], [0.2, 0.7], [0.4, 0.9]])
    array2 = np.array([[0.3, 0.8]])
    array3 = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    # Create a persistence diagram with these arrays
    diagram = PersistenceDiagram(
        run_id=uuid7(), embedding_model="Dummy", generators=[array1, array2, array3]
    )

    db_session.add(diagram)
    db_session.commit()

    # Retrieve the diagram from the database
    retrieved = db_session.get(PersistenceDiagram, diagram.id)

    assert retrieved is not None
    assert retrieved.embedding_model == "Dummy"

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
    connection_string = "sqlite:///test.sqlite"
    db = Database(connection_string)
    assert db.engine is not None
    assert str(db.engine.url) == connection_string


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
            max_length=1,
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
                max_length=1,
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


def test_incomplete_embeddings(db_session: Session):
    """Test the incomplete_embeddings function."""

    # Create a sample run
    sample_run = Run(
        initial_prompt="test incomplete embeddings",
        network=["model1"],
        seed=42,
        max_length=3,
    )

    # Create invocations
    invocation1 = Invocation(
        model="TextModel",
        type=InvocationType.TEXT,
        seed=42,
        run_id=sample_run.id,
        sequence_number=1,
        output_text="First output text",
    )

    invocation2 = Invocation(
        model="TextModel",
        type=InvocationType.TEXT,
        seed=43,
        run_id=sample_run.id,
        sequence_number=2,
        output_text="Second output text",
    )

    # Create embeddings - one complete and one incomplete
    complete_embedding = Embedding(
        invocation_id=invocation1.id, embedding_model="embedding_model-1"
    )
    complete_embedding.vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    incomplete_embedding1 = Embedding(
        invocation_id=invocation1.id, embedding_model="embedding_model-2"
    )

    incomplete_embedding2 = Embedding(
        invocation_id=invocation2.id, embedding_model="embedding_model-1"
    )

    # Add everything to the session
    db_session.add(sample_run)
    db_session.add(invocation1)
    db_session.add(invocation2)
    db_session.add(complete_embedding)
    db_session.add(incomplete_embedding1)
    db_session.add(incomplete_embedding2)
    db_session.commit()

    # Test the function
    results = incomplete_embeddings(db_session)

    # Verify the results
    assert len(results) == 2

    # Results should be ordered by embedding_model
    assert results[0].embedding_model == "embedding_model-1"
    assert results[0].invocation_id == invocation2.id
    assert results[1].embedding_model == "embedding_model-2"
    assert results[1].invocation_id == invocation1.id

    # Verify complete embedding is not in results
    for embedding in results:
        assert embedding.vector is None


def test_list_invocations(db_session: Session):
    """Test the list_invocations function."""

    # Create a sample run
    sample_run = Run(
        initial_prompt="test list invocations",
        network=["model1"],
        seed=42,
        max_length=3,
    )

    # Create invocations with different creation times
    invocation1 = Invocation(
        model="TextModel",
        type=InvocationType.TEXT,
        seed=42,
        run_id=sample_run.id,
        sequence_number=1,
        output_text="First output text",
    )

    invocation2 = Invocation(
        model="TextModel",
        type=InvocationType.TEXT,
        seed=43,
        run_id=sample_run.id,
        sequence_number=2,
        output_text="Second output text",
    )

    invocation3 = Invocation(
        model="ImageModel",
        type=InvocationType.IMAGE,
        seed=44,
        run_id=sample_run.id,
        sequence_number=3,
    )

    # Add everything to the session
    db_session.add(sample_run)
    db_session.add(invocation1)
    db_session.add(invocation2)
    db_session.add(invocation3)
    db_session.commit()

    # Test with no limit
    results = list_invocations(db_session)

    # Verify we got all invocations
    assert len(results) == 3


def test_list_embeddings(db_session: Session):
    """Test the list_embeddings function."""

    # Create a sample run
    sample_run = Run(
        initial_prompt="test list embeddings", network=["model1"], seed=42, max_length=3
    )

    # Create invocations
    invocation1 = Invocation(
        model="TextModel",
        type=InvocationType.TEXT,
        seed=42,
        run_id=sample_run.id,
        sequence_number=1,
        output_text="First output text",
    )

    invocation2 = Invocation(
        model="TextModel",
        type=InvocationType.TEXT,
        seed=43,
        run_id=sample_run.id,
        sequence_number=2,
        output_text="Second output text",
    )

    # Create embeddings with different models
    embedding1 = Embedding(
        invocation_id=invocation1.id, embedding_model="embedding_model-1"
    )
    embedding1.vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    embedding2 = Embedding(
        invocation_id=invocation1.id, embedding_model="embedding_model-2"
    )
    embedding2.vector = np.array([0.4, 0.5, 0.6], dtype=np.float32)

    embedding3 = Embedding(
        invocation_id=invocation2.id, embedding_model="embedding_model-1"
    )
    embedding3.vector = np.array([0.7, 0.8, 0.9], dtype=np.float32)

    # Add everything to the session
    db_session.add(sample_run)
    db_session.add(invocation1)
    db_session.add(invocation2)
    db_session.add(embedding1)
    db_session.add(embedding2)
    db_session.add(embedding3)
    db_session.commit()

    # Test with no filters
    results = list_embeddings(db_session)
    assert len(results) == 3


def test_get_session_from_connection_string():
    """Test the get_session_from_connection_string context manager."""

    # Create a temporary directory that will be automatically cleaned up
    temp_dir = tempfile.TemporaryDirectory()
    # Use a file-based SQLite database in the temporary directory
    connection_string = f"sqlite:///{temp_dir.name}/test_db.sqlite"

    # Test normal operation
    with get_session_from_connection_string(connection_string) as session:
        # Create a sample run
        sample_run = Run(
            initial_prompt="testing get_session_from_connection_string",
            network=["test"],
            seed=789,
            max_length=1,
        )
        session.add(sample_run)

    # Create a new session to verify the run was committed
    with get_session_from_connection_string(connection_string) as session:
        statement = select(Run).where(
            Run.initial_prompt == "testing get_session_from_connection_string"
        )
        runs = session.exec(statement).all()
        assert len(runs) == 1
        assert runs[0].seed == 789

    # Test rollback on exception
    try:
        with get_session_from_connection_string(connection_string) as session:
            # Create another sample run
            sample_run = Run(
                initial_prompt="should be rolled back too",
                network=["test"],
                seed=999,
                max_length=1,
            )
            session.add(sample_run)
            raise ValueError("Test exception to trigger rollback")
    except ValueError:
        pass

    # Verify the run was rolled back
    with get_session_from_connection_string(connection_string) as session:
        statement = select(Run).where(Run.initial_prompt == "should be rolled back too")
        runs = session.exec(statement).all()
        assert len(runs) == 0
