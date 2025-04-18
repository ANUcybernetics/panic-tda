import json
from datetime import datetime
from uuid import UUID

import numpy as np
import pytest
from PIL import Image
from sqlalchemy import MetaData, inspect
from sqlmodel import Session
from uuid_v7.base import uuid7

from panic_tda.db import (
    delete_invocation,
    get_engine_from_connection_string,
    incomplete_embeddings,
    latest_experiment,
    list_embeddings,
    list_invocations,
    read_invocation,
    read_run,
)
from panic_tda.schemas import (
    Embedding,
    ExperimentConfig,
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


def test_engine_initialization():
    """Test the engine initialization with connection pooling."""
    connection_string = "sqlite:///output/test/test.sqlite"
    engine = get_engine_from_connection_string(connection_string)
    assert engine is not None
    assert str(engine.url) == connection_string


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


def test_delete_invocation(db_session: Session):
    """Test the delete_invocation function."""

    # Create a sample run
    sample_run = Run(
        initial_prompt="test delete_invocation",
        network=["model1"],
        seed=42,
        max_length=3,
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

    # Create embeddings associated with the invocation
    embedding1 = Embedding(
        invocation_id=sample_invocation.id, embedding_model="embedding_model-1"
    )
    embedding1.vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    embedding2 = Embedding(
        invocation_id=sample_invocation.id, embedding_model="embedding_model-2"
    )
    embedding2.vector = np.array([0.4, 0.5, 0.6], dtype=np.float32)

    # Add everything to the session
    db_session.add(sample_run)
    db_session.add(sample_invocation)
    db_session.add(embedding1)
    db_session.add(embedding2)
    db_session.commit()

    # Verify invocation and embeddings exist
    assert db_session.get(Invocation, sample_invocation.id) is not None
    assert db_session.get(Embedding, embedding1.id) is not None
    assert db_session.get(Embedding, embedding2.id) is not None

    # Test delete with valid ID
    result = delete_invocation(sample_invocation.id, db_session)
    assert result is True

    # Verify invocation and related embeddings are deleted
    assert db_session.get(Invocation, sample_invocation.id) is None
    assert db_session.get(Embedding, embedding1.id) is None
    assert db_session.get(Embedding, embedding2.id) is None

    # Test delete with invalid ID
    nonexistent_id = uuid7()
    result = delete_invocation(nonexistent_id, db_session)
    assert result is False


def test_experiment_config_storage(db_session: Session):
    """Test storing and retrieving ExperimentConfig objects."""
    # Create a sample experiment config
    experiment_config = ExperimentConfig(
        networks=[["model1", "model2"], ["model3"]],
        seeds=[42, 43],
        prompts=["test prompt 1", "test prompt 2"],
        embedding_models=["embedding_model_1", "embedding_model_2"],
        max_length=5,
    )

    db_session.add(experiment_config)
    db_session.commit()

    # Retrieve the experiment config from the database
    retrieved = db_session.get(ExperimentConfig, experiment_config.id)

    assert retrieved is not None
    assert retrieved.id == experiment_config.id
    assert retrieved.networks == [["model1", "model2"], ["model3"]]
    assert retrieved.seeds == [42, 43]
    assert retrieved.prompts == ["test prompt 1", "test prompt 2"]
    assert retrieved.embedding_models == ["embedding_model_1", "embedding_model_2"]
    assert retrieved.max_length == 5

    # Test relationships
    # Create a run linked to this experiment
    sample_run = Run(
        initial_prompt="test experiment config",
        network=["model1", "model2"],
        seed=42,
        max_length=5,
        experiment_id=experiment_config.id,
    )

    db_session.add(sample_run)
    db_session.commit()

    # Refresh the experiment config to get the updated relationships
    db_session.refresh(retrieved)

    # Verify the relationship
    assert len(retrieved.runs) == 1
    assert retrieved.runs[0].id == sample_run.id


def test_experiment_config_cascading_delete(db_session: Session):
    """Test that deleting an ExperimentConfig cascades to all related entities."""
    # Create experiment config
    experiment_config = ExperimentConfig(
        networks=[["model1"]],
        seeds=[42],
        prompts=["test prompt"],
        embedding_models=["embedding_model"],
        max_length=3,
    )

    # Create a run linked to this experiment
    run = Run(
        initial_prompt="test cascade delete",
        network=["model1"],
        seed=42,
        max_length=3,
        experiment_id=experiment_config.id,
    )

    # Create an invocation linked to the run
    invocation = Invocation(
        model="TextModel",
        type=InvocationType.TEXT,
        seed=42,
        run_id=run.id,
        sequence_number=1,
        output_text="Sample text",
    )

    # Create an embedding linked to the invocation
    embedding = Embedding(
        invocation_id=invocation.id, embedding_model="embedding_model"
    )
    embedding.vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    # Add persistence diagram linked to the run
    diagram = PersistenceDiagram(
        run_id=run.id,
        embedding_model="embedding_model",
        generators=[np.array([[0.1, 0.5], [0.2, 0.7]])],
    )

    # Add everything to the session
    db_session.add(experiment_config)
    db_session.add(run)
    db_session.add(invocation)
    db_session.add(embedding)
    db_session.add(diagram)
    db_session.commit()

    # Verify all entities exist
    assert db_session.get(ExperimentConfig, experiment_config.id) is not None
    assert db_session.get(Run, run.id) is not None
    assert db_session.get(Invocation, invocation.id) is not None
    assert db_session.get(Embedding, embedding.id) is not None
    assert db_session.get(PersistenceDiagram, diagram.id) is not None

    # Delete the experiment config
    db_session.delete(experiment_config)
    db_session.commit()

    # Verify all related entities are deleted
    assert db_session.get(ExperimentConfig, experiment_config.id) is None
    assert db_session.get(Run, run.id) is None
    assert db_session.get(Invocation, invocation.id) is None
    assert db_session.get(Embedding, embedding.id) is None
    assert db_session.get(PersistenceDiagram, diagram.id) is None


def test_latest_experiment(db_session: Session):
    """Test the latest_experiment function."""
    # Create multiple experiment configs with different timestamps
    experiment_config1 = ExperimentConfig(
        networks=[["model1"]],
        seeds=[42],
        prompts=["test prompt 1"],
        embedding_models=["embedding_model_1"],
        max_length=3,
        started_at=datetime.now(),
    )

    experiment_config2 = ExperimentConfig(
        networks=[["model2"]],
        seeds=[43],
        prompts=["test prompt 2"],
        embedding_models=["embedding_model_2"],
        max_length=4,
        started_at=datetime.now(),
    )

    # Add to the session with a delay to ensure different timestamps
    db_session.add(experiment_config1)
    db_session.commit()

    db_session.add(experiment_config2)
    db_session.commit()

    # Import the function to test

    # Test the function
    latest = latest_experiment(db_session)

    # Verify we got the most recent experiment
    assert latest is not None
    assert latest.id == experiment_config2.id
    assert latest.prompts == ["test prompt 2"]
    assert latest.networks == [["model2"]]

    # Test with no experiments in the database
    db_session.delete(experiment_config1)
    db_session.delete(experiment_config2)
    db_session.commit()

    assert latest_experiment(db_session) is None


@pytest.mark.skip(
    reason="This test creates a file on disk, useful mostly for debugging"
)
def test_dump_schema(db_session: Session):
    """Test dumping the database schema to a file."""

    # Get the engine from the session
    engine = db_session.get_bind()

    # Create a metadata object
    metadata = MetaData()

    # Reflect the existing tables
    metadata.reflect(bind=engine)

    # Prepare a dictionary to hold the schema information
    schema_info = {}

    # Get inspector to access table details
    inspector = inspect(engine)

    # Iterate through all tables
    for table_name in inspector.get_table_names():
        schema_info[table_name] = {
            "columns": [],
            "primary_key": inspector.get_pk_constraint(table_name),
            "foreign_keys": inspector.get_foreign_keys(table_name),
            "indexes": inspector.get_indexes(table_name),
        }

        # Get column details
        for column in inspector.get_columns(table_name):
            col_info = {
                "name": column["name"],
                "type": str(column["type"]),
                "nullable": column["nullable"],
            }
            if "default" in column and column["default"] is not None:
                col_info["default"] = str(column["default"])

            schema_info[table_name]["columns"].append(col_info)

    # Write schema to file
    with open("database_schema.json", "w") as f:
        json.dump(schema_info, f, indent=2)

    # Verify the file was created and contains data
    import os

    assert os.path.exists("database_schema.json")

    # Read back the file to verify content
    with open("database_schema.json", "r") as f:
        loaded_schema = json.load(f)

    assert loaded_schema
    assert len(loaded_schema) > 0
    assert "run" in loaded_schema or "Run" in loaded_schema
