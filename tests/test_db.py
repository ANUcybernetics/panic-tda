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
    export_experiments,
    find_embedding_for_vector,
    get_engine_from_connection_string,
    incomplete_embeddings,
    latest_experiment,
    list_embeddings,
    list_invocations,
    read_embedding,
    read_invocation,
    read_run,
)
from panic_tda.local_modules.shared import droplet_and_leaf_invocations, list_completed_run_ids
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


def test_read_embedding(db_session: Session):
    """Test reading a specific embedding by ID."""
    # Create a sample run and invocation
    sample_run = Run(
        initial_prompt="test read embedding",
        network=["model1"],
        seed=42,
        max_length=1,
    )
    sample_invocation = Invocation(
        model="TextModel",
        type=InvocationType.TEXT,
        seed=42,
        run_id=sample_run.id,
        sequence_number=0,
        output_text="Test",
    )

    # Create a sample embedding
    original_vector = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    sample_embedding = Embedding(
        invocation_id=sample_invocation.id, embedding_model="test-model"
    )
    sample_embedding.vector = original_vector

    db_session.add(sample_run)
    db_session.add(sample_invocation)
    db_session.add(sample_embedding)
    db_session.commit()

    # Test reading the embedding with a valid ID
    retrieved_embedding = read_embedding(sample_embedding.id, db_session)

    assert retrieved_embedding is not None
    assert retrieved_embedding.id == sample_embedding.id
    assert retrieved_embedding.embedding_model == "test-model"
    assert isinstance(retrieved_embedding.vector, np.ndarray)
    assert retrieved_embedding.vector.shape == (3,)
    assert np.allclose(retrieved_embedding.vector, original_vector)

    # Test reading with a non-existent ID
    nonexistent_id = uuid7()
    retrieved_embedding = read_embedding(nonexistent_id, db_session)
    assert retrieved_embedding is None


def test_find_embedding_for_vector(db_session: Session):
    """Test the find_embedding_for_vector function."""
    # Create a sample run
    sample_run = Run(
        initial_prompt="test find embedding",
        network=["model1"],
        seed=42,
        max_length=3,
    )

    # Create two invocations for our embeddings
    invocation1 = Invocation(
        model="TextModel",
        type=InvocationType.TEXT,
        seed=42,
        run_id=sample_run.id,
        sequence_number=1,
        output_text="Sample output text 1",
    )

    invocation2 = Invocation(
        model="TextModel",
        type=InvocationType.TEXT,
        seed=43,
        run_id=sample_run.id,
        sequence_number=2,
        output_text="Sample output text 2",
    )

    # Create first embedding with a specific vector for testing
    test_vector1 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    embedding1 = Embedding(invocation_id=invocation1.id, embedding_model="test-model-1")
    embedding1.vector = test_vector1

    # Create second embedding with a different vector
    test_vector2 = np.array([0.4, 0.5, 0.6], dtype=np.float32)
    embedding2 = Embedding(invocation_id=invocation2.id, embedding_model="test-model-2")
    embedding2.vector = test_vector2

    # Add everything to the session
    db_session.add(sample_run)
    db_session.add(invocation1)
    db_session.add(invocation2)
    db_session.add(embedding1)
    db_session.add(embedding2)
    db_session.commit()

    # Test finding exact match for first embedding
    result1 = find_embedding_for_vector(test_vector1, db_session)
    assert result1 is not None
    assert result1.id == embedding1.id
    assert result1.embedding_model == "test-model-1"
    assert np.array_equal(result1.vector, test_vector1)

    # Test finding exact match for second embedding
    result2 = find_embedding_for_vector(test_vector2, db_session)
    assert result2 is not None
    assert result2.id == embedding2.id
    assert result2.embedding_model == "test-model-2"
    assert np.array_equal(result2.vector, test_vector2)

    # Test for vector that doesn't exist
    non_existent_vector = np.random.rand(3).astype(np.float32)
    with pytest.raises(ValueError, match="No embedding found with the given vector"):
        find_embedding_for_vector(non_existent_vector, db_session)


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


def test_droplet_and_leaf_invocations(db_session: Session):
    """Test the droplet_and_leaf_invocations function."""

    # Create a sample run
    sample_run = Run(
        initial_prompt="test leaf and droplet",
        network=["model1"],
        seed=42,
        max_length=3,
    )
    db_session.add(sample_run)
    db_session.commit()

    # Create text invocations to serve as inputs for image invocations
    leaf_input_invocation = Invocation(
        model="TextModel",
        type=InvocationType.TEXT,
        seed=40,
        run_id=sample_run.id,
        sequence_number=0,
        output_text="Generate an image of a green leaf",
    )

    droplet_input_invocation = Invocation(
        model="TextModel",
        type=InvocationType.TEXT,
        seed=41,
        run_id=sample_run.id,
        sequence_number=0,
        output_text="Show me a water droplet on a surface",
    )

    irrelevant_input_invocation = Invocation(
        model="TextModel",
        type=InvocationType.TEXT,
        seed=42,
        run_id=sample_run.id,
        sequence_number=0,
        output_text="Generate a picture of a mountain",
    )

    db_session.add(leaf_input_invocation)
    db_session.add(droplet_input_invocation)
    db_session.add(irrelevant_input_invocation)
    db_session.commit()

    # Image invocation with "leaf" in input
    leaf_img_invocation = Invocation(
        model="ImageModel",
        type=InvocationType.IMAGE,
        seed=42,
        run_id=sample_run.id,
        sequence_number=1,
        input_invocation_id=leaf_input_invocation.id,
    )

    # Image invocation with "droplet" in input
    droplet_img_invocation = Invocation(
        model="ImageModel",
        type=InvocationType.IMAGE,
        seed=43,
        run_id=sample_run.id,
        sequence_number=2,
        input_invocation_id=droplet_input_invocation.id,
    )

    # Image invocation without relevant keywords
    irrelevant_img_invocation = Invocation(
        model="ImageModel",
        type=InvocationType.IMAGE,
        seed=44,
        run_id=sample_run.id,
        sequence_number=3,
        input_invocation_id=irrelevant_input_invocation.id,
    )

    # Text invocations
    leaf_text_invocation = Invocation(
        model="TextModel",
        type=InvocationType.TEXT,
        seed=45,
        run_id=sample_run.id,
        sequence_number=4,
        output_text="The leaf fell gently from the tree.",
    )

    droplet_text_invocation = Invocation(
        model="TextModel",
        type=InvocationType.TEXT,
        seed=46,
        run_id=sample_run.id,
        sequence_number=5,
        output_text="A droplet of rain slid down the window.",
    )

    irrelevant_text_invocation = Invocation(
        model="TextModel",
        type=InvocationType.TEXT,
        seed=47,
        run_id=sample_run.id,
        sequence_number=6,
        output_text="The sky was clear and blue today.",
    )

    # Add all invocations to the session
    db_session.add(leaf_img_invocation)
    db_session.add(droplet_img_invocation)
    db_session.add(irrelevant_img_invocation)
    db_session.add(leaf_text_invocation)
    db_session.add(droplet_text_invocation)
    db_session.add(irrelevant_text_invocation)
    db_session.commit()

    # Call the function - now returns a tuple of (image_invocations, text_invocations)
    image_invocations, text_invocations = droplet_and_leaf_invocations(db_session)

    # Convert results to lists of IDs for easier checking
    image_ids = [invocation.id for invocation in image_invocations]
    text_ids = [invocation.id for invocation in text_invocations]

    # Assert no duplicates in results
    assert len(image_ids) == len(image_invocations), "Duplicate image invocations found"
    assert len(text_ids) == len(text_invocations), "Duplicate text invocations found"
    # Verify the results - check correct counts
    assert (
        len(image_invocations) == 2
    )  # Should include all image invocations linked to leaf/droplet text invocations
    assert (
        len(text_invocations) == 4
    )  # Should include all text invocations with leaf/droplet in the output

    # Verify the right invocations are included in the image list
    assert leaf_img_invocation.id in image_ids
    assert droplet_img_invocation.id in image_ids

    # Verify the right invocations are included in the text list
    assert leaf_text_invocation.id in text_ids
    assert droplet_text_invocation.id in text_ids

    # Verify irrelevant invocations are not included in either list
    assert irrelevant_img_invocation.id not in image_ids
    assert irrelevant_text_invocation.id not in text_ids


def test_list_completed_run_ids(db_session: Session):
    """Test the list_completed_run_ids function."""

    # Create runs with different prompts and networks
    # Group 1: prompt A, network 1 (many runs with this combination)
    run1 = Run(
        initial_prompt="test prompt A",
        network=["model1"],
        seed=42,
        max_length=3,
    )

    run2 = Run(
        initial_prompt="test prompt A",
        network=["model1"],
        seed=43,
        max_length=3,
    )

    run3 = Run(
        initial_prompt="test prompt A",
        network=["model1"],
        seed=44,
        max_length=3,
    )

    # Adding more runs for prompt A, network 1 to test imbalanced distribution
    run3a = Run(
        initial_prompt="test prompt A",
        network=["model1"],
        seed=101,
        max_length=3,
    )

    run3b = Run(
        initial_prompt="test prompt A",
        network=["model1"],
        seed=102,
        max_length=3,
    )

    run3c = Run(
        initial_prompt="test prompt A",
        network=["model1"],
        seed=103,
        max_length=3,
    )

    # Group 2: prompt A, network 2
    run4 = Run(
        initial_prompt="test prompt A",
        network=["model2"],
        seed=45,
        max_length=3,
    )

    run5 = Run(
        initial_prompt="test prompt A",
        network=["model2"],
        seed=46,
        max_length=3,
    )

    # Group 3: prompt B, network 1
    run6 = Run(
        initial_prompt="test prompt B",
        network=["model1"],
        seed=47,
        max_length=3,
    )

    run7 = Run(
        initial_prompt="test prompt B",
        network=["model1"],
        seed=48,
        max_length=3,
    )

    # Group 4: prompt C, network 3 (only one run)
    run8 = Run(
        initial_prompt="test prompt C",
        network=["model3"],
        seed=49,
        max_length=3,
    )

    # Add runs to the session
    db_session.add(run1)
    db_session.add(run2)
    db_session.add(run3)
    db_session.add(run3a)
    db_session.add(run3b)
    db_session.add(run3c)
    db_session.add(run4)
    db_session.add(run5)
    db_session.add(run6)
    db_session.add(run7)
    db_session.add(run8)
    db_session.commit()

    # Create persistence diagrams for most of the runs
    diagram1 = PersistenceDiagram(
        run_id=run1.id,
        embedding_model="test-model",
        generators=[np.array([[0.1, 0.5], [0.2, 0.7]])],
    )

    diagram2 = PersistenceDiagram(
        run_id=run2.id,
        embedding_model="test-model",
        generators=[np.array([[0.3, 0.6], [0.4, 0.8]])],
    )

    # Add diagrams for some of the additional runs
    diagram3a = PersistenceDiagram(
        run_id=run3a.id,
        embedding_model="test-model",
        generators=[np.array([[0.2, 0.3], [0.4, 0.5]])],
    )

    diagram3c = PersistenceDiagram(
        run_id=run3c.id,
        embedding_model="test-model",
        generators=[np.array([[0.6, 0.7], [0.8, 0.9]])],
    )

    diagram4 = PersistenceDiagram(
        run_id=run4.id,
        embedding_model="test-model",
        generators=[np.array([[0.5, 0.7], [0.6, 0.9]])],
    )

    diagram5 = PersistenceDiagram(
        run_id=run5.id,
        embedding_model="test-model",
        generators=[np.array([[0.7, 0.8], [0.8, 0.9]])],
    )

    diagram6 = PersistenceDiagram(
        run_id=run6.id,
        embedding_model="test-model",
        generators=[np.array([[0.9, 1.0], [1.0, 1.1]])],
    )

    diagram7 = PersistenceDiagram(
        run_id=run7.id,
        embedding_model="test-model",
        generators=[np.array([[1.1, 1.2], [1.2, 1.3]])],
    )

    diagram8 = PersistenceDiagram(
        run_id=run8.id,
        embedding_model="test-model",
        generators=[np.array([[1.3, 1.4], [1.4, 1.5]])],
    )

    # Add persistence diagrams to the session
    db_session.add(diagram1)
    db_session.add(diagram2)
    db_session.add(diagram3a)
    db_session.add(diagram3c)
    db_session.add(diagram4)
    db_session.add(diagram5)
    db_session.add(diagram6)
    db_session.add(diagram7)
    db_session.add(diagram8)
    db_session.commit()

    # Test with first_n=2 for each prompt and network combination
    results = list_completed_run_ids(db_session, 2)

    # Should include up to 2 runs from each prompt+network combination that have persistence diagrams
    assert (
        len(results) == 7
    )  # 2 from (A,model1), 2 from (A,model2), 2 from (B,model1), 1 from (C,model3)

    # Verify runs from Group 1 - should only include 2 despite having 4 valid runs
    group1_count = sum(
        1
        for run_id in [str(run1.id), str(run2.id), str(run3a.id), str(run3c.id)]
        if run_id in results
    )
    assert group1_count == 2, "Should only select 2 runs from prompt A, network 1"

    # Verify specific runs from other groups
    assert str(run4.id) in results
    assert str(run5.id) in results
    assert str(run6.id) in results
    assert str(run7.id) in results
    assert str(run8.id) in results

    # Verify runs without persistence diagrams are not included
    assert str(run3.id) not in results  # No persistence diagram
    assert str(run3b.id) not in results  # No persistence diagram

    # Test with first_n=3 for each prompt and network combination
    results = list_completed_run_ids(db_session, 3)

    # Should include up to 3 runs from each prompt+network combination
    # For prompt A, network 1: should find 3 runs with diagrams (run1, run2, run3a, run3c - but only take 3)
    # For prompt A, network 2: should find 2 runs with diagrams (run4, run5)
    # For prompt B, network 1: should find 2 runs with diagrams (run6, run7)
    # For prompt C, network 3: should find 1 run with diagram (run8)
    assert len(results) == 8  # 3 + 2 + 2 + 1 = 8

    # Group 1 should now have 3 runs since first_n=3
    group1_count = sum(
        1
        for run_id in [str(run1.id), str(run2.id), str(run3a.id), str(run3c.id)]
        if run_id in results
    )
    assert group1_count == 3, "Should select 3 runs from prompt A, network 1"

    # Test with first_n=1 for each prompt and network combination
    results = list_completed_run_ids(db_session, 1)

    # Should include only 1 run from each prompt+network combination
    assert len(results) == 4  # 1 from each unique combination

    # Verify only one run from Group 1 (prompt A, network 1)
    group1_count = sum(
        1
        for run_id in [str(run1.id), str(run2.id), str(run3a.id), str(run3c.id)]
        if run_id in results
    )
    assert group1_count == 1, "Should only select 1 run from prompt A, network 1"

    # Test with first_n=0
    results = list_completed_run_ids(db_session, 0)

    # Should return an empty list
    assert len(results) == 0


def test_export_experiments(db_session: Session, tmp_path):
    """Test the export_experiments function."""
    # Create two experiment configs
    experiment1 = ExperimentConfig(
        networks=[["model1", "model2"]],
        seeds=[42, 43],
        prompts=["test prompt 1"],
        embedding_models=["embedding_model_1"],
        max_length=3,
    )

    experiment2 = ExperimentConfig(
        networks=[["model3"]],
        seeds=[44],
        prompts=["test prompt 2", "test prompt 3"],
        embedding_models=["embedding_model_2"],
        max_length=2,
    )

    # Create runs for experiment1
    run1_1 = Run(
        experiment_id=experiment1.id,
        initial_prompt="test prompt 1",
        network=["model1", "model2"],
        seed=42,
        max_length=3,
    )

    run1_2 = Run(
        experiment_id=experiment1.id,
        initial_prompt="test prompt 1",
        network=["model1", "model2"],
        seed=43,
        max_length=3,
    )

    # Create run for experiment2
    run2_1 = Run(
        experiment_id=experiment2.id,
        initial_prompt="test prompt 2",
        network=["model3"],
        seed=44,
        max_length=2,
    )

    # Create invocations
    invocation1 = Invocation(
        run_id=run1_1.id,
        model="model1",
        type=InvocationType.TEXT,
        seed=42,
        sequence_number=0,
        output_text="Output from model1",
    )

    invocation2 = Invocation(
        run_id=run1_2.id,
        model="model1",
        type=InvocationType.TEXT,
        seed=43,
        sequence_number=0,
        output_text="Another output from model1",
    )

    invocation3 = Invocation(
        run_id=run2_1.id,
        model="model3",
        type=InvocationType.TEXT,
        seed=44,
        sequence_number=0,
        output_text="Output from model3",
    )

    # Create embeddings
    embedding1 = Embedding(
        invocation_id=invocation1.id,
        embedding_model="embedding_model_1",
        vector=np.array([0.1, 0.2, 0.3], dtype=np.float32),
    )

    embedding2 = Embedding(
        invocation_id=invocation2.id,
        embedding_model="embedding_model_1",
        vector=np.array([0.4, 0.5, 0.6], dtype=np.float32),
    )

    embedding3 = Embedding(
        invocation_id=invocation3.id,
        embedding_model="embedding_model_2",
        vector=np.array([0.7, 0.8, 0.9], dtype=np.float32),
    )

    # Create persistence diagrams
    diagram1 = PersistenceDiagram(
        run_id=run1_1.id,
        embedding_model="embedding_model_1",
        generators=[np.array([[0.1, 0.5], [0.2, 0.7]])],
    )

    diagram2 = PersistenceDiagram(
        run_id=run2_1.id,
        embedding_model="embedding_model_2",
        generators=[np.array([[0.3, 0.6], [0.4, 0.8]])],
    )

    # Add all to session
    db_session.add(experiment1)
    db_session.add(experiment2)
    db_session.add(run1_1)
    db_session.add(run1_2)
    db_session.add(run2_1)
    db_session.add(invocation1)
    db_session.add(invocation2)
    db_session.add(invocation3)
    db_session.add(embedding1)
    db_session.add(embedding2)
    db_session.add(embedding3)
    db_session.add(diagram1)
    db_session.add(diagram2)
    db_session.commit()

    # Test exporting only experiment1
    source_db_str = str(db_session.get_bind().url)
    target_db_path = tmp_path / "export_test.db"
    target_db_str = f"sqlite:///{target_db_path}"

    export_experiments(source_db_str, target_db_str, [str(experiment1.id)])

    # Verify the export by opening the target database
    from panic_tda.db import get_session_from_connection_string, list_experiments

    with get_session_from_connection_string(target_db_str) as target_session:
        # Check experiments
        experiments = list_experiments(target_session)
        assert len(experiments) == 1
        assert experiments[0].id == experiment1.id

        # Check runs
        from panic_tda.db import list_runs

        runs = list_runs(target_session)
        assert len(runs) == 2  # Only runs from experiment1
        run_ids = {run.id for run in runs}
        assert run1_1.id in run_ids
        assert run1_2.id in run_ids
        assert run2_1.id not in run_ids  # From experiment2, should not be exported

        # Check invocations
        invocations = list_invocations(target_session)
        assert len(invocations) == 2  # Only invocations from experiment1's runs
        inv_ids = {inv.id for inv in invocations}
        assert invocation1.id in inv_ids
        assert invocation2.id in inv_ids
        assert invocation3.id not in inv_ids

        # Check embeddings
        embeddings = list_embeddings(target_session)
        assert len(embeddings) == 2  # Only embeddings from experiment1's invocations
        emb_ids = {emb.id for emb in embeddings}
        assert embedding1.id in emb_ids
        assert embedding2.id in emb_ids
        assert embedding3.id not in emb_ids

        # Check persistence diagrams
        from panic_tda.db import list_persistence_diagrams

        diagrams = list_persistence_diagrams(target_session)
        assert len(diagrams) == 1  # Only diagram from experiment1's runs
        assert diagrams[0].id == diagram1.id

    # Test exporting multiple experiments
    target_db_path2 = tmp_path / "export_test2.db"
    target_db_str2 = f"sqlite:///{target_db_path2}"

    export_experiments(
        source_db_str, target_db_str2, [str(experiment1.id), str(experiment2.id)]
    )

    with get_session_from_connection_string(target_db_str2) as target_session:
        experiments = list_experiments(target_session)
        assert len(experiments) == 2
        exp_ids = {exp.id for exp in experiments}
        assert experiment1.id in exp_ids
        assert experiment2.id in exp_ids

        runs = list_runs(target_session)
        assert len(runs) == 3  # All runs

        invocations = list_invocations(target_session)
        assert len(invocations) == 3  # All invocations

        embeddings = list_embeddings(target_session)
        assert len(embeddings) == 3  # All embeddings

        diagrams = list_persistence_diagrams(target_session)
        assert len(diagrams) == 2  # All diagrams

    # Test with non-existent experiment ID
    with pytest.raises(ValueError, match="Experiment .* not found in source database"):
        export_experiments(
            source_db_str,
            target_db_str,
            [str(uuid7())],  # Non-existent UUID
        )
