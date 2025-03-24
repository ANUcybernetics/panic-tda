import hashlib
import io
from datetime import datetime
from uuid import UUID

import numpy as np
import ray
from PIL import Image
from sqlmodel import Session

from trajectory_tracer.db import (
    list_embeddings,
    list_invocations,
    list_persistence_diagrams,
    list_runs,
)
from trajectory_tracer.engine import (
    compute_embedding,
    compute_persistence_diagram,
    get_output_hash,
    perform_experiment,
    process_run_generators,
    run_generator,
)
from trajectory_tracer.schemas import (
    Embedding,
    ExperimentConfig,
    Invocation,
    PersistenceDiagram,
    Run,
)


def test_get_output_hash():
    """Test that get_output_hash correctly hashes different types of output."""

    # Test with string output
    text = "Hello, world!"
    text_hash = get_output_hash(text)
    expected_text_hash = hashlib.sha256(text.encode()).hexdigest()
    assert text_hash == expected_text_hash

    # Test with image output
    image = Image.new("RGB", (50, 50), color="blue")
    image_hash = get_output_hash(image)

    # Verify image hash by recreating it
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=30)
    expected_image_hash = hashlib.sha256(buffer.getvalue()).hexdigest()
    assert image_hash == expected_image_hash

    # Test with other type (e.g., integer)
    num = 42

    num_hash = get_output_hash(num)
    expected_num_hash = hashlib.sha256(str(num).encode()).hexdigest()
    assert num_hash == expected_num_hash


def test_run_generator(db_session: Session):
    """Test that run_generator correctly generates a sequence of invocations."""

    # Create a test run
    run = Run(
        network=["DummyT2I", "DummyI2T"],
        initial_prompt="Test prompt",
        seed=42,
        max_length=3,
    )
    db_session.add(run)
    db_session.commit()
    db_session.refresh(run)

    # Get the SQLite connection string from the session
    db_url = str(db_session.get_bind().engine.url)

    # Call the generator function with the same DB that db_session is using
    gen_ref = run_generator.remote(str(run.id), db_url)

    # Get the DynamicObjectRefGenerator object
    ref_generator = ray.get(gen_ref)

    # Get the invocation IDs by iterating through the DynamicObjectRefGenerator
    invocation_ids = []
    # Iterate directly over the DynamicObjectRefGenerator
    for invocation_id_ref in ref_generator:
        # Each item is an ObjectRef containing an invocation ID
        invocation_id = ray.get(invocation_id_ref)
        invocation_ids.append(invocation_id)
        if len(invocation_ids) >= 3:
            break

    # Verify we got the expected number of invocations
    assert len(invocation_ids) == 3

    # Verify the invocations are in the database with the right sequence numbers
    for i, invocation_id in enumerate(invocation_ids):
        # Convert string UUID to UUID object if needed
        if isinstance(invocation_id, str):
            invocation_id = UUID(invocation_id)

        invocation = db_session.get(Invocation, invocation_id)
        assert invocation is not None
        assert invocation.run_id == run.id
        assert invocation.sequence_number == i


def test_run_generator_duplicate_detection(db_session: Session):
    """Test that run_generator correctly detects and stops on duplicate outputs."""

    # Create a test run with network that will produce duplicates
    # Since DummyT2I always produces the same output for the same seed,
    # we should get a duplicate when the cycle repeats
    run = Run(
        network=["DummyT2I", "DummyI2T"],
        initial_prompt="Test prompt for duplication",
        seed=123,  # Use a fixed seed to ensure deterministic outputs
        max_length=10,  # Set higher than expected to verify early termination
    )
    db_session.add(run)
    db_session.commit()
    db_session.refresh(run)

    # Get the SQLite connection string from the session
    db_url = str(db_session.get_bind().engine.url)

    # Call the generator function
    gen_ref = run_generator.remote(str(run.id), db_url)
    ref_generator = ray.get(gen_ref)

    # Get all invocation IDs
    invocation_ids = []
    for invocation_id_ref in ref_generator:
        invocation_id = ray.get(invocation_id_ref)
        invocation_ids.append(invocation_id)

    # We expect 4 invocations before detecting a duplicate:
    # 1. DummyT2I (produces image A)
    # 2. DummyI2T (produces text B)
    # 3. DummyT2I (produces image A again - should be detected as duplicate)
    # 4. DummyI2T (this should not be executed due to duplicate detection)
    assert len(invocation_ids) == 3, (
        f"Expected 3 invocations but got {len(invocation_ids)}"
    )

    # Verify the invocations in the database
    for i, invocation_id in enumerate(invocation_ids):
        if isinstance(invocation_id, str):
            invocation_id = UUID(invocation_id)

        invocation = db_session.get(Invocation, invocation_id)
        assert invocation is not None
        assert invocation.run_id == run.id
        assert invocation.sequence_number == i

        # Check the model pattern matches our expectation
        expected_model = run.network[i % len(run.network)]
        assert invocation.model == expected_model


def test_compute_embedding(db_session: Session):
    """Test that compute_embedding correctly computes an embedding for an invocation."""

    # Create a test run
    run = Run(
        network=["DummyT2I"],
        initial_prompt="Test prompt for embedding",
        seed=42,
        max_length=1,
    )
    db_session.add(run)
    db_session.commit()
    db_session.refresh(run)

    # Create an invocation
    invocation = Invocation(
        model="DummyT2I",
        type="image",
        run_id=run.id,
        sequence_number=0,
        seed=42,
    )
    db_session.add(invocation)
    db_session.commit()
    db_session.refresh(invocation)

    # Generate output for the invocation
    image = Image.new("RGB", (50, 50), color="red")
    invocation.output = image
    db_session.add(invocation)
    db_session.commit()

    # Get the SQLite connection string from the session
    db_url = str(db_session.get_bind().engine.url)

    # Call the compute_embedding function
    embedding_id_ref = compute_embedding.remote(str(invocation.id), "Dummy", db_url)
    embedding_id = ray.get(embedding_id_ref)

    # Convert string UUID to UUID object
    embedding_uuid = UUID(embedding_id)

    # Verify the embedding is in the database
    embedding = db_session.get(Embedding, embedding_uuid)
    assert embedding is not None
    assert embedding.invocation_id == invocation.id
    assert embedding.embedding_model == "Dummy"
    assert embedding.vector is not None
    assert len(embedding.vector) > 0  # Vector should not be empty

    # Verify the embedding has start and complete timestamps
    assert embedding.started_at is not None
    assert embedding.completed_at is not None
    assert embedding.completed_at > embedding.started_at


def test_compute_persistence_diagram(db_session: Session):
    """Test that compute_persistence_diagram correctly computes a persistence diagram for a run."""
    # Create a test run
    run = Run(
        network=["DummyT2I", "DummyI2T"],
        initial_prompt="Test prompt for persistence diagram",
        seed=42,
        max_length=3,
    )
    db_session.add(run)
    db_session.commit()
    db_session.refresh(run)

    # Create multiple invocations for the run
    for i in range(3):
        invocation = Invocation(
            model=run.network[i % len(run.network)],
            type="image" if i % 2 == 0 else "text",
            run_id=run.id,
            sequence_number=i,
            seed=42,
        )
        db_session.add(invocation)
        db_session.commit()
        db_session.refresh(invocation)

        # Generate output for the invocation
        if i % 2 == 0:
            output = Image.new(
                "RGB", (50, 50), color=f"rgb({i * 20}, {i * 30}, {i * 40})"
            )
        else:
            output = f"Test output for invocation {i}"

        invocation.output = output
        db_session.add(invocation)
        db_session.commit()

        # Compute embedding for the invocation
        embedding = Embedding(
            invocation_id=invocation.id,
            embedding_model="Dummy",
            vector=None,
        )
        db_session.add(embedding)
        db_session.commit()
        db_session.refresh(embedding)

        # Set embedding vector (different for each invocation)

        embedding.vector = np.array(
            [float(i), float(i + 1), float(i + 2)], dtype=np.float32
        )
        embedding.started_at = datetime.now()
        embedding.completed_at = datetime.now()
        db_session.add(embedding)
        db_session.commit()

    # Get the SQLite connection string from the session
    db_url = str(db_session.get_bind().engine.url)

    # Call the compute_persistence_diagram function
    pd_id_ref = compute_persistence_diagram.remote(str(run.id), "Dummy", db_url)
    pd_id = ray.get(pd_id_ref)

    # Convert string UUID to UUID object

    pd_uuid = UUID(pd_id)

    # Verify the persistence diagram is in the database
    pd = db_session.get(PersistenceDiagram, pd_uuid)
    assert pd is not None
    assert pd.run_id == run.id
    assert pd.embedding_model == "Dummy"
    assert pd.generators is not None

    # Verify the persistence diagram has start and complete timestamps
    assert pd.started_at is not None
    assert pd.completed_at is not None
    assert pd.completed_at > pd.started_at

    # We should have at least some generators with birth-death pairs
    assert len(pd.generators) > 0


def test_process_run_generators(db_session: Session):
    """Test that process_run_generators correctly processes multiple run generators."""
    # Create multiple test runs
    runs = []
    for i in range(3):
        run = Run(
            network=["DummyT2I", "DummyI2T"],
            initial_prompt=f"Test prompt {i}",
            seed=42 + i,
            max_length=2,
        )
        db_session.add(run)
        db_session.commit()
        db_session.refresh(run)
        runs.append(run)

    run_ids = [str(run.id) for run in runs]

    # Get the SQLite connection string from the session
    db_url = str(db_session.get_bind().engine.url)

    # Call the process_run_generators function

    invocation_ids = process_run_generators(run_ids, db_url)

    # We expect 2 invocations for each run
    assert len(invocation_ids) == 6

    # Verify all invocations are in the database
    for invocation_id in invocation_ids:
        if isinstance(invocation_id, str):
            invocation_id = UUID(invocation_id)

        invocation = db_session.get(Invocation, invocation_id)
        assert invocation is not None
        assert invocation.run_id in [run.id for run in runs]


def test_perform_experiment(db_session: Session):
    """Test that perform_experiment correctly executes an experiment with multiple runs."""
    # Create a test experiment config with multiple embedding models and a -1 seed
    config = ExperimentConfig(
        networks=[["DummyT2I", "DummyI2T"]],
        seeds=[42, 43, -1],
        prompts=["Test prompt A", "Test prompt B"],
        embedding_models=["Dummy", "Dummy2"],  # Added second embedding model
        max_length=5,  # Increased max length (especially for -1 seed)
    )

    # Get the SQLite connection string from the session
    db_url = str(db_session.get_bind().engine.url)

    # Call the perform_experiment function
    perform_experiment(config, db_url)

    # We should have 3*2*1 = 6 runs (3 seeds, 2 prompts, 1 network)
    runs = list_runs(db_session)
    assert len(runs) == 6

    # Total invocations will depend on the runs
    # For -1 seed runs, we should have exactly max_length invocations (5 each)
    # Regular seeds (42, 43) might have fewer than max_length if duplicates are detected
    invocations = list_invocations(db_session)
    # At minimum, we should have 2 runs with -1 seed * max_length (5) = 10 invocations
    # Plus some invocations from the other 4 runs (at least 3 each) = at least 22 total
    assert len(invocations) >= 22

    # Each invocation should have 2 embeddings (one for each embedding model)
    embeddings = list_embeddings(db_session)
    assert len(embeddings) == len(invocations) * 2

    # We should have 6 runs * 2 embedding models = 12 persistence diagrams
    pds = list_persistence_diagrams(db_session)
    assert len(pds) == 12

    # Verify all persistence diagrams have generators
    for pd in pds:
        assert pd.generators is not None
        assert len(pd.generators) > 0
