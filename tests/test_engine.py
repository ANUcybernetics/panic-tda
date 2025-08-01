import hashlib
import io
import os
import tempfile
from datetime import datetime
from uuid import UUID

import numpy as np
import pytest
import ray
from PIL import Image
from sqlmodel import Session, SQLModel, create_engine, select

from panic_tda.db import (
    create_db_and_tables,
    list_embeddings,
    list_invocations,
    list_persistence_diagrams,
    list_runs,
)
from panic_tda.embeddings import get_actor_class as get_embedding_actor_class
from panic_tda.engine import (
    compute_embeddings,
    compute_persistence_diagram,
    experiment_doctor,
    get_output_hash,
    init_runs,
    perform_embeddings_stage,
    perform_experiment,
    perform_pd_stage,
    perform_runs_stage,
    run_generator,
)
from panic_tda.genai_models import get_actor_class as get_genai_actor_class
from panic_tda.genai_models import get_output_type, list_models
from panic_tda.schemas import (
    Embedding,
    ExperimentConfig,
    Invocation,
    InvocationType,
    PersistenceDiagram,
    Run,
)


@pytest.fixture(scope="module")
def genai_model_actors():
    """Module-scoped fixture for GenAI model actors."""
    actors = {}
    for model_name in list_models():
        actor_class = get_genai_actor_class(model_name)
        actors[model_name] = actor_class.remote()
    yield actors
    # Cleanup
    for actor in actors.values():
        ray.kill(actor)


@pytest.fixture(scope="module")
def embedding_model_actors():
    """Module-scoped fixture for embedding model actors."""
    from panic_tda.embeddings import list_models as list_embedding_models

    actors = {}
    for model_name in list_embedding_models():
        actor_class = get_embedding_actor_class(model_name)
        actors[model_name] = actor_class.remote()
    yield actors
    # Cleanup
    for actor in actors.values():
        ray.kill(actor)


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

    # Create model actors

    # Create a dictionary mapping model names to their actors
    model_actors = {}
    for model_name in ["DummyT2I", "DummyI2T"]:
        actor_class = get_genai_actor_class(model_name)
        model_actors[model_name] = actor_class.remote()

    # Call the generator function with the same DB that db_session is using and the model actors
    gen_ref = run_generator.remote(str(run.id), db_url, model_actors)

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

    # Clean up the actor references from the model_actors dictionary
    for actor in model_actors.values():
        ray.kill(actor)


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

    # Create model actors
    model_actors = {}
    for model_name in ["DummyT2I", "DummyI2T"]:
        actor_class = get_genai_actor_class(model_name)
        model_actors[model_name] = actor_class.remote()

    # Call the generator function with the model actors
    gen_ref = run_generator.remote(str(run.id), db_url, model_actors)
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

    # Clean up the actor references from the model_actors dictionary
    for actor in model_actors.values():
        ray.kill(actor)


def test_init_runs(db_session: Session):
    """Test that init_runs correctly creates and groups runs by network."""
    # Create a test experiment config with multiple networks, seeds, and prompts
    config = ExperimentConfig(
        networks=[["DummyT2I", "DummyI2T"], ["SDXLTurbo", "Moondream"]],
        seeds=[42, 43],
        prompts=["Test prompt A", "Test prompt B"],
        embedding_models=["Dummy"],
        max_length=3,
    )

    # Save to database
    db_session.add(config)
    db_session.commit()
    db_session.refresh(config)

    experiment_id = config.id

    # Get the SQLite connection string from the session
    db_url = str(db_session.get_bind().engine.url)

    # Call init_runs
    run_groups = init_runs(experiment_id, db_url)

    # Verify we have the correct number of groups (should be 2, one per network)
    assert len(run_groups) == 2

    # Calculate expected total runs (2 networks * 2 seeds * 2 prompts = 8)
    total_expected_runs = len(config.networks) * len(config.seeds) * len(config.prompts)

    # Verify we have the correct number of runs in total
    all_run_ids = [run_id for group in run_groups for run_id in group]
    assert len(all_run_ids) == total_expected_runs

    # Verify each group contains runs with the same network
    for group in run_groups:
        # Get the first run to compare its network
        sample_run_id = group[0]
        sample_run = db_session.get(Run, UUID(sample_run_id))
        reference_network = sample_run.network

        # Verify all runs in this group have the same network
        for run_id in group:
            run = db_session.get(Run, UUID(run_id))
            assert run.network == reference_network

    # Verify all runs have the correct experiment_id
    for run_id in all_run_ids:
        run = db_session.get(Run, UUID(run_id))
        assert run.experiment_id == experiment_id
        assert run.max_length == config.max_length


def test_perform_runs_stage(db_session: Session):
    """Test that perform_runs_stage correctly processes multiple runs."""
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

    # Call the perform_runs_stage function
    invocation_ids = perform_runs_stage(run_ids, db_url)

    # We expect 2 invocations for each run
    assert len(invocation_ids) == 6

    # Verify all invocations are in the database
    for invocation_id in invocation_ids:
        if isinstance(invocation_id, str):
            invocation_id = UUID(invocation_id)

        invocation = db_session.get(Invocation, invocation_id)
        assert invocation is not None
        assert invocation.run_id in [run.id for run in runs]


def test_compute_embeddings(db_session: Session):
    """Test that compute_embeddings correctly computes embeddings for multiple invocations."""

    # Create a test run
    run = Run(
        network=["DummyI2T"],
        initial_prompt="Test prompt for embedding",
        seed=42,
        max_length=2,
    )
    db_session.add(run)
    db_session.commit()
    db_session.refresh(run)

    # Create two text invocations
    invocations = []
    for i in range(2):
        invocation = Invocation(
            model="DummyI2T",
            type="text",  # Changed to text type
            run_id=run.id,
            sequence_number=i,
            seed=42,
        )
        db_session.add(invocation)
        db_session.commit()
        db_session.refresh(invocation)

        # Generate text output for the invocation
        text_output = f"Test text output for invocation {i}"
        invocation.output = text_output
        db_session.add(invocation)
        db_session.commit()

        invocations.append(invocation)

    # Get the SQLite connection string from the session
    db_url = str(db_session.get_bind().engine.url)

    # Create an embedding actor
    embedding_model = "Dummy"
    actor = get_embedding_actor_class(embedding_model).remote()

    # Get invocation IDs as strings for the batch
    invocation_ids = [str(inv.id) for inv in invocations]

    # Call the compute_embeddings function with the batch of invocation IDs
    embedding_ids_ref = compute_embeddings.remote(
        actor, invocation_ids, embedding_model, db_url
    )
    embedding_ids = ray.get(embedding_ids_ref)

    # Verify we got the right number of embeddings back
    assert len(embedding_ids) == len(invocations)

    # Verify each embedding in the database
    for i, embedding_id in enumerate(embedding_ids):
        # Convert string UUID to UUID object if needed
        if isinstance(embedding_id, str):
            embedding_uuid = UUID(embedding_id)
        else:
            embedding_uuid = embedding_id

        # Verify the embedding is in the database
        embedding = db_session.get(Embedding, embedding_uuid)
        assert embedding is not None
        assert embedding.invocation_id == invocations[i].id
        assert embedding.embedding_model == embedding_model
        assert embedding.vector is not None
        assert len(embedding.vector) > 0  # Vector should not be empty

        # Verify the embedding has start and complete timestamps
        assert embedding.started_at is not None
        assert embedding.completed_at is not None
        assert embedding.completed_at > embedding.started_at

    # Clean up the actor
    ray.kill(actor)


def test_compute_embeddings_skips_existing(db_session: Session):
    """Test that compute_embeddings skips invocations that already have embeddings."""

    # Create a test run
    run = Run(
        network=["DummyI2T"],
        initial_prompt="Test prompt for embedding skipping",
        seed=42,
        max_length=3,
    )
    db_session.add(run)
    db_session.commit()
    db_session.refresh(run)

    # Create three text invocations
    invocations = []
    for i in range(3):
        invocation = Invocation(
            model="DummyI2T",
            type="text",  # Changed to text type
            run_id=run.id,
            sequence_number=i,
            seed=42,
        )
        db_session.add(invocation)
        db_session.commit()
        db_session.refresh(invocation)

        # Generate text output for the invocation
        text_output = f"Test text output for invocation {i}"
        invocation.output = text_output
        db_session.add(invocation)
        db_session.commit()

        invocations.append(invocation)

    # Create an embedding for the first invocation
    embedding_model = "Dummy"
    existing_embedding = Embedding(
        invocation_id=invocations[0].id,
        embedding_model=embedding_model,
        vector=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        started_at=datetime.now(),
        completed_at=datetime.now(),
    )
    db_session.add(existing_embedding)
    db_session.commit()
    db_session.refresh(existing_embedding)

    # Get the SQLite connection string from the session
    db_url = str(db_session.get_bind().engine.url)

    # Create an embedding actor
    actor = get_embedding_actor_class(embedding_model).remote()

    # Get invocation IDs as strings for the batch
    invocation_ids = [str(inv.id) for inv in invocations]

    # Call the compute_embeddings function with the batch of invocation IDs
    embedding_ids_ref = compute_embeddings.remote(
        actor, invocation_ids, embedding_model, db_url
    )
    embedding_ids = ray.get(embedding_ids_ref)

    # Verify we got the right number of embeddings back (3 total)
    assert len(embedding_ids) == 3

    # Verify the first embedding ID matches our existing embedding
    assert str(existing_embedding.id) in embedding_ids

    # Verify only 2 new embeddings were actually computed
    # (by checking embeddings for invocations 1 and 2 are different from the existing one)
    new_embedding_ids = [
        eid for eid in embedding_ids if eid != str(existing_embedding.id)
    ]
    assert len(new_embedding_ids) == 2

    # Verify each embedding in the database
    for embedding_id in embedding_ids:
        embedding_uuid = UUID(embedding_id)
        embedding = db_session.get(Embedding, embedding_uuid)
        assert embedding is not None
        assert embedding.embedding_model == embedding_model
        assert embedding.vector is not None
        assert len(embedding.vector) > 0

    # Clean up the actor
    ray.kill(actor)


def test_perform_embeddings_stage(db_session: Session):
    """Test that perform_embeddings_stage correctly processes embeddings for multiple invocations."""
    # Create a test run
    run = Run(
        network=["DummyT2I", "DummyI2T"],
        initial_prompt="Test prompt for embeddings stage",
        seed=42,
        max_length=3,
    )
    db_session.add(run)
    db_session.commit()
    db_session.refresh(run)

    # Create multiple invocations with outputs
    invocation_ids = []
    text_invocation_ids = []
    for i in range(3):
        is_text = i % 2 != 0  # Only odd indexes are text invocations
        invocation = Invocation(
            model=run.network[i % len(run.network)],
            type="text" if is_text else "image",
            run_id=run.id,
            sequence_number=i,
            seed=42,
        )
        db_session.add(invocation)
        db_session.commit()
        db_session.refresh(invocation)

        # Generate output
        if not is_text:
            output = Image.new(
                "RGB", (50, 50), color=f"rgb({i * 20}, {i * 30}, {i * 40})"
            )
        else:
            output = f"Test output for invocation {i}"
            text_invocation_ids.append(str(invocation.id))

        invocation.output = output
        db_session.add(invocation)
        db_session.commit()

        invocation_ids.append(str(invocation.id))

    # Get the SQLite connection string from the session
    db_url = str(db_session.get_bind().engine.url)

    # Call the perform_embeddings_stage function with two embedding models
    embedding_models = ["Dummy", "Dummy2"]
    embedding_ids = perform_embeddings_stage(
        invocation_ids, embedding_models, db_url, num_actors=2
    )

    # We expect only text invocations to have embeddings
    # 1 text invocation * 2 embedding models = 2 embeddings
    assert len(embedding_ids) == len(text_invocation_ids) * len(embedding_models)

    # Verify all embeddings are in the database and associated with text invocations
    for embedding_id in embedding_ids:
        if isinstance(embedding_id, str):
            embedding_id = UUID(embedding_id)

        embedding = db_session.get(Embedding, embedding_id)
        assert embedding is not None
        assert str(embedding.invocation_id) in text_invocation_ids
        assert embedding.embedding_model in embedding_models
        assert embedding.vector is not None
        assert len(embedding.vector) > 0


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
    assert pd.diagram_data is not None

    # Verify the persistence diagram has start and complete timestamps
    assert pd.started_at is not None
    assert pd.completed_at is not None
    assert pd.completed_at > pd.started_at

    # We should have at least some data in the diagram
    assert "dgms" in pd.diagram_data
    assert len(pd.diagram_data["dgms"]) > 0


def test_perform_pd_stage(db_session: Session):
    """Test that perform_pd_stage correctly computes persistence diagrams for multiple runs."""
    # Create multiple test runs
    runs = []
    for i in range(2):
        run = Run(
            network=["DummyT2I", "DummyI2T"],
            initial_prompt=f"Test prompt for PD stage {i}",
            seed=42 + i,
            max_length=3,
        )
        db_session.add(run)
        db_session.commit()
        db_session.refresh(run)
        runs.append(run)

    # For each run, create invocations and embeddings
    embedding_models = ["Dummy", "Dummy2"]

    for run in runs:
        for i in range(3):  # 3 invocations per run
            invocation = Invocation(
                model=run.network[i % len(run.network)],
                type="image" if i % 2 == 0 else "text",
                run_id=run.id,
                sequence_number=i,
                seed=run.seed,
            )
            db_session.add(invocation)
            db_session.commit()
            db_session.refresh(invocation)

            # Generate output
            if i % 2 == 0:
                output = Image.new(
                    "RGB", (50, 50), color=f"rgb({i * 20}, {i * 30}, {i * 40})"
                )
            else:
                output = f"Test output for invocation {i} in run {run.id}"

            invocation.output = output
            db_session.add(invocation)
            db_session.commit()

            # Add embeddings for each model
            for model in embedding_models:
                embedding = Embedding(
                    invocation_id=invocation.id,
                    embedding_model=model,
                    vector=np.array(
                        [float(i), float(i + 1), float(i + 2)], dtype=np.float32
                    ),
                    started_at=datetime.now(),
                    completed_at=datetime.now(),
                )
                db_session.add(embedding)
                db_session.commit()

    # Get the SQLite connection string from the session
    db_url = str(db_session.get_bind().engine.url)

    # Get run IDs
    run_ids = [str(run.id) for run in runs]

    # Call the perform_pd_stage function
    pd_ids = perform_pd_stage(run_ids, embedding_models, db_url)

    # We expect 2 runs * 2 embedding models = 4 persistence diagrams
    assert len(pd_ids) == 4

    # Verify all persistence diagrams are in the database
    for pd_id in pd_ids:
        pd_uuid = UUID(pd_id)
        pd = db_session.get(PersistenceDiagram, pd_uuid)
        assert pd is not None
        assert pd.run_id in [run.id for run in runs]
        assert pd.embedding_model in embedding_models
        assert pd.diagram_data is not None
        assert "dgms" in pd.diagram_data
        assert len(pd.diagram_data["dgms"]) > 0


def test_perform_experiment(db_session: Session):
    """Test that perform_experiment correctly executes an experiment with multiple runs."""
    # Create a test experiment config with multiple embedding models and a -1 seed
    config = ExperimentConfig(
        networks=[["DummyT2I", "DummyI2T"]],
        seeds=[-1, -1],
        prompts=["Test prompt A", "Test prompt B"],
        embedding_models=["Dummy", "Dummy2"],  # Added second embedding model
        max_length=10,  # Increased max length (especially for -1 seed)
    )

    # Get the SQLite connection string from the session
    db_url = str(db_session.get_bind().engine.url)

    # Save the config to the database first
    db_session.add(config)
    db_session.commit()
    db_session.refresh(config)

    experiment_id = config.id

    # Call the perform_experiment function with the config ID
    perform_experiment(str(experiment_id), db_url)

    # Refresh the session to see changes made by perform_experiment
    db_session.expire_all()  # Clear any cached objects

    # Verify the experiment config is stored in the database
    stored_config = db_session.get(ExperimentConfig, experiment_id)
    assert stored_config is not None
    assert stored_config.networks == config.networks
    assert stored_config.seeds == config.seeds
    assert stored_config.prompts == config.prompts
    assert stored_config.embedding_models == config.embedding_models
    assert stored_config.max_length == config.max_length
    assert stored_config.started_at is not None
    assert stored_config.completed_at is not None

    # We should have 2*2*1 = 6 runs (2 seeds, 2 prompts, 1 network)
    runs = list_runs(db_session)
    assert len(runs) == 4

    # Verify that all runs are linked to the experiment
    for run in runs:
        assert run.experiment_id == experiment_id

    invocations = list_invocations(db_session)
    assert len(invocations) == 40

    # Each text invocation should have 2 embeddings (one for each embedding model),
    # and image invocations will have no embeddings, so the numbers should be the same
    embeddings = list_embeddings(db_session)
    assert len(embeddings) == len(invocations)

    # We should have 4 runs * 2 embedding models = 8 persistence diagrams
    pds = list_persistence_diagrams(db_session)
    assert len(pds) == 8

    # Verify all persistence diagrams have diagram_data
    for pd in pds:
        assert pd.diagram_data is not None
        assert "dgms" in pd.diagram_data
        assert len(pd.diagram_data["dgms"]) > 0


def test_restart_experiment(db_session: Session):
    """Test that perform_experiment can restart and complete a partially executed experiment."""
    # Create a test experiment config with two -1 seeds
    config = ExperimentConfig(
        networks=[["DummyT2I", "DummyI2T"]],
        seeds=[-1, -1],  # Two random seeds
        prompts=["Test prompt for restart"],
        embedding_models=["Dummy"],
        max_length=10,
    )

    # Save to database
    db_session.add(config)
    db_session.commit()
    db_session.refresh(config)
    experiment_id = str(config.id)

    # First, run perform_experiment to create complete runs
    db_url = str(db_session.get_bind().engine.url)
    perform_experiment(experiment_id, db_url)

    # Refresh the session to see all changes
    db_session.expire_all()

    # Check that the experiment was completed correctly
    experiment = db_session.get(ExperimentConfig, UUID(experiment_id))
    assert experiment.started_at is not None
    assert experiment.completed_at is not None

    # We should have 2 runs now
    runs = list_runs(db_session)
    assert len(runs) == 2
    run_id_1 = runs[0].id
    run_id_2 = runs[1].id

    # Get all invocations
    invocations = list_invocations(db_session)
    assert len(invocations) == 20  # 2 runs * 10 invocations each

    # Store the original invocation IDs we'll keep for each run (first half)
    original_invocation_ids_1 = []
    original_invocation_ids_2 = []

    for inv in invocations:
        if inv.sequence_number < config.max_length // 2:
            if inv.run_id == run_id_1:
                original_invocation_ids_1.append(inv.id)
            else:
                original_invocation_ids_2.append(inv.id)

    # Store the IDs of embeddings to keep (for text invocations in the first half)
    original_embedding_ids_1 = []
    original_embedding_ids_2 = []
    text_invocation_count_1 = 0
    text_invocation_count_2 = 0

    embeddings = list_embeddings(db_session)
    for emb in embeddings:
        # Get the invocation for this embedding
        invocation = db_session.get(Invocation, emb.invocation_id)
        if invocation.sequence_number < config.max_length // 2:
            if invocation.run_id == run_id_1:
                if invocation.type == InvocationType.TEXT:
                    text_invocation_count_1 += 1
                original_embedding_ids_1.append(emb.id)
            else:
                if invocation.type == InvocationType.TEXT:
                    text_invocation_count_2 += 1
                original_embedding_ids_2.append(emb.id)

    # Verify we have at least 2 text invocations with embeddings in first half for each run
    assert text_invocation_count_1 >= 2, (
        "Need at least 2 text invocations with embeddings for persistence diagram in run 1"
    )
    assert text_invocation_count_2 >= 2, (
        "Need at least 2 text invocations with embeddings for persistence diagram in run 2"
    )

    # Delete all invocations with sequence_number >= max_length/2 for both runs
    for run_id in [run_id_1, run_id_2]:
        invocations_to_delete = db_session.exec(
            select(Invocation).where(
                Invocation.run_id == run_id,
                Invocation.sequence_number >= config.max_length // 2,
            )
        ).all()

        for inv in invocations_to_delete:
            db_session.delete(inv)

    db_session.commit()

    # For the second run, get the most recent invocation and set its output to None
    # This simulates an incomplete invocation
    highest_seq_invocation = db_session.exec(
        select(Invocation)
        .where(Invocation.run_id == run_id_2)
        .order_by(Invocation.sequence_number.desc())
    ).first()

    if highest_seq_invocation:
        highest_seq_invocation.output = None
        db_session.add(highest_seq_invocation)
        db_session.commit()
        incomplete_invocation_id = highest_seq_invocation.id

    # Delete persistence diagrams for both runs (will be recreated)
    for run_id in [run_id_1, run_id_2]:
        pds_to_delete = db_session.exec(
            select(PersistenceDiagram).where(PersistenceDiagram.run_id == run_id)
        ).all()

        for pd in pds_to_delete:
            db_session.delete(pd)

    db_session.commit()

    # Verify that only the first half of invocations remain for each run
    remaining_invocations = list_invocations(db_session)
    expected_count = config.max_length // 2 * 2  # Two runs
    assert len(remaining_invocations) == expected_count

    # Reset the experiment's completed_at to simulate an interrupted experiment
    experiment.completed_at = None
    db_session.add(experiment)
    db_session.commit()

    # Call perform_experiment again to complete the experiment
    perform_experiment(experiment_id, db_url)

    # Refresh the session to see all changes
    db_session.expire_all()

    # Check that the experiment was completed correctly
    experiment = db_session.get(ExperimentConfig, UUID(experiment_id))
    assert experiment.started_at is not None
    assert experiment.completed_at is not None

    # We should still have just 2 runs
    runs = list_runs(db_session)
    assert len(runs) == 2
    assert {runs[0].id, runs[1].id} == {run_id_1, run_id_2}

    # We should now have 20 invocations (2 runs * max_length)
    invocations = list_invocations(db_session)
    assert len(invocations) == 20

    # Verify sequence numbers are complete for each run
    for run_id in [run_id_1, run_id_2]:
        run_invocations = [inv for inv in invocations if inv.run_id == run_id]
        sequence_numbers = [inv.sequence_number for inv in run_invocations]
        assert set(sequence_numbers) == set(range(10))

    # Verify our original invocations for run 1 are still there, except the final one
    current_invocation_ids = [inv.id for inv in invocations if inv.run_id == run_id_1]
    for inv_id in original_invocation_ids_1[:-1]:
        assert inv_id in current_invocation_ids

    # For run 2, the incomplete invocation should have been replaced
    current_run2_invocations = [inv for inv in invocations if inv.run_id == run_id_2]
    assert incomplete_invocation_id in [inv.id for inv in current_run2_invocations]

    # Verify our original embeddings are still there, except for embeddings from the final invocations
    current_embedding_ids = [emb.id for emb in list_embeddings(db_session)]
    for embedding_id in original_embedding_ids_1[:-1]:
        assert embedding_id in current_embedding_ids

    for embedding_id in original_embedding_ids_2[:-1]:
        assert embedding_id in current_embedding_ids

    # We should have embeddings for all text invocations
    embeddings = list_embeddings(db_session)
    text_invocations = [inv for inv in invocations if inv.type == InvocationType.TEXT]
    assert len(embeddings) == len(text_invocations)

    # We should have 2 persistence diagrams (2 runs * 1 embedding model)
    pds = list_persistence_diagrams(db_session)
    assert len(pds) == 2

    # Verify each persistence diagram has diagram data
    for pd in pds:
        assert pd.diagram_data is not None
        assert "dgms" in pd.diagram_data
        assert len(pd.diagram_data["dgms"]) > 0


@pytest.mark.slow
def test_perform_experiment_real_models(db_session: Session):
    """Test that perform_experiment correctly executes an experiment with real models."""
    # Create a test experiment config with multiple embedding models and a -1 seed
    config = ExperimentConfig(
        networks=[["FluxSchnell", "BLIP2"]],
        seeds=[-1],
        prompts=["A real prompt, for real models"],
        embedding_models=["Nomic"],
        max_length=10,
    )
    # Get the SQLite connection string from the session
    db_url = str(db_session.get_bind().engine.url)

    # Save the config to the database first
    db_session.add(config)
    db_session.commit()
    db_session.refresh(config)

    experiment_id = str(config.id)

    # Call the perform_experiment function with the config ID
    perform_experiment(experiment_id, db_url)

    # Verify the experiment config is stored in the database
    stored_config = db_session.get(ExperimentConfig, UUID(experiment_id))
    assert stored_config is not None
    assert stored_config.networks == config.networks
    assert stored_config.seeds == config.seeds
    assert stored_config.prompts == config.prompts
    assert stored_config.embedding_models == config.embedding_models
    assert stored_config.max_length == config.max_length

    # We should have 1*1*1 = 1 run (1 seed, 1 prompt, 1 network)
    runs = list_runs(db_session)
    assert len(runs) == 1

    # Verify the run is linked to the experiment
    assert runs[0].experiment_id == UUID(experiment_id)

    # Total invocations will depend on the runs
    # For -1 seed runs, we should have exactly max_length invocations (10 each)
    invocations = list_invocations(db_session)
    # We should have 1 run with -1 seed * max_length (10) = 10 invocations
    assert len(invocations) == 10

    # Only TEXT invocations have embeddings (approximately half of all invocations)
    embeddings = list_embeddings(db_session)
    assert len(embeddings) == len(invocations) // 2

    # We should have 1 run * 1 embedding model = 1 persistence diagram
    pds = list_persistence_diagrams(db_session)
    assert len(pds) == 1

    # Verify all persistence diagrams have diagram_data
    for pd in pds:
        assert pd.diagram_data is not None
        assert "dgms" in pd.diagram_data
        assert len(pd.diagram_data["dgms"]) > 0


@pytest.mark.slow
def test_perform_experiment_real_models_2(db_session: Session):
    # Create a test experiment config with multiple embedding models and -1 seeds
    config = ExperimentConfig(
        networks=[["SDXLTurbo", "BLIP2"]],
        seeds=[-1, -1],
        prompts=["Test prompt 1"],
        embedding_models=["Nomic", "JinaClip"],
        max_length=10,
    )

    # Get the SQLite connection string from the session
    db_url = str(db_session.get_bind().engine.url)

    # Save the config to the database first
    db_session.add(config)
    db_session.commit()
    db_session.refresh(config)

    experiment_id = str(config.id)

    # Call the perform_experiment function with the config ID
    perform_experiment(experiment_id, db_url)

    # Verify the experiment config is stored in the database
    stored_config = db_session.get(ExperimentConfig, UUID(experiment_id))
    assert stored_config is not None
    assert stored_config.networks == config.networks
    assert stored_config.seeds == config.seeds
    assert stored_config.prompts == config.prompts
    assert stored_config.embedding_models == config.embedding_models
    assert stored_config.max_length == config.max_length

    # We should have 2*1*1 = 2 runs (2 seeds, 1 prompt, 1 network)
    runs = list_runs(db_session)
    assert len(runs) == 2

    # Verify all runs are linked to the experiment
    for run in runs:
        assert run.experiment_id == UUID(experiment_id)

    # For -1 seed runs, we should have exactly max_length invocations (100 each)
    invocations = list_invocations(db_session)
    # We should have 2 runs with -1 seed * max_length (10) = 20 invocations
    assert len(invocations) == 20

    # Only TEXT invocations have embeddings (half of all invocations)
    # With 2 embedding models per text invocation
    embeddings = list_embeddings(db_session)
    assert len(embeddings) == (len(invocations) // 2) * 2

    # We should have 2 runs * 2 embedding models = 4 persistence diagrams
    pds = list_persistence_diagrams(db_session)
    assert len(pds) == 4

    # Verify all persistence diagrams have diagram_data
    for pd in pds:
        assert pd.diagram_data is not None
        assert "dgms" in pd.diagram_data
        assert len(pd.diagram_data["dgms"]) > 0


@pytest.mark.slow
def test_perform_experiment_with_real_models_3(db_session: Session):
    """Test that perform_experiment correctly executes an experiment with FluxSchnell and Moondream."""
    # Create a test experiment config from test-config.json
    config = ExperimentConfig(
        networks=[["FluxSchnell", "Moondream"]],
        seeds=[-1],
        prompts=["Look on my Works, ye Mighty, and despair!"],
        embedding_models=["JinaClip"],
        max_length=4,
    )

    # Get the SQLite connection string from the session
    db_url = str(db_session.get_bind().engine.url)

    # Save the config to the database first
    db_session.add(config)
    db_session.commit()
    db_session.refresh(config)

    experiment_id = str(config.id)

    # Call the perform_experiment function with the config ID
    perform_experiment(experiment_id, db_url)

    # Verify the experiment config is stored in the database
    stored_config = db_session.get(ExperimentConfig, UUID(experiment_id))
    assert stored_config is not None
    assert stored_config.networks == config.networks
    assert stored_config.seeds == config.seeds
    assert stored_config.prompts == config.prompts
    assert stored_config.embedding_models == config.embedding_models
    assert stored_config.max_length == config.max_length

    # We should have 1*1*1 = 1 run (1 seed, 1 prompt, 1 network)
    runs = list_runs(db_session)
    assert len(runs) == 1

    # Verify the run is linked to the experiment
    assert runs[0].experiment_id == UUID(experiment_id)

    # For -1 seed run with max_length 4, we should have exactly 2 invocations
    invocations = list_invocations(db_session)
    assert len(invocations) == 4

    # Only TEXT invocations (Moondream) have embeddings (half of all invocations)
    embeddings = list_embeddings(db_session)
    assert len(embeddings) == len(invocations) // 2

    # We should have 1 run * 1 embedding model = 1 persistence diagram
    pds = list_persistence_diagrams(db_session)
    assert len(pds) == 1

    # Verify the persistence diagram has diagram_data
    pd = pds[0]
    assert pd.diagram_data is not None
    assert "dgms" in pd.diagram_data
    assert len(pd.diagram_data["dgms"]) > 0

    # Verify the invocation models match our expected network
    assert invocations[0].model == "FluxSchnell"
    assert invocations[1].model == "Moondream"


@pytest.mark.slow
def test_perform_experiment_with_file_db():
    """Test perform_experiment with a file-based database instead of in-memory."""

    # Create a temporary directory for the database file
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a file-based SQLite database path
        db_path = os.path.join(temp_dir, "test_trajectory.db")
        db_url = f"sqlite:///{db_path}"
        # Create the database and tables
        create_db_and_tables(db_url)

        # Create a test experiment config identical to test_perform_experiment_real_models_2
        config = ExperimentConfig(
            networks=[["SDXLTurbo", "BLIP2"]],
            seeds=[-1, -1],
            prompts=["Test prompt 1"],
            embedding_models=["Nomic", "JinaClip"],
            max_length=10,
        )

        # Create an engine and session to save the config
        engine = create_engine(db_url)
        SQLModel.metadata.create_all(engine)

        with Session(engine) as session:
            session.add(config)
            session.commit()
            session.refresh(config)
            experiment_id = str(config.id)

        # Call the perform_experiment function with the config ID and file-based database
        perform_experiment(experiment_id, db_url)

        # Create a new session to verify the results
        with Session(engine) as session:
            # Verify the experiment config is stored in the database
            stored_config = session.get(ExperimentConfig, UUID(experiment_id))
            assert stored_config is not None
            assert stored_config.networks == config.networks
            assert stored_config.seeds == config.seeds
            assert stored_config.prompts == config.prompts
            assert stored_config.embedding_models == config.embedding_models
            assert stored_config.max_length == config.max_length

            # We should have 2*1*1 = 2 runs (2 seeds, 1 prompt, 1 network)
            runs = list_runs(session)
            assert len(runs) == 2

            # Verify all runs are linked to the experiment
            for run in runs:
                assert run.experiment_id == UUID(experiment_id)

            # For -1 seed runs, we should have exactly max_length invocations (10 each)
            invocations = list_invocations(session)
            # We should have 2 runs with -1 seed * max_length (10) = 20 invocations
            assert len(invocations) == 20

            # Only TEXT invocations have embeddings (half of all invocations)
            # With 2 embedding models per text invocation
            embeddings = list_embeddings(session)
            assert len(embeddings) == (len(invocations) // 2) * 2

            # We should have 2 runs * 2 embedding models = 4 persistence diagrams
            pds = list_persistence_diagrams(session)
            assert len(pds) == 4

            # Verify all persistence diagrams have diagram_data
            for pd in pds:
                assert pd.diagram_data is not None
                assert "dgms" in pd.diagram_data
                assert len(pd.diagram_data["dgms"]) > 0


@pytest.mark.slow
@pytest.mark.parametrize(
    "t2i_model,i2t_model",
    [
        (t2i, i2t)
        for t2i in [
            m for m in list_models() if get_output_type(m) == InvocationType.IMAGE
        ]
        for i2t in [
            m for m in list_models() if get_output_type(m) == InvocationType.TEXT
        ]
    ],
)
def test_model_combination(t2i_model, i2t_model, db_session: Session):
    """Test that a specific combination of T2I and I2T models works correctly in a run."""

    # Create a network with this T2I and I2T model
    network = [t2i_model, i2t_model]

    # Create a run with a short max_length
    run = Run(
        network=network,
        initial_prompt=f"Testing network {t2i_model}->{i2t_model}",
        seed=-1,  # Use random seed
        max_length=4,
    )
    db_session.add(run)
    db_session.commit()
    db_session.refresh(run)
    run_id = str(run.id)

    # Get the SQLite connection string from the session
    db_url = str(db_session.get_bind().engine.url)

    # Perform the runs stage
    invocation_ids = perform_runs_stage([run_id], db_url)

    # Verify we got some invocations
    assert len(invocation_ids) > 0

    # Compute embeddings for all invocations using a simple embedding model
    embedding_model = "Dummy"
    embedding_ids = perform_embeddings_stage(invocation_ids, [embedding_model], db_url)

    # Verify we got embeddings (only for TEXT invocations)
    assert len(embedding_ids) > 0

    # Compute persistence diagrams
    pd_ids = perform_pd_stage([run_id], [embedding_model], db_url)

    # Verify we got persistence diagrams
    assert len(pd_ids) > 0

    # Verify the run was processed successfully
    run = db_session.get(Run, UUID(run_id))
    assert run is not None
    assert len(run.invocations) > 0
    assert len(run.persistence_diagrams) > 0


def test_experiment_doctor_with_fix(db_session: Session):
    """Test that experiment_doctor correctly identifies and fixes issues with an experiment."""

    # Create a test experiment config
    config = ExperimentConfig(
        networks=[["DummyT2I", "DummyI2T"]],
        seeds=[42],
        prompts=["Test prompt for doctor"],
        embedding_models=["Dummy", "Dummy2"],  # Using two embedding models
        max_length=4,  # Increased to 4 to ensure at least 2 text invocations for PD
    )

    # Save to database
    db_session.add(config)
    db_session.commit()
    db_session.refresh(config)
    experiment_id = str(config.id)

    # Create a run with missing invocations (will be detected as an issue)
    run = Run(
        network=["DummyT2I", "DummyI2T"],
        initial_prompt="Test prompt for doctor",
        seed=42,
        max_length=4,  # Increased to 4 to ensure at least 2 text invocations for PD
        experiment_id=config.id,
    )
    db_session.add(run)
    db_session.commit()
    db_session.refresh(run)
    run_id = run.id

    # Create incomplete invocations with issues
    # We need at least 2 text invocations for persistence diagrams to work
    for i in range(4):  # Create all 4 invocations to avoid run regeneration
        # Alternate between image and text
        inv_type = InvocationType.IMAGE if i % 2 == 0 else InvocationType.TEXT
        model = "DummyT2I" if i % 2 == 0 else "DummyI2T"

        invocation = Invocation(
            model=model,
            type=inv_type,
            run_id=run_id,
            sequence_number=i,
            seed=42,
        )
        db_session.add(invocation)
        db_session.commit()
        db_session.refresh(invocation)

        # Set output based on type
        if inv_type == InvocationType.IMAGE:
            image = Image.new("RGB", (50, 50), color="blue")
            invocation.output = image
        else:
            invocation.output = f"Text output {i}"

            # 2. For text invocation, create one embedding with null vector
            # and completely missing the other embedding model
            embedding = Embedding(
                invocation_id=invocation.id,
                embedding_model="Dummy",
                vector=None,  # Null vector - will be detected as an issue
                started_at=datetime.now(),
                completed_at=datetime.now(),
            )
            db_session.add(embedding)
            # We don't create an embedding for "Dummy2" at all - will be detected as missing

        db_session.add(invocation)
        db_session.commit()

    # Create a persistence diagram for an embedding model not in the experiment config
    # This should be detected and removed during the doctor run
    invalid_pd = PersistenceDiagram(
        run_id=run_id,
        embedding_model="InvalidModel",  # Not in config.embedding_models
        diagram_data={"dgms": [np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)]},
        started_at=datetime.now(),
        completed_at=datetime.now(),
    )
    db_session.add(invalid_pd)
    db_session.commit()
    db_session.refresh(invalid_pd)
    invalid_pd_id = invalid_pd.id

    # Get the SQLite connection string from the session
    db_url = str(db_session.get_bind().engine.url)

    # Call experiment_doctor with fix=True
    experiment_doctor(experiment_id, db_url, fix=True)

    # Refresh the session to see changes
    db_session.expire_all()

    # Verify issues were fixed

    # 1. Check invocations - should now have 4 invocations (already created all of them)
    run = db_session.get(Run, run_id)
    assert len(run.invocations) == 4
    assert set(inv.sequence_number for inv in run.invocations) == {0, 1, 2, 3}

    # 2. Check embeddings - all text invocations should have valid embeddings for both models
    text_invocations = [
        inv for inv in run.invocations if inv.type == InvocationType.TEXT
    ]
    assert len(text_invocations) > 0  # Make sure we have at least one text invocation

    for inv in text_invocations:
        for model in config.embedding_models:
            embedding = inv.embedding(model)
            assert embedding is not None
            assert embedding.vector is not None

    # 3. Check persistence diagrams - should have one for each embedding model
    # Now that we have 2 text invocations with proper embeddings, this should work
    assert len(run.persistence_diagrams) == len(config.embedding_models)
    embedding_models_with_diagrams = {
        pd.embedding_model for pd in run.persistence_diagrams
    }
    assert embedding_models_with_diagrams == set(config.embedding_models)

    for pd in run.persistence_diagrams:
        assert pd.diagram_data is not None

    # 4. Verify that persistence diagrams from invalid embedding models were deleted
    invalid_pd = db_session.get(PersistenceDiagram, invalid_pd_id)
    assert invalid_pd is None, (
        "Persistence diagram with invalid embedding model should be deleted"
    )
