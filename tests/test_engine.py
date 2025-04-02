import hashlib
import io
import itertools
import os
import tempfile
from datetime import datetime
from uuid import UUID

import numpy as np
import pytest
import ray
from PIL import Image
from sqlmodel import Session, SQLModel, create_engine

from trajectory_tracer.db import (
    create_db_and_tables,
    list_embeddings,
    list_invocations,
    list_persistence_diagrams,
    list_runs,
)
from trajectory_tracer.embeddings import get_actor_class as get_embedding_actor_class
from trajectory_tracer.engine import (
    compute_embeddings,
    compute_persistence_diagram,
    get_output_hash,
    init_runs,
    perform_embeddings_stage,
    perform_experiment,
    perform_pd_stage,
    perform_runs_stage,
    run_generator,
)
from trajectory_tracer.genai_models import get_actor_class as get_genai_actor_class
from trajectory_tracer.genai_models import get_output_type, list_models
from trajectory_tracer.schemas import (
    Embedding,
    ExperimentConfig,
    Invocation,
    InvocationType,
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

    # Generate combinations
    combinations = list(
        itertools.product(
            config.networks,
            config.seeds,
            config.prompts,
        )
    )

    # Get the SQLite connection string from the session
    db_url = str(db_session.get_bind().engine.url)

    # Call init_runs
    run_groups = init_runs(experiment_id, combinations, config, db_url)

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
        network=["DummyT2I"],
        initial_prompt="Test prompt for embedding",
        seed=42,
        max_length=2,
    )
    db_session.add(run)
    db_session.commit()
    db_session.refresh(run)

    # Create two invocations
    invocations = []
    for i in range(2):
        invocation = Invocation(
            model="DummyT2I",
            type="image",
            run_id=run.id,
            sequence_number=i,
            seed=42,
        )
        db_session.add(invocation)
        db_session.commit()
        db_session.refresh(invocation)

        # Generate output for the invocation
        image = Image.new("RGB", (50, 50), color=f"rgb({i * 50}, {i * 50}, {i * 50})")
        invocation.output = image
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
    embedding_ids_ref = compute_embeddings.remote(actor, invocation_ids, embedding_model, db_url)
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
        network=["DummyT2I"],
        initial_prompt="Test prompt for embedding skipping",
        seed=42,
        max_length=3,
    )
    db_session.add(run)
    db_session.commit()
    db_session.refresh(run)

    # Create three invocations
    invocations = []
    for i in range(3):
        invocation = Invocation(
            model="DummyT2I",
            type="image",
            run_id=run.id,
            sequence_number=i,
            seed=42,
        )
        db_session.add(invocation)
        db_session.commit()
        db_session.refresh(invocation)

        # Generate output for the invocation
        image = Image.new("RGB", (50, 50), color=f"rgb({i * 50}, {i * 50}, {i * 50})")
        invocation.output = image
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
    embedding_ids_ref = compute_embeddings.remote(actor, invocation_ids, embedding_model, db_url)
    embedding_ids = ray.get(embedding_ids_ref)

    # Verify we got the right number of embeddings back (3 total)
    assert len(embedding_ids) == 3

    # Verify the first embedding ID matches our existing embedding
    assert str(existing_embedding.id) in embedding_ids

    # Verify only 2 new embeddings were actually computed
    # (by checking embeddings for invocations 1 and 2 are different from the existing one)
    new_embedding_ids = [eid for eid in embedding_ids if eid != str(existing_embedding.id)]
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

        # Generate output
        if i % 2 == 0:
            output = Image.new("RGB", (50, 50), color=f"rgb({i * 20}, {i * 30}, {i * 40})")
        else:
            output = f"Test output for invocation {i}"

        invocation.output = output
        db_session.add(invocation)
        db_session.commit()

        invocation_ids.append(str(invocation.id))

    # Get the SQLite connection string from the session
    db_url = str(db_session.get_bind().engine.url)

    # Call the perform_embeddings_stage function with two embedding models
    embedding_models = ["Dummy", "Dummy2"]
    embedding_ids = perform_embeddings_stage(invocation_ids, embedding_models, db_url, num_actors=2)

    # We expect 3 invocations * 2 embedding models = 6 embeddings
    assert len(embedding_ids) == 6

    # Verify all embeddings are in the database
    for embedding_id in embedding_ids:
        if isinstance(embedding_id, str):
            embedding_id = UUID(embedding_id)

        embedding = db_session.get(Embedding, embedding_id)
        assert embedding is not None
        assert str(embedding.invocation_id) in invocation_ids
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
                output = Image.new("RGB", (50, 50), color=f"rgb({i * 20}, {i * 30}, {i * 40})")
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
                    vector=np.array([float(i), float(i + 1), float(i + 2)], dtype=np.float32),
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
        seeds=[42, 43, -1],
        prompts=["Test prompt A", "Test prompt B"],
        embedding_models=["Dummy", "Dummy2"],  # Added second embedding model
        max_length=5,  # Increased max length (especially for -1 seed)
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

    # We should have 3*2*1 = 6 runs (3 seeds, 2 prompts, 1 network)
    runs = list_runs(db_session)
    assert len(runs) == 6

    # Verify that all runs are linked to the experiment
    for run in runs:
        assert run.experiment_id == experiment_id

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

    # Verify all persistence diagrams have diagram_data
    for pd in pds:
        assert pd.diagram_data is not None
        assert "dgms" in pd.diagram_data
        assert len(pd.diagram_data["dgms"]) > 0


@pytest.mark.slow
def test_perform_experiment_real_models(db_session: Session):
    """Test that perform_experiment correctly executes an experiment with real models."""
    # Create a test experiment config with multiple embedding models and a -1 seed
    config = ExperimentConfig(
        networks=[["FluxDev", "Moondream"]],
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

    # Each invocation should have 1 embedding (one embedding model)
    embeddings = list_embeddings(db_session)
    assert len(embeddings) == len(invocations)

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
    # We should have 2 runs with -1 seed * max_length (100) = 200 invocations
    assert len(invocations) == 20

    # Each invocation should have 2 embeddings (two embedding models)
    embeddings = list_embeddings(db_session)
    assert len(embeddings) == len(invocations) * 2

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
        max_length=2,
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

    # For -1 seed run with max_length 2, we should have exactly 2 invocations
    invocations = list_invocations(db_session)
    assert len(invocations) == 2

    # Each invocation should have 1 embedding (one embedding model)
    embeddings = list_embeddings(db_session)
    assert len(embeddings) == len(invocations)

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

            # Each invocation should have 2 embeddings (two embedding models)
            embeddings = list_embeddings(session)
            assert len(embeddings) == len(invocations) * 2

            # We should have 2 runs * 2 embedding models = 4 persistence diagrams
            pds = list_persistence_diagrams(session)
            assert len(pds) == 4

            # Verify all persistence diagrams have diagram_data
            for pd in pds:
                assert pd.diagram_data is not None
                assert "dgms" in pd.diagram_data
                assert len(pd.diagram_data["dgms"]) > 0


@pytest.mark.slow
@pytest.mark.parametrize("t2i_model,i2t_model", [
    (t2i, i2t) for t2i in [m for m in list_models() if get_output_type(m) == InvocationType.IMAGE]
    for i2t in [m for m in list_models() if get_output_type(m) == InvocationType.TEXT]
])
def test_model_combination(t2i_model, i2t_model, db_session: Session):
    """Test that a specific combination of T2I and I2T models works correctly in a run."""

    # Create a network with this T2I and I2T model
    network = [t2i_model, i2t_model]

    # Create a run with a short max_length
    run = Run(
        network=network,
        initial_prompt=f"Testing network {t2i_model}->{i2t_model}",
        seed=-1,  # Use random seed
        max_length=4
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

    # Verify we got embeddings
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
