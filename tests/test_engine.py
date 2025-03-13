from PIL import Image
from sqlmodel import Session, select
from uuid_v7.base import uuid7

from trajectory_tracer.engine import (
    create_embedding,
    create_invocation,
    create_run,
    perform_embedding,
    perform_experiment,
    perform_invocation,
    perform_run,
)
from trajectory_tracer.schemas import ExperimentConfig, Invocation, InvocationType, Run


def test_create_image_to_text_invocation(db_session: Session):
    """Test that create_invocation correctly initializes an invocation object with image input."""
    run_id = uuid7()
    # Create a test image as input instead of text
    image_input = Image.new('RGB', (100, 100), color='red')
    model = "DummyI2T"  # DummyI2T is an Image-to-Text model
    sequence_number = 0
    seed = 12345

    image_to_text_invocation = create_invocation(
        model=model,
        input=image_input,
        run_id=run_id,
        sequence_number=sequence_number,
        session=db_session,
        seed=seed
    )

    assert image_to_text_invocation.model == model
    assert image_to_text_invocation.type == InvocationType.TEXT
    assert image_to_text_invocation.run_id == run_id
    assert image_to_text_invocation.sequence_number == sequence_number
    assert image_to_text_invocation.seed == seed
    assert image_to_text_invocation.input_invocation_id is None
    assert image_to_text_invocation.output is None
    assert image_to_text_invocation.id is not None  # Should have an ID since it was saved to DB


def test_create_image_invocation(db_session: Session):
    """Test that create_invocation correctly initializes an invocation object with image input."""
    run_id = uuid7()
    image_input = Image.new('RGB', (100, 100), color='red')
    model = "DummyT2I"
    sequence_number = 1
    seed = 12345
    input_invocation_id = uuid7()

    image_invocation = create_invocation(
        model=model,
        input=image_input,
        run_id=run_id,
        sequence_number=sequence_number,
        input_invocation_id=input_invocation_id,
        session=db_session,
        seed=seed
    )

    assert image_invocation.model == model
    assert image_invocation.type == "image"
    assert image_invocation.run_id == run_id
    assert image_invocation.sequence_number == sequence_number
    assert image_invocation.seed == seed
    assert image_invocation.input_invocation_id == input_invocation_id
    assert image_invocation.output is None
    assert image_invocation.id is not None  # Should have an ID since it was saved to DB


def test_perform_invocation_text(db_session: Session):
    """Test that perform_invocation correctly handles text input with a dummy model."""

    run_id = uuid7()
    text_input = "A test prompt"

    # Create invocation object
    invocation = Invocation(
        model="DummyT2I",
        type=InvocationType.IMAGE,  # Changed to IMAGE since DummyT2I produces images
        run_id=run_id,
        sequence_number=0,
        seed=42
    )
    db_session.add(invocation)
    db_session.commit()
    db_session.refresh(invocation)

    # Perform the invocation
    result = perform_invocation(invocation, text_input, db_session)

    # Check the result
    assert result.output is not None
    assert isinstance(result.output, Image.Image)
    assert result.started_at is not None  # Check that started_at is set
    assert result.completed_at is not None  # Check that completed_at is set
    assert result.completed_at >= result.started_at  # Completion should be after or equal to start


def test_perform_invocation_image(db_session: Session):
    """Test that perform_invocation correctly handles image input with a dummy model."""

    run_id = uuid7()
    image_input = Image.new('RGB', (100, 100), color='blue')

    # Create invocation object
    invocation = Invocation(
        model="DummyI2T",
        type=InvocationType.TEXT,  # Changed to TEXT since DummyI2T produces text
        run_id=run_id,
        sequence_number=0,
        seed=42
    )
    db_session.add(invocation)
    db_session.commit()
    db_session.refresh(invocation)

    # Perform the invocation
    result = perform_invocation(invocation, image_input, db_session)

    # Check the result
    assert result.output is not None
    assert result.output == "dummy text caption"
    assert result.started_at is not None  # Check that started_at is set
    assert result.completed_at is not None  # Check that completed_at is set
    assert result.completed_at >= result.started_at  # Completion should be after or equal to start


def test_perform_run(db_session: Session):
    """Test that perform_run correctly executes a complete run with the specified network of models."""
    # Create a simple run with DummyT2I and DummyI2T models
    network = ["DummyT2I", "DummyI2T"]
    initial_prompt = "This is a test prompt"
    seed = 42

    # Create the run using the engine function
    run = create_run(
        network=network,
        initial_prompt=initial_prompt,
        run_length=10,
        seed=seed,
        session=db_session
    )
    db_session.add(run)
    db_session.commit()
    db_session.refresh(run)

    # Perform the run
    perform_run(run, db_session)

    # Check that the run completed successfully and has the correct number of invocations
    assert len(run.invocations) == run.length

    # Verify the sequence of invocations
    for i, invocation in enumerate(run.invocations):
        assert invocation.sequence_number == i
        assert invocation.run_id == run.id
        assert invocation.seed == seed
        assert invocation.started_at is not None  # Check that started_at is set
        assert invocation.completed_at is not None  # Check that completed_at is set
        assert invocation.completed_at >= invocation.started_at  # Completion should be after or equal to start

        # Check the model alternation pattern
        if i % 2 == 0:
            assert invocation.model == "DummyT2I"
            assert invocation.type == InvocationType.IMAGE
        else:
            assert invocation.model == "DummyI2T"
            assert invocation.type == InvocationType.TEXT


def test_run_is_complete(db_session: Session):
    """Test that Run.is_complete property returns correct values."""
    # Create a run with multiple invocations
    run = create_run(
        network=["DummyT2I", "DummyI2T"],
        initial_prompt="Test prompt for completeness check",
        run_length=3,
        seed=42,
        session=db_session
    )
    db_session.add(run)
    db_session.commit()

    # Initially, run should not be complete (no invocations)
    assert not run.is_complete

    # Add first invocation (sequence 0)
    invocation0 = create_invocation(
        model="DummyT2I",
        input="Test prompt for completeness check",
        run_id=run.id,
        sequence_number=0,
        session=db_session,
        seed=42
    )
    invocation0 = perform_invocation(invocation0, "Test prompt for completeness check", db_session)
    db_session.refresh(run)

    # Run not complete with just the first invocation
    assert not run.is_complete

    # Add second invocation (sequence 1)
    invocation1 = create_invocation(
        model="DummyI2T",
        input=invocation0.output,
        run_id=run.id,
        sequence_number=1,
        input_invocation_id=invocation0.id,
        session=db_session,
        seed=42
    )
    invocation1 = perform_invocation(invocation1, invocation0.output, db_session)
    db_session.refresh(run)

    # Run not complete yet
    assert not run.is_complete

    # Add final invocation (sequence 2)
    invocation2 = create_invocation(
        model="DummyT2I",
        input=invocation1.output,
        run_id=run.id,
        sequence_number=2,
        input_invocation_id=invocation1.id,
        session=db_session,
        seed=42
    )
    invocation2 = perform_invocation(invocation2, invocation1.output, db_session)
    db_session.refresh(run)

    # Now run should be complete
    assert run.is_complete

    # Let's test that a run with final invocation but no output is not complete
    run2 = create_run(
        network=["DummyT2I", "DummyI2T"],
        initial_prompt="Another test prompt",
        run_length=2,
        seed=43,
        session=db_session
    )
    db_session.add(run2)
    db_session.commit()

    # Add invocations but don't perform the last one
    invocation_first = create_invocation(
        model="DummyT2I",
        input="Another test prompt",
        run_id=run2.id,
        sequence_number=0,
        session=db_session,
        seed=43
    )
    invocation_first = perform_invocation(invocation_first, "Another test prompt", db_session)

    # Create final invocation but don't set its output
    invocation_final = create_invocation(
        model="DummyI2T",
        input=invocation_first.output,
        run_id=run2.id,
        sequence_number=1,
        input_invocation_id=invocation_first.id,
        session=db_session,
        seed=43
    )
    db_session.add(invocation_final)
    db_session.commit()
    db_session.refresh(run2)

    # Run should not be complete because final invocation has no output
    assert not run2.is_complete


def test_create_embedding(db_session: Session):
    """Test that create_embedding correctly initializes an embedding for an invocation."""
    # Create a test invocation
    input_text = "This is test text for embedding"
    invocation = create_invocation(
        model="DummyT2I",
        input=input_text,
        run_id=uuid7(),
        sequence_number=0,
        session=db_session,
        seed=42
    )
    # Use the real perform_invocation function to get a real output
    invocation = perform_invocation(invocation, input_text, db_session)
    db_session.add(invocation)
    db_session.commit()
    db_session.refresh(invocation)

    # Import the create_embedding function from engine
    from trajectory_tracer.engine import create_embedding

    # Create an embedding for the invocation
    embedding = create_embedding("Dummy", invocation, db_session)

    # Check that the embedding was created correctly
    assert embedding is not None
    assert embedding.invocation_id == invocation.id
    assert embedding.embedding_model == "Dummy"
    assert embedding.vector is None  # Vector is not calculated yet

    # Verify the relationship is established correctly
    db_session.refresh(invocation)
    assert len(invocation.embeddings) == 1
    assert invocation.embeddings[0].id == embedding.id


def test_perform_embedding(db_session: Session):
    """Test that perform_embedding correctly calculates and stores the embedding vector."""
    # Create a test invocation
    input_text = "This is test text for embedding calculation"
    invocation = create_invocation(
        model="DummyT2I",
        input=input_text,
        run_id=uuid7(),
        sequence_number=0,
        session=db_session,
        seed=42
    )
    # Use the real perform_invocation function to get a real output
    invocation = perform_invocation(invocation, input_text, db_session)
    db_session.add(invocation)
    db_session.commit()
    db_session.refresh(invocation)

    # Import the create_embedding and perform_embedding functions from engine
    from trajectory_tracer.engine import create_embedding, perform_embedding

    # Create an empty embedding
    embedding = create_embedding("Dummy", invocation, db_session)

    # Perform the actual embedding calculation
    embedding = perform_embedding(embedding, db_session)

    # Check that the embedding vector was calculated correctly
    assert embedding.vector is not None
    assert len(embedding.vector) == 768  # dummy embeddings are 768-dimensional
    assert embedding.started_at is not None  # Check that started_at is set
    assert embedding.completed_at is not None  # Check that completed_at is set
    assert embedding.completed_at >= embedding.started_at  # Completion should be after or equal to start


def test_multiple_embeddings_per_invocation(db_session: Session):
    """Test that multiple embedding models can be used on the same invocation."""
    # Create a test invocation
    input_text = "This is test text for multiple embeddings"
    invocation = create_invocation(
        model="DummyT2I",
        input=input_text,
        run_id=uuid7(),
        sequence_number=0,
        session=db_session,
        seed=42
    )
    invocation = perform_invocation(invocation, input_text, db_session)
    db_session.add(invocation)
    db_session.commit()
    db_session.refresh(invocation)

    # Create first embedding using Dummy

    # Create first empty embedding
    embedding1 = create_embedding("Dummy", invocation, db_session)
    # Compute the actual embedding vector
    embedding1 = perform_embedding(embedding1, db_session)

    # Create second empty embedding
    embedding2 = create_embedding("Dummy2", invocation, db_session)
    # Compute the actual embedding vector
    embedding2 = perform_embedding(embedding2, db_session)

    # Refresh invocation to see updated relationships
    db_session.refresh(invocation)

    # Check that both embeddings were created correctly
    assert len(invocation.embeddings) == 2

    # Check timestamps for first embedding
    assert embedding1.started_at is not None
    assert embedding1.completed_at is not None
    assert embedding1.completed_at >= embedding1.started_at

    # Check timestamps for second embedding
    assert embedding2.started_at is not None
    assert embedding2.completed_at is not None
    assert embedding2.completed_at >= embedding2.started_at

    # Check that we have one of each embedding model type
    embedding_models = [e.embedding_model for e in invocation.embeddings]
    assert "Dummy" in embedding_models
    assert "Dummy2" in embedding_models

    # Verify each embedding has correct properties
    for embedding in invocation.embeddings:
        assert embedding.invocation_id == invocation.id
        assert embedding.vector is not None
        assert embedding.dimension == 768
        assert embedding.started_at is not None
        assert embedding.completed_at is not None
        assert embedding.completed_at >= embedding.started_at


def test_run_embeddings_by_model(db_session: Session):
    """Test that Run.embeddings_by_model returns the correct embeddings for a specific model."""
    # Create a run
    run = create_run(
        network=["DummyT2I", "DummyI2T"],
        initial_prompt="Test prompt for embeddings by model",
        run_length=3,
        seed=42,
        session=db_session
    )
    db_session.add(run)
    db_session.commit()

    # Add invocations and perform them
    invocation0 = create_invocation(
        model="DummyT2I",
        input=run.initial_prompt,
        run_id=run.id,
        sequence_number=0,
        session=db_session,
        seed=42
    )
    invocation0 = perform_invocation(invocation0, run.initial_prompt, db_session)

    invocation1 = create_invocation(
        model="DummyI2T",
        input=invocation0.output,
        run_id=run.id,
        sequence_number=1,
        input_invocation_id=invocation0.id,
        session=db_session,
        seed=42
    )
    invocation1 = perform_invocation(invocation1, invocation0.output, db_session)

    # Create embeddings with different models for each invocation
    embed1_inv0 = create_embedding("Dummy", invocation0, db_session)
    embed1_inv0 = perform_embedding(embed1_inv0, db_session)

    embed2_inv0 = create_embedding("Dummy2", invocation0, db_session)
    embed2_inv0 = perform_embedding(embed2_inv0, db_session)

    embed1_inv1 = create_embedding("Dummy", invocation1, db_session)
    embed1_inv1 = perform_embedding(embed1_inv1, db_session)

    embed2_inv1 = create_embedding("Dummy2", invocation1, db_session)
    embed2_inv1 = perform_embedding(embed2_inv1, db_session)

    db_session.refresh(run)

    # Test embeddings_by_model for first model
    dummy_embeddings = run.embeddings_by_model("Dummy")
    assert len(dummy_embeddings) == 2
    assert all(e.embedding_model == "Dummy" for e in dummy_embeddings)
    assert {e.id for e in dummy_embeddings} == {embed1_inv0.id, embed1_inv1.id}

    # Test embeddings_by_model for second model
    dummy2_embeddings = run.embeddings_by_model("Dummy2")
    assert len(dummy2_embeddings) == 2
    assert all(e.embedding_model == "Dummy2" for e in dummy2_embeddings)
    assert {e.id for e in dummy2_embeddings} == {embed2_inv0.id, embed2_inv1.id}

    # Test with a model that doesn't exist
    nonexistent_embeddings = run.embeddings_by_model("NonExistentModel")
    assert len(nonexistent_embeddings) == 0


def test_compute_missing_embeds(db_session: Session):
    """Test that compute_missing_embeds correctly processes embeddings without vectors."""
    from trajectory_tracer.engine import compute_missing_embeds

    # Create two test invocations with outputs
    invocation1 = create_invocation(
        model="DummyT2I",
        input="Text for invocation 1",
        run_id=uuid7(),
        sequence_number=0,
        session=db_session,
        seed=42
    )
    invocation1.output = "Output text for invocation 1"
    db_session.add(invocation1)

    invocation2 = create_invocation(
        model="DummyT2I",
        input="Text for invocation 2",
        run_id=uuid7(),
        sequence_number=0,
        session=db_session,
        seed=43
    )
    invocation2.output = "Output text for invocation 2"
    db_session.add(invocation2)
    db_session.commit()

    # Create embedding objects without vectors
    from trajectory_tracer.schemas import Embedding

    # First embedding with Dummy model
    embedding1 = Embedding(
        invocation_id=invocation1.id,
        embedding_model="Dummy",
        vector=None
    )
    db_session.add(embedding1)

    # Second embedding with Dummy2 model
    embedding2 = Embedding(
        invocation_id=invocation2.id,
        embedding_model="Dummy2",
        vector=None
    )
    db_session.add(embedding2)
    db_session.commit()

    # Run the function to compute missing embeddings
    processed_count = compute_missing_embeds(db_session)

    # Verify results
    assert processed_count == 2

    # Check that embeddings now have vectors
    db_session.refresh(embedding1)
    db_session.refresh(embedding2)

    assert embedding1.vector is not None
    assert embedding2.vector is not None
    assert embedding1.dimension == 768
    assert embedding2.dimension == 768
    assert embedding1.started_at is not None
    assert embedding1.completed_at is not None
    assert embedding2.started_at is not None
    assert embedding2.completed_at is not None
    assert embedding1.completed_at >= embedding1.started_at
    assert embedding2.completed_at >= embedding2.started_at


def test_perform_experiment(db_session: Session):
    """Test that a small experiment with multiple runs can be executed successfully."""
    # Create a small experiment configuration
    config = ExperimentConfig(
        networks=[
            ["DummyT2I", "DummyI2T"],
            ["DummyI2T", "DummyT2I"]
        ],
        seeds=[42, 43],
        prompts=["Test prompt 1", "Test prompt 2"],
        embedding_models=["Dummy", "Dummy2"],
        run_length=10 # Short run length for testing
    )

    # Perform the experiment
    perform_experiment(config, db_session)

    # Query the database to verify created objects

    # Check that runs were created
    statement = select(Run)
    runs = db_session.exec(statement).all()
    assert len(runs) == 8  # 2 networks × 2 seeds × 2 prompts = 8 runs

    # Check that each run has the correct properties and relationships
    for run in runs:
        # Verify run properties
        assert run.length == 10
        assert run.seed in [42, 43]
        assert run.initial_prompt in ["Test prompt 1", "Test prompt 2"]
        assert len(run.network) == 2
        assert run.network in [["DummyT2I", "DummyI2T"], ["DummyI2T", "DummyT2I"]]

        # Verify invocations for this run
        assert len(run.invocations) == 10

        # Check invocation sequence and timing
        for i, invocation in enumerate(run.invocations):
            assert invocation.sequence_number == i
            assert invocation.started_at is not None
            assert invocation.completed_at is not None
            assert invocation.completed_at >= invocation.started_at

            # Check that each invocation has embeddings for both embedding models
            assert len(invocation.embeddings) == 2  # Should have 2 embeddings, one for each model in embedding_models

            # Get the embedding models used
            embedding_models = [e.embedding_model for e in invocation.embeddings]
            assert "Dummy" in embedding_models
            assert "Dummy2" in embedding_models

            # Check embedding properties
            for embedding in invocation.embeddings:
                assert embedding.embedding_model in ["Dummy", "Dummy2"]
                assert embedding.vector is not None
                assert embedding.dimension == 768
                assert embedding.started_at is not None
                assert embedding.completed_at is not None
                assert embedding.completed_at >= embedding.started_at


def test_perform_experiment_with_test_config(db_session: Session):
    """Test that an experiment with the test-config.json configuration can be executed successfully."""
    # Create experiment configuration matching the test-config.json
    config = ExperimentConfig(
        networks=[
            ["DummyT2I", "DummyI2T"],
            ["DummyI2T", "DummyT2I"]
        ],
        seeds=[42, 43],
        prompts=["Test prompt 1", "Test prompt 2"],
        embedding_models=["Dummy", "Dummy2"],
        run_length=10
    )

    # Perform the experiment
    perform_experiment(config, db_session)

    # Query the database to verify created objects
    statement = select(Run)
    runs = db_session.exec(statement).all()

    # Check that runs were created: 2 networks × 2 seeds × 2 prompts = 8 runs
    assert len(runs) == 8

    # Check network configurations
    networks_count = {
        str(["DummyT2I", "DummyI2T"]): 0,
        str(["DummyI2T", "DummyT2I"]): 0
    }

    # Check prompts usage
    prompts_count = {
        "Test prompt 1": 0,
        "Test prompt 2": 0
    }

    # Check seeds usage
    seeds_count = {
        42: 0,
        43: 0
    }

    # Check each run's properties
    for run in runs:
        assert run.length == 10
        networks_count[str(run.network)] += 1
        prompts_count[run.initial_prompt] += 1
        seeds_count[run.seed] += 1

        # Verify invocations
        assert len(run.invocations) == 10

        # Check that all invocations have embeddings from both models
        for invocation in run.invocations:
            embedding_models = [e.embedding_model for e in invocation.embeddings]
            assert "Dummy" in embedding_models
            assert "Dummy2" in embedding_models
            assert len(invocation.embeddings) == 2

            # Make sure all embeddings have vectors
            for embedding in invocation.embeddings:
                assert embedding.vector is not None
                assert embedding.dimension == 768

    # Verify all combinations were used
    assert networks_count[str(["DummyT2I", "DummyI2T"])] == 4
    assert networks_count[str(["DummyI2T", "DummyT2I"])] == 4
    assert prompts_count["Test prompt 1"] == 4
    assert prompts_count["Test prompt 2"] == 4
    assert seeds_count[42] == 4
    assert seeds_count[43] == 4

def test_experiment_config_validation():
    """Test that ExperimentConfig validates input parameters correctly."""
    # Test with valid configuration
    valid_config = ExperimentConfig(
        networks=[["DummyT2I"]],
        seeds=[42],
        prompts=["Test prompt"],
        embedding_models=["Dummy"],
        run_length=5
    )
    assert valid_config.run_length == 5

    # Test with empty networks list
    try:
        ExperimentConfig(
            networks=[],
            seeds=[42],
            prompts=["Test prompt"],
            embedding_models=["Dummy"],
            run_length=5
        )
        assert False, "Should have raised ValueError for empty networks list"
    except ValueError:
        pass

    # Test with empty prompts
