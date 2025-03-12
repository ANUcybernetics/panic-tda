from PIL import Image
from sqlmodel import Session, select
from uuid_v7.base import uuid7

from trajectory_tracer.engine import (
    create_invocation,
    create_run,
    embed_invocation,
    embed_run,
    perform_invocation,
    perform_run,
)
from trajectory_tracer.schemas import Invocation, InvocationType, Run


def test_create_text_invocation(db_session: Session):
    """Test that create_invocation correctly initializes an invocation object with text input."""
    run_id = uuid7()
    text_input = "A test prompt"
    model = "DummyT2I"
    sequence_number = 0
    seed = 12345

    text_invocation = create_invocation(
        model=model,
        input=text_input,
        run_id=run_id,
        sequence_number=sequence_number,
        session=db_session,
        seed=seed
    )

    assert text_invocation.model == model
    assert text_invocation.type == "text"
    assert text_invocation.run_id == run_id
    assert text_invocation.sequence_number == sequence_number
    assert text_invocation.seed == seed
    assert text_invocation.input_invocation_id is None
    assert text_invocation.output is None
    assert text_invocation.id is not None  # Should have an ID since it was saved to DB


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
        else:
            assert invocation.model == "DummyI2T"

def test_embed_invocation(db_session: Session):
    """Test that embed_invocation correctly generates and associates an embedding with an invocation."""
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
    invocation.output = "This is test text for embedding"
    db_session.add(invocation)
    db_session.commit()
    db_session.refresh(invocation)

    # Create an embedding for the invocation
    embedding = embed_invocation(invocation, "Dummy", db_session)

    # Check that the embedding was created correctly
    assert embedding is not None
    assert embedding.invocation_id == invocation.id
    assert embedding.embedding_model == "dummy-embedding"
    assert embedding.vector is not None
    assert embedding.dimension == 768  # dummy embeddings are 768-dimensional
    assert embedding.started_at is not None  # Check that started_at is set
    assert embedding.completed_at is not None  # Check that completed_at is set
    assert embedding.completed_at >= embedding.started_at  # Completion should be after or equal to start

    # Verify the relationship is established correctly
    db_session.refresh(invocation)
    assert len(invocation.embeddings) == 1
    assert invocation.embeddings[0].id == embedding.id


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
    invocation.output = "This is test text for multiple embeddings"
    db_session.add(invocation)
    db_session.commit()
    db_session.refresh(invocation)

    # Create first embedding using Dummy
    embedding1 = embed_invocation(invocation, "Dummy", db_session)

    # Create second embedding using Dummy2
    embedding2 = embed_invocation(invocation, "Dummy2", db_session)

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
    assert "dummy-embedding" in embedding_models
    assert "dummy2-embedding" in embedding_models

    # Verify each embedding has correct properties
    for embedding in invocation.embeddings:
        assert embedding.invocation_id == invocation.id
        assert embedding.vector is not None
        assert embedding.dimension == 768
        assert embedding.started_at is not None
        assert embedding.completed_at is not None
        assert embedding.completed_at >= embedding.started_at


def test_embed_run(db_session: Session):
    """Test that embed_run correctly generates embeddings for all invocations in a run."""

    # Create a test run
    network = ["DummyT2I", "DummyI2T"]
    initial_prompt = "Test prompt for embedding run"
    seed = 42

    # Create and perform the run
    run = create_run(network=network, initial_prompt=initial_prompt, run_length=6, session=db_session, seed=seed)
    db_session.add(run)
    db_session.commit()
    db_session.refresh(run)

    run = perform_run(run, db_session)

    # Verify run was created with correct number of invocations
    assert len(run.invocations) == 6

    # Create embeddings for all invocations
    embeddings = embed_run(run, "Dummy", db_session)

    # Check that embeddings were created for each invocation
    assert len(embeddings) == 6

    # Verify each embedding has timestamps
    for embedding in embeddings:
        assert embedding.started_at is not None
        assert embedding.completed_at is not None
        assert embedding.completed_at >= embedding.started_at

    # Verify each invocation has an embedding
    for invocation in run.invocations:
        assert len(invocation.embeddings) == 1
        assert invocation.embeddings[0].embedding_model == "dummy-embedding"
        assert invocation.embeddings[0].dimension == 768
        assert invocation.embeddings[0].started_at is not None
        assert invocation.embeddings[0].completed_at is not None
        assert invocation.embeddings[0].completed_at >= invocation.embeddings[0].started_at


def test_perform_experiment(db_session: Session):
    """Test that a small experiment with multiple runs can be executed successfully."""
    from trajectory_tracer.engine import perform_experiment
    from trajectory_tracer.schemas import ExperimentConfig

    # Create a small experiment configuration
    config = ExperimentConfig(
        networks=[
            ["DummyT2I", "DummyI2T"],
            ["DummyI2T", "DummyT2I"]
        ],
        seeds=[42, 43],
        prompts=["Test prompt 1", "Test prompt 2"],
        embedders=["Dummy", "Dummy2"],
        run_length=10 # Short run length for testing
    )

    # Perform the experiment
    perform_experiment(config, db_session)

    # Query the database to verify created objects

    # Check that runs were created
    runs = db_session.exec(select(Run)).all()
    assert len(runs) == 16  # 2 networks × 2 seeds × 2 prompts × 2 embedders = 16 runs

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

            # Check that each invocation has an embedding
            assert len(invocation.embeddings) == 1

            # Check embedding properties
            for embedding in invocation.embeddings:
                assert embedding.embedding_model in ["dummy-embedding", "dummy2-embedding"]
                assert embedding.vector is not None
                assert embedding.dimension == 768
                assert embedding.started_at is not None
                assert embedding.completed_at is not None
                assert embedding.completed_at >= embedding.started_at
