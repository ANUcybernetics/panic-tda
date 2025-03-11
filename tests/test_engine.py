from uuid import uuid4

from PIL import Image
from sqlmodel import Session

from trajectory_tracer.embeddings import dummy
from trajectory_tracer.engine import (
    create_invocation,
    create_run,
    embed_invocation,
    embed_run,
    perform_invocation,
    perform_run,
)
from trajectory_tracer.schemas import Invocation, InvocationType


def test_create_text_invocation(db_session: Session):
    """Test that create_invocation correctly initializes an invocation object with text input."""
    run_id = uuid4()
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
    run_id = uuid4()
    image_input = Image.new('RGB', (100, 100), color='red')
    model = "DummyT2I"
    sequence_number = 1
    seed = 12345
    input_invocation_id = uuid4()

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

    run_id = uuid4()
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


def test_perform_invocation_image(db_session: Session):
    """Test that perform_invocation correctly handles image input with a dummy model."""

    run_id = uuid4()
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
        seed=seed,
        session=db_session
    )
    run.length = 10  # Override length for testing
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
        run_id=uuid4(),
        sequence_number=0,
        session=db_session,
        seed=42
    )
    invocation.output = "This is test text for embedding"
    db_session.add(invocation)
    db_session.commit()
    db_session.refresh(invocation)

    # Create an embedding for the invocation
    embedding = embed_invocation(invocation, dummy, db_session)

    # Check that the embedding was created correctly
    assert embedding is not None
    assert embedding.invocation_id == invocation.id
    assert embedding.embedding_model == "dummy-embedding"
    assert embedding.vector is not None
    assert embedding.dimension == 768  # dummy embeddings are 768-dimensional

    # Verify the relationship is established correctly
    db_session.refresh(invocation)
    assert len(invocation.embeddings) == 1
    assert invocation.embeddings[0].id == embedding.id


def test_embed_run(db_session: Session):
    """Test that embed_run correctly generates embeddings for all invocations in a run."""

    # Create a test run
    network = ["DummyT2I", "DummyI2T"]
    initial_prompt = "Test prompt for embedding run"
    seed = 42

    # Create and perform the run
    run = create_run(network=network, initial_prompt=initial_prompt, session=db_session, seed=seed)
    run.length = 6  # Override length to have multiple cycles through the network
    db_session.add(run)
    db_session.commit()
    db_session.refresh(run)

    run = perform_run(run, db_session)

    # Verify run was created with correct number of invocations
    assert len(run.invocations) == 6

    # Create embeddings for all invocations
    embeddings = embed_run(run, dummy, db_session)

    # Check that embeddings were created for each invocation
    assert len(embeddings) == 6

    # Verify each invocation has an embedding
    for invocation in run.invocations:
        assert len(invocation.embeddings) == 1
        assert invocation.embeddings[0].embedding_model == "dummy-embedding"
        assert invocation.embeddings[0].dimension == 768
