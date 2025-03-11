from uuid import uuid4

from PIL import Image
from sqlmodel import Session

from trajectory_tracer.engine import create_invocation, perform_run
from trajectory_tracer.schemas import Invocation, Run


def test_create_text_invocation():
    """Test that create_invocation correctly initializes an invocation object with text input."""
    run_id = str(uuid4())
    text_input = "A test prompt"
    model = "DummyT2I"
    sequence_number = 0
    seed = 12345

    text_invocation = create_invocation(
        model=model,
        input=text_input,
        run_id=run_id,
        sequence_number=sequence_number,
        seed=seed
    )

    assert text_invocation.model == model
    assert text_invocation.type == "text"
    assert text_invocation.run_id == run_id
    assert text_invocation.sequence_number == sequence_number
    assert text_invocation.seed == seed
    assert text_invocation.input_invocation_id is None
    assert text_invocation.output is None


def test_create_image_invocation():
    """Test that create_invocation correctly initializes an invocation object with image input."""
    run_id = str(uuid4())
    image_input = Image.new('RGB', (100, 100), color='red')
    model = "DummyT2I"
    sequence_number = 1
    seed = 12345
    input_invocation_id = str(uuid4())

    image_invocation = create_invocation(
        model=model,
        input=image_input,
        run_id=run_id,
        sequence_number=sequence_number,
        input_invocation_id=input_invocation_id,
        seed=seed
    )

    assert image_invocation.model == model
    assert image_invocation.type == "image"
    assert image_invocation.run_id == run_id
    assert image_invocation.sequence_number == sequence_number
    assert image_invocation.seed == seed
    assert image_invocation.input_invocation_id == input_invocation_id
    assert image_invocation.output is None


def test_perform_invocation_text():
    """Test that perform_invocation correctly handles text input with a dummy model."""
    from trajectory_tracer.engine import perform_invocation
    from trajectory_tracer.schemas import InvocationType

    run_id = str(uuid4())
    text_input = "A test prompt"

    # Create invocation object
    invocation = Invocation(
        model="DummyT2I",
        type=InvocationType.IMAGE,  # Changed to IMAGE since DummyT2I produces images
        run_id=run_id,
        sequence_number=0,
        seed=42
    )

    # Perform the invocation
    result = perform_invocation(invocation, text_input)

    # Check the result
    assert result.output is not None
    assert isinstance(result.output, Image.Image)


def test_perform_invocation_image():
    """Test that perform_invocation correctly handles image input with a dummy model."""
    from trajectory_tracer.engine import perform_invocation
    from trajectory_tracer.schemas import InvocationType

    run_id = str(uuid4())
    image_input = Image.new('RGB', (100, 100), color='blue')

    # Create invocation object
    invocation = Invocation(
        model="DummyI2T",
        type=InvocationType.TEXT,  # Changed to TEXT since DummyI2T produces text
        run_id=run_id,
        sequence_number=0,
        seed=42
    )

    # Perform the invocation
    result = perform_invocation(invocation, image_input)

    # Check the result
    assert result.output is not None
    assert result.output == "dummy text caption"


def test_perform_run(db_session: Session):
    """Test that perform_run correctly executes a complete run with the specified network of models."""
    # Create a simple run with DummyT2I and DummyI2T models
    network = ["DummyT2I", "DummyI2T"]
    initial_prompt = "This is a test prompt"
    seed = 42

    # Create the run - directly use SQLModel methods with session
    run = Run(
        network=network,
        initial_prompt=initial_prompt,
        seed=seed,
        length=10
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

        # Check the model alternation pattern
        if i % 2 == 0:
            assert invocation.model == "DummyT2I"
        else:
            assert invocation.model == "DummyI2T"
