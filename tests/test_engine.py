from uuid import uuid4

from PIL import Image

from trajectory_tracer.engine import create_invocation


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
