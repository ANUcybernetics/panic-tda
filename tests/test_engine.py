import hashlib
import io

import ray
from PIL import Image
from sqlmodel import Session
from uuid_v7.base import uuid7

from trajectory_tracer.engine import (
    create_invocation,
    run_generator,
)
from trajectory_tracer.schemas import (
    InvocationType,
    Run,
)


def test_create_image_to_text_invocation(db_session: Session):
    """Test that create_invocation correctly initializes an invocation object with image input."""
    run_id = uuid7()
    # Create a test image as input instead of text
    image_input = Image.new("RGB", (100, 100), color="red")
    model = "DummyI2T"  # DummyI2T is an Image-to-Text model
    sequence_number = 0
    seed = 12345

    image_to_text_invocation = create_invocation(
        model=model,
        input=image_input,
        run_id=run_id,
        sequence_number=sequence_number,
        session=db_session,
        seed=seed,
    )

    assert image_to_text_invocation.model == model
    assert image_to_text_invocation.type == InvocationType.TEXT
    assert image_to_text_invocation.run_id == run_id
    assert image_to_text_invocation.sequence_number == sequence_number
    assert image_to_text_invocation.seed == seed
    assert image_to_text_invocation.input_invocation_id is None
    assert image_to_text_invocation.output is None
    assert (
        image_to_text_invocation.id is not None
    )  # Should have an ID since it was saved to DB


def test_get_output_hash():
    """Test that get_output_hash correctly hashes different types of output."""

    from trajectory_tracer.engine import get_output_hash

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

    # Get the generator object
    gen_obj = ray.get(gen_ref)

    # Get the first few invocation IDs
    invocation_ids = []
    for _ in range(3):
        try:
            invocation_id = next(gen_obj)
            invocation_ids.append(invocation_id)
        except StopIteration:
            break

    # Verify we got the expected number of invocations
    assert len(invocation_ids) == 3

    # Verify the invocations are in the database with the right sequence numbers
    for i, invocation_id in enumerate(invocation_ids):
        invocation = db_session.get(InvocationType, invocation_id)
        assert invocation is not None
        assert invocation.run_id == run.id
        assert invocation.sequence_number == i
