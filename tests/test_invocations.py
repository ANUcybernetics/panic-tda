from uuid import UUID

from PIL import Image

from panic_tda.schemas import Invocation, InvocationType, Run


def test_create_initial_invocation():
    initial_invocation = Invocation(
        model="DummyT2I", type=InvocationType.IMAGE, seed=12345
    )

    initial_invocation.output = Image.new("RGB", (100, 100), color="red")

    assert initial_invocation.model == "DummyT2I"  # First model in network
    assert initial_invocation.output
    assert initial_invocation.seed == 12345
    assert initial_invocation.sequence_number == 0
    assert initial_invocation.input_invocation_id is None
    assert isinstance(initial_invocation.id, UUID)


def test_invocation_duration():
    import time
    from datetime import datetime, timedelta

    # Create invocation with different timestamps
    start_time = datetime.now() - timedelta(seconds=5)
    end_time = datetime.now()

    invocation = Invocation(
        model="DummyModel",
        type=InvocationType.TEXT,
        seed=12345,
        started_at=start_time,
        completed_at=end_time,
    )

    # Test that duration is calculated correctly
    assert invocation.duration == (end_time - start_time).total_seconds()
    assert invocation.duration >= 5.0  # Should be around 5 seconds
    assert invocation.duration < 6.0  # With some tolerance for test execution

    # Test with real time delay
    invocation = Invocation(model="DummyModel", type=InvocationType.TEXT, seed=12345)

    invocation.started_at = datetime.now()
    time.sleep(0.1)  # Small delay
    invocation.completed_at = datetime.now()

    assert invocation.duration > 0.0
    assert invocation.duration < 0.5  # Should be small but positive


def test_invocation_input_property():
    # Test case for sequence_number=0
    run = Run(
        network=["ModelA", "ModelB"],
        seed=12345,
        max_length=5,
        initial_prompt="test prompt",
    )

    initial_invocation = Invocation(
        model="ModelA", type=InvocationType.TEXT, seed=12345, sequence_number=0, run=run
    )

    # For sequence_number=0, input should be the run's initial_prompt
    assert initial_invocation.input == "test prompt"

    # Test case for sequence_number > 0
    input_invocation = Invocation(
        model="ModelB", type=InvocationType.TEXT, seed=12345, sequence_number=0, run=run
    )
    input_invocation.output = "output from previous invocation"

    second_invocation = Invocation(
        model="ModelA",
        type=InvocationType.TEXT,
        seed=12345,
        sequence_number=1,
        run=run,
        input_invocation=input_invocation,
    )

    # For sequence_number > 0, input should be the output of the previous invocation
    assert second_invocation.input == "output from previous invocation"

    # Test when input_invocation is None for sequence_number > 0
    orphan_invocation = Invocation(
        model="ModelA", type=InvocationType.TEXT, seed=12345, sequence_number=1, run=run
    )

    assert orphan_invocation.input is None


def test_invocation_input_property_with_images():
    # Create a Run
    run = Run(
        network=["ModelA", "ModelB"],
        seed=12345,
        max_length=5,
        initial_prompt="test prompt",
    )

    # Create an image input invocation
    input_invocation = Invocation(
        model="DummyT2I",
        type=InvocationType.IMAGE,
        seed=12345,
        sequence_number=0,
        run=run,
    )

    # Set an image as output
    test_image = Image.new("RGB", (100, 100), color="blue")
    input_invocation.output = test_image

    # Create a second invocation that takes the first as input
    second_invocation = Invocation(
        model="DummyI2T",
        type=InvocationType.TEXT,
        seed=12345,
        sequence_number=1,
        run=run,
        input_invocation=input_invocation,
    )

    # Test that the input property returns an image
    input_image = second_invocation.input
    assert isinstance(input_image, Image.Image)

    # Check basic properties of the image
    assert input_image.width == 100
    assert input_image.height == 100


def test_invocation_type():
    # Test TEXT type invocation
    text_invocation = Invocation(model="DummyT2T", type=InvocationType.TEXT, seed=12345)

    assert text_invocation.type == InvocationType.TEXT

    # Test IMAGE type invocation
    image_invocation = Invocation(
        model="DummyT2I", type=InvocationType.IMAGE, seed=12345
    )

    assert image_invocation.type == InvocationType.IMAGE

    # Test setting output based on type
    text_invocation.output = "text output"
    assert text_invocation.output_text == "text output"
    assert text_invocation.output_image_data is None

    test_image = Image.new("RGB", (50, 50), color="green")
    image_invocation.output = test_image
    assert image_invocation.output_text is None
    assert image_invocation.output_image_data is not None
