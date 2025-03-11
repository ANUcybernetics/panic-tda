from uuid import UUID

from PIL import Image

from trajectory_tracer.schemas import Invocation, InvocationType


def test_create_initial_invocation():

    initial_invocation = Invocation(
        model="DummyT2I",
        type=InvocationType.IMAGE,
        seed=12345
    )

    initial_invocation.output=Image.new('RGB', (100, 100), color='red')

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
        completed_at=end_time
    )

    # Test that duration is calculated correctly
    assert invocation.duration == (end_time - start_time).total_seconds()
    assert invocation.duration >= 5.0  # Should be around 5 seconds
    assert invocation.duration < 6.0   # With some tolerance for test execution

    # Test with real time delay
    invocation = Invocation(
        model="DummyModel",
        type=InvocationType.TEXT,
        seed=12345
    )

    invocation.started_at = datetime.now()
    time.sleep(0.1)  # Small delay
    invocation.completed_at = datetime.now()

    assert invocation.duration > 0.0
    assert invocation.duration < 0.5  # Should be small but positive
