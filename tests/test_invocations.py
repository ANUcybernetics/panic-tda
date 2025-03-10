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
