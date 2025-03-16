import json

from PIL import Image
from sqlmodel import Session

from trajectory_tracer.engine import create_run, perform_run
from trajectory_tracer.genai_models import IMAGE_SIZE
from trajectory_tracer.utils import export_run_images


def test_export_run_images(db_session: Session, tmp_path):
    """Test that export_run_images correctly exports images from a run to a directory."""
    # Create a test run with dummy models that will generate images
    network = ["DummyT2I", "DummyI2T"]
    initial_prompt = "Test prompt for image export"
    seed = 42

    # Create and execute the run
    run = create_run(network=network, initial_prompt=initial_prompt, run_length=6, session=db_session, seed=seed)
    db_session.add(run)
    db_session.commit()
    db_session.refresh(run)

    run = perform_run(run, db_session)

    # The run should have image outputs from the DummyT2I model
    # at sequence numbers 0, 2, 4

    # Create a temporary output directory
    output_dir = tmp_path / "test_images"

    # Export the images
    export_run_images(run, db_session, output_dir=str(output_dir))

    # Check that image files were created
    image_files = list(output_dir.glob("*.jpg"))

    # We should have image outputs from DummyT2I (at positions 0, 2, 4)
    assert len(image_files) == 3

    # Verify each image file exists and is a valid image
    for image_file in image_files:
        assert image_file.exists()
        # Try opening the image to make sure it's valid
        img = Image.open(image_file)
        assert img.size == (IMAGE_SIZE, IMAGE_SIZE)  # Check it matches our IMAGE_SIZE constant

        # Get EXIF data from the image
        exif_data = img.getexif()
        assert 0x9286 in exif_data  # UserComment tag should exist

        # Parse metadata from EXIF
        metadata_bytes = exif_data[0x9286]
        metadata = json.loads(metadata_bytes.decode('utf-8'))

        # Verify metadata content
        assert "prompt" in metadata
        assert "model" in metadata
        assert "sequence_number" in metadata
        assert "seed" in metadata

        # Verify specific metadata values
        assert metadata["seed"] == str(seed)
        assert metadata["model"] == "DummyT2I"  # Image invocations come from DummyT2I in this test
