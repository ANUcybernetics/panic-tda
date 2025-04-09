import json
import os

from PIL import Image
from sqlmodel import Session

from trajectory_tracer.genai_models import IMAGE_SIZE
from trajectory_tracer.schemas import Invocation, InvocationType, Run
from trajectory_tracer.export import export_run_images, export_run_mosaic


def test_export_run_images(db_session: Session, tmp_path):
    """Test that export_run_images correctly exports images from a run to a directory."""
    # Create a test run with dummy models that will generate images
    network = ["DummyT2I", "DummyI2T"]
    initial_prompt = "Test prompt for image export"
    seed = 42

    # Create the run directly
    run = Run(
        network=network,
        initial_prompt=initial_prompt,
        max_length=4,
        seed=seed,
    )
    db_session.add(run)
    db_session.commit()
    db_session.refresh(run)

    # Create invocations manually instead of using perform_run
    # First invocation - DummyT2I (text to image)
    text_to_image = Invocation(
        model="DummyT2I",
        type=InvocationType.IMAGE,
        seed=seed,
        run_id=run.id,
        sequence_number=0,
    )

    # Create a test image for the output
    img1 = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color="red")
    text_to_image.output = img1

    # Second invocation - DummyI2T (image to text)
    image_to_text = Invocation(
        model="DummyI2T",
        type=InvocationType.TEXT,
        seed=seed,
        run_id=run.id,
        sequence_number=1,
        input_invocation_id=text_to_image.id,
        output_text="This is a description of the image",
    )

    # Third invocation - DummyT2I again (text to image)
    text_to_image2 = Invocation(
        model="DummyT2I",
        type=InvocationType.IMAGE,
        seed=seed,
        run_id=run.id,
        sequence_number=2,
        input_invocation_id=image_to_text.id,
    )

    # Create a second test image
    img2 = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color="blue")
    text_to_image2.output = img2

    # Add invocations to the database
    db_session.add(text_to_image)
    db_session.add(image_to_text)
    db_session.add(text_to_image2)
    db_session.commit()

    # Create a temporary output directory
    output_dir = tmp_path / "test_images"

    # Export the images
    export_run_images(run, db_session, output_dir=str(output_dir))

    # Check that the run-specific directory was created
    run_dir = output_dir / str(run.id)
    assert run_dir.exists()
    assert run_dir.is_dir()

    # Check that image files were created in the run-specific directory
    image_files = list(run_dir.glob("*.jpg"))

    # We should have image outputs from DummyT2I (at positions 0 and 2)
    assert len(image_files) == 2

    # Verify each image file exists and is a valid image
    for image_file in image_files:
        assert image_file.exists()
        # Try opening the image to make sure it's valid
        img = Image.open(image_file)
        assert img.size == (
            IMAGE_SIZE,
            IMAGE_SIZE,
        )  # Check it matches our IMAGE_SIZE constant

        # Get EXIF data from the image
        exif_data = img.getexif()
        assert 0x9286 in exif_data  # UserComment tag should exist

        # Parse metadata from EXIF
        metadata_bytes = exif_data[0x9286]
        metadata = json.loads(metadata_bytes.decode("utf-8"))

        # Verify metadata content
        assert "prompt" in metadata
        assert "model" in metadata
        assert "sequence_number" in metadata
        assert "seed" in metadata

        # Verify specific metadata values
        assert metadata["seed"] == str(seed)
        assert (
            metadata["model"] == "DummyT2I"
        )  # Image invocations come from DummyT2I in this test


def test_export_run_mosaic(db_session: Session, tmp_path):
    """Test that export_run_mosaic correctly creates a mosaic grid from multiple runs."""
    # Create 4 test runs with dummy models that will generate images
    run_ids = []
    for i in range(4):
        # Create the run
        run = Run(
            network=["DummyT2I"],
            initial_prompt=f"Test prompt {i} for mosaic export",
            max_length=2,
            seed=-1,
        )
        db_session.add(run)
        db_session.commit()
        db_session.refresh(run)
        run_ids.append(str(run.id))

        # Create an image invocation for each run
        text_to_image = Invocation(
            model="DummyT2I",
            type=InvocationType.IMAGE,
            seed=-1,
            run_id=run.id,
            sequence_number=0,
        )

        # Create a different colored test image for each run
        colors = ["red", "blue", "green", "yellow"]
        img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=colors[i])
        text_to_image.output = img
        db_session.add(text_to_image)
        db_session.commit()

    # Create a temporary output directory
    output_dir = tmp_path / "test_mosaic"
    os.makedirs(str(output_dir), exist_ok=True)

    # Create output file path
    output_video = output_dir / "test-mosaic.mp4"

    # Call the export_run_mosaic function with 2 columns
    fps = 10
    export_run_mosaic(
        run_ids,
        db_session,
        cols=4,
        resolution="HD",
        output_video=str(output_video),
        fps=fps,
    )
    # Check that the mosaic directory was created
    assert output_dir.exists()
    assert output_dir.is_dir()
    assert output_video.exists()
