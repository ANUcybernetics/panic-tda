import json
import logging
import os
import subprocess
from io import BytesIO
from uuid import UUID

from PIL import Image
from sqlmodel import Session

from trajectory_tracer.db import read_run
from trajectory_tracer.schemas import InvocationType, Run

logger = logging.getLogger(__name__)


def export_run_images(
    run: Run, session: Session, output_dir: str = "output/images"
) -> None:
    """
    Export all image invocations from a run to webp files.

    Args:
        run: The Run object containing invocations
        session: SQLModel Session for database operations
        output_dir: Directory where images will be saved (default: "output/images")
    """
    # Create run-specific directory
    run_dir = os.path.join(output_dir, str(run.id))

    # Ensure output directory exists
    os.makedirs(run_dir, exist_ok=True)

    # Ensure invocations are loaded
    session.refresh(run)

    # Process each invocation
    for invocation in run.invocations:
        # Skip non-image invocations
        if invocation.type != InvocationType.IMAGE:
            continue

        try:
            # Check if output_image_data exists and is not empty
            if not invocation.output_image_data:
                logger.warning(f"No image data found for invocation {invocation.id}")
                continue

            # Create BytesIO object and reset position
            image_data = BytesIO(invocation.output_image_data)
            image_data.seek(0)

            # Load image from output_image_data
            img = Image.open(image_data)

            # Force loading of image data to catch format issues early
            img.load()

            # Get prompt text from invocation's input property
            prompt_text = invocation.input

            # Prepare file path for image
            # Use JPEG format instead of WebP for better EXIF support
            file_path = os.path.join(run_dir, f"{invocation.sequence_number:05d}--{invocation.id}.jpg")

            # Create metadata
            metadata = {
                "prompt": prompt_text,
                "model": invocation.model,
                "sequence_number": str(invocation.sequence_number),
                "seed": str(invocation.seed),
            }

            # Convert to RGB mode for JPEG format
            img_with_metadata = img.convert("RGB")

            # Create EXIF data with metadata
            exif_data = img_with_metadata.getexif()

            # Store metadata in EXIF - UserComment tag (0x9286)
            exif_data[0x9286] = json.dumps(metadata).encode("utf-8")

            # Save image with EXIF metadata
            img_with_metadata.save(file_path, format="JPEG", quality=95, exif=exif_data)

        except Exception as e:
            logger.error(f"Error exporting image for invocation {invocation.id}: {e}")


def export_run_mosaic(
    run_ids: list[str], session: Session, cols: int, cell_size: int, output_dir: str, fps: int, output_video: str) -> None:
    """
    Export a mosaic of images from multiple runs and create a video from the mosaic images.

    Args:
        run_ids: List of run IDs to include in the mosaic
        session: SQLModel Session for database operations
        cols: Number of columns in the mosaic grid
        cell_size: Size of each cell in pixels
        output_dir: Directory where mosaic images will be saved
        fps: Frames per second for the output video
        output_video: Name of the output video file
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load all specified runs using the db helper, maintaining the order from run_ids
    runs = []
    run_map = {}  # Dictionary to map run_id to run object

    for run_id in run_ids:
        run = read_run(UUID(run_id), session)
        if run:
            run_map[run_id] = run

    # Maintain original order from run_ids
    for run_id in run_ids:
        if run_id in run_map:
            runs.append(run_map[run_id])

    # Find max sequence number across all runs
    max_seq = 0
    for run in runs:
        for invocation in run.invocations:
            if invocation.type == InvocationType.IMAGE and invocation.sequence_number > max_seq:
                max_seq = invocation.sequence_number

    # Calculate rows based on number of runs
    rows = (len(runs) + cols - 1) // cols  # Ceiling division

    # Process each sequence number
    for seq_num in range(0, max_seq + 1, 2):
        # Create a blank canvas for the mosaic
        mosaic = Image.new('RGB', (cols * cell_size, rows * cell_size), (0, 0, 0))

        # Place each run's image at this sequence number into the mosaic
        # The order is preserved from the original run_ids list
        for idx, run in enumerate(runs):
            row = idx // cols
            col = idx % cols

            # Find the invocation with this sequence number in this run
            for invocation in run.invocations:
                if (invocation.type == InvocationType.IMAGE and
                    invocation.sequence_number == seq_num and
                    invocation.output_image_data):

                    # Load the image
                    image_data = BytesIO(invocation.output_image_data)
                    image_data.seek(0)
                    img = Image.open(image_data).convert("RGB")

                    # If this is our first image, use its dimensions
                    if cell_size == 512 and idx == 0:
                        cell_size = img.width
                        # Recreate the mosaic with the correct dimensions
                        mosaic = Image.new('RGB', (cols * cell_size, rows * cell_size), (0, 0, 0))

                    # Calculate position and paste
                    x_offset = col * cell_size
                    y_offset = row * cell_size
                    mosaic.paste(img, (x_offset, y_offset))

                    # Only use the first matching invocation
                    break

        # Save the mosaic
        output_path = os.path.join(output_dir, f"{seq_num:05d}.jpg")
        mosaic.save(output_path, format="JPEG", quality=95)

        logger.info(f"Saved mosaic for sequence {seq_num}")

    # Create video from the mosaic images using ffmpeg
    try:
        # Get full path to output video
        video_path = os.path.join(output_dir, output_video)

        # Construct the ffmpeg command with settings optimized for 8K Samsung TV
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-framerate', str(fps),
            '-pattern_type', 'glob',
            '-i', os.path.join(output_dir, '*.jpg'),

            # Video codec settings - using H.265/HEVC for 8K TV compatibility
            '-c:v', 'libx265',
            '-preset', 'medium',  # Balance between quality and encoding speed
            '-crf', '22',         # Good quality-size balance (18-28 range)

            # Add HEVC tag for better compatibility with Samsung TVs
            '-tag:v', 'hvc1',

            # Add audio (silent if no audio is available)
            '-f', 'lavfi',
            '-i', 'anullsrc=r=48000:cl=stereo',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-shortest',

            # Color handling
            '-color_primaries', 'bt709',
            '-color_trc', 'bt709',
            '-colorspace', 'bt709',

            # Properly handle pixel format
            '-pix_fmt', 'yuv420p',

            # Use movflags to enable streaming
            '-movflags', '+faststart',

            video_path
        ]

        # Execute the command
        logger.info(f"Creating video with command: {' '.join(ffmpeg_cmd)}")
        subprocess.run(ffmpeg_cmd, check=True)

        logger.info(f"Created mosaic video at {video_path}")
    except Exception as e:
        logger.error(f"Error creating mosaic video: {e}")
