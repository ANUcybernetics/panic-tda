import json
import logging
import os
import subprocess
from io import BytesIO
from uuid import UUID

from PIL import Image, ImageDraw, ImageFont
from sqlmodel import Session

from trajectory_tracer.db import read_run
from trajectory_tracer.genai_models import IMAGE_SIZE
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


def order_runs_for_mosaic(run_ids: list[str], session: Session) -> list[str]:
    """
    Orders run IDs for mosaic creation based on their prompts and networks.

    Args:
        run_ids: List of run IDs to be ordered
        session: SQLModel Session for database operations

    Returns:
        List of run IDs ordered first by prompt, then by network structure
    """
    # Load all runs
    runs = []
    for run_id in run_ids:
        run = read_run(UUID(run_id), session)
        if run:
            runs.append(run)

    if not runs:
        logger.warning("No valid runs found for ordering")
        return run_ids

    # Sort runs first by prompt, then by network
    sorted_runs = sorted(runs, key=lambda r: (r.initial_prompt, *r.network))

    # Return the ordered run IDs
    return [str(run.id) for run in sorted_runs]


def export_run_mosaic( run_ids: list[str], session: Session, cols: int, fps: int, resolution: str, output_video: str) -> None:
    """
    Export a mosaic of images from multiple runs and create a video from the mosaic images.

    Args:
        run_ids: List of run IDs to include in the mosaic
        session: SQLModel Session for database operations
        cols: Number of columns in the mosaic grid
        fps: Frames per second for the output video
        output_video: Name of the output video file (can include subfolders)
        resolution: Target resolution for the output video ("HD", "4K", or "8K")
    """
    # Extract directory from output_video path
    output_dir = os.path.dirname(output_video)

    # If output_dir is not empty, ensure it exists
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        # Use current directory if no directory specified
        output_dir = "."
        output_video = os.path.join(output_dir, output_video)

    # Load all specified runs in the given order
    runs = []
    run_ids = order_runs_for_mosaic(run_ids, session)

    for run_id in run_ids:
        run = read_run(UUID(run_id), session)
        if run:
            runs.append(run)

    if not runs:
        logger.error("No valid runs found for the provided IDs")
        return

    # Get initial prompt from the first run for display on the progress bar
    initial_prompt = runs[0].initial_prompt if runs[0].initial_prompt else "No prompt available"

    # Filter and organize invocations
    run_invocations = []

    for run in runs:
        # Filter to include only image invocations and sort by sequence number
        invocations = [inv for inv in run.invocations if inv.type == InvocationType.IMAGE]
        invocations.sort(key=lambda inv: inv.sequence_number)

        # Make sure all sequence numbers start from zero and are consecutive
        if not invocations or invocations[0].sequence_number != 0:
            logger.warning(f"Run {run.id} doesn't start with sequence number 0")

        run_invocations.append(invocations)

    # Verify all runs have the same number of invocations
    invocation_counts = [len(invs) for invs in run_invocations]
    if len(set(invocation_counts)) > 1:
        logger.warning(f"Runs have different numbers of invocations: {invocation_counts}")

    # Find max sequence number across all runs
    max_seq = max([invs[-1].sequence_number if invs else 0 for invs in run_invocations])

    # Calculate rows based on number of runs
    rows = (len(runs) + cols - 1) // cols  # Ceiling division

    # Calculate basic mosaic dimensions without progress bar
    base_height = rows * IMAGE_SIZE
    base_width = cols * IMAGE_SIZE

    # Adjust progress_bar_offset to achieve 16:9 aspect ratio
    target_aspect_ratio = 16/9
    current_aspect_ratio = base_width / base_height

    if current_aspect_ratio >= target_aspect_ratio:
        # Width is already proportionally wider than or equal to 16:9
        # Add progress bar space to maintain width but increase height
        progress_bar_offset = int(base_width / target_aspect_ratio) - base_height
        progress_bar_offset = max(144, progress_bar_offset)  # Ensure minimum space for text
    else:
        # Height is proportionally taller than 16:9 aspect ratio allows
        # We require a 16:9 aspect ratio for compatibility
        raise ValueError(f"Cannot achieve 16:9 aspect ratio with current dimensions. Current ratio: {current_aspect_ratio:.2f}. Consider reducing the number of rows or increasing the number of columns.")

    # Apply the calculated offset
    canvas_height = base_height + progress_bar_offset
    canvas_width = base_width
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 72)

    # Create a blank canvas for the mosaic (reuse for each sequence)
    mosaic = Image.new('RGB', (canvas_width, canvas_height), (0, 0, 0))
    draw = ImageDraw.Draw(mosaic) if font else None

    # Create temp directory for frames
    temp_dir = os.path.join(output_dir, "temp_frames")
    os.makedirs(temp_dir, exist_ok=True)

    # Process each sequence number (by groups of 2 to match original)
    for seq_num in range(0, max_seq + 1, 2):
        # Clear the canvas by filling with black
        draw = ImageDraw.Draw(mosaic)
        draw.rectangle((0, 0, canvas_width, canvas_height), fill=(0, 0, 0))

        # Place each run's image at this sequence number into the mosaic
        for idx, invocations in enumerate(run_invocations):
            row = idx // cols
            col = idx % cols

            # Find the invocation with this sequence number in this run
            matching_inv = next((inv for inv in invocations
                                if inv.sequence_number == seq_num and inv.output_image_data),
                                None)

            if matching_inv:
                # Load and paste the image
                image_data = BytesIO(matching_inv.output_image_data)
                image_data.seek(0)
                img = Image.open(image_data).convert("RGB")

                # Calculate position and paste
                x_offset = col * IMAGE_SIZE
                y_offset = row * IMAGE_SIZE
                mosaic.paste(img, (x_offset, y_offset))

        # Draw progress bar (10px wide, 100px from bottom)
        if draw:
            progress_percent = seq_num / max_seq if max_seq > 0 else 0
            progress_width = int(canvas_width * progress_percent)

            # Progress bar background
            draw.rectangle(
                (0, canvas_height - progress_bar_offset, canvas_width, canvas_height - progress_bar_offset + 10),
                fill=(50, 50, 50)
            )

            # Progress bar fill - white
            draw.rectangle(
                (0, canvas_height - progress_bar_offset, progress_width, canvas_height - progress_bar_offset + 10),
                fill=(255, 255, 255)
            )

            # Draw the full prompt text without truncating
            draw.text(
                (20, canvas_height - progress_bar_offset + 20),
                initial_prompt,  # Don't limit the text width - show full prompt
                fill=(255, 255, 255),
                font=font
            )

        # Save the mosaic
        output_path = os.path.join(temp_dir, f"{seq_num:05d}.jpg")
        mosaic.save(output_path, format="JPEG", quality=95)

        logger.info(f"Saved mosaic for sequence {seq_num} ({int(progress_percent*100)}%)")

    # Define the output resolution based on the resolution parameter
    # Resolution standards: HD (1920x1080), 4K (3840x2160), 8K (7680x4320)
    resolution_settings = {
        "HD": {"width": 1920, "height": 1080},
        "4K": {"width": 3840, "height": 2160},
        "8K": {"width": 7680, "height": 4320}
    }

    # Use default HD resolution if specified resolution is not recognized
    if resolution not in resolution_settings:
        logger.warning(f"Unknown resolution '{resolution}'. Using HD (1920x1080) as default.")
        resolution = "HD"

    target_width = resolution_settings[resolution]["width"]
    target_height = resolution_settings[resolution]["height"]

    # Construct the ffmpeg command with settings for the target resolution
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',  # Overwrite output file if it exists
        '-framerate', str(fps),
        '-pattern_type', 'glob',
        '-i', os.path.join(temp_dir, '*.jpg'),

        # Video codec settings - using H.265/HEVC for better compression at high resolutions
        '-c:v', 'libx265',
        '-preset', 'medium',  # Balance between quality and encoding speed
        '-crf', '22',         # Good quality-size balance (18-28 range)

        # Set specific resolution
        '-vf', f'scale={target_width}:{target_height}',

        '-tag:v', 'hvc1',

        # Color handling
        '-color_primaries', 'bt709',
        '-color_trc', 'bt709',
        '-colorspace', 'bt709',

        # Properly handle pixel format
        '-pix_fmt', 'yuv420p',

        # Use movflags to enable streaming
        '-movflags', '+faststart',

        output_video
    ]

    # Execute the command
    logger.info(f"Creating {resolution} video with resolution {target_width}x{target_height}")
    logger.info(f"Creating video with command: {' '.join(ffmpeg_cmd)}")
    subprocess.run(ffmpeg_cmd, check=True)

    # Clean up temporary files
    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))
    os.rmdir(temp_dir)

    logger.info(f"Mosaic video created successfully at {output_video} with {resolution} resolution")
