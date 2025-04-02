import json
import logging
import math
import os
import shutil
import subprocess
import textwrap
from io import BytesIO
from typing import Union
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
            file_path = os.path.join(
                run_dir, f"{invocation.sequence_number:05d}--{invocation.id}.jpg"
            )

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


# Helper function for text wrapping on image
def draw_text_wrapped(draw, text, position, max_width, font, fill=(255, 255, 255)):
    """Draws text on an image, wrapped to a max width."""
    # Estimate characters per line based on font size (adjust multiplier as needed)
    chars_per_line = int(max_width / (font.size * 0.6)) if font.size > 0 else 20
    lines = textwrap.wrap(
        text, width=max(10, chars_per_line)
    )  # Ensure width is at least 10

    x, y = position
    line_height = (
        font.getbbox("A")[3] if hasattr(font, "getbbox") else font.size * 1.2
    )  # Approximate line height

    total_text_height = len(lines) * (line_height + 2)
    # Recalculate start_y to vertically center based on actual lines
    start_y = (
        y + (max_width - total_text_height) / 2
    )  # Center vertically within the available space (using max_width assuming square)

    current_y = start_y
    for line in lines:
        # Center each line horizontally
        try:  # Use textbbox for more accurate centering if available
            bbox = draw.textbbox((0, 0), line, font=font)
            line_width = bbox[2] - bbox[0]
            text_x = x + (max_width - line_width) / 2
        except AttributeError:  # Fallback for older PIL/Pillow or basic fonts
            line_width = draw.textlength(line, font=font)  # Deprecated but fallback
            text_x = x + (max_width - line_width) / 2

        draw.text((text_x, current_y), line, font=font, fill=fill)
        current_y += line_height + 2  # Add some spacing between lines


# Helper function to create a title card image
def create_prompt_title_card(
    prompt_text: str, size: int, font_path: str, base_font_size: int
) -> Image.Image:
    """Creates a square image with wrapped text, centered vertically and horizontally."""
    img = Image.new("RGB", (size, size), color="black")
    draw = ImageDraw.Draw(img)

    title_font_size = max(12, int(base_font_size * 0.8))
    try:
        font = ImageFont.truetype(font_path, title_font_size)
    except IOError:
        logger.warning(f"Font not found at {font_path}. Using default PIL font.")
        try:
            font = ImageFont.load_default(size=title_font_size)
        except Exception:
            logger.error("Could not load any default font.")
            # Draw simple error text on the blank image
            try:  # Use a known basic font if possible
                error_font = ImageFont.load_default()
                draw.text((10, 10), "FONT ERR", font=error_font, fill="red")
            except Exception:
                pass  # If even default fails, return blank
            return img

    # Calculate max width allowing for padding
    padding = size * 0.1
    max_text_width = size - (2 * padding)

    # Use the wrapping helper, passing the top-left corner for positioning (padding, padding)
    # The helper function will handle the vertical centering within the drawing area
    draw_text_wrapped(draw, prompt_text, (padding, padding), max_text_width, font)

    return img


def export_run_mosaic(
    run_ids: list[str],
    session: Session,
    cols: Union[int, str],
    fps: int,
    resolution: str,
    output_video: str,
) -> None:
    """
    Export a mosaic of images from multiple runs and create a video.
    Title cards are added for each new prompt.

    Args:
        run_ids: List of run IDs to include in the mosaic
        session: SQLModel Session for database operations
        cols: Number of columns in the mosaic grid or "auto" to calculate the optimal
              number of columns for a 16:9 aspect ratio
        fps: Frames per second for the output video
        output_video: Name of the output video file
        resolution: Target resolution ("HD", "4K", or "8K")
    """
    # Setup output directory
    output_dir = os.path.dirname(output_video)
    os.makedirs(output_dir, exist_ok=True)

    # Load runs
    runs = []
    ordered_run_ids = order_runs_for_mosaic(run_ids, session)
    for run_id in ordered_run_ids:
        run = read_run(UUID(run_id), session)
        if run:
            session.refresh(run)
            runs.append(run)

    if not runs:
        logger.error("No valid runs found for the provided IDs")
        return

    # Step 1: Find prompt changes and create title cards
    title_cards = []
    title_card_positions = []
    run_positions = []

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    base_title_font_size = int(IMAGE_SIZE * 0.1)

    # First, collect all runs and identify where prompt changes occur
    prompt_changes = []

    # Always include the first prompt at position 0
    if runs:
        prompt_changes.append((runs[0].initial_prompt, 0))

    # Add runs to run_positions and find prompt changes
    for idx, run in enumerate(runs):
        run_positions.append(run)

        # Check if next run has a different prompt
        if idx < len(runs) - 1:
            current_prompt = run.initial_prompt or "No prompt"
            next_prompt = runs[idx + 1].initial_prompt or "No prompt"

            if next_prompt != current_prompt:
                # Record the position where the next run will be placed
                prompt_changes.append((next_prompt, idx + 1))

    # Now create title cards and record their positions
    for prompt, position in prompt_changes:
        # Create a new title card for this prompt
        title_card = create_prompt_title_card(
            prompt, IMAGE_SIZE, font_path, base_title_font_size
        )

        title_cards.append(title_card)
        title_card_positions.append(position)
        logger.debug(f"Added title card for prompt: '{prompt}'")

    # Step 2: Create list-of-lists of images for each run
    run_image_lists = []

    for run in runs:
        # Get and sort invocations by sequence number
        image_list = [
            inv.output for inv in run.invocations if inv.type == InvocationType.IMAGE
        ]
        run_image_lists.append(image_list)

    # Determine the number of columns
    total_positions = len(run_positions) + len(title_cards)

    if cols == "auto":
        # Calculate optimal columns for 16:9 aspect ratio
        # The formula is sqrt((16/9) * total_positions)
        target_aspect_ratio = 16 / 9
        ideal_cols = math.sqrt(target_aspect_ratio * total_positions)
        cols = max(1, round(ideal_cols))
        logger.info(
            f"Auto-calculated {cols} columns for optimal 16:9 aspect ratio with {total_positions} positions"
        )
    elif not isinstance(cols, int) or cols < 1:
        logger.warning(f"Invalid columns value: {cols}. Using 1 column.")
        cols = 1

    # Create dimensions for the canvas
    rows = (total_positions + cols - 1) // cols
    base_width = cols * IMAGE_SIZE
    base_height = rows * IMAGE_SIZE
    progress_bar_area_height = 30
    canvas_width = base_width
    canvas_height = base_height + progress_bar_area_height

    # Create the black canvas
    mosaic = Image.new("RGB", (canvas_width, canvas_height), (0, 0, 0))

    # Create temporary directory for frame storage
    temp_dir = os.path.join(output_dir, "temp_frames")
    os.makedirs(temp_dir, exist_ok=True)

    # Step 3 & 4: Create each frame by directly iterating through images
    # Find the maximum length of any image list
    max_images = max([len(img_list) for img_list in run_image_lists], default=0)

    # Now iterate through each image position instead of by sequence number
    for frame_idx in range(0, max_images):
        # Clear canvas for new frame
        draw = ImageDraw.Draw(mosaic)
        draw.rectangle((0, 0, canvas_width, canvas_height), fill=(0, 0, 0))

        # Get frame images for this position from each run
        frame_images = []
        for run_idx, image_list in enumerate(run_image_lists):
            # Get image for this position if it exists
            # If position exceeds available images, use the last image
            image = None
            if frame_idx < len(image_list):
                image = image_list[frame_idx]
            elif image_list:  # Use last available image if we've run out
                image = image_list[-1]
            frame_images.append(image)

        # Insert title cards at appropriate positions
        for card_pos, card_idx in sorted(
            zip(title_card_positions, range(len(title_cards))), reverse=True
        ):
            # Insert a copy of the title card at the appropriate position
            frame_images.insert(card_pos, title_cards[card_idx])

        # Paste each image to the canvas at its grid position
        for idx, img in enumerate(frame_images):
            if img is not None:
                row = idx // cols
                col = idx % cols
                x_offset = col * IMAGE_SIZE
                y_offset = row * IMAGE_SIZE
                mosaic.paste(img, (x_offset, y_offset))

        # Step 5: Draw progress bar
        progress_percent = (
            frame_idx / (max_images - 1)
            if max_images > 1
            else (1.0 if frame_idx == 0 else 0.0)
        )
        progress_bar_pixel_width = int(canvas_width * progress_percent)
        progress_bar_y_top = int(base_height + (progress_bar_area_height - 10) / 2)
        progress_bar_y_bottom = progress_bar_y_top + 10

        draw.rectangle(
            (0, progress_bar_y_top, canvas_width, progress_bar_y_bottom),
            fill=(50, 50, 50),
        )
        if progress_bar_pixel_width > 0:
            draw.rectangle(
                (
                    0,
                    progress_bar_y_top,
                    progress_bar_pixel_width,
                    progress_bar_y_bottom,
                ),
                fill=(255, 255, 255),
            )

        # Step 6: Save frame (save all frames without checking for uniqueness)
        output_path = os.path.join(temp_dir, f"{frame_idx:05d}.jpg")
        mosaic.save(output_path, format="JPEG", quality=95)

        if frame_idx % (max(1, max_images // 20)) == 0 or frame_idx == max_images - 1:
            if frame_idx == 0:
                logger.info("Started saving mosaic frames...")
            logger.info(
                f"Saved frame {frame_idx + 1}/{max_images} ({int(progress_percent * 100)}%)"
            )

    # Step 7: Video creation with ffmpeg
    resolution_settings = {
        "HD": {"width": 1920, "height": 1080},
        "4K": {"width": 3840, "height": 2160},
        "8K": {"width": 7680, "height": 4320},
    }

    if resolution not in resolution_settings:
        logger.warning(f"Unknown resolution '{resolution}'. Using HD.")
        resolution = "HD"

    target_width = resolution_settings[resolution]["width"]
    target_height = resolution_settings[resolution]["height"]

    filtergraph = (
        f"scale=w={target_width}:h={target_height}:force_original_aspect_ratio=decrease,"
        f"pad=w={target_width}:h={target_height}:x=(ow-iw)/2:y=(oh-ih)/2:color=black"
    )

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-pattern_type",
        "glob",
        "-i",
        os.path.join(temp_dir, "*.jpg"),
        "-c:v",
        "libx265",
        "-preset",
        "medium",
        "-crf",
        "22",
        "-vf",
        filtergraph,
        "-tag:v",
        "hvc1",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-colorspace",
        "bt709",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        output_video,
    ]

    logger.info(f"Creating {resolution} video with ffmpeg")
    try:
        subprocess.run(
            ffmpeg_cmd,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        logger.info(f"Video created successfully: {output_video}")
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg failed: {e.returncode}")
        stderr_str = e.stderr if hasattr(e, "stderr") else "N/A"
        logger.error(f"stderr: {stderr_str}")
        raise e
    finally:
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.error(f"Error cleaning up temp dir: {e}")
