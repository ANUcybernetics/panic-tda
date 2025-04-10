import json
import logging
import os
import shutil
import subprocess
import textwrap
from io import BytesIO
from uuid import UUID

from PIL import Image, ImageDraw, ImageFont
from sqlmodel import Session

from panic_tda.db import read_run
from panic_tda.genai_models import IMAGE_SIZE
from panic_tda.schemas import InvocationType, Run

logger = logging.getLogger(__name__)


# TODO maybe remove this function, and if you need still just use the frames
# that are the by-product of the video export
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
    font = ImageFont.truetype(font_path, title_font_size)

    # Calculate max width allowing for padding
    padding = size * 0.1
    max_text_width = size - (2 * padding)

    # Use the wrapping helper, passing the top-left corner for positioning (padding, padding)
    # The helper function will handle the vertical centering within the drawing area
    draw_text_wrapped(draw, prompt_text, (padding, padding), max_text_width, font)

    return img


def export_video(
    run_ids: list[str],
    session: Session,
    fps: int,
    resolution: str,
    output_video: str,
    prompt_order: list[str] = None,
) -> None:
    """
    Export a mosaic video of images from multiple runs, organized by prompt (rows)
    and network (columns) with informative borders. Each prompt can span multiple rows
    to distribute networks evenly.

    Args:
        run_ids: List of run IDs to include in the mosaic
        session: SQLModel Session for database operations
        fps: Frames per second for the output video
        resolution: Target resolution ("HD", "4K", or "8K")
        output_video: Name of the output video file
        prompt_order: Optional custom ordering for prompts (default: alphabetical)
    """
    # Setup output directory
    output_dir = os.path.dirname(output_video)
    os.makedirs(output_dir, exist_ok=True)

    # Load runs
    runs = []
    for run_id in run_ids:
        run = read_run(UUID(run_id), session)
        if run:
            session.refresh(run)
            runs.append(run)

    if not runs:
        logger.error("No valid runs found for the provided IDs")
        return

    # Group runs by prompt and network
    prompt_network_runs = {}
    all_prompts = set()
    all_networks = []
    network_to_idx = {}  # For ordered networks

    for run in runs:
        prompt = run.initial_prompt
        network_tuple = tuple(run.network)  # Convert list to hashable tuple

        if network_tuple not in network_to_idx:
            network_to_idx[network_tuple] = len(all_networks)
            all_networks.append(network_tuple)

        all_prompts.add(prompt)

        # Create nested dictionary structure for prompt -> network -> runs
        if prompt not in prompt_network_runs:
            prompt_network_runs[prompt] = {}

        if network_tuple not in prompt_network_runs[prompt]:
            prompt_network_runs[prompt][network_tuple] = []

        prompt_network_runs[prompt][network_tuple].append(run)

    # Determine prompt order
    if prompt_order:
        # Use specified order, adding any missing prompts at the end
        ordered_prompts = [p for p in prompt_order if p in prompt_network_runs]
        # Add any prompts not in prompt_order
        ordered_prompts.extend(
            sorted([p for p in prompt_network_runs if p not in ordered_prompts])
        )
    else:
        # Default: sort prompts alphabetically
        ordered_prompts = sorted(prompt_network_runs.keys())

    # Sort networks consistently
    all_networks.sort()

    # Step 1: Count seeds for each prompt-network combination and verify consistency
    seeds_counts = {}
    for prompt in ordered_prompts:
        for network in all_networks:
            if network in prompt_network_runs[prompt]:
                seed_count = len(prompt_network_runs[prompt][network])
                key = (prompt, network)
                seeds_counts[key] = seed_count

    # Verify all prompt-network combinations have the same number of seeds
    n_seeds = None
    for (prompt, network), count in seeds_counts.items():
        if n_seeds is None:
            n_seeds = count
        elif n_seeds != count:
            logger.error(
                f"Inconsistent number of seeds: {prompt} + {network} has {count} seeds, expected {n_seeds}"
            )
            return

    if n_seeds is None or n_seeds == 0:
        logger.error("No valid runs found with seeds")
        return

    # Step 2: Calculate the grid dimensions
    n_prompts = len(ordered_prompts)
    n_networks = len(all_networks)

    # Calculate possible seeds_per_row values
    # Start with all seeds in one row, then try dividing by 2 repeatedly
    possible_seeds_per_row = []
    seeds_per_row = n_seeds
    while seeds_per_row >= 1:
        possible_seeds_per_row.append(seeds_per_row)
        if seeds_per_row % 2 == 0:
            seeds_per_row = seeds_per_row // 2
        else:
            break

    # Step 3: Find the seeds_per_row that produces the aspect ratio closest to 16:9
    target_ratio = 16 / 9
    best_ratio_diff = float("inf")
    best_seeds_per_row = n_seeds

    for seeds_per_row in possible_seeds_per_row:
        # For each prompt, we need (n_seeds / seeds_per_row) rows
        rows_per_prompt = (
            n_seeds + seeds_per_row - 1
        ) // seeds_per_row  # Ceiling division
        total_rows = n_prompts * rows_per_prompt

        # Each row has n_networks * seeds_per_row columns
        total_cols = n_networks * seeds_per_row

        # Account for borders and progress bar in aspect ratio calculation
        progress_bar_height = 30
        width_with_borders = (total_cols + 2) * IMAGE_SIZE
        height_with_borders = ((total_rows + 2) * IMAGE_SIZE) + progress_bar_height

        ratio = width_with_borders / height_with_borders
        ratio_diff = abs(ratio - target_ratio)

        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_seeds_per_row = seeds_per_row

    # Step 4: Create a mapping of run_id to grid position (r, c)
    seeds_per_row = best_seeds_per_row
    rows_per_prompt = (n_seeds + seeds_per_row - 1) // seeds_per_row
    run_grid_positions = {}
    grid_cells = {}  # Maps (row, col) -> run

    # Calculate the final grid dimensions
    rows = n_prompts * rows_per_prompt
    cols = n_networks * seeds_per_row

    # Assign positions for each run
    for prompt_idx, prompt in enumerate(ordered_prompts):
        for network_idx, network in enumerate(all_networks):
            if network in prompt_network_runs[prompt]:
                # Sort runs with same prompt and network by run_id
                sorted_runs = sorted(
                    prompt_network_runs[prompt][network], key=lambda r: str(r.id)
                )

                for seed_idx, run in enumerate(sorted_runs):
                    # Calculate position in grid
                    row_within_prompt = seed_idx // seeds_per_row
                    seed_within_row = seed_idx % seeds_per_row

                    # Final grid position
                    row = (prompt_idx * rows_per_prompt) + row_within_prompt
                    col = (network_idx * seeds_per_row) + seed_within_row

                    run_grid_positions[str(run.id)] = (row, col)
                    grid_cells[(row, col)] = run

    logger.info(
        f"Optimized mosaic grid for 16:9 aspect ratio: {rows} rows × {cols} columns "
        f"({n_prompts} prompts, {n_networks} networks, {seeds_per_row} seeds per row)"
    )

    # Step 5: Create the canvas
    base_width = (cols + 2) * IMAGE_SIZE
    base_height = (rows + 2) * IMAGE_SIZE
    progress_bar_area_height = 30
    canvas_width = base_width
    canvas_height = base_height + progress_bar_area_height

    base_canvas = Image.new("RGB", (canvas_width, canvas_height), (0, 0, 0))
    draw = ImageDraw.Draw(base_canvas)

    # Set up font for border text
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    base_font_size = int(IMAGE_SIZE * 0.1)

    # Step 6: Draw prompt information in left and right borders
    for prompt_idx, prompt in enumerate(ordered_prompts):
        for row_within_prompt in range(rows_per_prompt):
            row_idx = (prompt_idx * rows_per_prompt) + row_within_prompt

            # Left border
            left_border_x = 0
            left_border_y = (row_idx + 1) * IMAGE_SIZE

            # Create and paste prompt tile
            prompt_tile = create_prompt_title_card(
                prompt, IMAGE_SIZE, font_path, base_font_size
            )
            base_canvas.paste(prompt_tile, (left_border_x, left_border_y))

            # Right border - same content as left
            right_border_x = (cols + 1) * IMAGE_SIZE
            right_border_y = left_border_y
            base_canvas.paste(prompt_tile, (right_border_x, right_border_y))

    # Define abbreviations for network names
    abbreviations = {"FluxSchnell": "Flux", "SDXLTurbo": "SDXL"}

    # Step 7: Draw network information in top and bottom borders
    for network_idx, network in enumerate(all_networks):
        # Create abbreviated network string
        abbreviated_network = [abbreviations.get(net, net) for net in network]
        network_str = " → ".join(abbreviated_network)

        # For each column in this network's group
        for seed_within_row in range(seeds_per_row):
            col_idx = (network_idx * seeds_per_row) + seed_within_row

            # Top border
            top_border_x = (col_idx + 1) * IMAGE_SIZE
            top_border_y = 0

            # Create and paste network tile
            network_tile = create_prompt_title_card(
                network_str, IMAGE_SIZE, font_path, base_font_size
            )
            base_canvas.paste(network_tile, (top_border_x, top_border_y))

            # Bottom border - same content as top
            bottom_border_x = top_border_x
            bottom_border_y = (rows + 1) * IMAGE_SIZE
            base_canvas.paste(network_tile, (bottom_border_x, bottom_border_y))

    # Create temporary directory for frame storage
    temp_dir = os.path.join(output_dir, "temp_frames")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Find the maximum number of frames (images) across all runs
        max_images = 0
        run_image_counts = {}

        # Get all run IDs in grid order
        all_grid_runs = []
        for r in range(rows):
            for c in range(cols):
                if (r, c) in grid_cells:
                    all_grid_runs.append(grid_cells[(r, c)])

        for run in all_grid_runs:
            # Count image invocations for each run
            image_count = sum(
                1 for inv in run.invocations if inv.type == InvocationType.IMAGE
            )
            run_image_counts[str(run.id)] = image_count
            max_images = max(max_images, image_count)

        if max_images == 0:
            logger.warning("No images found in any run")
            return

        # Now iterate through each image position (frame)
        for frame_idx in range(max_images):
            # Start with a copy of the base canvas with borders
            mosaic = base_canvas.copy()
            draw = ImageDraw.Draw(mosaic)

            # Paste each run's image at the current sequence position
            for row in range(rows):
                for col in range(cols):
                    if (row, col) in grid_cells:
                        run = grid_cells[(row, col)]

                        # Get this run's invocations of type IMAGE
                        image_invocations = [
                            inv
                            for inv in run.invocations
                            if inv.type == InvocationType.IMAGE
                        ]

                        # Get the specific image for this frame if it exists
                        image = None
                        if frame_idx < len(image_invocations):
                            image = image_invocations[frame_idx].output
                        elif (
                            image_invocations
                        ):  # Use last available image if we've run out
                            image = image_invocations[-1].output

                        # Paste the image onto the canvas
                        if image is not None:
                            # Account for borders (+1 to both row and column)
                            x_offset = (col + 1) * IMAGE_SIZE
                            y_offset = (row + 1) * IMAGE_SIZE
                            mosaic.paste(image, (x_offset, y_offset))

            # Draw progress bar
            progress_percent = (
                frame_idx / (max_images - 1)
                if max_images > 1
                else (1.0 if frame_idx == 0 else 0.0)
            )
            progress_bar_pixel_width = int(canvas_width * progress_percent)
            progress_bar_y_top = int(base_height + (progress_bar_area_height - 10) / 2)
            progress_bar_y_bottom = progress_bar_y_top + 10

            # Background bar
            draw.rectangle(
                (0, progress_bar_y_top, canvas_width, progress_bar_y_bottom),
                fill=(50, 50, 50),
            )

            # Progress bar
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

            # Save frame
            output_path = os.path.join(temp_dir, f"{frame_idx:05d}.jpg")
            mosaic.save(output_path, format="JPEG", quality=95)

            # Allow mosaic and other objects to be garbage collected
            del mosaic
            del draw

            if (
                frame_idx % (max(1, max_images // 20)) == 0
                or frame_idx == max_images - 1
            ):
                if frame_idx == 0:
                    logger.info("Started saving mosaic frames...")
                logger.info(
                    f"Saved frame {frame_idx + 1}/{max_images} ({int(progress_percent * 100)}%)"
                )

        # Video creation with ffmpeg
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
