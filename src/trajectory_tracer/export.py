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

from trajectory_tracer.db import read_run
from trajectory_tracer.genai_models import IMAGE_SIZE
from trajectory_tracer.schemas import InvocationType, Run

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

    # Group runs by prompt
    prompt_to_runs = {}
    for run in runs:
        prompt = run.initial_prompt
        if prompt not in prompt_to_runs:
            prompt_to_runs[prompt] = []
        prompt_to_runs[prompt].append(run)

    # Determine prompt order
    if prompt_order:
        # Use specified order, adding any missing prompts at the end
        ordered_prompts = [p for p in prompt_order if p in prompt_to_runs]
        # Add any prompts not in prompt_order
        ordered_prompts.extend(
            sorted([p for p in prompt_to_runs if p not in prompt_order])
        )
    else:
        # Default: sort prompts alphabetically
        ordered_prompts = sorted(prompt_to_runs.keys())

    # Optimize grid layout for 16:9 aspect ratio
    # First, gather all runs organized by prompt and network
    all_sorted_runs = []
    prompt_network_counts = {}  # For tracking networks per prompt

    for prompt in ordered_prompts:
        # Group the runs by network
        network_to_runs = {}
        for run in prompt_to_runs[prompt]:
            network_key = tuple(run.network)  # Convert list to hashable tuple
            if network_key not in network_to_runs:
                network_to_runs[network_key] = []
            network_to_runs[network_key].append(run)

        # Sort networks
        networks = sorted(network_to_runs.keys())

        # Track networks per prompt for later layout calculation
        prompt_network_counts[prompt] = len(networks)

        # Add all runs from this prompt in network order
        prompt_runs = []
        for network in networks:
            # Sort runs with same prompt and network by run_id
            sorted_runs = sorted(network_to_runs[network], key=lambda r: str(r.id))
            prompt_runs.extend(sorted_runs)

        all_sorted_runs.append((prompt, prompt_runs))

    # Calculate total cells needed in the grid
    total_cells = sum(len(runs) for _, runs in all_sorted_runs)

    # Find all possible divisors for the total cells
    divisors = [i for i in range(1, total_cells + 1) if total_cells % i == 0]

    # Target aspect ratio is 16:9
    target_ratio = 16 / 9

    # Find the configuration closest to 16:9, considering borders
    best_ratio_diff = float("inf")
    best_cols = total_cells

    for rows in divisors:
        cols = total_cells // rows

        # Account for borders in aspect ratio calculation (+2 for both dimensions)
        # We also need to account for progress bar height
        progress_bar_height = 30
        width_with_borders = (cols + 2) * IMAGE_SIZE
        height_with_borders = ((rows + 2) * IMAGE_SIZE) + progress_bar_height

        ratio = width_with_borders / height_with_borders
        ratio_diff = abs(ratio - target_ratio)

        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_cols = cols

    # Now create the grid layout based on the optimal dimensions
    grid_layout = []

    # Distribute runs across the grid while maintaining consistent prompts in rows
    # and networks in columns as much as possible
    remaining_cells = total_cells
    run_index = 0

    # Track which prompts have been completely processed
    processed_prompts = set()

    # Keep filling rows until we've placed all cells
    while remaining_cells > 0:
        current_row_cells = min(best_cols, remaining_cells)
        row_runs = []

        # Try to keep prompt consistency in rows
        # Find the prompt that hasn't been fully processed with most runs
        largest_remaining_prompt = None
        largest_count = 0

        for prompt, runs in all_sorted_runs:
            remaining_runs = [r for r in runs if r.id not in processed_prompts]
            if len(remaining_runs) > largest_count:
                largest_remaining_prompt = prompt
                largest_count = len(remaining_runs)

        # If we found a prompt that fits in this row, use it
        if largest_remaining_prompt and largest_count >= current_row_cells:
            # Find all runs for this prompt
            for prompt, runs in all_sorted_runs:
                if prompt == largest_remaining_prompt:
                    # Take the first current_row_cells runs that haven't been processed
                    prompt_runs = [r for r in runs if r.id not in processed_prompts][
                        :current_row_cells
                    ]
                    row_runs = prompt_runs

                    # Mark these runs as processed
                    for run in prompt_runs:
                        processed_prompts.add(run.id)

                    break
        else:
            # Fill from remaining runs in order
            for i in range(current_row_cells):
                if run_index < total_cells:
                    # Find the next unprocessed run
                    for prompt, runs in all_sorted_runs:
                        for run in runs:
                            if run not in processed_prompts:
                                row_runs.append(run)
                                processed_prompts.add(run.id)
                                break
                        if len(row_runs) > i:  # We found a run for this position
                            break
                run_index += 1

        # Add this row to the grid layout
        if row_runs:
            # Use the prompt of the first run in the row for the entire row
            prompt = row_runs[0].initial_prompt if row_runs else ""
            grid_layout.append((prompt, row_runs))

        remaining_cells -= current_row_cells

    # Final grid dimensions
    rows = len(grid_layout)
    cols = best_cols  # The optimal column count we calculated

    logger.info(
        f"Optimized mosaic grid for 16:9 aspect ratio: {rows} rows × {cols} columns (plus borders)"
    )

    # Create dimensions for the canvas
    # Adding 2 to rows and cols for the borders
    base_width = (cols + 2) * IMAGE_SIZE
    base_height = (rows + 2) * IMAGE_SIZE
    progress_bar_area_height = 30
    canvas_width = base_width
    canvas_height = base_height + progress_bar_area_height

    # Create the canvas
    base_canvas = Image.new("RGB", (canvas_width, canvas_height), (0, 0, 0))

    # Set up font for border text
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    base_font_size = int(IMAGE_SIZE * 0.08)

    # Draw the border tiles with prompt and network information
    draw = ImageDraw.Draw(base_canvas)

    # Draw prompt information in left and right borders
    for row_idx, (prompt, _) in enumerate(grid_layout):
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

    # Draw network information in top and bottom borders
    # Collect all networks in order they appear in the grid
    all_networks = []
    for _, row_runs in grid_layout:
        for run in row_runs:
            network_str = " → ".join(run.network)
            if network_str not in all_networks:
                all_networks.append(network_str)

    # Draw network labels in the top and bottom borders
    for col_idx, network_str in enumerate(all_networks[:cols]):  # Limit to max columns
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
        # Collect all runs in grid order and their image lists
        all_grid_runs = []
        for _, row_runs in grid_layout:
            all_grid_runs.extend(row_runs)

        # Get image lists for each run
        run_image_lists = []
        for run in all_grid_runs:
            image_list = [
                inv.output
                for inv in run.invocations
                if inv.type == InvocationType.IMAGE
            ]
            run_image_lists.append(image_list)

        # Find the maximum length of any image list
        max_images = max([len(img_list) for img_list in run_image_lists], default=0)

        if max_images == 0:
            logger.warning("No images found in any run")
            return

        # Now iterate through each image position
        for frame_idx in range(max_images):
            # Start with a copy of the base canvas with borders
            mosaic = base_canvas.copy()
            draw = ImageDraw.Draw(mosaic)

            # Paste each run's image at the current sequence position
            run_idx = 0
            for row_idx, (_, row_runs) in enumerate(grid_layout):
                for col_idx, run in enumerate(row_runs):
                    if col_idx >= cols:  # Skip if exceeds max columns
                        continue

                    # Get image list for this run
                    if run_idx < len(run_image_lists):
                        image_list = run_image_lists[run_idx]

                        # Get image for this position if it exists
                        image = None
                        if frame_idx < len(image_list):
                            image = image_list[frame_idx]
                        elif image_list:  # Use last available image if we've run out
                            image = image_list[-1]

                        # Paste the image onto the canvas
                        if image is not None:
                            # Account for borders (+1 to both row and column)
                            x_offset = (col_idx + 1) * IMAGE_SIZE
                            y_offset = (row_idx + 1) * IMAGE_SIZE
                            mosaic.paste(image, (x_offset, y_offset))

                    run_idx += 1

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
