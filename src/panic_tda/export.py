import json
import logging
import math
import os
import shutil
import subprocess
import textwrap
from io import BytesIO
from typing import Dict, List
from uuid import UUID

from PIL import Image, ImageDraw, ImageFont
from sqlmodel import Session

from panic_tda.db import read_invocation, read_run
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
def draw_text_wrapped(
    draw, text, position, max_width, font, fill=(255, 255, 255), max_height=None
):
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

    # Use max_height for vertical centering if provided, otherwise use max_width
    height_for_centering = max_height if max_height is not None else max_width

    # Recalculate start_y to vertically center based on actual lines
    start_y = y + (height_for_centering - total_text_height) / 2

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
    """Creates a rectangular image with wrapped text, centered vertically and horizontally."""
    # Create rectangle with 2x width of height
    img = Image.new("RGB", (size * 2, size), color="black")
    draw = ImageDraw.Draw(img)

    # Use the same font size as network banners (no reduction)
    title_font_size = base_font_size
    font = ImageFont.truetype(font_path, title_font_size)

    # Calculate max width allowing for padding
    padding = size * 0.1
    max_text_width = (size * 2) - (2 * padding)  # Adjust for 2x width

    # Use the wrapping helper, passing the top-left corner for positioning (padding, padding)
    # Pass the actual height (size) for proper vertical centering
    draw_text_wrapped(
        draw,
        prompt_text,
        (padding, padding),
        max_text_width,
        font,
        max_height=size - 2 * padding,
    )

    return img


# New helper function to create a network title banner that spans multiple columns
def create_network_title_banner(
    network_str: str, width: int, height: int, font_path: str, font_size: int
) -> Image.Image:
    """Creates a wide banner with network text centered."""
    img = Image.new("RGB", (width, height), color="black")
    draw = ImageDraw.Draw(img)

    font = ImageFont.truetype(font_path, font_size)

    # Center text horizontally and vertically
    try:
        bbox = draw.textbbox((0, 0), network_str, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except AttributeError:  # Fallback for older PIL/Pillow
        text_width = draw.textlength(network_str, font=font)
        text_height = font.size

    x = (width - text_width) / 2
    y = (height - text_height) / 2

    draw.text((x, y), network_str, font=font, fill=(255, 255, 255))

    return img


def export_video(
    run_ids: list[str],
    session: Session,
    fps: int,
    resolution: str,
    output_file: str,
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
        output_file: Name of the output video file
        prompt_order: Optional custom ordering for prompts (default: alphabetical)
    """
    # Setup output directory
    output_dir = os.path.dirname(output_file)
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

    # Define the gaps between network blocks and prompt blocks
    network_gap = 50  # 50px gap between network blocks
    prompt_gap = 50  # 50px gap between prompt blocks

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

        # Account for borders, gaps, and progress bar in aspect ratio calculation
        progress_bar_height = 30

        # Calculate width with borders and network gaps
        width_with_borders = (total_cols * IMAGE_SIZE) + (
            2 * IMAGE_SIZE * 2
        )  # Add left and right borders (2x width)
        if n_networks > 1:
            width_with_borders += (
                n_networks - 1
            ) * network_gap  # Add gaps between networks

        # Calculate height with borders and prompt gaps
        height_with_borders = (
            (total_rows * IMAGE_SIZE) + (2 * IMAGE_SIZE) + progress_bar_height
        )  # Add top and bottom borders
        if n_prompts > 1:
            height_with_borders += (
                n_prompts - 1
            ) * prompt_gap  # Add gaps between prompts

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

    # Step 5: Calculate the actual canvas dimensions including gaps
    # Calculate the width of each network block (seeds_per_row * IMAGE_SIZE)
    network_block_width = seeds_per_row * IMAGE_SIZE

    # Calculate total width including borders and gaps
    base_width = (n_networks * network_block_width) + (
        2 * IMAGE_SIZE * 2
    )  # Add left and right borders (2x width)
    if n_networks > 1:
        base_width += (n_networks - 1) * network_gap  # Add gaps between networks

    # Calculate total height including borders, gaps, and progress bar
    base_height = (rows * IMAGE_SIZE) + (2 * IMAGE_SIZE)  # Add top and bottom borders
    if n_prompts > 1:
        base_height += (n_prompts - 1) * prompt_gap  # Add gaps between prompts

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

            # Calculate y-position accounting for prompt gaps
            y_offset = (row_idx + 1) * IMAGE_SIZE
            if prompt_idx > 0:
                y_offset += prompt_idx * prompt_gap

            # Left border
            left_border_x = 0
            left_border_y = y_offset

            # Create and paste prompt tile (now 2x width)
            prompt_tile = create_prompt_title_card(
                prompt, IMAGE_SIZE, font_path, base_font_size
            )
            base_canvas.paste(prompt_tile, (left_border_x, left_border_y))

            # Right border - same content as left
            right_border_x = base_width - (IMAGE_SIZE * 2)  # Adjust for 2x width
            right_border_y = left_border_y
            base_canvas.paste(prompt_tile, (right_border_x, right_border_y))

    # Define abbreviations for network names
    abbreviations = {"FluxSchnell": "Flux", "SDXLTurbo": "SDXL"}

    # Step 7: Draw network information in top and bottom borders - once per network block
    for network_idx, network in enumerate(all_networks):
        # Create abbreviated network string
        abbreviated_network = [abbreviations.get(net, net) for net in network]
        network_str = "→".join(abbreviated_network)

        # Calculate the width of this network's block
        network_width = seeds_per_row * IMAGE_SIZE

        # Calculate x-position accounting for network gaps
        x_offset = IMAGE_SIZE * 2 + (
            network_idx * network_width
        )  # Adjust for 2x width prompt cards
        if network_idx > 0:
            x_offset += network_idx * network_gap

        # Create a network banner that spans the entire network block
        network_banner = create_network_title_banner(
            network_str, network_width, IMAGE_SIZE, font_path, base_font_size
        )

        # Top border
        top_border_y = 0
        base_canvas.paste(network_banner, (x_offset, top_border_y))

        # Bottom border - same content as top
        bottom_border_y = base_height - IMAGE_SIZE
        base_canvas.paste(network_banner, (x_offset, bottom_border_y))

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
            # Calculate background color that oscillates between black and 5% grey (0.1Hz)
            # 0.1Hz means one complete cycle every 10 seconds, or every (10 * fps) frames

            cycle_frames = 10 * fps  # Number of frames for one complete oscillation
            # Calculate value between 0 and 1 representing position in the cycle
            cycle_position = math.sin(2 * math.pi * frame_idx / cycle_frames)
            # Convert to a value between 0 and 13 (5% grey is roughly RGB(13,13,13))
            grey_value = int(6.5 + 6.5 * cycle_position)  # Oscillates between 0 and 13
            background_color = (grey_value, grey_value, grey_value)

            # Create a new canvas with the calculated background color
            mosaic = Image.new("RGB", (canvas_width, canvas_height), background_color)
            # Paste the base borders and titles onto the background
            mosaic.paste(base_canvas, (0, 0), base_canvas.convert("RGBA"))
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
                            # Calculate prompt and network indexes for this cell
                            prompt_idx = row // rows_per_prompt
                            network_idx = col // seeds_per_row

                            # Account for borders (+1 to both row and column)
                            x_offset = (
                                (col % seeds_per_row)
                                + (network_idx * seeds_per_row)
                                + 2
                            )  # +2 for 2x width prompt cards
                            y_offset = (
                                (row % rows_per_prompt)
                                + (prompt_idx * rows_per_prompt)
                                + 1
                            )

                            # Convert to pixel coordinates
                            x_offset = x_offset * IMAGE_SIZE
                            y_offset = y_offset * IMAGE_SIZE

                            # Add gaps where there are prompt or network changes
                            if prompt_idx > 0:
                                y_offset += prompt_idx * prompt_gap

                            if network_idx > 0:
                                x_offset += network_idx * network_gap

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
            output_file,
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
            logger.info(f"Video created successfully: {output_file}")
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


def export_timeline(
    run_ids: list[str],
    session: Session,
    images_per_run: int,
    output_file: str,
    prompt_order: list[str] = None,
) -> None:
    """
    Export a timeline image showing the progression of multiple runs, organized by
    prompt (rows) and network (columns) with informative borders.

    Args:
        run_ids: List of run IDs to include in the timeline
        session: SQLModel Session for database operations
        images_per_run: Number of evenly-spaced images to show from each run
        output_file: Path to save the output image
        prompt_order: Optional custom ordering for prompts (default: alphabetical)
    """
    # Setup output directory
    output_dir = os.path.dirname(output_file)
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

    # Calculate seeds per row (runs per prompt-network combination)
    max_seeds = 0
    for prompt in ordered_prompts:
        for network in all_networks:
            if network in prompt_network_runs[prompt]:
                max_seeds = max(max_seeds, len(prompt_network_runs[prompt][network]))

    # Calculate grid dimensions
    n_prompts = len(ordered_prompts)
    n_networks = len(all_networks)
    rows = n_prompts * max_seeds
    cols = n_networks

    # Calculate the total width of the timeline for each run
    timeline_width = images_per_run * IMAGE_SIZE

    # Add spacing between network columns
    network_spacing = 20
    timeline_total_width = cols * timeline_width
    if cols > 1:
        timeline_total_width += (cols - 1) * network_spacing

    # Calculate canvas width without left border (since we're removing it)
    canvas_width = timeline_total_width

    # Set up font for text - single font size for all text
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font_size = int(IMAGE_SIZE * 0.25)
    font = ImageFont.truetype(font_path, font_size)

    # Calculate spacing based on font size
    prompt_text_height = font_size * 2  # Height for prompt text area
    prompt_spacing = prompt_text_height + font_size  # Spacing between prompt blocks
    network_banner_height = font_size * 2  # Height for network banner

    # Calculate canvas height
    # Network banner + all rows of images + spacing between prompt blocks
    canvas_height = (
        network_banner_height
        + (rows * IMAGE_SIZE)
        + ((n_prompts - 1) * prompt_spacing if n_prompts > 1 else 0)
    )

    # Create the canvas
    canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))

    # Helper function to create a horizontal text banner for prompts
    def create_prompt_banner(text, width, height, font):
        """Creates a horizontal banner with centered text"""
        banner = Image.new(
            "RGB", (width, height), color=(240, 240, 240)
        )  # Light gray background
        draw = ImageDraw.Draw(banner)

        # Center text horizontally and vertically
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            text_width = draw.textlength(text, font=font)
            text_height = font.size

        x = (width - text_width) / 2
        y = (height - text_height) / 2

        draw.text((x, y), text, font=font, fill=(0, 0, 0))
        return banner

    # Create single network banner for the top
    network_banner = Image.new(
        "RGB", (canvas_width, network_banner_height), color="white"
    )
    draw = ImageDraw.Draw(network_banner)

    # Create network labels with positions
    for network_idx, network in enumerate(all_networks):
        network_str = "network: " + "→".join(network)

        # Calculate position for this network
        col_start = network_idx * (timeline_width + network_spacing)
        col_center = col_start + (timeline_width / 2)

        # Draw network text centered in its column
        try:
            bbox = draw.textbbox((0, 0), network_str, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            text_width = draw.textlength(network_str, font=font)
            text_height = font.size

        x = col_center - (text_width / 2)
        y = (network_banner_height - text_height) / 2

        draw.text((x, y), network_str, font=font, fill=(0, 0, 0))

    # Paste the network banner at the top
    canvas.paste(network_banner, (0, 0))

    # Calculate the starting Y position for placing images (after network banner)
    content_start_y = network_banner_height

    # Now place prompt headers and images
    for prompt_idx, prompt in enumerate(ordered_prompts):
        # Calculate Y position for this prompt's block
        block_start_y = content_start_y
        if prompt_idx > 0:
            # Add height of previous blocks + prompt spacing
            previous_blocks_height = prompt_idx * max_seeds * IMAGE_SIZE
            previous_spacing = prompt_idx * prompt_spacing
            block_start_y += previous_blocks_height + previous_spacing

        # Create and place the prompt banner above this block
        prompt_banner = create_prompt_banner(
            "initial prompt: " + prompt, canvas_width, prompt_text_height, font
        )
        prompt_banner_y = (
            block_start_y - prompt_text_height if prompt_idx > 0 else content_start_y
        )
        canvas.paste(prompt_banner, (0, prompt_banner_y))

        # Place images for each run with this prompt
        for network_idx, network in enumerate(all_networks):
            if network in prompt_network_runs[prompt]:
                # Sort runs with same prompt and network by run_id
                sorted_runs = sorted(
                    prompt_network_runs[prompt][network], key=lambda r: str(r.id)
                )

                for seed_idx, run in enumerate(sorted_runs):
                    if seed_idx >= max_seeds:
                        continue  # Skip if we have more runs than max_seeds

                    # Get all image invocations
                    image_invocations = [
                        inv
                        for inv in run.invocations
                        if inv.type == InvocationType.IMAGE
                    ]
                    image_invocations.sort(key=lambda inv: inv.sequence_number)

                    if not image_invocations:
                        logger.warning(f"No images found for run {run.id}")
                        continue

                    # Select evenly-spaced images
                    selected_images = []
                    if len(image_invocations) <= images_per_run:
                        # Use all available images if we have fewer than requested
                        selected_images = image_invocations
                    else:
                        # Select evenly-spaced images
                        step = (
                            (len(image_invocations) - 1) / (images_per_run - 1)
                            if images_per_run > 1
                            else 0
                        )
                        for i in range(images_per_run):
                            idx = min(int(i * step), len(image_invocations) - 1)
                            selected_images.append(image_invocations[idx])

                    # Calculate row position within this prompt block
                    if prompt_idx == 0:
                        # For first prompt, start after the prompt banner
                        y_offset = (
                            block_start_y + prompt_text_height + (seed_idx * IMAGE_SIZE)
                        )
                    else:
                        y_offset = block_start_y + (seed_idx * IMAGE_SIZE)

                    # Calculate horizontal position
                    col_start = network_idx * (timeline_width + network_spacing)

                    # Paste the selected images
                    for img_idx, invocation in enumerate(selected_images):
                        if invocation.output is not None:
                            x_offset = col_start + img_idx * IMAGE_SIZE
                            canvas.paste(invocation.output, (x_offset, y_offset))

    # Save the final image
    canvas.save(output_file, format="JPEG", quality=95)
    logger.info(f"Timeline image saved to: {output_file}")


def export_mosaic_image(
    label_invocations: Dict[str, List[UUID]],
    session: Session,
    output_file: str,
) -> None:
    """
    Export a mosaic image where each row corresponds to a label and contains all images
    associated with that label.

    Args:
        label_invocations: Dictionary mapping labels to lists of Invocation UUIDs
        session: SQLModel Session for database operations
        output_file: Path to save the output mosaic image
        font_size: Font size for label text (default: 16)
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    font_size = 32

    # Set up font for labels
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, font_size)

    # Determine dimensions
    n_labels = len(label_invocations)
    # Verify all lists have the same length and get that length
    list_lengths = [len(invs) for invs in label_invocations.values()]
    if not all(length == list_lengths[0] for length in list_lengths):
        logger.warning(
            "Not all label lists have the same length. Using maximum length."
        )
    n_images_per_row = max(list_lengths)

    # Calculate label area height (font size + padding)
    label_height = font_size + 10

    # Define row gap as 2x the font size
    row_gap = 2 * font_size

    # Calculate dimensions of the entire mosaic
    mosaic_width = n_images_per_row * IMAGE_SIZE
    # Add row gaps between rows (n_labels - 1 gaps)
    mosaic_height = (
        n_labels * (IMAGE_SIZE + label_height) + (n_labels - 1) * row_gap
        if n_labels > 1
        else n_labels * (IMAGE_SIZE + label_height)
    )

    # Create canvas for the mosaic
    mosaic = Image.new("RGB", (mosaic_width, mosaic_height), (255, 255, 255))
    draw = ImageDraw.Draw(mosaic)

    # For each label, create a row of images with the label above
    row_y = 0
    for label, invocation_uuids in label_invocations.items():
        # Draw label
        draw.text((10, row_y), label, font=font, fill=(0, 0, 0))

        # Move down to start the image row
        row_y += label_height

        # Place each image in the row
        for i, inv_uuid in enumerate(invocation_uuids):
            if i >= n_images_per_row:
                break  # Only include up to n_images_per_row images

            # Read invocation from database
            invocation = read_invocation(inv_uuid, session)
            x_offset = i * IMAGE_SIZE
            # the image is actually the output of the previous invocation
            mosaic.paste(invocation.input_invocation.output, (x_offset, row_y))

        # Move down to the next row, adding the row gap
        row_y += IMAGE_SIZE + row_gap

    # Save the final mosaic
    mosaic.save(output_file, format="JPEG", quality=95)
    logger.info(f"Mosaic image saved to: {output_file}")


def image_grid(
    images: list[Image.Image],
    image_size: int = 64,
    aspect_ratio: float = 16 / 9,
    output_file: str = "output/vis/grid.jpg",
) -> None:
    """
    Create a grid of images with an aspect ratio as close as possible to the target.

    Args:
        images: List of PIL Image objects to arrange in a grid
        image_size: Size to resize each image to (square)
        aspect_ratio: Target aspect ratio for the overall grid (width/height)
        output_file: Path to save the output grid image
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    # Calculate optimal number of columns and rows
    n_images = len(images)

    # Direct calculation of columns based on aspect ratio
    cols = int(math.sqrt(n_images * aspect_ratio))
    rows = math.ceil(n_images / cols)

    # Create the grid image
    grid_width = cols * image_size
    grid_height = rows * image_size
    grid = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))

    # Paste each image into the grid
    for idx, img in enumerate(images):
        if idx >= n_images or img is None:
            continue

        # Calculate position
        row = idx // cols
        col = idx % cols

        resized_img = img.resize((image_size, image_size))

        # Calculate coordinates
        x = col * image_size
        y = row * image_size

        # Paste the image
        grid.paste(resized_img, (x, y))

    # Save the grid
    grid.save(output_file, format="JPEG", quality=95)
    logger.info(f"Grid image saved to: {output_file}")
