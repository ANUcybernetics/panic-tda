import os
import logging
from io import BytesIO
from typing import List, Optional
from PIL import Image
from sqlmodel import Session
from trajectory_tracer.schemas import Run, InvocationType

logger = logging.getLogger(__name__)

def export_run_images(run: Run, session: Session, output_dir: str = "image_outputs") -> None:
    """
    Export all image invocations from a run to webp files.

    Args:
        run: The Run object containing invocations
        session: SQLModel Session for database operations
        output_dir: Directory where images will be saved (default: "image_outputs")
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Ensure invocations are loaded
    session.refresh(run)

    # Process each invocation
    for invocation in run.invocations:
        # Skip non-image invocations
        if invocation.type != InvocationType.IMAGE or not invocation.output_image_data:
            continue

        try:
            # Load image from output_image_data
            img = Image.open(BytesIO(invocation.output_image_data))

            # Get prompt text from previous invocation if available
            prompt_text = None
            if invocation.input_invocation_id:
                input_invocation = session.get(invocation.__class__, invocation.input_invocation_id)
                if input_invocation and input_invocation.type == InvocationType.TEXT:
                    prompt_text = input_invocation.output_text

            # Prepare file path
            file_path = os.path.join(output_dir, f"{invocation.id}.webp")

            # Save image with metadata
            metadata = {"prompt": prompt_text} if prompt_text else {}
            img.save(file_path, format="WEBP", lossless=True, quality=100, exif=metadata)

            logger.info(f"Saved image for invocation {invocation.id} to {file_path}")

        except Exception as e:
            logger.error(f"Error exporting image for invocation {invocation.id}: {e}")
