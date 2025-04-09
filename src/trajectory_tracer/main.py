import json
import logging
from datetime import datetime
from pathlib import Path
from uuid import UUID

import typer

import trajectory_tracer.engine as engine
from trajectory_tracer.datavis import paper_charts
from trajectory_tracer.db import (
    count_invocations,
    create_db_and_tables,
    get_session_from_connection_string,
    latest_experiment,
    list_experiments,
    list_runs,
    print_experiment_info,
)
from trajectory_tracer.db import delete_experiment as db_delete_experiment
from trajectory_tracer.embeddings import list_models as list_embedding_models
from trajectory_tracer.export import (
    export_video,
    order_runs_for_mosaic,
)
from trajectory_tracer.genai_models import get_output_type
from trajectory_tracer.genai_models import list_models as list_genai_models
from trajectory_tracer.schemas import ExperimentConfig

# NOTE: all these logging shenanigans are required because it's not otherwise
# possible to shut pyvips (a dep of moondream) up

# Set up logging first, before any handlers might be added by other code
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Create a special NullHandler that will silently discard all VIPS messages
class VIPSNullHandler(logging.Handler):
    def emit(self, record):
        pass


# Create a separate logger for VIPS messages
vips_logger = logging.Logger("VIPS")
vips_logger.addHandler(VIPSNullHandler())
vips_logger.propagate = False  # Don't propagate to root logger

# Store the original getLogger method
original_getLogger = logging.getLogger


# Define a replacement getLogger that catches VIPS loggers
def patched_getLogger(name=None):
    if name == "VIPS" or (isinstance(name, str) and "VIPS" in name):
        return vips_logger
    return original_getLogger(name)


# Replace the standard getLogger method
logging.getLogger = patched_getLogger

# Also capture direct root logger messages about VIPS
original_log = logging.Logger._log


def patched_log(self, level, msg, args, exc_info=None, extra=None, stack_info=False):
    if isinstance(msg, str) and "VIPS:" in msg:
        return None  # Skip logging VIPS messages
    return original_log(self, level, msg, args, exc_info, extra, stack_info)


logging.Logger._log = patched_log

# Get a logger for this module
logger = logging.getLogger(__name__)

app = typer.Typer()


@app.command("perform-experiment")
def perform_experiment(
    config_file: Path = typer.Argument(
        ...,
        help="Path to the configuration JSON file",
        exists=True,
        readable=True,
        file_okay=True,
        dir_okay=False,
    ),
    db_path: Path = typer.Option(
        "db/trajectory_data.sqlite",
        "--db-path",
        "-d",
        help="Path to the SQLite database file",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
):
    """
    Run a trajectory tracer experiment defined in CONFIG_FILE.

    The CONFIG_FILE should be a JSON file containing experiment parameters that can be
    parsed into an ExperimentConfig object.
    """
    # Configure logging based on verbosity
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    logger.info(f"Loading configuration from {config_file}")

    try:
        with open(config_file, "r") as f:
            config_data = json.load(f)

        # Handle seed_count if present
        if "seed_count" in config_data:
            seed_count = config_data.pop("seed_count")
            config_data["seeds"] = [-1] * seed_count
            logger.info(
                f"Using seed_count={seed_count}, generated {seed_count} seeds with value -1"
            )

        # Create database engine and tables
        db_str = f"sqlite:///{db_path}"
        logger.info(f"Creating/connecting to database at {db_path}")

        # Call the create_db_and_tables function
        create_db_and_tables(db_str)

        # Create experiment config from JSON and save to database
        with get_session_from_connection_string(db_str) as session:
            config = ExperimentConfig(**config_data)
            session.add(config)
            session.commit()
            session.refresh(config)

            # Calculate total number of runs that will be generated
            total_runs = len(config.networks) * len(config.seeds) * len(config.prompts)

            logger.info(
                f"Configuration loaded successfully: {len(config.networks)} networks, "
                f"{len(config.seeds)} seeds, {len(config.prompts)} prompts, "
                f"for a total of {total_runs} runs"
            )

            # Get the config ID to pass to the engine
            config_id = str(config.id)

        # Run the experiment
        logger.info(f"Starting experiment with config ID: {config_id}")
        engine.perform_experiment(config_id, db_str)

        logger.info(f"Experiment completed successfully. Results saved to {db_path}")
    except Exception as e:
        logger.error(f"Early termination of experiment: {e}")
        raise typer.Exit(code=1)


@app.command("list-experiments")
def list_experiments_command(
    db_path: Path = typer.Option(
        "db/trajectory_data.sqlite",
        "--db-path",
        "-d",
        help="Path to the SQLite database file",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show full experiment details"
    ),
):
    """
    List all experiments stored in the database.

    Displays experiment IDs and basic information about each experiment.
    Use --verbose for more detailed output.
    """
    # Create database connection
    db_str = f"sqlite:///{db_path}"
    logger.info(f"Connecting to database at {db_path}")

    # List all experiments using the function from db module
    with get_session_from_connection_string(db_str) as session:
        experiments = list_experiments(session)

        if not experiments:
            typer.echo("No experiments found in the database.")
            return

        typer.echo(f"Found {len(experiments)} experiments:")

        for experiment in experiments:
            run_count = len(experiment.runs)

            if verbose:
                # Detailed output
                # Print experiment status
                print_experiment_info(experiment, session)
                typer.echo("\n---\n")
            else:
                # Simple output
                elapsed = (
                    (experiment.completed_at or datetime.now()) - experiment.started_at
                ).total_seconds()
                typer.echo(
                    f"{experiment.id} - runs: {run_count}, "
                    f"max_length: {experiment.max_length}, "
                    f"started: {experiment.started_at.strftime('%Y-%m-%d %H:%M')}, "
                    f"elapsed: {elapsed:.1f}s"
                )


@app.command("experiment-status")
def experiment_status(
    experiment_id: str = typer.Argument(
        None,
        help="ID of the experiment to check status for (defaults to the most recent experiment)",
    ),
    db_path: Path = typer.Option(
        "db/trajectory_data.sqlite",
        "--db-path",
        "-d",
        help="Path to the SQLite database file",
    ),
):
    """
    Get the status of a trajectory tracer experiment.

    Shows the progress of the experiment, including invocation, embedding, and
    persistence diagram completion percentages.
    """
    # Create database connection
    db_str = f"sqlite:///{db_path}"
    logger.info(f"Connecting to database at {db_path}")

    # Get the experiment and print status
    with get_session_from_connection_string(db_str) as session:
        experiment = None

        if experiment_id is None or experiment_id.lower() == "latest":
            experiment = latest_experiment(session)
            if not experiment:
                logger.error("No experiments found in the database")
                raise typer.Exit(code=1)
            logger.info(f"Using latest experiment with ID: {experiment.id}")
        else:
            experiment = session.get(ExperimentConfig, UUID(experiment_id))
            if not experiment:
                logger.error(f"Experiment with ID {experiment_id} not found")
                raise typer.Exit(code=1)

        print_experiment_info(experiment, session)


@app.command("delete-experiment")
def delete_experiment(
    experiment_id: str = typer.Argument(
        ...,
        help="ID of the experiment to delete",
    ),
    db_path: Path = typer.Option(
        "db/trajectory_data.sqlite",
        "--db-path",
        "-d",
        help="Path to the SQLite database file",
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompt"),
):
    """
    Delete an experiment and all associated data from the database.

    This will permanently remove the experiment and all its runs, invocations,
    embeddings, and persistence diagrams.
    """
    try:
        # Create database connection
        db_str = f"sqlite:///{db_path}"
        logger.info(f"Connecting to database at {db_path}")

        # Get the experiment and confirm deletion
        with get_session_from_connection_string(db_str) as session:
            try:
                experiment = session.get(ExperimentConfig, UUID(experiment_id))
                if not experiment:
                    logger.error(f"Experiment with ID {experiment_id} not found")
                    raise typer.Exit(code=1)
            except ValueError as e:
                logger.error(f"Invalid experiment ID format: {e}")
                raise typer.Exit(code=1)

            # Show experiment details and confirm deletion
            run_count = len(experiment.runs)
            typer.echo(f"Experiment ID: {experiment.id}")
            typer.echo(f"Started: {experiment.started_at}")
            typer.echo(f"Runs: {run_count}")

            if not force:
                confirm = typer.confirm(
                    f"Are you sure you want to delete this experiment and all its {run_count} runs?",
                    default=False,
                )
                if not confirm:
                    typer.echo("Deletion cancelled.")
                    return

            # Delete the experiment
            result = db_delete_experiment(experiment.id, session)

            if result:
                typer.echo(f"Experiment {experiment_id} successfully deleted.")
            else:
                typer.echo(f"Failed to delete experiment {experiment_id}.")

    except Exception as e:
        logger.error(f"Error deleting experiment: {e}")
        raise typer.Exit(code=1)


@app.command("list-runs")
def list_runs_command(
    db_path: Path = typer.Option(
        "db/trajectory_data.sqlite",
        "--db-path",
        "-d",
        help="Path to the SQLite database file",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show full run details"
    ),
):
    """
    List all runs stored in the database.

    Displays run IDs and basic information about each run.
    Use --verbose for more detailed output.
    """
    try:
        # Create database connection
        db_str = f"sqlite:///{db_path}"
        logger.info(f"Connecting to database at {db_path}")

        # List all runs
        with get_session_from_connection_string(db_str) as session:
            runs = list_runs(session)

            if not runs:
                typer.echo("No runs found in the database.")
                return

            for run in runs:
                if verbose:
                    # Detailed output
                    typer.echo(f"\nRun ID: {run.id}")
                    typer.echo(f"  Network: {run.network}")
                    typer.echo(f"  Initial prompt: {run.initial_prompt}")
                    typer.echo(f"  Seed: {run.seed}")
                    typer.echo(f"  Length: {len(run.invocations)}")
                    typer.echo(f"  Stop reason: {run.stop_reason}")
                else:
                    # Simple output
                    typer.echo(
                        f"{run.id} (seed {run.seed}) - length: {len(run.invocations)}/{run.max_length}, stop reason: {run.stop_reason}"
                    )
            typer.echo(
                f"Found {len(runs)} runs ({count_invocations(session)} invocations in total):"
            )

    except Exception as e:
        logger.error(f"Error listing runs: {e}")
        raise typer.Exit(code=1)


@app.command("list-models")
def list_models():
    """
    List all available GenAI and embedding models with their output types.

    This is useful when creating experiment configurations and you need to know
    what models are available and their expected output types.
    """

    typer.echo("## Available GenAI Models:")

    # Group models by output type
    models_by_type = {}
    for model_name in list_genai_models():
        output_type = get_output_type(model_name).value
        if output_type not in models_by_type:
            models_by_type[output_type] = []
        models_by_type[output_type].append(model_name)

    # Print models grouped by output type
    for output_type, models in models_by_type.items():
        typer.echo(f"\n  Output Type: {output_type}")
        for model_name in models:
            typer.echo(f"    {model_name}")

    typer.echo("\n## Available Embedding Models:")

    # List embedding models using the helper function
    for model_name in list_embedding_models():
        typer.echo(f"  {model_name}")


@app.command("export-video")
def export_video_command(
    experiment_ids: list[str] = typer.Argument(
        ...,
        help="One or more Experiment IDs to include in the mosaic video",
    ),
    fps: int = typer.Option(
        2,
        "--fps",
        "-f",
        help="Frames per second for the output video (default: 2)",
    ),
    resolution: str = typer.Option(
        "HD",
        "--resolution",
        "-r",
        help="Target resolution for the output video: HD, 4K, or 8K (default: HD)",
    ),
    db_path: Path = typer.Option(
        "db/trajectory_data.sqlite",
        "--db-path",
        "-d",
        help="Path to the SQLite database file",
    ),
    output_dir: Path = typer.Option(
        "output/mosaic",
        "--output-dir",
        "-o",
        help="Directory to save the mosaic video",
    ),
):
    """
    Generate a mosaic video from all runs in one or more specified experiments.

    Creates a grid of images showing the progression of multiple runs side by side,
    and renders them as a video file named 'mosaic.mp4' in a subdirectory named
    after the first experiment ID within the specified output directory.
    """
    try:
        # Create database connection
        db_str = f"sqlite:///{db_path}"
        logger.info(f"Connecting to database at {db_path}")

        all_runs = []
        valid_experiment_ids = []

        # Get the experiments and collect runs
        with get_session_from_connection_string(db_str) as session:
            for experiment_id_str in experiment_ids:
                # Validate UUID format
                try:
                    experiment_uuid = UUID(experiment_id_str)
                except ValueError:
                    logger.error(
                        f"Invalid experiment ID format: '{experiment_id_str}'. Please provide a valid UUID."
                    )
                    raise typer.Exit(code=1)

                # Fetch experiment
                experiment = session.get(ExperimentConfig, experiment_uuid)
                if not experiment:
                    logger.warning(
                        f"Experiment with ID {experiment_id_str} not found. Skipping."
                    )
                    continue  # Skip to the next experiment ID

                # Check for runs
                if not experiment.runs:
                    logger.warning(
                        f"No runs found for experiment {experiment_id_str}. Skipping."
                    )
                    continue  # Skip to the next experiment ID

                # Collect runs and valid ID
                all_runs.extend(experiment.runs)
                valid_experiment_ids.append(experiment_id_str)
                logger.info(
                    f"Added {len(experiment.runs)} runs from experiment {experiment_id_str}"
                )

            # Check if any runs were collected at all
            if not all_runs:
                logger.error(
                    "No valid runs found for any of the specified experiment IDs."
                )
                raise typer.Exit(code=1)

            # Get run IDs as strings
            run_ids = [str(run.id) for run in all_runs]
            logger.info(f"Total runs collected for mosaic: {len(run_ids)}")

            # Sort them so they're in nice orders
            run_ids = order_runs_for_mosaic(run_ids, session)

            # Define output path using the first valid experiment ID for the subdirectory name
            first_experiment_id = valid_experiment_ids[0]
            output_subdir = output_dir / first_experiment_id
            output_subdir.mkdir(parents=True, exist_ok=True)
            output_video_path = output_subdir / "mosaic.mp4"

            logger.info(f"Preparing to export mosaic video to {output_video_path}")

            # Create the mosaic video
            export_video(
                run_ids=run_ids,
                session=session,
                fps=fps,
                resolution=resolution,
                output_video=str(output_video_path),
            )

            logger.info(f"Mosaic video successfully created at {output_video_path}")

    except Exception as e:
        # Catch any other unexpected errors during the process
        logger.error(
            f"An error occurred during mosaic video creation: {e}", exc_info=True
        )
        raise typer.Exit(code=1)


@app.command("doctor")
def doctor_command(
    experiment_id: str = typer.Argument(
        ...,
        help="ID of the experiment to diagnose and fix",
    ),
    db_path: Path = typer.Option(
        "db/trajectory_data.sqlite",
        "--db-path",
        "-d",
        help="Path to the SQLite database file",
    ),
    fix: bool = typer.Option(
        False, "--fix", "-f", help="Fix issues that are found (default: report only)"
    ),
):
    """
    Diagnose and optionally fix issues with an experiment's data.

    Performs checks on the experiment's runs, invocations, embeddings, and persistence
    diagrams to ensure data integrity and completeness. Use the --fix flag to
    automatically repair issues that are found.
    """
    # Create database connection
    db_str = f"sqlite:///{db_path}"
    logger.info(f"Connecting to database at {db_path}")

    # Validate experiment ID format
    try:
        UUID(experiment_id)
    except ValueError:
        logger.error(f"Invalid experiment ID format: {experiment_id}")
        raise typer.Exit(code=1)

    logger.info(f"Running diagnostics on experiment {experiment_id}")
    if fix:
        logger.info("Fix mode enabled - will attempt to repair issues found")

    # Call the experiment_doctor function from the engine module
    engine.experiment_doctor(experiment_id, db_str, fix)

    logger.info("Experiment diagnostic completed")


@app.command("paper-charts")
def paper_charts_command(
    db_path: Path = typer.Option(
        "db/trajectory_data.sqlite",
        "--db-path",
        "-d",
        help="Path to the SQLite database file",
    ),
):
    """
    Generate charts for publication using data from specific experiments.
    """

    # Create database connection
    db_str = f"sqlite:///{db_path}"
    logger.info(f"Connecting to database at {db_path}")

    # Call the paper_charts function with a database session
    with get_session_from_connection_string(db_str) as session:
        paper_charts(session)
        logger.info("Paper charts generation completed successfully.")


@app.command("script")
def script():
    # Create database connection
    db_str = "sqlite:///db/trajectory_data.sqlite"
    logger.info("Connecting to database to process all experiments...")

    with get_session_from_connection_string(db_str) as session:
        session


if __name__ == "__main__":
    app()
