import json
import logging
from pathlib import Path
from uuid import UUID

import typer

import trajectory_tracer.engine as engine
from trajectory_tracer.db import (
    count_invocations,
    create_db_and_tables,
    get_session_from_connection_string,
    latest_experiment,
    list_experiments,
    list_runs,
)
from trajectory_tracer.db import delete_experiment as db_delete_experiment
from trajectory_tracer.embeddings import list_models as list_embedding_models
from trajectory_tracer.genai_models import get_output_type
from trajectory_tracer.genai_models import list_models as list_genai_models
from trajectory_tracer.schemas import ExperimentConfig
from trajectory_tracer.utils import export_run_mosaic, order_runs_for_mosaic

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
            logger.info(f"Using seed_count={seed_count}, generated {seed_count} seeds with value -1")

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
    try:
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
                    experiment.print_status()
                    typer.echo("\n---\n")
                else:
                    # Simple output
                    elapsed = (experiment.completed_at - experiment.started_at).total_seconds()
                    typer.echo(
                        f"{experiment.id} - runs: {run_count}, "
                        f"max_length: {experiment.max_length}, "
                        f"started: {experiment.started_at.strftime('%Y-%m-%d %H:%M')}, "
                        f"elapsed: {elapsed:.1f}s"
                    )

    except Exception as e:
        logger.error(f"Error listing experiments: {e}")
        raise typer.Exit(code=1)


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

        experiment.print_status()


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
    force: bool = typer.Option(
        False, "--force", "-f", help="Skip confirmation prompt"
    ),
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
                    default=False
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
def export_video(
    experiment_id: str = typer.Argument(
        ...,
        help="ID of the experiment to create a mosaic video from",
    ),
    cols: int = typer.Option(
        4,
        "--cols",
        "-c",
        help="Number of columns in the mosaic grid (default: 4)",
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
):
    """
    Generate a mosaic video from all runs in an experiment.

    Creates a grid of images showing the progression of multiple runs side by side,
    and renders them as a video file.
    """
    try:
        # Create database connection
        db_str = f"sqlite:///{db_path}"
        logger.info(f"Connecting to database at {db_path}")

        # Get the experiment and export video
        with get_session_from_connection_string(db_str) as session:

            # Find the experiment by ID
            try:
                experiment = session.get(ExperimentConfig, UUID(experiment_id))
                if not experiment:
                    logger.error(f"Experiment with ID {experiment_id} not found")
                    raise typer.Exit(code=1)
            except ValueError as e:
                logger.error(f"Invalid experiment ID format: {e}")
                raise typer.Exit(code=1)

            # Get all runs for this experiment
            runs = experiment.runs
            if not runs:
                logger.error(f"No runs found for experiment {experiment_id}")
                raise typer.Exit(code=1)

            # Get run IDs as strings
            run_ids = [str(run.id) for run in runs]

            # sort them so they're in nice orders
            run_ids = order_runs_for_mosaic(run_ids, session)

            output_video = f"output/mosaic/{experiment_id}/mosaic.mp4"

            # Create the mosaic video
            export_run_mosaic(
                run_ids=run_ids,
                session=session,
                cols=cols,
                fps=fps,
                resolution=resolution,
                output_video=output_video
            )

    except Exception as e:
        logger.error(f"Error creating mosaic video: {e}")
        raise typer.Exit(code=1)


@app.command("script")
def script():
    """
    Execute a Python script in the context of the application.

    This provides access to the application's database and other utilities,
    allowing for quick development and testing of scripts without needing to set up
    the environment manually.
    """
    # Create database connection
    db_str = "sqlite:///db/trajectory_data.sqlite"
    logger.info("Connecting to database...")

    with get_session_from_connection_string(db_str) as session:
        from trajectory_tracer.visualisation import paper_charts
        paper_charts(session)

    logger.info("Script execution completed")


if __name__ == "__main__":
    app()
