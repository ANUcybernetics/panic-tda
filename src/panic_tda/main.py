import json
import logging
from datetime import datetime
from pathlib import Path
from uuid import UUID

import typer
from sqlmodel import Session, func, select

import panic_tda.engine as engine
from panic_tda.clustering_manager import (
    get_cluster_details,
)
from panic_tda.db import (
    count_invocations,
    create_db_and_tables,
    export_experiments,
    get_session_from_connection_string,
    latest_experiment,
    list_experiments,
    list_runs,
    print_experiment_info,
)
from panic_tda.db import delete_experiment as db_delete_experiment
from panic_tda.embeddings import list_models as list_embedding_models
from panic_tda.export import (
    export_video,
    order_runs_for_mosaic,
)
from panic_tda.genai_models import get_output_type
from panic_tda.genai_models import list_models as list_genai_models
from panic_tda.local import paper_charts
from panic_tda.schemas import ExperimentConfig, EmbeddingCluster

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

# Create subcommand groups
experiment_app = typer.Typer()
run_app = typer.Typer()
cluster_app = typer.Typer()
model_app = typer.Typer()
export_app = typer.Typer()

# Add subcommand groups to main app
app.add_typer(experiment_app, name="experiment", help="Manage experiments")
app.add_typer(
    experiment_app,
    name="experiments",
    help="Manage experiments (alias for experiment)",
    hidden=True,
)
app.add_typer(run_app, name="run", help="Manage runs")
app.add_typer(run_app, name="runs", help="Manage runs (alias for run)", hidden=True)
app.add_typer(cluster_app, name="cluster", help="Manage clustering")
app.add_typer(
    cluster_app,
    name="clusters",
    help="Manage clustering (alias for cluster)",
    hidden=True,
)
app.add_typer(model_app, name="model", help="Manage models")
app.add_typer(export_app, name="export", help="Export data and visualizations")


@experiment_app.command("perform")
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
    Run a panic-tda experiment defined in CONFIG_FILE.

    The CONFIG_FILE should be a JSON file containing experiment parameters that can be
    parsed into an ExperimentConfig object.
    """
    # Configure logging based on verbosity
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    logger.info(f"Loading configuration from {config_file}")

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


@experiment_app.command("resume")
def resume_experiment(
    experiment_id: str = typer.Argument(
        ...,
        help="UUID of the experiment configuration to resume",
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
    Resume a panic-tda experiment by its UUID.

    This will continue processing existing runs or resume runs that didn't complete
    successfully in the original experiment.
    """
    # Configure logging based on verbosity
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create database engine and tables
    db_str = f"sqlite:///{db_path}"
    logger.info(f"Connecting to database at {db_path}")

    # Validate experiment exists
    with get_session_from_connection_string(db_str) as session:
        experiment = session.get(ExperimentConfig, UUID(experiment_id))
        if not experiment:
            logger.error(f"Experiment with ID {experiment_id} not found")
            raise typer.Exit(code=1)

        # Calculate total number of runs
        total_runs = (
            len(experiment.networks) * len(experiment.seeds) * len(experiment.prompts)
        )

        logger.info(
            f"Found experiment with ID {experiment_id}: {len(experiment.networks)} networks, "
            f"{len(experiment.seeds)} seeds, {len(experiment.prompts)} prompts, "
            f"for a total of {total_runs} runs"
        )

    # Run the experiment
    logger.info(f"Resuming experiment with ID: {experiment_id}")
    engine.perform_experiment(experiment_id, db_str)

    logger.info(f"Experiment resumeed successfully. Results saved to {db_path}")


@experiment_app.command("list")
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


@experiment_app.command("show")
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
    Get the status of a panic-tda experiment.

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


@experiment_app.command("delete")
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
    # Create database connection
    db_str = f"sqlite:///{db_path}"
    logger.info(f"Connecting to database at {db_path}")

    with get_session_from_connection_string(db_str) as session:
        # Get the experiment and confirm deletion
        experiment = session.get(ExperimentConfig, UUID(experiment_id))
        if not experiment:
            logger.error(f"Experiment with ID {experiment_id} not found")
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


@run_app.command("list")
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


@model_app.command("list")
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


@export_app.command("video")
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
    output_file: Path = typer.Option(
        "output/videos/mosaic.mp4",
        "--output-file",
        "-o",
        help="Filename to save the mosaic video",
    ),
):
    """
    Generate a mosaic video from all runs in one or more specified experiments.

    Creates a grid of images showing the progression of multiple runs side by side,
    and renders them as a video file named 'mosaic.mp4' in a subdirectory named
    after the first experiment ID within the specified output directory.
    """
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
            logger.error("No valid runs found for any of the specified experiment IDs.")
            raise typer.Exit(code=1)

        # Get run IDs as strings
        run_ids = [str(run.id) for run in all_runs]
        logger.info(f"Total runs collected for mosaic: {len(run_ids)}")

        # Sort them so they're in nice orders
        run_ids = order_runs_for_mosaic(run_ids, session)

        logger.info(f"Preparing to export mosaic video to {output_file}")

        # Create the mosaic video
        export_video(
            run_ids=run_ids,
            session=session,
            fps=fps,
            resolution=resolution,
            output_file=str(output_file),
        )

        logger.info(f"Mosaic video successfully created at {output_file}")


@experiment_app.command("doctor")
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


@export_app.command("charts")
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


@cluster_app.command("embeddings")
def cluster_embeddings_command(
    embedding_model_id: str = typer.Argument(
        "all",
        help="Embedding model ID to cluster (default: 'all' for all models)",
    ),
    db_path: Path = typer.Option(
        "db/trajectory_data.sqlite",
        "--db-path",
        "-d",
        help="Path to the SQLite database file",
    ),
    downsample: int = typer.Option(
        1,
        "--downsample",
        help="Downsampling factor (1 = no downsampling, 10 = every 10th embedding)",
    ),
    epsilon: float = typer.Option(
        0.4,
        "--epsilon",
        "-e",
        help="Epsilon value for HDBSCAN cluster selection (default: 0.4)",
    ),
):
    """
    Run clustering on embeddings in the database.

    By default clusters all available embeddings across all experiments.
    You can specify a specific embedding model to cluster only that model's embeddings.
    """
    # Create database connection
    db_str = f"sqlite:///{db_path}"
    logger.info(f"Connecting to database at {db_path}")
    logger.info(f"Downsampling factor: {downsample}")

    # Use a longer timeout for clustering operations (5 minutes)
    with get_session_from_connection_string(db_str, timeout=300) as session:
        # Import here to avoid circular imports
        from panic_tda.clustering_manager import cluster_all_data

        result = cluster_all_data(session, downsample, embedding_model_id, epsilon)

        if result["status"] == "success":
            typer.echo(
                f"Successfully clustered {result['clustered_embeddings']:,}/{result['total_embeddings']:,} embeddings"
            )
            typer.echo(
                f"Created {result['total_clusters']:,} clusters across {result['embedding_models_count']} embedding models"
            )
        else:
            typer.echo(f"Clustering failed: {result.get('message', 'Unknown error')}")


@cluster_app.command("list")
def list_clusters_command(
    db_path: Path = typer.Option(
        "db/trajectory_data.sqlite",
        "--db-path",
        "-d",
        help="Path to the SQLite database file",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show full cluster details"
    ),
):
    """
    List all clustering results stored in the database.

    Displays clustering result IDs and basic information about each clustering.
    Use --verbose for more detailed output.
    """
    # Create database connection
    db_str = f"sqlite:///{db_path}"
    logger.info(f"Connecting to database at {db_path}")

    with get_session_from_connection_string(db_str) as session:
        # Import here to avoid circular imports
        from panic_tda.clustering_manager import list_clustering_results

        results = list_clustering_results(session)

        if not results:
            typer.echo("No clustering results found in the database.")
            return

        typer.echo(f"Found {len(results)} clustering result(s):")

        for result in results:
            # Count total assignments and outliers
            assignments_count = session.exec(
                select(func.count(EmbeddingCluster.id)).where(
                    EmbeddingCluster.clustering_result_id == result.id
                )
            ).one()

            # Count outlier assignments (cluster_id == -1)
            outlier_count = session.exec(
                select(func.count(EmbeddingCluster.id)).where(
                    EmbeddingCluster.clustering_result_id == result.id,
                    EmbeddingCluster.cluster_id == -1,
                )
            ).one()

            outlier_percentage = (
                (outlier_count / assignments_count * 100)
                if assignments_count > 0
                else 0
            )

            # Count regular clusters (excluding outliers)
            regular_clusters = len([c for c in result.clusters if c.get("id") != -1])

            if verbose:
                # Detailed output
                typer.echo(f"\nClustering Result ID: {result.id}")
                typer.echo(f"  Embedding Model: {result.embedding_model}")
                typer.echo(f"  Algorithm: {result.algorithm}")
                typer.echo(f"  Parameters: {result.parameters}")
                typer.echo(f"  Created: {result.created_at}")
                typer.echo(f"  Total Clusters: {regular_clusters}")
                typer.echo(f"  Total Assignments: {assignments_count}")
                typer.echo(
                    f"  Outliers: {outlier_percentage:.1f}% ({outlier_count:,} embeddings)"
                )
            else:
                # Simple output with cluster_selection_epsilon parameter
                epsilon = result.parameters.get("cluster_selection_epsilon", "N/A")
                typer.echo(
                    f"{result.id} - model: {result.embedding_model}, "
                    f"epsilon: {epsilon}, "
                    f"clusters: {regular_clusters}, "
                    f"assignments: {assignments_count:,}, "
                    f"outliers: {outlier_percentage:.1f}%, "
                    f"created: {result.created_at.strftime('%Y-%m-%d %H:%M')}"
                )


@cluster_app.command("delete")
def delete_cluster_command(
    clustering_id: str = typer.Argument(
        ...,
        help="ID of the clustering result to delete (use 'all' to delete all)",
    ),
    db_path: Path = typer.Option(
        "db/trajectory_data.sqlite",
        "--db-path",
        "-d",
        help="Path to the SQLite database file",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation prompt",
    ),
):
    """
    Delete a clustering result and all associated data from the database.

    This will permanently remove the clustering result and all its cluster assignments.
    Use 'all' as the clustering ID to delete all clustering results.
    """
    # Create database connection
    db_str = f"sqlite:///{db_path}"
    logger.info(f"Connecting to database at {db_path}")

    with get_session_from_connection_string(db_str) as session:
        # Import here to avoid circular imports
        from panic_tda.clustering_manager import (
            delete_cluster_data,
            delete_single_cluster,
            get_clustering_result_by_id,
        )

        if clustering_id.lower() == "all":
            # Delete all clusters
            if not force:
                response = typer.confirm(
                    "Are you sure you want to delete ALL clustering results?"
                )
                if not response:
                    typer.echo("Deletion cancelled.")
                    raise typer.Exit()

            result = delete_cluster_data(session, "all")

            if result["status"] == "success":
                typer.echo(
                    f"Successfully deleted {result['deleted_results']} clustering result(s) "
                    f"and {result['deleted_assignments']} cluster assignments."
                )
            elif result["status"] == "not_found":
                typer.echo("No clustering results found in the database.")
            else:
                typer.echo(f"Deletion failed: {result.get('message', 'Unknown error')}")
        else:
            # Delete single cluster
            try:
                cluster_uuid = UUID(clustering_id)
            except ValueError:
                logger.error(f"Invalid clustering ID format: {clustering_id}")
                raise typer.Exit(code=1)

            # Get the clustering result to show details
            clustering_result = get_clustering_result_by_id(cluster_uuid, session)
            if not clustering_result:
                logger.error(f"Clustering result with ID {clustering_id} not found")
                raise typer.Exit(code=1)

            # Count assignments
            assignments_count = session.exec(
                select(func.count(EmbeddingCluster.id)).where(
                    EmbeddingCluster.clustering_result_id == clustering_result.id
                )
            ).one()

            # Show details and confirm deletion
            typer.echo(f"Clustering Result ID: {clustering_result.id}")
            typer.echo(f"Embedding Model: {clustering_result.embedding_model}")
            typer.echo(f"Algorithm: {clustering_result.algorithm}")
            typer.echo(f"Created: {clustering_result.created_at}")
            typer.echo(f"Clusters: {len(clustering_result.clusters)}")
            typer.echo(f"Assignments: {assignments_count}")

            if not force:
                confirm = typer.confirm(
                    f"Are you sure you want to delete this clustering result?",
                    default=False,
                )
                if not confirm:
                    typer.echo("Deletion cancelled.")
                    return

            # Delete the clustering result
            result = delete_single_cluster(cluster_uuid, session)

            if result["status"] == "success":
                typer.echo(f"Clustering result {clustering_id} successfully deleted.")
            else:
                typer.echo(
                    f"Failed to delete clustering result: {result.get('message', 'Unknown error')}"
                )


@cluster_app.command("show")
def cluster_status_command(
    clustering_id: str = typer.Argument(
        None,
        help="ID of the clustering result to check status for (defaults to the most recent)",
    ),
    db_path: Path = typer.Option(
        "db/trajectory_data.sqlite",
        "--db-path",
        "-d",
        help="Path to the SQLite database file",
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        "-n",
        help="Maximum number of clusters to show",
    ),
):
    """
    Get the status of a clustering result.

    Shows detailed information about a clustering result, including all clusters
    with their medoid text and size, sorted by size.
    """
    # Create database connection
    db_str = f"sqlite:///{db_path}"
    logger.info(f"Connecting to database at {db_path}")

    with get_session_from_connection_string(db_str) as session:
        # Import here to avoid circular imports
        from panic_tda.clustering_manager import (
            get_cluster_details_by_id,
            get_latest_clustering_result,
        )

        details = None

        if clustering_id is None or clustering_id.lower() == "latest":
            # Get the latest clustering result
            clustering_result = get_latest_clustering_result(session)
            if not clustering_result:
                logger.error("No clustering results found in the database")
                raise typer.Exit(code=1)
            logger.info(
                f"Using latest clustering result with ID: {clustering_result.id}"
            )
            details = get_cluster_details_by_id(clustering_result.id, session, limit)
        else:
            # Validate UUID format
            try:
                cluster_uuid = UUID(clustering_id)
            except ValueError:
                logger.error(f"Invalid clustering ID format: {clustering_id}")
                raise typer.Exit(code=1)

            # Get cluster details
            details = get_cluster_details_by_id(cluster_uuid, session, limit)

            if not details:
                logger.error(f"Clustering result with ID {clustering_id} not found")
                raise typer.Exit(code=1)

        # Calculate outlier percentage
        outlier_cluster = next((c for c in details["clusters"] if c["id"] == -1), None)
        outlier_count = outlier_cluster["size"] if outlier_cluster else 0
        outlier_percentage = (
            (outlier_count / details["total_assignments"] * 100)
            if details["total_assignments"] > 0
            else 0
        )

        # Filter out outliers from regular clusters
        regular_clusters = [c for c in details["clusters"] if c["id"] != -1]

        # Print header
        typer.echo(f"Clustering Result ID: {details['clustering_id']}")
        typer.echo(f"Embedding Model: {details['embedding_model']}")
        typer.echo(f"Algorithm: {details['algorithm']}")
        typer.echo(f"Parameters: {details['parameters']}")
        typer.echo(f"Created: {details['created_at']}")
        typer.echo(f"Total clusters: {len(regular_clusters)}")
        typer.echo(f"Total assignments: {details['total_assignments']}")
        typer.echo(
            f"Outliers: {outlier_percentage:.1f}% ({outlier_count:,} embeddings)"
        )
        typer.echo("")

        # Print clusters (already limited by the query)
        if regular_clusters:
            typer.echo(f"Top {len(regular_clusters)} clusters by size:")
            typer.echo("-" * 80)

            for i, cluster in enumerate(regular_clusters):
                text = cluster["medoid_text"]
                if len(text) > 60:
                    text = text[:57] + "..."

                percentage = (
                    (cluster["size"] / details["total_assignments"] * 100)
                    if details["total_assignments"] > 0
                    else 0
                )
                typer.echo(f"{i + 1:3d}. {text:<60} {percentage:>5.1f}%")


@export_app.command("db")
def export_db_command(
    experiment_ids: list[str] = typer.Argument(
        ...,
        help="One or more Experiment IDs to export",
    ),
    target_db: Path = typer.Argument(
        ...,
        help="Path to the target SQLite database file",
    ),
    source_db: Path = typer.Option(
        "db/trajectory_data.sqlite",
        "--source-db",
        "-s",
        help="Path to the source SQLite database file",
    ),
    timeout: int = typer.Option(
        30,
        "--timeout",
        "-t",
        help="SQLite timeout in seconds",
    ),
):
    """
    Export a subset of experiments from one database to another.

    This command copies all data related to the specified experiments while
    maintaining referential integrity. It creates the target database and
    tables if they don't exist.

    Example:
        panic-tda export-db experiment-id-1 experiment-id-2 target.db
    """
    # Create database connections
    source_db_str = f"sqlite:///{source_db}"
    target_db_str = f"sqlite:///{target_db}"

    logger.info(f"Exporting {len(experiment_ids)} experiments")
    logger.info(f"Source database: {source_db}")
    logger.info(f"Target database: {target_db}")

    try:
        export_experiments(
            source_db_str, target_db_str, experiment_ids, timeout=timeout
        )

        typer.echo(
            f"Successfully exported {len(experiment_ids)} experiment(s) to {target_db}"
        )

        # Show summary of what was exported
        with get_session_from_connection_string(target_db_str) as session:
            experiments = list_experiments(session)
            typer.echo(
                f"\nTarget database now contains {len(experiments)} experiment(s):"
            )
            for exp in experiments:
                run_count = len(exp.runs)
                typer.echo(f"  - {exp.id}: {run_count} runs")

    except ValueError as e:
        logger.error(f"Export failed: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"Unexpected error during export: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
