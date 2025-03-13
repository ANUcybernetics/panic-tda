import json
import logging
from pathlib import Path

import typer

from trajectory_tracer.db import get_database
from trajectory_tracer.engine import perform_experiment
from trajectory_tracer.schemas import ExperimentConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = typer.Typer()

@app.command()
def main(
    config_file: Path = typer.Argument(..., help="Path to the configuration JSON file", exists=True, readable=True, file_okay=True, dir_okay=False),
    db_path: Path = typer.Option("trajectory_data.sqlite", "--db-path", "-d", help="Path to the SQLite database file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
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
        with open(config_file, 'r') as f:
            config_data = json.load(f)

        # Create experiment config from JSON
        config = ExperimentConfig(**config_data)
        logger.info(f"Configuration loaded successfully: {len(config.networks)} networks, "
                   f"{len(config.seeds)} seeds, {len(config.prompts)} prompts")

        # Create database engine and tables
        db_url = f"sqlite:///{db_path}"
        logger.info(f"Creating/connecting to database at {db_path}")

        # Get database instance with the specified path
        database = get_database(connection_string=db_url)

        # Run the experiment
        with database.get_session() as session:
            logger.info("Starting experiment...")
            perform_experiment(config=config, session=session)

        logger.info(f"Experiment completed successfully. Results saved to {db_path}")

    except Exception as e:
        logger.error(f"Early termination of experiment: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
