import os

import pytest

from panic_tda.analysis import load_invocations_df, load_embeddings_df, load_runs_df
from panic_tda.datavis import (
    plot_invocation_duration,
    plot_persistence_diagram,
    plot_persistence_diagram_by_run,
    plot_persistence_diagram_faceted,
    plot_persistence_entropy,
    plot_semantic_drift,
)
from panic_tda.engine import perform_experiment
from panic_tda.schemas import ExperimentConfig


@pytest.fixture
def mock_experiment_data(db_session):
    experiment = ExperimentConfig(
        networks=[
            ["DummyT2I", "DummyI2T"],
            ["DummyT2I", "DummyI2T2"],
            ["DummyT2I2", "DummyI2T2"],
            ["DummyT2I2", "DummyI2T"],
        ],
        prompts=["one fish", "two fish", "red fish", "blue fish"],
        seeds=[-1] * 4,
        embedding_models=["Dummy", "Dummy2"],
        max_length=10,  # Short sequences for testing
    )

    # Save experiment to database to get an ID
    db_session.add(experiment)
    db_session.commit()
    db_session.refresh(experiment)

    # Run the experiment to populate database with dummy model runs
    db_url = str(db_session.get_bind().engine.url)
    perform_experiment(str(experiment.id), db_url)
    db_session.refresh(experiment)
    # Run the actual experiment using the configuration
    # The dummy models are efficient and won't take long
    runs_df = load_runs_df(db_session, use_cache=False)
    embeddings_df = load_embeddings_df(db_session, use_cache=False)
    invocations_df = load_invocations_df(db_session, use_cache=False)

    return {
        "runs_df": runs_df,
        "embeddings_df": embeddings_df,
        "invocations_df": invocations_df,
    }


def test_plot_persistence_diagram(mock_experiment_data):
    runs_df = mock_experiment_data["runs_df"]

    # Verify we have persistence diagram data
    assert runs_df.height > 0
    assert "homology_dimension" in runs_df.columns
    assert "birth" in runs_df.columns
    assert "persistence" in runs_df.columns

    # Define output file
    output_file = "output/test/persistence_diagram.png"

    # Generate the plot
    plot_persistence_diagram(runs_df, output_file)

    # Verify file was created
    assert os.path.exists(output_file), f"File was not created: {output_file}"


def test_plot_persistence_diagram_faceted(mock_experiment_data):
    runs_df = mock_experiment_data["runs_df"]

    # Verify we have persistence diagram data
    assert runs_df.height > 0
    assert "homology_dimension" in runs_df.columns
    assert "birth" in runs_df.columns
    assert "persistence" in runs_df.columns
    assert "text_model" in runs_df.columns
    assert "image_model" in runs_df.columns

    # Define output file
    output_file = "output/test/persistence_diagram_faceted.png"

    # Generate the plot
    plot_persistence_diagram_faceted(runs_df, output_file)

    # Verify file was created
    assert os.path.exists(output_file), f"File was not created: {output_file}"


def test_plot_persistence_diagram_by_run(mock_experiment_data):
    runs_df = mock_experiment_data["runs_df"]

    # Verify we have persistence diagram data
    assert runs_df.height > 0
    assert "homology_dimension" in runs_df.columns
    assert "birth" in runs_df.columns
    assert "persistence" in runs_df.columns
    assert "run_id" in runs_df.columns

    # Define output file
    output_file = "output/test/persistence_diagram_by_run.png"

    plot_persistence_diagram_by_run(runs_df, output_file)

    # Verify file was created
    assert os.path.exists(output_file), f"File was not created: {output_file}"


def test_plot_semantic_drift(mock_experiment_data):
    embeddings_df = mock_experiment_data["embeddings_df"]

    # Verify we have semantic dispersion data
    assert embeddings_df.height > 0
    assert "semantic_drift_instantaneous" in embeddings_df.columns
    assert "sequence_number" in embeddings_df.columns
    assert "run_id" in embeddings_df.columns
    assert "initial_prompt" in embeddings_df.columns
    assert "embedding_model" in embeddings_df.columns

    # Define output file
    output_file = "output/test/semantic_drift.png"

    # Generate the plot
    plot_semantic_drift(embeddings_df, output_file)

    # Verify file was created
    assert os.path.exists(output_file), f"File was not created: {output_file}"


def test_plot_persistence_entropy(mock_experiment_data):
    runs_df = mock_experiment_data["runs_df"]

    # Verify we have persistence diagram data with entropy values
    assert runs_df.height > 0
    assert "homology_dimension" in runs_df.columns
    assert "entropy" in runs_df.columns
    assert "embedding_model" in runs_df.columns
    assert "text_model" in runs_df.columns
    assert "image_model" in runs_df.columns

    # Define output file
    output_file = "output/test/persistence_entropy.png"

    # Generate the plot
    plot_persistence_entropy(runs_df, output_file)

    # Verify file was created
    assert os.path.exists(output_file), f"File was not created: {output_file}"


def test_plot_invocation_duration(mock_experiment_data):
    invocations_df = mock_experiment_data["invocations_df"]

    # Verify we have the necessary columns
    assert invocations_df.height > 0
    assert "duration" in invocations_df.columns
    assert "model" in invocations_df.columns

    # Define output file
    output_file = "output/test/invocation_duration.png"

    # Generate the plot
    plot_invocation_duration(invocations_df, output_file)

    # Verify file was created
    assert os.path.exists(output_file), f"File was not created: {output_file}"
