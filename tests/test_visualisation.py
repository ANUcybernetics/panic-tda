import os

from trajectory_tracer.analysis import load_embeddings_df, load_runs_df
from trajectory_tracer.engine import perform_experiment
from trajectory_tracer.schemas import ExperimentConfig
from trajectory_tracer.visualisation import (
    plot_persistence_diagram_faceted,
    plot_persistence_entropy_faceted,
    plot_semantic_drift,
)


def test_plot_persistence_diagram(db_session):
    # Create a simple test configuration with persistence diagrams
    config = ExperimentConfig(
        networks=[["DummyT2I", "DummyI2T"]],
        seeds=[-1],
        prompts=["single fish"],
        embedding_models=["Dummy"],
        max_length=100,  # Longer run to get more features in persistence diagram
    )

    # Save config to database to get an ID
    db_session.add(config)
    db_session.commit()
    db_session.refresh(config)

    # Run the experiment to populate database with real data
    db_url = str(db_session.get_bind().engine.url)
    perform_experiment(str(config.id), db_url)

    # Load the actual runs data with persistence diagram information
    df = load_runs_df(db_session, use_cache=False)

    # Verify we have persistence diagram data
    assert df.height > 0
    assert "homology_dimension" in df.columns
    assert "birth" in df.columns
    assert "death" in df.columns

    # Import the function we want to test
    from trajectory_tracer.visualisation import plot_persistence_diagram

    # Define output file
    output_file = "output/vis/persistence_diagram_single_test.html"

    # Generate the plot
    plot_persistence_diagram(df, output_file)

    # Verify the file was created
    assert os.path.exists(output_file), f"File was not created: {output_file}"


def test_plot_persistence_diagram_faceted(db_session):
    # Import necessary modules

    # Create a simple test configuration with persistence diagrams
    config = ExperimentConfig(
        networks=[["DummyT2I", "DummyI2T"]],
        seeds=[-1],
        prompts=["one fish", "two fish"],
        embedding_models=["Dummy"],
        max_length=100,  # Longer run to get more features in persistence diagram
    )

    # Save config to database to get an ID
    db_session.add(config)
    db_session.commit()
    db_session.refresh(config)

    # Run the experiment to populate database with real data
    db_url = str(db_session.get_bind().engine.url)
    perform_experiment(str(config.id), db_url)

    # Load the actual runs data with persistence diagram information
    df = load_runs_df(db_session, use_cache=False)

    # Verify we have persistence diagram data
    assert df.height > 0
    assert "homology_dimension" in df.columns
    assert "birth" in df.columns
    assert "death" in df.columns

    # Define output file
    output_file = "output/vis/persistence_diagram_faceted_test.html"

    # Generate the plot
    plot_persistence_diagram_faceted(df, output_file)

    # Verify the file was created
    assert os.path.exists(output_file), f"File was not created: {output_file}"


def test_plot_persistence_entropy(db_session):
    # Create a simple test configuration with persistence diagrams
    config = ExperimentConfig(
        networks=[["DummyT2I", "DummyI2T"]],
        seeds=[-1],
        prompts=["red fish", "blue fish"],
        embedding_models=["Dummy"],
        max_length=100,  # Longer run to get more features in persistence diagram
    )

    # Save config to database to get an ID
    db_session.add(config)
    db_session.commit()
    db_session.refresh(config)

    # Run the experiment to populate database with real data
    db_url = str(db_session.get_bind().engine.url)
    perform_experiment(str(config.id), db_url)

    # Load the actual runs data with persistence diagram information
    df = load_runs_df(db_session, use_cache=False)

    # Verify we have persistence diagram data with entropy values
    assert df.height > 0
    assert "homology_dimension" in df.columns
    assert "entropy" in df.columns

    # Import the function we want to test
    from trajectory_tracer.visualisation import plot_persistence_entropy

    # Define output file
    output_file = "output/vis/persistence_entropy_test.html"

    # Generate the plot
    plot_persistence_entropy(df, output_file)

    # Verify the file was created
    assert os.path.exists(output_file), f"File was not created: {output_file}"


def test_plot_semantic_drift(db_session):
    # Create a simple test configuration
    config = ExperimentConfig(
        networks=[["DummyT2I", "DummyI2T"]],
        seeds=[-1],
        prompts=["red fish", "blue fish"],
        embedding_models=["Dummy"],
        max_length=100,
    )

    # Save config to database to get an ID
    db_session.add(config)
    db_session.commit()
    db_session.refresh(config)

    # Run the experiment to populate database with real data
    db_url = str(db_session.get_bind().engine.url)
    perform_experiment(str(config.id), db_url)

    # Load the embeddings data with semantic drift information
    df = load_embeddings_df(db_session, use_cache=False)

    # Verify we have semantic dispersion data
    assert df.height > 0
    assert "drift_euclidean" in df.columns
    assert "drift_cosine" in df.columns
    assert "sequence_number" in df.columns

    # Define output file
    output_file = "output/vis/semantic_drift.html"

    # Generate the plot
    plot_semantic_drift(df, output_file)

    # Verify the file was created
    assert os.path.exists(output_file), f"File was not created: {output_file}"


def test_plot_persistence_entropy_faceted(db_session):
    # Create a simple test configuration with persistence diagrams
    config = ExperimentConfig(
        networks=[["DummyT2I", "DummyI2T"]],
        seeds=[-1, -2],  # Use two seeds to get multiple runs
        prompts=["black fish", "green fish"],
        embedding_models=["Dummy"],
        max_length=100,  # Longer run to get more features in persistence diagram
    )

    # Save config to database to get an ID
    db_session.add(config)
    db_session.commit()
    db_session.refresh(config)

    # Run the experiment to populate database with real data
    db_url = str(db_session.get_bind().engine.url)
    perform_experiment(str(config.id), db_url)

    # Load the actual runs data with persistence diagram information
    df = load_runs_df(db_session, use_cache=False)

    # Verify we have persistence diagram data with entropy values
    assert df.height > 0
    assert "homology_dimension" in df.columns
    assert "entropy" in df.columns
    assert "run_id" in df.columns

    # Import the function we want to test

    # Define output file
    output_file = "output/vis/persistence_entropy_faceted_test.html"

    # Generate the plot
    plot_persistence_entropy_faceted(df, output_file)

    # Verify the file was created
    assert os.path.exists(output_file), f"File was not created: {output_file}"
