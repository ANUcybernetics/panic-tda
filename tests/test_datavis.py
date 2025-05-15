import os

import pytest
import polars as pl

from panic_tda.data_prep import (
    add_cluster_labels,
    add_persistence_entropy,
    load_embeddings_df,
    load_invocations_df,
    load_runs_df,
)
from panic_tda.datavis import (
    plot_cluster_example_images,
    plot_cluster_histograms,
    plot_cluster_histograms_top_n,
    plot_cluster_timelines,
    plot_cluster_transitions,
    plot_invocation_duration,
    plot_persistence_diagram,
    plot_persistence_diagram_by_prompt,
    plot_persistence_diagram_faceted,
    plot_persistence_entropy,
    plot_persistence_entropy_by_prompt,
    plot_semantic_drift,
    read_existing_label_map,
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
        max_length=100,  # Short sequences for testing
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
    runs_df = load_runs_df(db_session)
    runs_df = add_persistence_entropy(runs_df, db_session)
    embeddings_df = load_embeddings_df(db_session)
    embeddings_df = add_cluster_labels(embeddings_df, 1, db_session)
    invocations_df = load_invocations_df(db_session)

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


def test_plot_persistence_diagram_by_prompt(mock_experiment_data):
    runs_df = mock_experiment_data["runs_df"]

    # Verify we have persistence diagram data
    assert runs_df.height > 0
    assert "homology_dimension" in runs_df.columns
    assert "birth" in runs_df.columns
    assert "persistence" in runs_df.columns
    assert "run_id" in runs_df.columns

    # Define output file
    output_file = "output/test/persistence_diagram_by_prompt.png"

    plot_persistence_diagram_by_prompt(runs_df, output_file)

    # Verify file was created
    assert os.path.exists(output_file), f"File was not created: {output_file}"


def test_plot_persistence_entropy_by_prompt(mock_experiment_data):
    runs_df = mock_experiment_data["runs_df"]

    # Verify we have persistence diagram data
    assert runs_df.height > 0
    assert "homology_dimension" in runs_df.columns
    assert "birth" in runs_df.columns
    assert "persistence" in runs_df.columns
    assert "run_id" in runs_df.columns

    # Define output file
    output_file = "output/test/persistence_entropy_by_prompt.pdf"

    plot_persistence_entropy_by_prompt(runs_df, output_file)

    # Verify file was created
    assert os.path.exists(output_file), f"File was not created: {output_file}"


@pytest.mark.skip(
    reason="Not using semantic drift atm, so it's not in the mock db (and therefore skip this test)"
)
def test_plot_semantic_drift(mock_experiment_data):
    embeddings_df = mock_experiment_data["embeddings_df"]

    # Verify we have semantic dispersion data
    assert embeddings_df.height > 0
    assert "sequence_number" in embeddings_df.columns
    assert "run_id" in embeddings_df.columns
    assert "initial_prompt" in embeddings_df.columns
    assert "embedding_model" in embeddings_df.columns

    # Define output file
    output_file = "output/test/semantic_drift.pdf"

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
    output_file = "output/test/persistence_entropy.pdf"

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
    output_file = "output/test/invocation_duration.pdf"

    # Generate the plot
    plot_invocation_duration(invocations_df, output_file)

    # Verify file was created
    assert os.path.exists(output_file), f"File was not created: {output_file}"


def test_plot_cluster_timelines(mock_experiment_data):
    embeddings_df = mock_experiment_data["embeddings_df"]

    # Verify we have cluster timeline data
    assert embeddings_df.height > 0
    assert "run_id" in embeddings_df.columns
    assert "sequence_number" in embeddings_df.columns
    assert "initial_prompt" in embeddings_df.columns

    # Define output file
    output_file = "output/test/cluster_timelines.pdf"

    # Generate the plot
    plot_cluster_timelines(embeddings_df, output_file)

    # Verify file was created
    assert os.path.exists(output_file), f"File was not created: {output_file}"


def test_plot_cluster_example_images(mock_experiment_data, db_session):
    embeddings_df = mock_experiment_data["embeddings_df"]

    # Verify we have the necessary columns
    assert embeddings_df.height > 0
    assert "invocation_id" in embeddings_df.columns
    assert "embedding_model" in embeddings_df.columns

    # Get the first embedding model for testing
    embedding_model = embeddings_df.select("embedding_model").unique().to_series()[0]

    # Define output file
    output_file = "output/test/cluster_examples.jpg"

    # Generate the plot
    plot_cluster_example_images(
        embeddings_df,
        num_examples=10,
        embedding_model=embedding_model,
        session=db_session,
        output_file=output_file,
    )

    # Verify file was created
    assert os.path.exists(output_file), f"File was not created: {output_file}"


def test_plot_cluster_histograms(mock_experiment_data):
    embeddings_df = mock_experiment_data["embeddings_df"]

    # Verify we have the necessary columns
    assert embeddings_df.height > 0
    assert "run_id" in embeddings_df.columns
    assert "sequence_number" in embeddings_df.columns
    assert "initial_prompt" in embeddings_df.columns

    # Define output file
    output_file = "output/test/cluster_histograms.pdf"

    # Generate the plot
    plot_cluster_histograms(embeddings_df, output_file)

    # Verify file was created
    assert os.path.exists(output_file), f"File was not created: {output_file}"


def test_plot_cluster_histograms_top_n(mock_experiment_data):
    embeddings_df = mock_experiment_data["embeddings_df"]

    # Verify we have the necessary columns
    assert embeddings_df.height > 0
    assert "run_id" in embeddings_df.columns
    assert "sequence_number" in embeddings_df.columns
    assert "initial_prompt" in embeddings_df.columns

    # Define output file
    output_file = "output/test/cluster_histograms_top_n.pdf"

    # Generate the plot
    plot_cluster_histograms_top_n(embeddings_df, 2, output_file=output_file)

    # Verify file was created
    assert os.path.exists(output_file), f"File was not created: {output_file}"


def test_plot_cluster_transitions(mock_experiment_data):
    embeddings_df = mock_experiment_data["embeddings_df"]

    # Verify we have the necessary columns
    assert embeddings_df.height > 0
    assert "run_id" in embeddings_df.columns
    assert "sequence_number" in embeddings_df.columns
    assert "cluster_label" in embeddings_df.columns

    # Define output file
    output_file = "output/test/cluster_transitions.pdf"

    # Generate the plot
    plot_cluster_transitions(embeddings_df, True, output_file)

    # Verify file was created
    assert os.path.exists(output_file), f"File was not created: {output_file}"


def test_cache_labels_and_read_from_cache(tmp_path):
    """Test caching cluster labels to JSON and reading them back."""

    # Create a polars Series with cluster labels
    labels = ["Cluster_A", "Cluster_B", "OUTLIER", "Cluster_A", "Cluster_C", "OUTLIER"]
    cluster_labels = pl.Series("cluster_labels", labels)

    # Create a temporary file path for testing
    cache_path = os.path.join(tmp_path, "test_cluster_map.json")

    # Test create_label_map function
    label_map = create_label_map(cluster_labels, output_path=cache_path)

    # Validate the output dictionary
    assert isinstance(label_map, dict)
    assert len(label_map) == 4  # 3 clusters + OUTLIER
    assert label_map["OUTLIER"] == -1
    assert label_map["Cluster_A"] > 0
    assert label_map["Cluster_B"] > 0
    assert label_map["Cluster_C"] > 0
    assert label_map["Cluster_A"] != label_map["Cluster_B"]
    assert label_map["Cluster_A"] != label_map["Cluster_C"]
    assert label_map["Cluster_B"] != label_map["Cluster_C"]

    # Verify file was created
    assert os.path.exists(cache_path)

    # Test read_labels_from_cache function
    loaded_map = read_existing_label_map(input_path=cache_path)

    # Verify loaded map matches original
    assert loaded_map == label_map

    # Test with different labels to ensure IDs are assigned in order
    new_labels = ["Group_1", "OUTLIER", "Group_2", "Group_3", "Group_1"]
    new_cluster_labels = pl.Series("cluster_labels", new_labels)
    new_cache_path = os.path.join(tmp_path, "test_cluster_map_2.json")

    new_label_map = create_label_map(new_cluster_labels, output_path=new_cache_path)

    # Validate the new mapping
    assert new_label_map["OUTLIER"] == -1
    assert new_label_map["Group_1"] == 1
    assert new_label_map["Group_2"] == 2
    assert new_label_map["Group_3"] == 3

    # Load and verify
    loaded_new_map = read_labels_from_cache(input_path=new_cache_path)
    assert loaded_new_map == new_label_map
