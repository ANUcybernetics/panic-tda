import os

import polars as pl

from panic_tda.clustering_manager import cluster_all_data
from panic_tda.data_prep import (
    add_persistence_entropy,
    load_clusters_df,
    load_embeddings_df,
    load_invocations_df,
    load_pd_df,
    load_runs_df,
)
from panic_tda.datavis import (
    create_label_map_df,
    format_label_map_as_markdown,
    plot_cluster_bubblegrid,
    plot_cluster_example_images,
    plot_cluster_histograms,
    plot_cluster_histograms_top_n,
    plot_cluster_run_length_bubblegrid,
    plot_cluster_run_length_violin,
    plot_cluster_run_lengths,
    plot_cluster_timelines,
    plot_cluster_transitions,
    plot_invocation_duration,
    plot_persistence_diagram,
    plot_persistence_diagram_by_prompt,
    plot_persistence_diagram_faceted,
    plot_persistence_entropy,
    plot_persistence_entropy_by_prompt,
    plot_semantic_drift,
)  # Temporary import for clarity
from panic_tda.engine import perform_experiment
from panic_tda.schemas import ExperimentConfig


def setup_minimal_experiment(db_session):
    """Set up a minimal experiment with just enough data for basic tests."""
    experiment = ExperimentConfig(
        networks=[["DummyT2I", "DummyI2T"]],
        prompts=["test prompt"],
        seeds=[-1],
        embedding_models=["DummyText"],
        max_length=10,  # Very short sequences for testing
    )

    # Save experiment to database to get an ID
    db_session.add(experiment)
    db_session.commit()
    db_session.refresh(experiment)

    # Run the experiment to populate database with dummy model runs
    db_url = str(db_session.get_bind().engine.url)
    perform_experiment(str(experiment.id), db_url)
    db_session.refresh(experiment)

    return experiment


def setup_persistence_experiment(db_session):
    """Set up an experiment with enough data for persistence diagram tests."""
    experiment = ExperimentConfig(
        networks=[["DummyT2I", "DummyI2T"], ["DummyT2I2", "DummyI2T2"]],
        prompts=["one fish", "two fish"],
        seeds=[-1] * 2,
        embedding_models=["DummyText"],
        max_length=20,  # Short sequences for testing
    )

    # Save experiment to database to get an ID
    db_session.add(experiment)
    db_session.commit()
    db_session.refresh(experiment)

    # Run the experiment to populate database with dummy model runs
    db_url = str(db_session.get_bind().engine.url)
    perform_experiment(str(experiment.id), db_url)
    db_session.refresh(experiment)

    return experiment


def setup_cluster_experiment(db_session):
    """Set up an experiment with enough data for cluster analysis tests."""
    experiment = ExperimentConfig(
        networks=[["DummyT2I", "DummyI2T"]],
        prompts=["one fish", "red fish"],
        seeds=[-1] * 2,
        embedding_models=["DummyText", "DummyText2"],
        max_length=15,  # Short sequences for testing
    )

    # Save experiment to database to get an ID
    db_session.add(experiment)
    db_session.commit()
    db_session.refresh(experiment)

    # Run the experiment to populate database with dummy model runs
    db_url = str(db_session.get_bind().engine.url)
    perform_experiment(str(experiment.id), db_url)
    db_session.refresh(experiment)

    return experiment


def test_plot_persistence_diagram(db_session):
    # Setup a minimal experiment for persistence diagram
    setup_minimal_experiment(db_session)

    # Load the persistence diagram data
    pd_df = load_pd_df(db_session)

    # Verify we have persistence diagram data
    assert pd_df.height > 0
    assert "homology_dimension" in pd_df.columns
    assert "birth" in pd_df.columns
    assert "persistence" in pd_df.columns

    # Define output file
    output_file = "output/test/persistence_diagram.png"

    # Generate the plot
    plot_persistence_diagram(pd_df, output_file)

    # Verify file was created
    assert os.path.exists(output_file), f"File was not created: {output_file}"


def test_plot_persistence_diagram_faceted(db_session):
    # Setup an experiment with multiple networks for faceted diagram
    setup_persistence_experiment(db_session)

    # Load the persistence diagram data
    pd_df = load_pd_df(db_session)

    # Verify we have persistence diagram data
    assert pd_df.height > 0
    assert "homology_dimension" in pd_df.columns
    assert "birth" in pd_df.columns
    assert "persistence" in pd_df.columns
    assert "text_model" in pd_df.columns
    assert "image_model" in pd_df.columns

    # Define output file
    output_file = "output/test/persistence_diagram_faceted.png"

    # Generate the plot
    plot_persistence_diagram_faceted(pd_df, output_file)

    # Verify file was created
    assert os.path.exists(output_file), f"File was not created: {output_file}"


def test_plot_persistence_diagram_by_prompt(db_session):
    # Setup experiment with multiple prompts
    setup_persistence_experiment(db_session)

    # Load the persistence diagram data
    pd_df = load_pd_df(db_session)

    # Verify we have persistence diagram data
    assert pd_df.height > 0
    assert "homology_dimension" in pd_df.columns
    assert "birth" in pd_df.columns
    assert "persistence" in pd_df.columns
    assert "run_id" in pd_df.columns
    assert "initial_prompt" in pd_df.columns

    # Define output file
    output_file = "output/test/persistence_diagram_by_prompt.png"

    plot_persistence_diagram_by_prompt(pd_df, output_file)

    # Verify file was created
    assert os.path.exists(output_file), f"File was not created: {output_file}"


def test_plot_persistence_entropy_by_prompt(db_session):
    # Setup experiment with multiple prompts
    setup_persistence_experiment(db_session)

    # Load the necessary data
    runs_df = load_runs_df(db_session)
    runs_df = add_persistence_entropy(runs_df, db_session)

    # Verify we have persistence entropy data
    assert runs_df.height > 0
    assert "homology_dimension" in runs_df.columns
    assert "entropy" in runs_df.columns
    assert "run_id" in runs_df.columns

    # Define output file
    output_file = "output/test/persistence_entropy_by_prompt.pdf"

    plot_persistence_entropy_by_prompt(runs_df, output_file)

    # Verify file was created
    assert os.path.exists(output_file), f"File was not created: {output_file}"


def test_plot_semantic_drift():
    """Test the plot_semantic_drift function with synthetic data."""
    import numpy as np

    # Create synthetic test data
    np.random.seed(42)
    data = []

    # Generate data for 2 networks with different drift patterns
    for network in ["T2I→I2T", "T2I→I2T→T2I"]:
        for seq_num in range(100):
            # Create different drift patterns for each network
            if network == "T2I→I2T":
                # Gradual increase in drift
                base_drift = seq_num / 100 * 0.8
                noise = np.random.normal(0, 0.05)
            else:
                # Oscillating drift pattern
                base_drift = 0.4 + 0.3 * np.sin(seq_num / 10)
                noise = np.random.normal(0, 0.08)

            drift_value = max(0, min(1, base_drift + noise))

            data.append({
                "network": network,
                "sequence_number": seq_num,
                "semantic_drift": drift_value,
                "initial_prompt": "test prompt",
                "embedding_model": "TestModel",
                "run_id": f"run_{network}_{seq_num % 5}",  # 5 runs per network
            })

    # Create DataFrame
    test_df = pl.DataFrame(data)

    # Verify the data structure
    assert test_df.height == 200  # 100 sequences * 2 networks
    assert "network" in test_df.columns
    assert "sequence_number" in test_df.columns
    assert "semantic_drift" in test_df.columns

    # Define output file
    output_file = "output/test/semantic_drift_ridgeline.pdf"

    # Generate the plot
    plot_semantic_drift(test_df, output_file)

    # Verify file was created
    assert os.path.exists(output_file), f"File was not created: {output_file}"

    # Clean up
    os.remove(output_file)


def test_plot_persistence_entropy(db_session):
    # Setup a minimal experiment
    setup_minimal_experiment(db_session)

    # Load the necessary data
    runs_df = load_runs_df(db_session)
    runs_df = add_persistence_entropy(runs_df, db_session)

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


def test_plot_cluster_run_lengths(db_session):
    # Setup the experiment with cluster data
    setup_cluster_experiment(db_session)

    # Run clustering
    cluster_all_data(db_session, downsample=1)
    db_session.commit()

    # Load clusters data
    clusters_df = load_clusters_df(db_session)

    # Define output file
    output_file = "output/test/cluster_run_lengths.pdf"

    # Generate the plot - now pass the clusters dataframe!
    plot_cluster_run_lengths(clusters_df, output_file)

    # Verify file was created
    assert os.path.exists(output_file), f"File was not created: {output_file}"


def test_plot_cluster_run_length_violin(db_session):
    # Setup the experiment with cluster data
    setup_cluster_experiment(db_session)

    # Run clustering
    cluster_all_data(db_session, downsample=1)
    db_session.commit()

    # Load clusters data
    clusters_df = load_clusters_df(db_session)

    # Define output file
    output_file = "output/test/cluster_run_length_violin.pdf"

    # Generate the plot - now pass the clusters dataframe!
    plot_cluster_run_length_violin(clusters_df, output_file=output_file)

    # Verify file was created
    assert os.path.exists(output_file), f"File was not created: {output_file}"


def test_plot_invocation_duration(db_session):
    # Setup a minimal experiment
    setup_minimal_experiment(db_session)

    # Load the necessary data
    invocations_df = load_invocations_df(db_session)

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


def test_plot_cluster_timelines(db_session):
    # Setup the experiment with cluster data
    setup_cluster_experiment(db_session)

    # Load the necessary data
    embeddings_df = load_embeddings_df(db_session)
    # Run clustering
    cluster_all_data(db_session, downsample=1)
    db_session.commit()
    clusters_df = load_clusters_df(db_session)

    # Join embeddings with clusters
    df_with_clusters = embeddings_df.join(
        clusters_df.select(["embedding_id", "cluster_label"]),
        left_on="id",
        right_on="embedding_id",
        how="left",
    )

    # Verify we have cluster timeline data
    assert df_with_clusters.height > 0
    assert "run_id" in df_with_clusters.columns
    assert "sequence_number" in df_with_clusters.columns
    assert "initial_prompt" in df_with_clusters.columns

    # Define output file
    output_file = "output/test/cluster_timelines.pdf"

    # Generate the plot - now pass clusters_df instead of session
    plot_cluster_timelines(df_with_clusters, clusters_df, output_file)

    # Verify file was created
    assert os.path.exists(output_file), f"File was not created: {output_file}"


def test_plot_cluster_example_images(db_session):
    # Setup the experiment with cluster data
    setup_cluster_experiment(db_session)

    # Load the necessary data
    embeddings_df = load_embeddings_df(db_session)
    # Run clustering
    cluster_all_data(db_session, downsample=1)
    db_session.commit()
    clusters_df = load_clusters_df(db_session)

    # Join embeddings with clusters
    df_with_clusters = embeddings_df.join(
        clusters_df.select(["embedding_id", "cluster_label"]),
        left_on="id",
        right_on="embedding_id",
        how="left",
    )

    # Verify we have the necessary columns
    assert df_with_clusters.height > 0
    assert "invocation_id" in df_with_clusters.columns
    assert "embedding_model" in df_with_clusters.columns

    # Get the first embedding model for testing
    embedding_model = df_with_clusters.select("embedding_model").unique().to_series()[0]

    # Define output file
    output_file = "output/test/cluster_examples.jpg"

    # Generate the plot
    plot_cluster_example_images(
        df_with_clusters,
        num_examples=2,  # Reduced from 10 to 2 for faster tests
        embedding_model=embedding_model,
        session=db_session,
        output_file=output_file,
    )

    # Verify file was created
    assert os.path.exists(output_file), f"File was not created: {output_file}"

    # Test with examples_per_row parameter
    output_file_wrapped = "output/test/cluster_examples_wrapped.jpg"
    plot_cluster_example_images(
        df_with_clusters,
        num_examples=4,
        embedding_model=embedding_model,
        session=db_session,
        examples_per_row=2,  # Force wrapping after 2 images
        output_file=output_file_wrapped,
    )

    # Verify wrapped file was created
    assert os.path.exists(output_file_wrapped), (
        f"File was not created: {output_file_wrapped}"
    )

    # Test with rescale parameter
    output_file_rescaled = "output/test/cluster_examples_rescaled.jpg"
    plot_cluster_example_images(
        df_with_clusters,
        num_examples=2,
        embedding_model=embedding_model,
        session=db_session,
        rescale=0.5,  # Scale down to half size
        output_file=output_file_rescaled,
    )

    # Verify rescaled file was created
    assert os.path.exists(output_file_rescaled), (
        f"File was not created: {output_file_rescaled}"
    )

    # Verify that the rescaled image is smaller
    from PIL import Image

    original = Image.open(output_file)
    rescaled = Image.open(output_file_rescaled)
    assert rescaled.width == original.width // 2
    assert rescaled.height == original.height // 2


def test_plot_cluster_histograms(db_session):
    # Setup the experiment with cluster data
    setup_cluster_experiment(db_session)

    # Run clustering
    cluster_all_data(db_session, downsample=1)
    db_session.commit()

    # Load clusters data
    clusters_df = load_clusters_df(db_session)

    # Define output file
    output_file = "output/test/cluster_histograms.pdf"

    # Generate the plot - now pass the clusters dataframe!
    plot_cluster_histograms(clusters_df, output_file=output_file)

    # Verify file was created
    assert os.path.exists(output_file), f"File was not created: {output_file}"


def test_plot_cluster_histograms_top_n(db_session):
    # Setup the experiment with cluster data
    setup_cluster_experiment(db_session)

    # Load the necessary data
    embeddings_df = load_embeddings_df(db_session)
    # Run clustering
    cluster_all_data(db_session, downsample=1)
    db_session.commit()
    clusters_df = load_clusters_df(db_session)

    # Join embeddings with clusters
    df_with_clusters = embeddings_df.join(
        clusters_df.select(["embedding_id", "cluster_label"]),
        left_on="id",
        right_on="embedding_id",
        how="left",
    )

    # Verify we have the necessary columns
    assert df_with_clusters.height > 0
    assert "run_id" in df_with_clusters.columns
    assert "sequence_number" in df_with_clusters.columns
    assert "initial_prompt" in df_with_clusters.columns

    # Define output file
    output_file = "output/test/cluster_histograms_top_n.pdf"

    # Generate the plot - using just 1 top cluster instead of 2 for faster test
    plot_cluster_histograms_top_n(df_with_clusters, 1, output_file=output_file)

    # Verify file was created
    assert os.path.exists(output_file), f"File was not created: {output_file}"


def test_plot_cluster_transitions(db_session):
    # Setup the experiment with cluster data
    setup_cluster_experiment(db_session)

    # Run clustering
    cluster_all_data(db_session, downsample=1)
    db_session.commit()

    # Load clusters data
    clusters_df = load_clusters_df(db_session)

    # Define output file
    output_file = "output/test/cluster_transitions.pdf"

    # Generate the plot - now pass the clusters dataframe!
    plot_cluster_transitions(clusters_df, True, output_file)

    # Verify file was created
    assert os.path.exists(output_file), f"File was not created: {output_file}"


def test_create_label_map_df(db_session, capsys):
    """Test that label_map_df is constructed properly and verifies that cluster_index
    is unique for each embedding_model + cluster_label combination."""

    # Setup the experiment with cluster data
    setup_cluster_experiment(db_session)

    # Run clustering to get some data
    cluster_all_data(db_session, downsample=1)
    db_session.commit()

    # Load clusters data
    clusters_df = load_clusters_df(db_session)

    # Test create_label_map_df function with clusters dataframe
    label_map_df = create_label_map_df(clusters_df)

    # Validate the output DataFrame
    assert isinstance(label_map_df, pl.DataFrame)
    assert label_map_df.height > 0

    # Check the columns
    assert set(label_map_df.columns) == {
        "embedding_model",
        "cluster_label",
        "cluster_index",
    }

    # Verify that OUTLIER labels are excluded (cluster_id != -1)
    assert not label_map_df.filter(pl.col("cluster_label") == "OUTLIER").height

    # Verify that cluster_index values start from 1
    assert label_map_df["cluster_index"].min() == 1

    # Verify indices are sequential within each embedding model
    for model in label_map_df["embedding_model"].unique():
        model_df = label_map_df.filter(pl.col("embedding_model") == model)
        indices = sorted(model_df["cluster_index"].to_list())
        # Check they're sequential starting from 1
        assert indices == list(range(1, len(indices) + 1))

    # Test the markdown printing functionality
    _label_map_df_with_print = create_label_map_df(clusters_df, print_mapping=True)
    captured = capsys.readouterr()

    # Verify the output contains expected markdown table elements
    assert "Cluster Label Mapping:" in captured.out
    assert "==" in captured.out  # Separator line
    assert "##" in captured.out  # Markdown headers
    assert "|" in captured.out  # Table pipes
    assert "Index" in captured.out  # Table header
    assert "Cluster Label" in captured.out  # Table header

    # Verify each embedding model appears in the output
    for model in label_map_df["embedding_model"].unique():
        assert model in captured.out


def test_plot_cluster_bubblegrid(db_session):
    # Setup the experiment with cluster data
    setup_cluster_experiment(db_session)

    # Run clustering
    cluster_all_data(db_session, downsample=1)
    db_session.commit()

    # Load clusters data
    clusters_df = load_clusters_df(db_session)

    # Define output file
    output_file = "output/test/cluster_bubblegrid.pdf"

    # without outliers - now pass the clusters dataframe!
    plot_cluster_bubblegrid(clusters_df, False, output_file)
    assert os.path.exists(output_file), (
        f"File was not created (without outliers): {output_file}"
    )

    # with outliers
    plot_cluster_bubblegrid(clusters_df, True, output_file)
    assert os.path.exists(output_file), (
        f"File was not created (with outliers): {output_file}"
    )


def test_plot_cluster_run_length_bubblegrid(db_session):
    # Setup the experiment with cluster data
    setup_cluster_experiment(db_session)

    # Run clustering
    cluster_all_data(db_session, downsample=1)
    db_session.commit()

    # Load clusters data
    clusters_df = load_clusters_df(db_session)

    # Define output file
    output_file = "output/test/cluster_run_length_bubblegrid.pdf"

    # without outliers - now pass clusters dataframe!
    plot_cluster_run_length_bubblegrid(clusters_df, False, output_file)
    assert os.path.exists(output_file), (
        f"File was not created (without outliers): {output_file}"
    )

    # with outliers
    plot_cluster_run_length_bubblegrid(clusters_df, True, output_file)
    assert os.path.exists(output_file), (
        f"File was not created (with outliers): {output_file}"
    )


def test_format_label_map_as_markdown():
    """Test the format_label_map_as_markdown function."""
    # Create test data
    test_data = {
        "embedding_model": ["Model1", "Model1", "Model1", "Model2", "Model2"],
        "cluster_label": [
            "Short label",
            "A very long label that should be truncated because it exceeds the maximum allowed length for display",
            "Medium length label that fits",
            "Another short one",
            "Yet another medium length label",
        ],
        "cluster_index": [1, 2, 3, 1, 2],
    }
    map_df = pl.DataFrame(test_data)

    # Test basic functionality
    markdown = format_label_map_as_markdown(map_df)
    assert "|" in markdown
    assert "Index" in markdown
    assert "Cluster Label" in markdown

    # Test filtering by model
    markdown_model1 = format_label_map_as_markdown(map_df, embedding_model="Model1")
    assert "Short label" in markdown_model1
    assert "Another short one" not in markdown_model1

    # Test label truncation
    assert "..." in markdown_model1  # Long label should be truncated
    assert (
        "A very long label that should be truncated because it exceeds the maximum allowed length for display"
        not in markdown_model1
    )

    # Test custom max length
    markdown_short = format_label_map_as_markdown(
        map_df, embedding_model="Model1", max_label_length=20
    )
    # Count occurrences of "..." - should have more with shorter max length
    assert markdown_short.count("...") >= 1
