from pathlib import Path

import numpy as np
import polars as pl
import polars.testing
import pytest

from panic_tda.data_prep import (
    add_semantic_drift,
    cache_dfs,
    calculate_cluster_bigrams,
    calculate_cluster_run_lengths,
    calculate_cluster_transitions,
    calculate_semantic_drift,
    embed_initial_prompts,
    filter_top_n_clusters,
    load_clusters_df,
    load_embeddings_df,
    load_invocations_df,
    load_pd_df,
    load_runs_df,
)
from panic_tda.db import list_runs
from panic_tda.embeddings import EMBEDDING_DIM
from panic_tda.engine import (
    perform_experiment,
)
from panic_tda.schemas import ExperimentConfig
from panic_tda.clustering_manager import cluster_all_data


def test_load_invocations_df(db_session):
    """Test that load_invocations_df returns a polars DataFrame with correct data."""

    # Create a simple test configuration
    config = ExperimentConfig(
        networks=[["DummyT2I", "DummyI2T"]],
        seeds=[-1],
        prompts=["test invocations dataframe"],
        embedding_models=["Dummy", "Dummy2"],
        max_length=10,
    )

    # Save config to database to get an ID
    db_session.add(config)
    db_session.commit()
    db_session.refresh(config)

    # Run the experiment to populate database
    db_url = str(db_session.get_bind().engine.url)
    perform_experiment(str(config.id), db_url)

    # Call function under test
    df = load_invocations_df(db_session)

    # Assertions
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 10  # Adjusted to match actual number of invocations

    # Check column names
    expected_columns = [
        "id",
        "run_id",
        "experiment_id",
        "model",
        "type",
        "sequence_number",
        "started_at",
        "completed_at",
        "duration",
        "initial_prompt",
        "seed",
    ]
    assert all(col in df.columns for col in expected_columns)
    # Check there are no extraneous columns
    assert set(df.columns) == set(expected_columns)

    # Check experiment_id is correctly stored
    assert df.filter(pl.col("experiment_id") == str(config.id)).height > 0

    # Verify field values using named columns instead of indices
    image_rows = df.filter(pl.col("model") == "DummyT2I")
    assert image_rows.height > 0
    image_row = image_rows.row(0, named=True)
    assert image_row["initial_prompt"] == "test invocations dataframe"
    assert image_row["model"] == "DummyT2I"
    assert image_row["seed"] == -1

    text_rows = df.filter(pl.col("model") == "DummyI2T")
    assert text_rows.height > 0
    text_row = text_rows.row(0, named=True)
    assert text_row["initial_prompt"] == "test invocations dataframe"
    assert text_row["model"] == "DummyI2T"
    assert text_row["seed"] == -1
    assert text_row["experiment_id"] == str(config.id)


def test_load_embeddings_df(db_session):
    """Test that load_embeddings_df returns a polars DataFrame with correct data."""

    # Create a simple test configuration
    config = ExperimentConfig(
        networks=[["DummyT2I", "DummyI2T"]],
        seeds=[-1],
        prompts=["test embedding dataframe"],
        embedding_models=["Dummy", "Dummy2"],
        max_length=100,
    )

    # Save config to database to get an ID
    db_session.add(config)
    db_session.commit()
    db_session.refresh(config)

    # Run the experiment to populate database
    db_url = str(db_session.get_bind().engine.url)
    perform_experiment(str(config.id), db_url)

    # Call function under test
    df = load_embeddings_df(db_session)

    # Assertions
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 100

    # Check column names
    expected_columns = {
        "id",
        "invocation_id",
        "run_id",
        "embedding_model",
        "started_at",
        "completed_at",
        "sequence_number",
        "initial_prompt",
        "text_model",
        "text",
        "network",
        "experiment_id",
    }
    # Check for missing columns using set difference
    missing_columns = expected_columns - set(df.columns)
    if missing_columns:
        raise AssertionError(f"Missing columns in DataFrame: {missing_columns}")

    # Check for extra columns
    unexpected_columns = set(df.columns) - expected_columns
    if unexpected_columns:
        raise AssertionError(f"Unexpected columns in DataFrame: {unexpected_columns}")
    # Check there are no extraneous columns
    assert set(df.columns) == set(expected_columns)

    # Check values
    assert df.filter(pl.col("embedding_model") == "Dummy").height == 50
    assert df.filter(pl.col("embedding_model") == "Dummy2").height == 50

    # Verify field values using named columns instead of indices
    text_rows = df.filter(pl.col("text_model") == "DummyI2T")
    assert text_rows.height > 0
    text_row = text_rows.row(0, named=True)
    assert text_row["initial_prompt"] == "test embedding dataframe"
    assert text_row["text_model"] == "DummyI2T"
    assert text_row["sequence_number"] == 1
    assert text_row["network"] == "DummyT2I→DummyI2T"

    # Check output text for text model (DummyI2T)
    assert "text" in text_row
    assert text_row["text"] is not None
    assert isinstance(text_row["text"], str)
    assert len(text_row["text"]) > 0


def test_load_clusters_df(db_session):
    """Test that load_clusters_df properly creates cluster assignments."""

    # Create a simple test configuration with minimal data
    config = ExperimentConfig(
        networks=[["DummyT2I", "DummyI2T"]],
        seeds=[-1],
        prompts=["test embedding clustering"],
        embedding_models=["Dummy", "Dummy2"],
        max_length=100,
    )

    # Save config to database to get an ID
    db_session.add(config)
    db_session.commit()
    db_session.refresh(config)

    # Run the experiment to populate database
    db_url = str(db_session.get_bind().engine.url)
    perform_experiment(str(config.id), db_url)

    # Run clustering on the embeddings
    cluster_all_data(db_session, downsample=1)

    # Commit the transaction to ensure data is persisted
    db_session.commit()

    # Load clusters dataframe
    clusters_df = load_clusters_df(db_session)

    # Check that the expected columns exist
    expected_columns = [
        "clustering_result_id",
        "embedding_id",
        "embedding_model",
        "cluster_id",
        "cluster_label",
        "run_id",
        "sequence_number",
        "invocation_id",
        "algorithm",
        "epsilon",
        "initial_prompt",
        "network",
    ]
    assert all(col in clusters_df.columns for col in expected_columns)
    assert set(clusters_df.columns) == set(expected_columns)

    # Check that each embedding has a cluster assignment
    assert (
        clusters_df.height == 100
    )  # All embeddings should be present (no downsampling)

    # Check that cluster labels are strings
    assert clusters_df.filter(
        pl.col("cluster_label").cast(pl.Utf8).is_not_null()
    ).height == len(clusters_df)

    # Check that cluster labels are non-empty strings
    assert clusters_df.filter(
        pl.col("cluster_label").str.len_chars() > 0
    ).height == len(clusters_df)

    # Check that each embedding model has its own clustering result
    clustering_results = clusters_df.group_by([
        "embedding_model",
        "clustering_result_id",
    ]).count()
    assert clustering_results.height == 2  # One clustering result per embedding model

    # Check algorithm and epsilon values
    assert (
        clusters_df.filter(pl.col("algorithm") == "hdbscan").height
        == clusters_df.height
    )
    assert clusters_df.filter(pl.col("epsilon") == 0.4).height == clusters_df.height

    # Verify that cluster IDs are integers (including -1 for outliers)
    assert clusters_df.select("cluster_id").dtypes[0] == pl.Int64

    # Load embeddings to compare
    embeddings_df = load_embeddings_df(db_session)

    # Verify that all cluster labels are from the set of output texts
    unique_labels = clusters_df.select("cluster_label").unique()
    all_texts = embeddings_df.select("text").unique()

    # Extract the unique texts and labels as sets for easier comparison
    text_set = set(all_texts.select("text").to_series().to_list())
    label_set = set(unique_labels.select("cluster_label").to_series().to_list())

    # Check that each cluster label is either from the set of output texts or is "OUTLIER"
    for label in label_set:
        assert label in text_set or label == "OUTLIER", (
            f"Cluster label '{label}' not found in the set of output texts and is not 'OUTLIER'"
        )


def test_load_clusters_df_with_downsampling(db_session):
    """Test that load_clusters_df properly works with downsampling."""

    # Create a test configuration with more data to demonstrate downsampling
    config = ExperimentConfig(
        networks=[["DummyT2I", "DummyI2T"]],
        seeds=[-1],
        prompts=["test embedding clustering with downsampling"],
        embedding_models=["Dummy", "Dummy2"],
        max_length=200,  # More data points
    )

    # Save config to database to get an ID
    db_session.add(config)
    db_session.commit()
    db_session.refresh(config)

    # Run the experiment to populate database
    db_url = str(db_session.get_bind().engine.url)
    perform_experiment(str(config.id), db_url)

    # Run clustering with downsampling factor of 2
    downsample = 2
    cluster_all_data(db_session, downsample=downsample)

    # Commit the transaction to ensure data is persisted
    db_session.commit()

    # Load clusters dataframe
    clusters_df = load_clusters_df(db_session)

    # Check that the expected columns exist
    expected_columns = [
        "clustering_result_id",
        "embedding_id",
        "embedding_model",
        "cluster_id",
        "cluster_label",
        "run_id",
        "sequence_number",
        "invocation_id",
        "algorithm",
        "epsilon",
        "initial_prompt",
        "network",
    ]
    assert all(col in clusters_df.columns for col in expected_columns)

    # With downsample=2, we should have approximately half the embeddings
    # Each run produces 100 text outputs, so 100 embeddings per model
    # With 2 models and downsample=2, we should have ~100 total cluster assignments
    assert clusters_df.height == 100

    # Check that each embedding model has clusters (considering downsampling)
    dummy_clusters = clusters_df.filter(pl.col("embedding_model") == "Dummy")
    dummy2_clusters = clusters_df.filter(pl.col("embedding_model") == "Dummy2")

    # Verify that both models have at least one cluster assignment
    assert dummy_clusters.height > 0
    assert dummy2_clusters.height > 0

    # Load embeddings to compare
    embeddings_df = load_embeddings_df(db_session)

    # Verify that all cluster labels are from the set of output texts or "OUTLIER"
    unique_labels = clusters_df.select("cluster_label").unique()
    all_texts = embeddings_df.select("text").unique()

    # Extract the unique texts and labels as sets for easier comparison
    text_set = set(all_texts.select("text").to_series().to_list())
    label_set = set(unique_labels.select("cluster_label").to_series().to_list())

    # Check that each cluster label is either from the set of output texts or is "OUTLIER"
    for label in label_set:
        assert label in text_set or label == "OUTLIER", (
            f"Cluster label '{label}' not found in the set of output texts and is not 'OUTLIER'"
        )


def test_initial_prompt_embeddings(db_session):
    """Test that embed_initial_prompts correctly creates embeddings for initial prompts."""

    # Create a simple test configuration with multiple prompts
    config = ExperimentConfig(
        networks=[["DummyT2I", "DummyI2T"]],
        seeds=[-1],
        prompts=["test prompt embeddings", "another test prompt"],
        embedding_models=["Dummy"],
        max_length=10,
    )

    # Save config to database to get an ID
    db_session.add(config)
    db_session.commit()
    db_session.refresh(config)

    # Run the experiment to populate database
    db_url = str(db_session.get_bind().engine.url)
    perform_experiment(str(config.id), db_url)

    # Call function under test
    prompt_vectors = embed_initial_prompts(db_session)

    # Assertions
    #
    # The dictionary is keyed by (initial_prompt, embedding_model) tuples
    assert isinstance(prompt_vectors, dict)
    assert len(prompt_vectors) > 0

    # Check that embeddings exist for all the prompts and embedding models
    prompts = config.prompts
    embedding_models = config.embedding_models

    for prompt in prompts:
        for model in embedding_models:
            key = (prompt, model)
            assert key in prompt_vectors, (
                f"No embedding found for prompt '{prompt}' with model '{model}'"
            )

            # Check that each embedding is a numpy array with length EMBEDDING_DIM
            embedding = prompt_vectors[key]
            assert embedding is not None
            assert isinstance(embedding, np.ndarray), (
                f"Expected numpy array, got {type(embedding)}"
            )
            assert len(embedding) == EMBEDDING_DIM, (
                f"Expected embedding dimension {EMBEDDING_DIM}, got {len(embedding)}"
            )


def test_add_semantic_drift(db_session):
    """Test that add_semantic_drift correctly adds semantic drift measurements to embeddings."""

    # Create a simple test configuration with minimal data
    config = ExperimentConfig(
        networks=[["DummyT2I", "DummyI2T"]],
        seeds=[-1],
        prompts=["test semantic drift"],
        embedding_models=["Dummy"],
        max_length=100,
    )

    # Save config to database to get an ID
    db_session.add(config)
    db_session.commit()
    db_session.refresh(config)

    # Run the experiment to populate database
    db_url = str(db_session.get_bind().engine.url)
    perform_experiment(str(config.id), db_url)

    # Load embeddings and add semantic drift
    df = load_embeddings_df(db_session)
    df = add_semantic_drift(df, db_session)

    # Check that the drift column was added
    assert "semantic_drift" in df.columns

    # Check that drift values are calculated properly
    # All drift values should be non-negative
    assert df.height > 0
    assert (df["semantic_drift"] >= 0.0).all()

    # Check that drift values are in reasonable range [0, 2]
    # (normalized euclidean distance for unit vectors is bounded by 2)
    assert (df["semantic_drift"] <= 2.0).all()

    # Check we have variation in drift values (not all the same)
    unique_drifts = df.select("semantic_drift").unique()
    assert unique_drifts.height > 1, "Expected variation in semantic drift values"


def test_load_runs_df(db_session):
    """Test that load_runs_df returns a polars DataFrame with correct data."""

    # Create a simple test configuration
    config = ExperimentConfig(
        networks=[["DummyT2I", "DummyI2T"]],
        seeds=[-1],
        prompts=["test runs dataframe"],
        embedding_models=["Dummy"],
        max_length=10,
    )

    # Save config to database to get an ID
    db_session.add(config)
    db_session.commit()
    db_session.refresh(config)

    # Run the experiment to populate database
    db_url = str(db_session.get_bind().engine.url)
    perform_experiment(str(config.id), db_url)

    # Call function under test
    df = load_runs_df(db_session)

    # Assertions
    assert isinstance(df, pl.DataFrame)
    assert df.height > 0  # Should have at least one row

    # Check column names
    expected_columns = [
        "run_id",
        "experiment_id",
        "network",
        "image_model",
        "text_model",
        "initial_prompt",
        "prompt_category",
        "seed",
        "max_length",
        "num_invocations",
        "persistence_diagram_id",
        "embedding_model",
        "homology_dimension",
        "entropy",
    ]

    # Check for missing columns using set difference
    missing_columns = set(expected_columns) - set(df.columns)
    if missing_columns:
        raise AssertionError(f"Missing columns in DataFrame: {missing_columns}")

    # Check for extra columns
    extra_columns = [col for col in df.columns if col not in expected_columns]
    if extra_columns:
        raise AssertionError(f"Unexpected extra columns in DataFrame: {extra_columns}")

    # Check experiment_id is correctly stored
    assert df.filter(pl.col("experiment_id") == str(config.id)).height > 0

    # Verify field values
    first_row = df.row(0, named=True)
    assert first_row["initial_prompt"] == "test runs dataframe"
    assert first_row["seed"] == -1  # Updated to match actual seed value
    assert first_row["network"] == ["DummyT2I", "DummyI2T"]
    assert first_row["max_length"] == 10  # Updated to match configured max_length
    assert first_row["num_invocations"] == 10
    assert first_row["experiment_id"] == str(config.id)

    # Verify image_model and text_model fields
    assert first_row["image_model"] == "DummyT2I"
    assert first_row["text_model"] == "DummyI2T"


def test_load_pd_df(db_session):
    """Test that load_pd_df correctly loads persistence diagram data."""
    # Create a test configuration with a longer run length
    config = ExperimentConfig(
        networks=[["DummyT2I", "DummyI2T"]],
        seeds=[-1, -1],
        prompts=["test persistence diagram data loading"],
        embedding_models=["Dummy"],
        max_length=100,  # Longer run to get more interesting persistence diagrams
    )

    # Save config to database to get an ID
    db_session.add(config)
    db_session.commit()
    db_session.refresh(config)

    # Run the experiment to populate database
    db_url = str(db_session.get_bind().engine.url)
    perform_experiment(str(config.id), db_url)

    # Load the persistence diagram data
    pd_df = load_pd_df(db_session)

    # Verify we have rows in the DataFrame
    assert pd_df.height > 0

    # Check column names for required columns
    required_columns = [
        "persistence_diagram_id",
        "run_id",
        "embedding_model",
        "homology_dimension",
        "birth",
        "death",
        "persistence",
        "initial_prompt",
        "network",
        "experiment_id",
    ]

    # Check that all required columns exist
    for column in required_columns:
        assert column in pd_df.columns, f"Column {column} missing from result"

    # Filter for just this experiment
    exp_df = pd_df.filter(pl.col("experiment_id") == str(config.id))
    assert exp_df.height > 0

    # Verify that persistence = death - birth (excluding infinite values)
    persistence_check = exp_df.with_columns(
        (pl.col("death") - pl.col("birth")).alias("calculated_persistence")
    )
    # Filter out rows with infinite persistence values for this check
    finite_persistence = persistence_check.filter(pl.col("persistence").is_finite())
    assert (
        finite_persistence.filter(
            (pl.col("persistence") - pl.col("calculated_persistence")).abs() > 1e-6
        ).height
        == 0
    ), "Persistence values don't match death - birth calculation"

    # Verify that infinite persistence corresponds to infinite death
    infinite_persistence = persistence_check.filter(pl.col("persistence").is_infinite())
    assert infinite_persistence.filter(~pl.col("death").is_infinite()).height == 0, (
        "Infinite persistence should only occur with infinite death values"
    )

    # Check data types
    assert pd_df["homology_dimension"].dtype == pl.Int64
    assert pd_df["birth"].dtype == pl.Float64
    assert pd_df["death"].dtype == pl.Float64
    assert pd_df["persistence"].dtype == pl.Float64

    # Verify we have data for different homology dimensions
    dimensions = pd_df.select("homology_dimension").unique().to_series().to_list()
    assert len(dimensions) > 0, "No homology dimensions found"

    # Check that network is properly formatted as a string
    network_values = pd_df.select("network").unique()
    assert network_values.height > 0
    assert all("→" in net for net in network_values.to_series().to_list())


def test_add_persistence_entropy(db_session):
    """Test that add_persistence_entropy correctly adds entropy scores to runs DataFrame."""

    # Create a test configuration with a longer run length
    config = ExperimentConfig(
        networks=[["DummyT2I", "DummyI2T"]],
        seeds=[-1, -1],
        prompts=["test persistence entropy"],
        embedding_models=["Dummy"],
        max_length=100,  # Longer run to get more interesting persistence diagrams
    )

    # Save config to database to get an ID
    db_session.add(config)
    db_session.commit()
    db_session.refresh(config)

    # Run the experiment to populate database
    db_url = str(db_session.get_bind().engine.url)
    perform_experiment(str(config.id), db_url)

    # Get the runs DataFrame - note this now includes entropy via add_persistence_entropy
    df = load_runs_df(db_session)

    # Verify we have rows in the DataFrame
    assert df.height > 0

    # Check that entropy-related columns exist after add_persistence_entropy
    entropy_columns = [
        "persistence_diagram_id",
        "embedding_model",
        "homology_dimension",
        "entropy",
    ]

    # Check that all entropy columns exist
    for column in entropy_columns:
        assert column in df.columns, f"Column {column} missing from result"

    # Filter for just this experiment
    exp_df = df.filter(pl.col("experiment_id") == str(config.id))

    # Verify we have entropy data
    entropy_rows = exp_df.filter(pl.col("entropy").is_not_null())
    assert entropy_rows.height > 0, "No entropy data found"

    # Check that we have multiple homology dimensions
    dimensions = (
        entropy_rows.select("homology_dimension").unique().to_series().to_list()
    )
    assert len(dimensions) > 0, "No homology dimensions found"

    # Verify entropy values are reasonable (non-negative)
    assert entropy_rows.filter(pl.col("entropy") < 0).height == 0, (
        "Found negative entropy values"
    )

    # Check that each persistence diagram has entropy for its dimensions
    pd_ids = (
        entropy_rows.select("persistence_diagram_id").unique().to_series().to_list()
    )
    for pd_id in pd_ids:
        pd_rows = entropy_rows.filter(pl.col("persistence_diagram_id") == pd_id)
        # Should have entropy for each dimension in the persistence diagram
        assert pd_rows.height > 0

    # Get all runs associated with this experiment for verification
    runs = list_runs(db_session)

    # Verify that the entropy data matches what's in the database
    for run in runs:
        if str(run.experiment_id) != str(config.id):
            continue
        # Check each persistence diagram for this run
        for pd in run.persistence_diagrams:
            # Filter DataFrame rows for this persistence diagram
            pd_rows = entropy_rows.filter(
                pl.col("persistence_diagram_id") == str(pd.id)
            )

            # Check entropy if available
            if (
                pd.diagram_data
                and "entropy" in pd.diagram_data
                and pd.diagram_data["entropy"] is not None
            ):
                if isinstance(pd.diagram_data["entropy"], np.ndarray):
                    # If entropy is an array of values (one per dimension)
                    for dim, ent_val in enumerate(pd.diagram_data["entropy"]):
                        dim_rows = pd_rows.filter(pl.col("homology_dimension") == dim)
                        if dim_rows.height > 0:
                            # Should have exactly one row per dimension for entropy
                            assert dim_rows.height == 1, (
                                f"Expected 1 row for dim {dim}, got {dim_rows.height}"
                            )
                            row = dim_rows.row(0, named=True)
                            assert abs(float(row["entropy"]) - float(ent_val)) < 1e-6


@pytest.mark.skip(
    reason="This will blow away the real cache, which is probably not what you want."
)
def test_cache_dfs(db_session):
    """Test that cache_dfs successfully writes DataFrames to Parquet files."""

    # 1. Setup: Create config, run experiment
    config = ExperimentConfig(
        networks=[["DummyT2I", "DummyI2T"]],
        seeds=[-1],
        prompts=["test caching dataframes"],
        embedding_models=["Dummy"],
        max_length=100,  # Keep it small for testing
    )
    db_session.add(config)
    db_session.commit()
    db_session.refresh(config)
    db_url = str(db_session.get_bind().engine.url)
    perform_experiment(str(config.id), db_url)

    cache_dfs(db_session, clusters=False)  # Don't cache clusters in this test
    # Check that cache directory exists
    cache_dir = Path("output/cache")

    # Check that all expected cache files exist
    assert (cache_dir / "runs.parquet").exists(), "Runs cache file not found"
    assert (cache_dir / "invocations.parquet").exists(), (
        "Invocations cache file not found"
    )
    assert (cache_dir / "embeddings.parquet").exists(), (
        "Embeddings cache file not found"
    )


def test_calculate_semantic_drift():
    """Test that calculate_semantic_drift correctly computes cosine distances using normalized vectors."""

    # Create test vectors with known cosine relationships
    vectors = np.array(
        [
            [1, 0, 0],  # Unit vector along x-axis
            [0, 1, 0],  # Unit vector along y-axis (orthogonal to v1)
            [-1, 0, 0],  # Opposite direction to v1
            [1, 1, 0],  # 45 degrees from v1 in xy-plane
            [1, 0, 0],  # Same as v1 (should have 0 distance)
            [2, 0, 0],  # Same direction as v1 but different magnitude
        ],
        dtype=np.float32,
    )

    reference = np.array([1, 0, 0], dtype=np.float32)  # Reference vector

    # Calculate semantic drift
    distances = calculate_semantic_drift(vectors, reference)

    # Expected distances (using normalized euclidean which is monotonic with cosine distance)
    # For normalized vectors: euclidean_dist = sqrt(2 - 2*cos_similarity)
    # cos_dist = 1 - cos_similarity, so:
    # - Same direction (cos_sim=1): euclidean=0
    # - Orthogonal (cos_sim=0): euclidean=sqrt(2)
    # - Opposite (cos_sim=-1): euclidean=2

    # Check specific cases
    assert abs(distances[0]) < 1e-6, "Same vector should have 0 distance"
    assert abs(distances[1] - np.sqrt(2)) < 1e-6, (
        "Orthogonal vectors should have sqrt(2) distance"
    )
    assert abs(distances[2] - 2.0) < 1e-6, "Opposite vectors should have distance 2"
    assert abs(distances[3] - np.sqrt(2 - np.sqrt(2))) < 1e-6, "45 degree angle check"
    assert abs(distances[4]) < 1e-6, "Identical direction should have 0 distance"
    assert abs(distances[5]) < 1e-6, (
        "Same direction different magnitude should have 0 distance after normalization"
    )

    # Test with a different reference vector
    reference2 = np.array([1, 1, 1], dtype=np.float32)
    distances2 = calculate_semantic_drift(vectors, reference2)

    # All distances should be non-negative
    assert np.all(distances2 >= 0), "All distances should be non-negative"
    assert np.all(distances2 <= 2), (
        "All distances should be <= 2 for normalized vectors"
    )


def test_semantic_drift_with_known_values():
    """Test semantic drift calculation with specific known values for validation."""

    # Test 1: Identical vectors should have 0 drift
    v1 = np.array([1, 2, 3], dtype=np.float32)
    result = calculate_semantic_drift(np.array([v1]), v1)
    assert abs(result[0]) < 1e-6, (
        f"Identical vectors should have 0 drift, got {result[0]}"
    )

    # Test 2: Scaled versions of the same vector should have 0 drift (after normalization)
    v2 = np.array([2, 4, 6], dtype=np.float32)  # 2x v1
    result = calculate_semantic_drift(np.array([v2]), v1)
    assert abs(result[0]) < 1e-6, f"Scaled vectors should have 0 drift, got {result[0]}"

    # Test 3: Orthogonal vectors
    v3 = np.array([1, 0, 0], dtype=np.float32)
    v4 = np.array([0, 1, 0], dtype=np.float32)
    result = calculate_semantic_drift(np.array([v4]), v3)
    # For orthogonal normalized vectors, euclidean distance = sqrt(2)
    assert abs(result[0] - np.sqrt(2)) < 1e-6, (
        f"Orthogonal vectors should have sqrt(2) drift, got {result[0]}"
    )

    # Test 4: Opposite vectors
    v5 = np.array([1, 0, 0], dtype=np.float32)
    v6 = np.array([-1, 0, 0], dtype=np.float32)
    result = calculate_semantic_drift(np.array([v6]), v5)
    # For opposite normalized vectors, euclidean distance = 2
    assert abs(result[0] - 2.0) < 1e-6, (
        f"Opposite vectors should have drift 2, got {result[0]}"
    )

    # Test 5: Multiple vectors at once
    vectors = np.array(
        [
            [1, 0, 0],  # Same as reference
            [0, 1, 0],  # Orthogonal
            [-1, 0, 0],  # Opposite
            [1, 1, 0],  # 45 degrees
        ],
        dtype=np.float32,
    )
    reference = np.array([1, 0, 0], dtype=np.float32)

    results = calculate_semantic_drift(vectors, reference)

    assert abs(results[0]) < 1e-6, "First vector (same as ref) should have 0 drift"
    assert abs(results[1] - np.sqrt(2)) < 1e-6, (
        "Second vector (orthogonal) should have sqrt(2) drift"
    )
    assert abs(results[2] - 2.0) < 1e-6, "Third vector (opposite) should have drift 2"
    # For 45 degree angle: cos(45°) = 1/sqrt(2), so euclidean = sqrt(2 - 2/sqrt(2)) = sqrt(2 - sqrt(2))
    expected_45_drift = np.sqrt(2 - np.sqrt(2))
    assert abs(results[3] - expected_45_drift) < 1e-6, (
        f"45 degree vector should have drift {expected_45_drift}, got {results[3]}"
    )


def test_filter_top_n_clusters():
    """Test that filter_top_n_clusters properly filters and retains top clusters."""

    # Create a synthetic DataFrame that mimics the structure of clusters_df
    # We need the columns that filter_top_n_clusters expects
    data = {
        "clustering_result_id": ["result1"] * 60 + ["result2"] * 40,
        "embedding_id": [f"00000000-0000-0000-0000-{i:012d}" for i in range(100)],
        "embedding_model": ["Dummy"] * 60 + ["Dummy2"] * 40,
        "cluster_id": list(range(100)),  # Unique cluster IDs
        "cluster_label": (
            # Dummy model clusters with varying frequencies
            ["cluster_A"] * 25  # Most common
            + ["cluster_B"] * 15  # Second most common
            + ["cluster_C"] * 10  # Third most common
            + ["cluster_D"] * 5  # Fourth most common
            + ["cluster_E"] * 5  # Fifth most common
            +
            # Dummy2 model clusters with varying frequencies
            ["cluster_X"] * 20  # Most common
            + ["cluster_Y"] * 10  # Second most common
            + ["cluster_Z"] * 8  # Third most common
            + ["cluster_W"] * 2  # Fourth most common
        ),
        "run_id": [f"run_{i % 10}" for i in range(100)],
        "sequence_number": list(range(100)),
        "invocation_id": [f"inv_{i}" for i in range(100)],
        "algorithm": ["hdbscan"] * 100,
        "epsilon": [0.6] * 100,
        "initial_prompt": [f"Test prompt {i % 5 + 1}" for i in range(100)],
    }

    # Create the DataFrame
    df = pl.DataFrame(data)

    # Use the filter_top_n_clusters function with n=2
    filtered_df = filter_top_n_clusters(df, 2)

    # Extract the top clusters for each embedding model
    top_clusters_by_model = filtered_df.group_by("embedding_model").agg(
        pl.col("cluster_label").unique().alias("top_clusters")
    )

    # Check results
    for model in ["Dummy", "Dummy2"]:
        model_row = top_clusters_by_model.filter(
            pl.col("embedding_model") == model
        ).row(0, named=True)
        top_clusters = model_row["top_clusters"]

        # Verify we got the expected number of top clusters
        expected_clusters = (
            ["cluster_A", "cluster_B"]
            if model == "Dummy"
            else ["cluster_X", "cluster_Y"]
        )
        assert set(top_clusters) == set(expected_clusters), (
            f"Expected top clusters for {model} to be {set(expected_clusters)}, got {set(top_clusters)}"
        )

    # Verify other clusters were filtered out
    for cluster in ["cluster_C", "cluster_D", "cluster_E", "cluster_Z", "cluster_W"]:
        assert cluster not in filtered_df.select("cluster_label").unique(), (
            f"Cluster {cluster} should have been filtered out"
        )

    # Test with n=1 to keep only the most common cluster per model
    filtered_df_n1 = filter_top_n_clusters(df, 1)
    top_clusters_n1 = filtered_df_n1.group_by("embedding_model").agg(
        pl.col("cluster_label").unique().alias("top_clusters")
    )

    # Check each model keeps only the single most common cluster
    for model in ["Dummy", "Dummy2"]:
        model_row = top_clusters_n1.filter(pl.col("embedding_model") == model).row(
            0, named=True
        )
        top_clusters = model_row["top_clusters"]

        expected_top = ["cluster_A"] if model == "Dummy" else ["cluster_X"]
        assert set(top_clusters) == set(expected_top), (
            f"Expected top cluster for {model} to be {expected_top}, got {set(top_clusters)}"
        )

    # Test with both embedding_model and initial_prompt in the grouping variables
    filtered_df_combined = filter_top_n_clusters(df, 1, ["initial_prompt"])

    # Extract the top clusters for each combination of embedding_model and initial_prompt
    top_clusters_combined = filtered_df_combined.group_by([
        "embedding_model",
        "initial_prompt",
    ]).agg(pl.col("cluster_label").unique().alias("top_clusters"))

    # Check that we have entries for all combinations
    assert top_clusters_combined.height == 10, (
        "Expected 10 unique combinations (2 models × 5 prompts)"
    )

    # Check each combination has the expected top cluster
    for model in ["Dummy", "Dummy2"]:
        for prompt_num in range(1, 6):
            prompt = f"Test prompt {prompt_num}"

            # Get the row for this combination
            combination_row = top_clusters_combined.filter(
                (pl.col("embedding_model") == model)
                & (pl.col("initial_prompt") == prompt)
            ).row(0, named=True)

            top_clusters = combination_row["top_clusters"]

            # Verify the top cluster for each combination
            expected_top = ["cluster_A"] if model == "Dummy" else ["cluster_X"]
            assert set(top_clusters) == set(expected_top), (
                f"Expected top cluster for {model}/{prompt} to be {expected_top}, got {set(top_clusters)}"
            )


def test_calculate_cluster_transitions():
    """Test that calculate_cluster_transitions correctly identifies transitions between clusters."""

    # Create a synthetic DataFrame that mimics the structure expected by calculate_cluster_transitions
    # The function expects a dataframe with at least: run_id, embedding_model, sequence_number, cluster_label
    data = {
        "clustering_result_id": ["result1"] * 20,
        "embedding_id": [f"emb_{i}" for i in range(20)],
        "embedding_model": ["model1"] * 20,
        "cluster_id": list(range(20)),
        "cluster_label": (
            ["cluster_A"] * 5  # First 5 invocations are in cluster A
            + ["cluster_B"] * 3  # Next 3 invocations are in cluster B
            + ["cluster_A"] * 2  # Then back to cluster A
            + ["cluster_C"] * 4  # Then to cluster C
            + ["cluster_B"] * 4  # Then back to cluster B
            + ["cluster_A"] * 2  # Finally back to cluster A
        ),
        "run_id": ["run1"] * 20,
        "sequence_number": list(range(1, 21)),  # 20 sequential invocations
        "invocation_id": [f"inv_{i}" for i in range(20)],
        "algorithm": ["hdbscan"] * 20,
        "epsilon": [0.6] * 20,
    }

    # Create the DataFrame
    df = pl.DataFrame(data)

    # Apply the function
    transitions_df = calculate_cluster_transitions(df, ["embedding_model"])
    print(transitions_df)

    # Check that the output is a DataFrame
    assert isinstance(transitions_df, pl.DataFrame)

    # Expected transitions and counts based on our synthetic data:
    # A->B: 1 time (after the first 5 A's)
    # B->A: 1 time (after the first 3 B's)
    # A->C: 1 time (after the next 2 A's)
    # C->B: 1 time (after the 4 C's)
    # B->A: 1 time (after the last 4 B's)

    # Check columns
    expected_columns = [
        "embedding_model",
        "from_cluster",
        "to_cluster",
        "transition_count",
    ]
    assert all(col in transitions_df.columns for col in expected_columns)

    # Convert to a more easily testable format
    transitions_dict = {
        (row["from_cluster"], row["to_cluster"]): row["transition_count"]
        for row in transitions_df.filter(pl.col("embedding_model") == "model1").rows(
            named=True
        )
    }

    # Check expected transitions
    expected_transitions = {
        ("cluster_A", "cluster_B"): 1,
        ("cluster_B", "cluster_A"): 2,
        ("cluster_A", "cluster_C"): 1,
        ("cluster_C", "cluster_B"): 1,
    }

    # Verify transitions exist and have correct counts
    for transition, expected_count in expected_transitions.items():
        from_cluster, to_cluster = transition
        assert transition in transitions_dict, (
            f"Transition {from_cluster}->{to_cluster} not found"
        )
        assert transitions_dict[transition] == expected_count, (
            f"Expected count {expected_count} for transition {from_cluster}->{to_cluster}, "
            f"got {transitions_dict[transition]}"
        )

    # Test with multiple run_ids
    multi_run_data = {
        "clustering_result_id": ["result1"] * 20,
        "embedding_id": [f"emb_{i}" for i in range(20)],
        "embedding_model": ["model1"] * 20,
        "cluster_id": list(range(20)),
        "cluster_label": (
            ["cluster_A"] * 5
            + ["cluster_B"] * 5  # run1: A->B
            + ["cluster_C"] * 5
            + ["cluster_D"] * 5  # run2: C->D
        ),
        "run_id": ["run1"] * 10 + ["run2"] * 10,
        "sequence_number": list(range(1, 11)) + list(range(1, 11)),  # 1-10 for each run
        "invocation_id": [f"inv_{i}" for i in range(20)],
        "algorithm": ["hdbscan"] * 20,
        "epsilon": [0.6] * 20,
    }

    multi_run_df = pl.DataFrame(multi_run_data)
    multi_transitions_df = calculate_cluster_transitions(
        multi_run_df, ["embedding_model"]
    )

    # Convert to dictionary for easy testing
    multi_transitions_dict = {
        (row["from_cluster"], row["to_cluster"]): row["transition_count"]
        for row in multi_transitions_df.filter(
            pl.col("embedding_model") == "model1"
        ).rows(named=True)
    }

    # Check expected transitions for multiple runs
    expected_multi_transitions = {
        ("cluster_A", "cluster_B"): 1,  # From run1
        ("cluster_C", "cluster_D"): 1,  # From run2
    }

    for transition, expected_count in expected_multi_transitions.items():
        from_cluster, to_cluster = transition
        assert transition in multi_transitions_dict, (
            f"Transition {from_cluster}->{to_cluster} not found"
        )
        assert multi_transitions_dict[transition] == expected_count, (
            f"Expected count {expected_count} for transition {from_cluster}->{to_cluster}, "
            f"got {multi_transitions_dict[transition]}"
        )


def test_calculate_cluster_transitions_include_outliers():
    """Test that calculate_cluster_transitions correctly filters outliers when requested."""

    # Create synthetic data with OUTLIER clusters
    data = {
        "clustering_result_id": ["result1"] * 15,
        "embedding_id": [f"emb_{i}" for i in range(15)],
        "embedding_model": ["model1"] * 15,
        "cluster_id": [
            -1 if i in [3, 4, 7, 10, 13] else i for i in range(15)
        ],  # -1 for outliers
        "cluster_label": (
            ["cluster_A"] * 3  # First 3 invocations are in cluster A
            + ["OUTLIER"] * 2  # Next 2 are outliers
            + ["cluster_B"] * 3  # Then to cluster B
            + ["OUTLIER"] * 1  # Then an outlier
            + ["cluster_C"] * 3  # Then to cluster C
            + ["OUTLIER"] * 1  # Another outlier
            + ["cluster_A"] * 2  # Finally back to cluster A
        ),
        "run_id": ["run1"] * 15,
        "sequence_number": list(range(1, 16)),  # 15 sequential invocations
        "invocation_id": [f"inv_{i}" for i in range(15)],
        "algorithm": ["hdbscan"] * 15,
        "epsilon": [0.6] * 15,
    }

    # Create the DataFrame
    df = pl.DataFrame(data)

    # Apply the function with include_outliers=False
    transitions_df_filtered = calculate_cluster_transitions(
        df, ["embedding_model"], include_outliers=False
    )

    # Check that the output is a DataFrame
    assert isinstance(transitions_df_filtered, pl.DataFrame)

    # Convert to dictionary for easy testing
    transitions_dict_filtered = {
        (row["from_cluster"], row["to_cluster"]): row["transition_count"]
        for row in transitions_df_filtered.filter(
            pl.col("embedding_model") == "model1"
        ).rows(named=True)
    }

    # Check that no transitions involving OUTLIER exist
    for transition in transitions_dict_filtered.keys():
        from_cluster, to_cluster = transition
        assert from_cluster != "OUTLIER" and to_cluster != "OUTLIER", (
            f"Transition {from_cluster}->{to_cluster} involves outliers but should be filtered out"
        )

    # Expected transitions with include_outliers=False
    # We should see only:
    # A->B: 1 (skipping the OUTLIER in between)
    # B->C: 1 (skipping the OUTLIER in between)
    # C->A: 1 (skipping the OUTLIER in between)
    expected_filtered_transitions = {
        ("cluster_A", "cluster_B"): 1,
        ("cluster_B", "cluster_C"): 1,
        ("cluster_C", "cluster_A"): 1,
    }

    # Verify filtered transitions
    for transition, expected_count in expected_filtered_transitions.items():
        assert transition in transitions_dict_filtered, (
            f"Expected transition {transition[0]}->{transition[1]} not found when filtering outliers"
        )
        assert transitions_dict_filtered[transition] == expected_count, (
            f"Expected count {expected_count} for transition {transition[0]}->{transition[1]}, "
            f"got {transitions_dict_filtered[transition]} when filtering outliers"
        )

    # Now apply the function with include_outliers=True
    transitions_df_unfiltered = calculate_cluster_transitions(
        df, ["embedding_model"], include_outliers=True
    )

    # Convert to dictionary for easy testing
    transitions_dict_unfiltered = {
        (row["from_cluster"], row["to_cluster"]): row["transition_count"]
        for row in transitions_df_unfiltered.filter(
            pl.col("embedding_model") == "model1"
        ).rows(named=True)
    }

    # Expected transitions with include_outliers=True
    # Now we should see:
    # A->OUTLIER: 1
    # OUTLIER->B: 1
    # B->OUTLIER: 1
    # OUTLIER->C: 1
    # C->OUTLIER: 1
    # OUTLIER->A: 1
    expected_unfiltered_transitions = {
        ("cluster_A", "OUTLIER"): 1,
        ("OUTLIER", "cluster_B"): 1,
        ("cluster_B", "OUTLIER"): 1,
        ("OUTLIER", "cluster_C"): 1,
        ("cluster_C", "OUTLIER"): 1,
        ("OUTLIER", "cluster_A"): 1,
    }

    # Verify unfiltered transitions
    for transition, expected_count in expected_unfiltered_transitions.items():
        assert transition in transitions_dict_unfiltered, (
            f"Expected transition {transition[0]}->{transition[1]} not found when including outliers"
        )
        assert transitions_dict_unfiltered[transition] == expected_count, (
            f"Expected count {expected_count} for transition {transition[0]}->{transition[1]}, "
            f"got {transitions_dict_unfiltered[transition]} when including outliers"
        )


def test_calculate_cluster_run_lengths():
    """Test that calculate_cluster_run_lengths correctly calculates run lengths."""

    # Create a synthetic DataFrame that mimics the structure expected by calculate_cluster_run_lengths
    data = {
        "clustering_result_id": ["result1"] * 20,
        "embedding_id": [f"emb_{i}" for i in range(20)],
        "embedding_model": ["model1"] * 20,
        "cluster_id": list(range(20)),
        "cluster_label": (
            ["cluster_A"] * 5  # Run of 5 As
            + ["cluster_B"] * 3  # Run of 3 Bs
            + ["cluster_A"] * 2  # Run of 2 As
            + ["cluster_C"] * 4  # Run of 4 Cs
            + ["cluster_B"] * 4  # Run of 4 Bs
            + ["cluster_A"] * 2  # Run of 2 As
        ),
        "run_id": ["run1"] * 20,
        "sequence_number": list(range(1, 21)),
        "invocation_id": [f"inv_{i}" for i in range(20)],
        "algorithm": ["hdbscan"] * 20,
        "epsilon": [0.6] * 20,
        "initial_prompt": ["prompt1"] * 20,
        "network": ["networkA"] * 20,
    }

    # Create the DataFrame
    df = pl.DataFrame(data)

    # Apply the function
    runs_df = calculate_cluster_run_lengths(df)

    # Check that the output is a DataFrame
    assert isinstance(runs_df, pl.DataFrame)

    # Expected run lengths based on our synthetic data as a Polars DataFrame
    # The order of runs should match their appearance in the sequence for each group,
    # as preserved by the function's internal sorting by sequence_number_min.
    expected_runs_df = pl.DataFrame({
        "run_id": ["run1"] * 6,
        "embedding_model": ["model1"] * 6,
        "initial_prompt": ["prompt1"] * 6,
        "network": ["networkA"] * 6,
        "cluster_label": [
            "cluster_A",
            "cluster_B",
            "cluster_A",
            "cluster_C",
            "cluster_B",
            "cluster_A",
        ],
        "run_length": pl.Series([5, 3, 2, 4, 4, 2], dtype=pl.UInt32),
    })

    # Check columns exist
    expected_columns = [
        "run_id",
        "embedding_model",
        "initial_prompt",
        "network",
        "cluster_label",
        "run_length",
    ]
    assert all(col in runs_df.columns for col in expected_columns)
    # Check there are no extraneous columns
    assert set(runs_df.columns) == set(expected_columns)

    # Compare the DataFrames
    # The function sorts by group_cols and then by the start of each run (sequence_number_min internally)
    # So, if expected_runs_df is defined in that precise order, direct comparison with check_row_order=True is valid.
    pl.testing.assert_frame_equal(
        runs_df,
        expected_runs_df,
        check_dtypes=True,
        check_row_order=True,  # Row order matters as function preserves run order
        check_column_order=False,  # Column order is checked by set comparison above
    )

    # Test with multiple run_ids
    multi_run_data = {
        "clustering_result_id": ["result1"] * 20,
        "embedding_id": [f"emb_{i}" for i in range(20)],
        "embedding_model": ["model1"] * 20,
        "cluster_id": list(range(20)),
        "cluster_label": (
            ["cluster_A"] * 5
            + ["cluster_B"] * 5  # run1: A(5), B(5)
            + ["cluster_C"] * 7
            + ["cluster_D"] * 3  # run2: C(7), D(3)
        ),
        "run_id": ["run1"] * 10 + ["run2"] * 10,
        "sequence_number": list(range(1, 11)) + list(range(1, 11)),
        "invocation_id": [f"inv_{i}" for i in range(20)],
        "algorithm": ["hdbscan"] * 20,
        "epsilon": [0.6] * 20,
        "initial_prompt": ["prompt1"] * 10 + ["prompt2"] * 10,
        "network": ["networkA"] * 10 + ["networkB"] * 10,
    }

    multi_run_df = pl.DataFrame(multi_run_data)
    # Apply the function
    actual_multi_runs_df = calculate_cluster_run_lengths(multi_run_df)

    # Expected runs for multiple run_ids as a Polars DataFrame
    # Order of groups (run1 then run2) should be maintained from input if input is sorted by group_cols.
    # Order of runs within groups should be maintained based on sequence_number.
    expected_multi_runs_df = pl.DataFrame({
        "run_id": ["run1", "run1", "run2", "run2"],
        "embedding_model": ["model1", "model1", "model1", "model1"],
        "initial_prompt": ["prompt1", "prompt1", "prompt2", "prompt2"],
        "network": ["networkA", "networkA", "networkB", "networkB"],
        "cluster_label": ["cluster_A", "cluster_B", "cluster_C", "cluster_D"],
        "run_length": pl.Series([5, 5, 7, 3], dtype=pl.UInt32),
    })

    pl.testing.assert_frame_equal(
        actual_multi_runs_df,
        expected_multi_runs_df,
        check_dtypes=True,
        check_row_order=True,
        check_column_order=False,
    )


def test_calculate_cluster_run_lengths_include_outliers():
    """Test that calculate_cluster_run_lengths correctly filters outliers when requested."""

    # Create synthetic data with OUTLIER clusters
    data = {
        "clustering_result_id": ["result1"] * 15,
        "embedding_id": [f"emb_{i}" for i in range(15)],
        "embedding_model": ["model1"] * 15,
        "cluster_id": [
            -1 if i in [3, 4, 7, 10, 13] else i for i in range(15)
        ],  # -1 for outliers
        "cluster_label": (
            ["cluster_A"] * 3  # Run of 3 As
            + ["OUTLIER"] * 2  # Run of 2 outliers
            + ["cluster_B"] * 3  # Run of 3 Bs
            + ["OUTLIER"] * 1  # Single outlier
            + ["cluster_C"] * 3  # Run of 3 Cs
            + ["OUTLIER"] * 1  # Single outlier
            + ["cluster_A"] * 2  # Run of 2 As
        ),
        "run_id": ["run1"] * 15,
        "sequence_number": list(range(1, 16)),
        "invocation_id": [f"inv_{i}" for i in range(15)],
        "algorithm": ["hdbscan"] * 15,
        "epsilon": [0.6] * 15,
        "initial_prompt": ["prompt_outlier_test"] * 15,
        "network": ["network_outlier_test"] * 15,
    }

    # Create the DataFrame
    df = pl.DataFrame(data)

    # Apply the function with include_outliers=False
    runs_df_filtered = calculate_cluster_run_lengths(df, include_outliers=False)

    # Check that the output is a DataFrame
    assert isinstance(runs_df_filtered, pl.DataFrame)

    # Expected run lengths with include_outliers=False.
    # Outliers are filtered first, then runs are calculated on remaining data.
    # Sequence of non-outlier labels: A,A,A, B,B,B, C,C,C, A,A
    # Corresponding runs (ordered by appearance): A(3), B(3), C(3), A(2)
    expected_filtered_runs_df = pl.DataFrame({
        "run_id": ["run1"] * 4,
        "embedding_model": ["model1"] * 4,
        "initial_prompt": ["prompt_outlier_test"] * 4,
        "network": ["network_outlier_test"] * 4,
        "cluster_label": ["cluster_A", "cluster_B", "cluster_C", "cluster_A"],
        "run_length": pl.Series([3, 3, 3, 2], dtype=pl.UInt32),
    })

    # Check columns exist
    expected_columns = [
        "run_id",
        "embedding_model",
        "initial_prompt",
        "network",
        "cluster_label",
        "run_length",
    ]
    assert all(col in runs_df_filtered.columns for col in expected_columns)
    # Check there are no extraneous columns
    assert set(runs_df_filtered.columns) == set(expected_columns)

    # Compare the DataFrames
    pl.testing.assert_frame_equal(
        runs_df_filtered,
        expected_filtered_runs_df,
        check_dtypes=True,
        check_row_order=True,
        check_column_order=False,
    )

    # Now apply the function with include_outliers=True
    runs_df_unfiltered = calculate_cluster_run_lengths(df, include_outliers=True)

    # Expected run lengths with include_outliers=True.
    # Original sequence: A,A,A, O,O, B,B,B, O, C,C,C, O, A,A
    # Corresponding runs (ordered by appearance): A(3), O(2), B(3), O(1), C(3), O(1), A(2)
    expected_unfiltered_runs_df = pl.DataFrame({
        "run_id": ["run1"] * 7,
        "embedding_model": ["model1"] * 7,
        "initial_prompt": ["prompt_outlier_test"] * 7,
        "network": ["network_outlier_test"] * 7,
        "cluster_label": [
            "cluster_A",
            "OUTLIER",
            "cluster_B",
            "OUTLIER",
            "cluster_C",
            "OUTLIER",
            "cluster_A",
        ],
        "run_length": pl.Series([3, 2, 3, 1, 3, 1, 2], dtype=pl.UInt32),
    })

    # Compare the DataFrames
    pl.testing.assert_frame_equal(
        runs_df_unfiltered,
        expected_unfiltered_runs_df,
        check_dtypes=True,
        check_row_order=True,
        check_column_order=False,
    )


def test_filter_top_n_clusters_by_model_and_network():
    """Test that filter_top_n_clusters properly filters with embedding_model and network grouping."""

    # Create a synthetic DataFrame that mimics the structure of clusters_df
    data = {
        "clustering_result_id": ["result1"] * 60 + ["result2"] * 60,
        "embedding_id": [f"00000000-0000-0000-0000-{i:012d}" for i in range(120)],
        "embedding_model": (["Dummy"] * 60 + ["Dummy2"] * 60),
        "cluster_id": list(range(120)),
        "cluster_label": (
            # Dummy model, Network1 clusters
            ["cluster_A1"] * 15  # Most common
            + ["cluster_B1"] * 10  # Second most common
            + ["cluster_C1"] * 5  # Third most common
            +
            # Dummy model, Network2 clusters
            ["cluster_A2"] * 12  # Most common
            + ["cluster_B2"] * 10  # Second most common
            + ["cluster_C2"] * 8  # Third most common
            +
            # Dummy2 model, Network1 clusters
            ["cluster_X1"] * 20  # Most common
            + ["cluster_Y1"] * 8  # Second most common
            + ["cluster_Z1"] * 2  # Third most common
            +
            # Dummy2 model, Network2 clusters
            ["cluster_X2"] * 18  # Most common
            + ["cluster_Y2"] * 12  # Second most common
            + ["cluster_Z2"] * 0  # No instances
        ),
        "run_id": [f"run_{i % 12}" for i in range(120)],
        "sequence_number": list(range(120)),
        "invocation_id": [f"inv_{i}" for i in range(120)],
        "algorithm": ["hdbscan"] * 120,
        "epsilon": [0.6] * 120,
        "network": (["Network1"] * 30 + ["Network2"] * 30) * 2,
    }

    # Create the DataFrame
    df = pl.DataFrame(data)

    # Use the filter_top_n_clusters function with n=2 and grouping by both model and network
    filtered_df = filter_top_n_clusters(df, 2, ["network"])

    # Extract the top clusters for each combination of embedding_model and network
    top_clusters_by_combo = filtered_df.group_by(["embedding_model", "network"]).agg(
        pl.col("cluster_label").unique().alias("top_clusters")
    )

    # Check results for each combination
    expected_top_clusters = {
        ("Dummy", "Network1"): ["cluster_A1", "cluster_B1"],
        ("Dummy", "Network2"): ["cluster_A2", "cluster_B2"],
        ("Dummy2", "Network1"): ["cluster_X1", "cluster_Y1"],
        ("Dummy2", "Network2"): ["cluster_X2", "cluster_Y2"],
    }

    for model in ["Dummy", "Dummy2"]:
        for network in ["Network1", "Network2"]:
            combo_row = top_clusters_by_combo.filter(
                (pl.col("embedding_model") == model) & (pl.col("network") == network)
            ).row(0, named=True)
            top_clusters = set(combo_row["top_clusters"])
            expected_clusters = set(expected_top_clusters[(model, network)])

            assert top_clusters == expected_clusters, (
                f"Expected top clusters for {model}/{network} to be {expected_clusters}, got {top_clusters}"
            )

    # Verify other clusters were filtered out
    filtered_out_clusters = ["cluster_C1", "cluster_C2", "cluster_Z1", "cluster_Z2"]
    for cluster in filtered_out_clusters:
        cluster_count = filtered_df.filter(pl.col("cluster_label") == cluster).height
        assert cluster_count == 0, (
            f"Cluster {cluster} should have been filtered out but has {cluster_count} rows"
        )

    # Test with n=1 to keep only the most common cluster per combination
    filtered_df_n1 = filter_top_n_clusters(df, 1, ["network"])
    top_clusters_n1 = filtered_df_n1.group_by(["embedding_model", "network"]).agg(
        pl.col("cluster_label").unique().alias("top_clusters")
    )

    # Check each combination keeps only the single most common cluster
    expected_top_one = {
        ("Dummy", "Network1"): ["cluster_A1"],
        ("Dummy", "Network2"): ["cluster_A2"],
        ("Dummy2", "Network1"): ["cluster_X1"],
        ("Dummy2", "Network2"): ["cluster_X2"],
    }

    for model in ["Dummy", "Dummy2"]:
        for network in ["Network1", "Network2"]:
            combo_row = top_clusters_n1.filter(
                (pl.col("embedding_model") == model) & (pl.col("network") == network)
            ).row(0, named=True)
            top_clusters = set(combo_row["top_clusters"])
            expected_clusters = set(expected_top_one[(model, network)])

            assert top_clusters == expected_clusters, (
                f"Expected top cluster for {model}/{network} to be {expected_clusters}, got {top_clusters}"
            )


def test_calculate_cluster_bigrams():
    """Test that calculate_cluster_bigrams correctly creates bigrams from cluster sequences."""

    # Create synthetic data representing a few runs with cluster sequences
    data = {
        "clustering_result_id": (
            ["result1"] * 4  # run1
            + ["result1"] * 5  # run2
            + ["result1"] * 6  # run3
            + ["result1"] * 3  # run4
            + ["result2"] * 3  # run5
            + ["result2"] * 1  # run6
            + ["result2"] * 11  # run7
            + ["result2"] * 2  # run8
        ),
        "embedding_id": [f"emb_{i}" for i in range(35)],
        "embedding_model": ["model1"] * 20 + ["model2"] * 15,
        "cluster_id": list(range(35)),
        "cluster_label": (
            # Run 1: A->B->B->C
            ["cluster_A", "cluster_B", "cluster_B", "cluster_C"]
            +
            # Run 2: D->E->F->E->D
            ["cluster_D", "cluster_E", "cluster_F", "cluster_E", "cluster_D"]
            +
            # Run 3: A->A->A->B->C->C
            [
                "cluster_A",
                "cluster_A",
                "cluster_A",
                "cluster_B",
                "cluster_C",
                "cluster_C",
            ]
            +
            # Run 4: G->H->I
            ["cluster_G", "cluster_H", "cluster_I"]
            +
            # Run 5: X->Y->Z (for result2)
            ["cluster_X", "cluster_Y", "cluster_Z"]
            +
            # Run 6: Single element (no bigrams)
            ["cluster_W"]
            +
            # Run 7: P->Q->P->Q->P->Q->P->Q->P->Q->P
            [
                "cluster_P",
                "cluster_Q",
                "cluster_P",
                "cluster_Q",
                "cluster_P",
                "cluster_Q",
                "cluster_P",
                "cluster_Q",
                "cluster_P",
                "cluster_Q",
                "cluster_P",
            ]
            +
            # Run 8: Two elements
            ["cluster_M", "cluster_N"]
        ),
        "run_id": (
            ["run1"] * 4
            + ["run2"] * 5
            + ["run3"] * 6
            + ["run4"] * 3
            + ["run5"] * 3
            + ["run6"] * 1
            + ["run7"] * 11
            + ["run8"] * 2
        ),
        "sequence_number": (
            list(range(1, 5))
            + list(range(1, 6))
            + list(range(1, 7))
            + list(range(1, 4))
            + list(range(3, 6))
            + [1]
            + list(range(1, 12))
            + list(range(1, 3))
        ),
        "invocation_id": [f"inv_{i}" for i in range(35)],
        "algorithm": ["hdbscan"] * 35,
        "epsilon": [0.6] * 35,
    }

    # Create the DataFrame
    df = pl.DataFrame(data)

    # Apply the function
    bigrams_df = calculate_cluster_bigrams(df)

    # Check that the output is a DataFrame with the correct columns
    assert isinstance(bigrams_df, pl.DataFrame)
    expected_columns = ["clustering_result_id", "run_id", "from_cluster", "to_cluster"]
    assert set(bigrams_df.columns) == set(expected_columns)

    # Expected bigrams for each run:
    # Run 1: A->B, B->B, B->C
    # Run 2: D->E, E->F, F->E, E->D
    # Run 3: A->A, A->A, A->B, B->C, C->C
    # Run 4: G->H, H->I
    # Run 5: X->Y, Y->Z
    # Run 6: (no bigrams - single element)
    # Run 7: P->Q (5 times), Q->P (5 times)
    # Run 8: M->N

    # Convert to dictionary for easier testing
    bigrams_dict = {}
    for row in bigrams_df.iter_rows(named=True):
        key = (
            row["clustering_result_id"],
            row["run_id"],
            row["from_cluster"],
            row["to_cluster"],
        )
        bigrams_dict[key] = bigrams_dict.get(key, 0) + 1

    # Check specific expected bigrams
    expected_bigrams = {
        ("result1", "run1", "cluster_A", "cluster_B"): 1,
        ("result1", "run1", "cluster_B", "cluster_B"): 1,
        ("result1", "run1", "cluster_B", "cluster_C"): 1,
        ("result1", "run2", "cluster_D", "cluster_E"): 1,
        ("result1", "run2", "cluster_E", "cluster_F"): 1,
        ("result1", "run2", "cluster_F", "cluster_E"): 1,
        ("result1", "run2", "cluster_E", "cluster_D"): 1,
        ("result1", "run3", "cluster_A", "cluster_A"): 2,
        ("result1", "run3", "cluster_A", "cluster_B"): 1,
        ("result1", "run3", "cluster_B", "cluster_C"): 1,
        ("result1", "run3", "cluster_C", "cluster_C"): 1,
        ("result1", "run4", "cluster_G", "cluster_H"): 1,
        ("result1", "run4", "cluster_H", "cluster_I"): 1,
        ("result2", "run5", "cluster_X", "cluster_Y"): 1,
        ("result2", "run5", "cluster_Y", "cluster_Z"): 1,
        ("result2", "run7", "cluster_P", "cluster_Q"): 5,
        ("result2", "run7", "cluster_Q", "cluster_P"): 5,
        ("result2", "run8", "cluster_M", "cluster_N"): 1,
    }

    for bigram, expected_count in expected_bigrams.items():
        assert bigram in bigrams_dict, f"Expected bigram {bigram} not found"
        assert bigrams_dict[bigram] == expected_count, (
            f"Expected count {expected_count} for bigram {bigram}, got {bigrams_dict[bigram]}"
        )

    # Check that run6 (single element) produces no bigrams
    run6_bigrams = bigrams_df.filter(pl.col("run_id") == "run6")
    assert run6_bigrams.height == 0, "Single-element run should produce no bigrams"

    # Check that run8 produces one bigram
    run8_bigrams = bigrams_df.filter(pl.col("run_id") == "run8")
    assert run8_bigrams.height == 1, "Run8 should produce one bigram"

    # Check total number of bigrams
    # Run1: 3, Run2: 4, Run3: 5, Run4: 2, Run5: 2, Run6: 0, Run7: 10, Run8: 1
    expected_total = 3 + 4 + 5 + 2 + 2 + 0 + 10 + 1
    assert bigrams_df.height == expected_total, (
        f"Expected {expected_total} total bigrams, got {bigrams_df.height}"
    )


def test_calculate_cluster_bigrams_with_outliers():
    """Test that calculate_cluster_bigrams correctly handles outliers."""

    # Create synthetic data with outliers
    data = {
        "clustering_result_id": ["result1"] * 10,
        "embedding_id": [f"emb_{i}" for i in range(10)],
        "embedding_model": ["model1"] * 10,
        "cluster_id": [-1, 0, 1, -1, 2, 3, -1, -1, 4, 5],
        "cluster_label": [
            "OUTLIER",
            "cluster_A",
            "cluster_B",
            "OUTLIER",
            "cluster_C",
            "cluster_D",
            "OUTLIER",
            "OUTLIER",
            "cluster_E",
            "cluster_F",
        ],
        "run_id": ["run1"] * 10,
        "sequence_number": list(range(1, 11)),
        "invocation_id": [f"inv_{i}" for i in range(10)],
        "algorithm": ["hdbscan"] * 10,
        "epsilon": [0.6] * 10,
    }

    df = pl.DataFrame(data)

    # Test with include_outliers=False
    bigrams_filtered = calculate_cluster_bigrams(df, include_outliers=False)

    # Expected bigrams without outliers: A->B, B->C, C->D, D->E, E->F
    filtered_dict = {}
    for row in bigrams_filtered.iter_rows(named=True):
        key = (row["from_cluster"], row["to_cluster"])
        filtered_dict[key] = filtered_dict.get(key, 0) + 1

    expected_filtered = {
        ("cluster_A", "cluster_B"): 1,
        ("cluster_B", "cluster_C"): 1,
        ("cluster_C", "cluster_D"): 1,
        ("cluster_D", "cluster_E"): 1,
        ("cluster_E", "cluster_F"): 1,
    }

    assert len(filtered_dict) == len(expected_filtered), (
        f"Expected {len(expected_filtered)} unique bigrams without outliers, got {len(filtered_dict)}"
    )

    for bigram, count in expected_filtered.items():
        assert bigram in filtered_dict, f"Expected bigram {bigram} not found"
        assert filtered_dict[bigram] == count, (
            f"Expected count {count} for bigram {bigram}, got {filtered_dict[bigram]}"
        )

    # Test with include_outliers=True
    bigrams_unfiltered = calculate_cluster_bigrams(df, include_outliers=True)

    # Expected bigrams with outliers included
    unfiltered_dict = {}
    for row in bigrams_unfiltered.iter_rows(named=True):
        key = (row["from_cluster"], row["to_cluster"])
        unfiltered_dict[key] = unfiltered_dict.get(key, 0) + 1

    expected_unfiltered = {
        ("OUTLIER", "cluster_A"): 1,
        ("cluster_A", "cluster_B"): 1,
        ("cluster_B", "OUTLIER"): 1,
        ("OUTLIER", "cluster_C"): 1,
        ("cluster_C", "cluster_D"): 1,
        ("cluster_D", "OUTLIER"): 1,
        ("OUTLIER", "OUTLIER"): 1,
        ("OUTLIER", "cluster_E"): 1,
        ("cluster_E", "cluster_F"): 1,
    }

    assert len(unfiltered_dict) == len(expected_unfiltered), (
        f"Expected {len(expected_unfiltered)} unique bigrams with outliers, got {len(unfiltered_dict)}"
    )

    for bigram, count in expected_unfiltered.items():
        assert bigram in unfiltered_dict, f"Expected bigram {bigram} not found"
        assert unfiltered_dict[bigram] == count, (
            f"Expected count {count} for bigram {bigram}, got {unfiltered_dict[bigram]}"
        )


def test_calculate_cluster_bigrams_empty():
    """Test that calculate_cluster_bigrams handles empty DataFrames correctly."""

    # Create empty DataFrame with correct schema
    empty_df = pl.DataFrame(
        schema={
            "clustering_result_id": pl.Utf8,
            "embedding_id": pl.Utf8,
            "embedding_model": pl.Utf8,
            "cluster_id": pl.Int64,
            "cluster_label": pl.Utf8,
            "run_id": pl.Utf8,
            "sequence_number": pl.Int64,
            "invocation_id": pl.Utf8,
            "algorithm": pl.Utf8,
            "epsilon": pl.Float64,
        }
    )

    # Calculate bigrams
    bigrams_df = calculate_cluster_bigrams(empty_df)

    # Should return empty DataFrame with correct schema
    assert bigrams_df.height == 0
    assert set(bigrams_df.columns) == {
        "clustering_result_id",
        "run_id",
        "from_cluster",
        "to_cluster",
    }
    assert bigrams_df.schema["clustering_result_id"] == pl.Utf8
    assert bigrams_df.schema["run_id"] == pl.Utf8
    assert bigrams_df.schema["from_cluster"] == pl.Utf8
    assert bigrams_df.schema["to_cluster"] == pl.Utf8
