from pathlib import Path

import numpy as np
import polars as pl
import polars.testing
import pytest

from panic_tda.data_prep import (
    add_cluster_labels,
    add_semantic_drift_cosine,
    add_semantic_drift_euclid,
    cache_dfs,
    calculate_cluster_run_lengths,
    calculate_cluster_transitions,
    calculate_cosine_distance,
    calculate_euclidean_distances,
    embed_initial_prompts,
    filter_top_n_clusters,
    load_embeddings_df,
    load_invocations_df,
    load_runs_df,
)
from panic_tda.db import list_runs
from panic_tda.embeddings import EMBEDDING_DIM
from panic_tda.engine import (
    perform_experiment,
)
from panic_tda.schemas import ExperimentConfig


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
    assert text_row["network"] == "DummyT2I → DummyI2T"

    # Check output text for text model (DummyI2T)
    assert "text" in text_row
    assert text_row["text"] is not None
    assert isinstance(text_row["text"], str)
    assert len(text_row["text"]) > 0


def test_add_cluster_labels(db_session):
    """Test that add_cluster_labels properly adds clustering to embeddings."""

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

    # Load embeddings
    df = load_embeddings_df(db_session)

    # Add cluster labels with smaller min_cluster_size to avoid hanging
    df_with_clusters = add_cluster_labels(df, 1, db_session)

    # Check that the cluster_label column was added
    assert "cluster_label" in df_with_clusters.columns

    # Check that each embedding has a cluster label assigned
    assert df_with_clusters.filter(pl.col("cluster_label").is_null()).height == 0

    # Check that cluster labels are strings
    assert df_with_clusters.filter(
        pl.col("cluster_label").cast(pl.Utf8).is_not_null()
    ).height == len(df_with_clusters)

    # Check that cluster labels are non-empty strings
    assert df_with_clusters.filter(
        pl.col("cluster_label").str.len_chars() > 0
    ).height == len(df_with_clusters)

    # Check that each embedding model has its own set of clusters
    dummy_clusters = (
        df_with_clusters.filter(pl.col("embedding_model") == "Dummy")
        .select("cluster_label")
        .unique()
    )
    dummy2_clusters = (
        df_with_clusters.filter(pl.col("embedding_model") == "Dummy2")
        .select("cluster_label")
        .unique()
    )

    # Verify that both models have at least one cluster
    assert dummy_clusters.height > 0
    assert dummy2_clusters.height > 0

    # Verify that all cluster labels are from the set of output texts
    unique_labels = df_with_clusters.select("cluster_label").unique()
    all_texts = df_with_clusters.select("text").unique()

    # Extract the unique texts and labels as sets for easier comparison
    text_set = set(all_texts.select("text").to_series().to_list())
    label_set = set(unique_labels.select("cluster_label").to_series().to_list())

    # Check that each cluster label is either from the set of output texts or is "OUTLIER"
    for label in label_set:
        assert label in text_set or label == "OUTLIER", (
            f"Cluster label '{label}' not found in the set of output texts and is not 'OUTLIER'"
        )

    # Another way to verify this is to check that each label is either a text or "OUTLIER"
    assert all(label in text_set or label == "OUTLIER" for label in label_set), (
        "Some cluster labels are neither from the set of outputs nor 'OUTLIER'"
    )


def test_add_cluster_labels_with_downsampling(db_session):
    """Test that add_cluster_labels properly works with downsampling."""

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

    # Load embeddings
    df = load_embeddings_df(db_session)

    # Record original size
    original_size = df.height

    # Add cluster labels with downsampling factor of 2
    downsample = 2
    df_with_clusters = add_cluster_labels(df, downsample, db_session)

    # Check that the cluster_label column was added
    assert "cluster_label" in df_with_clusters.columns

    # With downsample=2, approximately half of the entries should have null cluster labels
    null_labels_count = df_with_clusters.filter(
        pl.col("cluster_label").is_null()
    ).height
    assert null_labels_count > 0, "Expected some null cluster labels with downsampling"
    assert abs(null_labels_count - original_size / 2) < original_size * 0.1, (
        "Expected approximately half null labels"
    )

    assert (
        df_with_clusters.filter(
            ~pl.col("cluster_label").is_null()
            & (pl.col("cluster_label").str.len_chars() > 0)
        ).height
        == 200 / downsample
    )

    # Check that each embedding model has its own set of clusters (only considering non-null labels)
    dummy_clusters = (
        df_with_clusters.filter(
            (pl.col("embedding_model") == "Dummy")
            & (~pl.col("cluster_label").is_null())
        )
        .select("cluster_label")
        .unique()
    )
    dummy2_clusters = (
        df_with_clusters.filter(
            (pl.col("embedding_model") == "Dummy2")
            & (~pl.col("cluster_label").is_null())
        )
        .select("cluster_label")
        .unique()
    )

    # Verify that both models have at least one cluster
    assert dummy_clusters.height > 0
    assert dummy2_clusters.height > 0

    # Verify that all cluster labels are from the set of output texts or "OUTLIER"
    non_null_df = df_with_clusters.filter(~pl.col("cluster_label").is_null())
    unique_labels = non_null_df.select("cluster_label").unique()
    all_texts = df_with_clusters.select("text").unique()

    # Extract the unique texts and labels as sets for easier comparison
    text_set = set(all_texts.select("text").to_series().to_list())
    label_set = set(unique_labels.select("cluster_label").to_series().to_list())

    # Check that each cluster label is either from the set of output texts or is "OUTLIER"
    for label in label_set:
        assert label in text_set or label == "OUTLIER", (
            f"Cluster label '{label}' not found in the set of output texts and is not 'OUTLIER'"
        )

    # Verify that the output DataFrame has the same number of rows as input
    assert df_with_clusters.height == original_size


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

    # Load embeddings (inc. semantic drift measures)
    df = load_embeddings_df(db_session)
    df = add_semantic_drift_euclid(df, db_session)
    df = add_semantic_drift_cosine(df, db_session)

    # Check that the drift columns were added
    assert "drift_euclid" in df.columns
    assert "drift_cosine" in df.columns

    # Check that drift values are calculated properly
    # All drift values should be non-negative
    assert df.height > 0
    assert (df["drift_euclid"] >= 0.0).all()
    assert (df["drift_cosine"] >= 0.0).all()


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
        "persistence_diagram_started_at",
        "persistence_diagram_completed_at",
        "persistence_diagram_duration",
        "homology_dimension",
        "feature_id",
        "birth",
        "death",
        "persistence",
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


def test_add_persistence_entropy(db_session):
    """Test that add_persistence_entropy correctly adds persistence diagram data to runs DataFrame."""

    # Create a test configuration with a longer run length
    config = ExperimentConfig(
        networks=[["DummyT2I", "DummyI2T"]],
        seeds=[-1, -1],
        prompts=["test persistence data transfer"],
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

    # Get the runs DataFrame
    df = load_runs_df(db_session)

    # Verify we have rows in the DataFrame
    assert df.height > 0

    # Check column names for persistence-related columns
    persistence_columns = [
        "persistence_diagram_id",
        "embedding_model",
        "persistence_diagram_started_at",
        "persistence_diagram_completed_at",
        "persistence_diagram_duration",
        "homology_dimension",
        "feature_id",
        "birth",
        "death",
        "persistence",
        "entropy",
    ]

    # Check that all persistence columns exist
    for column in persistence_columns:
        assert column in df.columns, f"Column {column} missing from result"

    # Filter for just this experiment
    exp_df = df.filter(pl.col("experiment_id") == str(config.id))

    # Get all runs associated with this experiment
    runs = list_runs(db_session)

    # Check each persistence diagram in the database
    for run in runs:
        # Check each persistence diagram for this run
        for pd in run.persistence_diagrams:
            # Filter DataFrame rows for this persistence diagram
            pd_rows = exp_df.filter(pl.col("persistence_diagram_id") == str(pd.id))
            assert pd_rows.height > 0, f"No rows found for persistence diagram {pd.id}"

            # Extract birth-death pairs from the persistence diagram
            dgms = pd.diagram_data.get("dgms", [])

            # Count the total number of birth-death pairs across all dimensions
            total_bd_pairs = sum(
                len(dim_dgm) for dim_dgm in dgms if isinstance(dim_dgm, np.ndarray)
            )

            # There should be one row in the DataFrame for each birth-death pair
            assert pd_rows.height == total_bd_pairs, (
                f"Expected {total_bd_pairs} rows for PD {pd.id}, got {pd_rows.height}"
            )

            # Check each homology dimension
            for dim, dim_dgm in enumerate(dgms):
                if not isinstance(dim_dgm, np.ndarray) or len(dim_dgm) == 0:
                    continue

                # Filter rows for this dimension
                dim_rows = pd_rows.filter(pl.col("homology_dimension") == dim)
                assert dim_rows.height == len(dim_dgm), (
                    f"Expected {len(dim_dgm)} rows for dim {dim}, got {dim_rows.height}"
                )

                # Convert pd_rows to dict for easier comparison
                rows_dict = {}
                for i, row in enumerate(dim_rows.rows(named=True)):
                    rows_dict[row["feature_id"]] = row

                # Check each birth-death pair
                for i, bd_pair in enumerate(dim_dgm):
                    # Skip pairs with infinite death values
                    if np.isinf(bd_pair[1]):
                        continue

                    # Find the matching row in the DataFrame
                    found = False
                    for feature_id, row in rows_dict.items():
                        # Compare birth-death values with some tolerance for floating point
                        if (
                            abs(row["birth"] - bd_pair[0]) < 1e-6
                            and abs(row["death"] - bd_pair[1]) < 1e-6
                            and row["homology_dimension"] == dim
                        ):
                            found = True
                            # Check persistence value
                            assert (
                                abs(row["persistence"] - (bd_pair[1] - bd_pair[0]))
                                < 1e-6
                            )
                            break

                    assert found, (
                        f"Birth-death pair {bd_pair} for dim {dim} not found in DataFrame"
                    )

            # Check entropy if available
            if "entropy" in pd.diagram_data and pd.diagram_data["entropy"] is not None:
                if isinstance(pd.diagram_data["entropy"], np.ndarray):
                    # If entropy is an array of values (one per dimension)
                    for dim, ent_val in enumerate(pd.diagram_data["entropy"]):
                        dim_rows = pd_rows.filter(pl.col("homology_dimension") == dim)
                        if dim_rows.height > 0:
                            # Check each row individually to avoid scalar indexing issues
                            for row in dim_rows.rows(named=True):
                                if "entropy" in row and row["entropy"] is not None:
                                    assert (
                                        abs(float(row["entropy"]) - float(ent_val))
                                        < 1e-6
                                    )
                elif isinstance(pd.diagram_data["entropy"], (float, int)):
                    # If entropy is a single value for the entire diagram
                    for row in pd_rows.rows(named=True):
                        if "entropy" in row and row["entropy"] is not None:
                            assert (
                                abs(
                                    float(row["entropy"])
                                    - float(pd.diagram_data["entropy"])
                                )
                                < 1e-6
                            )


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

    cache_dfs(db_session)
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


def test_calculate_euclidean_distances():
    """Test that calculate_euclidean_distances correctly computes Euclidean distances."""

    # Create vectors with known distances
    vectors = {
        "v1": np.array([0, 0, 0], dtype=np.float32),  # Origin
        "v2": np.array([1, 0, 0], dtype=np.float32),  # Unit distance along x-axis
        "v3": np.array([0, 3, 0], dtype=np.float32),  # 3 units along y-axis
        "v4": np.array([0, 0, 5], dtype=np.float32),  # 5 units along z-axis
        "v5": np.array([1, 1, 1], dtype=np.float32),  # sqrt(3) from origin
    }

    # Expected distances from origin (v1)
    expected_distances = {
        "v1": 0.0,  # Distance to self should be 0
        "v2": 1.0,  # Unit distance
        "v3": 3.0,  # 3 units distance
        "v4": 5.0,  # 5 units distance
        "v5": np.sqrt(3),  # sqrt(1^2 + 1^2 + 1^2) = sqrt(3)
    }

    # Calculate distances using the function
    computed_distances = calculate_euclidean_distances(
        vectors["v1"],
        [vectors["v1"], vectors["v2"], vectors["v3"], vectors["v4"], vectors["v5"]],
    )

    # Assert computed distances match expected with small tolerance for floating-point errors
    for i, key in enumerate(["v1", "v2", "v3", "v4", "v5"]):
        assert abs(computed_distances[i] - expected_distances[key]) < 1e-6, (
            f"Distance for {key} incorrect. Expected {expected_distances[key]}, got {computed_distances[i]}"
        )

    # Test with some other vector as reference
    reference = vectors["v5"]  # [1, 1, 1]

    # Calculate expected distances from v5 to others
    expected_from_v5 = {
        "v1": np.sqrt(3),  # sqrt((1-0)^2 + (1-0)^2 + (1-0)^2)
        "v2": np.sqrt(2),  # sqrt((1-1)^2 + (1-0)^2 + (1-0)^2)
        "v3": np.sqrt(6),  # sqrt((1-0)^2 + (1-3)^2 + (1-0)^2)
        "v4": np.sqrt(18),  # sqrt((1-0)^2 + (1-0)^2 + (1-5)^2)
        "v5": 0.0,  # Distance to self should be 0
    }

    # Calculate distances using the function
    computed_from_v5 = calculate_euclidean_distances(
        reference,
        [vectors["v1"], vectors["v2"], vectors["v3"], vectors["v4"], vectors["v5"]],
    )

    # Assert computed distances match expected with small tolerance for floating-point errors
    for i, key in enumerate(["v1", "v2", "v3", "v4", "v5"]):
        assert np.isclose(computed_from_v5[i], expected_from_v5[key]), (
            f"Distance from v5 to {key} incorrect. Expected {expected_from_v5[key]}, got {computed_from_v5[i]}"
        )


def test_calculate_cosine_distance():
    """Test that calculate_cosine_distance correctly computes cosine distances."""

    # Create vectors with known cosine distances
    vectors = {
        "v1": np.array([1, 0, 0], dtype=np.float32),  # Unit vector along x-axis
        "v2": np.array([0, 1, 0], dtype=np.float32),  # Unit vector along y-axis
        "v3": np.array([0, 0, 1], dtype=np.float32),  # Unit vector along z-axis
        "v4": np.array([1, 1, 0], dtype=np.float32),  # Vector in xy-plane
        "v5": np.array([-1, 0, 0], dtype=np.float32),  # Opposite to v1
        "v6": np.array([1, 1, 1], dtype=np.float32),  # Equal components
    }

    # Expected cosine distances from v1 (normalized to unit length)
    expected_distances = {
        "v1": 0.0,  # Same direction, distance should be 0
        "v2": 1.0,  # Orthogonal, distance should be 1
        "v3": 1.0,  # Orthogonal, distance should be 1
        "v4": 1 - 1 / np.sqrt(2),  # cosine distance = 1 - 0.7071 ≈ 0.2929
        "v5": 2.0,  # Opposite direction, distance should be 2
        "v6": 1 - 1 / np.sqrt(3),  # cosine distance = 1 - 0.5774 ≈ 0.4226
    }

    # Calculate distances using the function
    computed_distances = calculate_cosine_distance(
        np.array([
            vectors["v1"],
            vectors["v2"],
            vectors["v3"],
            vectors["v4"],
            vectors["v5"],
            vectors["v6"],
        ]),
        vectors["v1"],
    )

    # Assert computed distances match expected with small tolerance for floating-point errors
    for i, key in enumerate(["v1", "v2", "v3", "v4", "v5", "v6"]):
        assert np.isclose(computed_distances[i], expected_distances[key]), (
            f"Cosine distance for {key} incorrect. Expected {expected_distances[key]}, got {computed_distances[i]}"
        )

    # Test with some other vector as reference
    reference = vectors["v6"]  # [1, 1, 1]

    # Calculate expected cosine distances from v6 to others
    expected_from_v6 = {
        "v1": 1 - 1 / np.sqrt(3),  # cosine distance = 1 - 0.5774 ≈ 0.4226
        "v2": 1 - 1 / np.sqrt(3),  # cosine distance = 1 - 0.5774 ≈ 0.4226
        "v3": 1 - 1 / np.sqrt(3),  # cosine distance = 1 - 0.5774 ≈ 0.4226
        "v4": 1 - 2 / np.sqrt(6),  # cosine distance = 1 - 0.8165 ≈ 0.1835
        "v5": 1 - (-1) / np.sqrt(3),  # cosine distance = 1 - (-0.5774) ≈ 1.5774
        "v6": 0.0,  # Distance to self should be 0
    }

    # Calculate distances using the function
    computed_from_v6 = calculate_cosine_distance(
        np.array([
            vectors["v1"],
            vectors["v2"],
            vectors["v3"],
            vectors["v4"],
            vectors["v5"],
            vectors["v6"],
        ]),
        reference,
    )

    # Assert computed distances match expected with small tolerance for floating-point errors
    for i, key in enumerate(["v1", "v2", "v3", "v4", "v5", "v6"]):
        assert np.isclose(computed_from_v6[i], expected_from_v6[key]), (
            f"Cosine distance from v6 to {key} incorrect. Expected {expected_from_v6[key]}, got {computed_from_v6[i]}"
        )


def test_filter_top_n_clusters():
    """Test that filter_top_n_clusters properly filters and retains top clusters."""

    # Create a synthetic DataFrame with embedding_model and cluster_label columns
    # We don't need an actual experiment, just the minimal columns required
    data = {
        "id": [f"00000000-0000-0000-0000-{i:012d}" for i in range(100)],
        "initial_prompt": [f"Test prompt {i % 5 + 1}" for i in range(100)],
        "embedding_model": ["Dummy"] * 60 + ["Dummy2"] * 40,
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

    # Create a synthetic DataFrame with sequential invocations and cluster labels
    # We'll set up a sequence of transitions between various clusters
    data = {
        "run_id": ["run1"] * 20,
        "embedding_model": ["model1"] * 20,
        "sequence_number": list(range(1, 21)),  # 20 sequential invocations
        "cluster_label": (
            ["cluster_A"] * 5  # First 5 invocations are in cluster A
            + ["cluster_B"] * 3  # Next 3 invocations are in cluster B
            + ["cluster_A"] * 2  # Then back to cluster A
            + ["cluster_C"] * 4  # Then to cluster C
            + ["cluster_B"] * 4  # Then back to cluster B
            + ["cluster_A"] * 2  # Finally back to cluster A
        ),
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
        "run_id": ["run1"] * 10 + ["run2"] * 10,
        "embedding_model": ["model1"] * 20,
        "sequence_number": list(range(1, 11)) + list(range(1, 11)),  # 1-10 for each run
        "cluster_label": (
            ["cluster_A"] * 5
            + ["cluster_B"] * 5  # run1: A->B
            + ["cluster_C"] * 5
            + ["cluster_D"] * 5  # run2: C->D
        ),
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
        "run_id": ["run1"] * 15,
        "embedding_model": ["model1"] * 15,
        "sequence_number": list(range(1, 16)),  # 15 sequential invocations
        "cluster_label": (
            ["cluster_A"] * 3  # First 3 invocations are in cluster A
            + ["OUTLIER"] * 2  # Next 2 are outliers
            + ["cluster_B"] * 3  # Then to cluster B
            + ["OUTLIER"] * 1  # Then an outlier
            + ["cluster_C"] * 3  # Then to cluster C
            + ["OUTLIER"] * 1  # Another outlier
            + ["cluster_A"] * 2  # Finally back to cluster A
        ),
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

    # Create a synthetic DataFrame with sequential invocations and cluster labels
    data = {
        "run_id": ["run1"] * 20,
        "embedding_model": ["model1"] * 20,
        "sequence_number": list(range(1, 21)),
        "cluster_label": (
            ["cluster_A"] * 5  # Run of 5 As
            + ["cluster_B"] * 3  # Run of 3 Bs
            + ["cluster_A"] * 2  # Run of 2 As
            + ["cluster_C"] * 4  # Run of 4 Cs
            + ["cluster_B"] * 4  # Run of 4 Bs
            + ["cluster_A"] * 2  # Run of 2 As
        ),
    }

    # Create the DataFrame
    df = pl.DataFrame(data)

    # Apply the function
    runs_df = calculate_cluster_run_lengths(df)

    # Check that the output is a DataFrame
    assert isinstance(runs_df, pl.DataFrame)

    # Expected run lengths based on our synthetic data as a Polars DataFrame
    # The order of runs should match their appearance in the sequence for each group.
    expected_runs_df = pl.DataFrame({
        "run_id": ["run1"] * 6,
        "embedding_model": ["model1"] * 6,
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
    expected_columns = ["run_id", "embedding_model", "cluster_label", "run_length"]
    assert all(col in runs_df.columns for col in expected_columns)
    # Check there are no extraneous columns
    assert set(runs_df.columns) == set(expected_columns)

    # Compare the DataFrames
    pl.testing.assert_frame_equal(
        runs_df, expected_runs_df, check_dtypes=True, check_row_order=True, check_column_order=False
    )

    # Test with multiple run_ids
    multi_run_data = {
        "run_id": ["run1"] * 10 + ["run2"] * 10,
        "embedding_model": ["model1"] * 20,
        "sequence_number": list(range(1, 11)) + list(range(1, 11)),
        "cluster_label": (
            ["cluster_A"] * 5
            + ["cluster_B"] * 5  # run1: A(5), B(5)
            + ["cluster_C"] * 7
            + ["cluster_D"] * 3  # run2: C(7), D(3)
        ),
    }

    multi_run_df = pl.DataFrame(multi_run_data)
    # Apply the function
    actual_multi_runs_df = calculate_cluster_run_lengths(multi_run_df)

    # Expected runs for multiple run_ids as a Polars DataFrame
    # Order of groups (run1 then run2) should be maintained from input.
    # Order of runs within groups should be maintained.
    expected_multi_runs_df = pl.DataFrame({
        "run_id": ["run1", "run1", "run2", "run2"],
        "embedding_model": ["model1", "model1", "model1", "model1"],
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
        "run_id": ["run1"] * 15,
        "embedding_model": ["model1"] * 15,
        "sequence_number": list(range(1, 16)),
        "cluster_label": (
            ["cluster_A"] * 3  # Run of 3 As
            + ["OUTLIER"] * 2  # Run of 2 outliers
            + ["cluster_B"] * 3  # Run of 3 Bs
            + ["OUTLIER"] * 1  # Single outlier
            + ["cluster_C"] * 3  # Run of 3 Cs
            + ["OUTLIER"] * 1  # Single outlier
            + ["cluster_A"] * 2  # Run of 2 As
        ),
    }

    # Create the DataFrame
    df = pl.DataFrame(data)

    # Apply the function with include_outliers=False
    runs_df_filtered = calculate_cluster_run_lengths(df, include_outliers=False)

    # Check that the output is a DataFrame
    assert isinstance(runs_df_filtered, pl.DataFrame)

    # Expected run lengths with include_outliers=False.
    # Outliers are filtered first, then runs are calculated on remaining data.
    # Sequence becomes: A,A,A, B,B,B, C,C,C, A,A
    # Runs: A(3), B(3), C(3), A(2)
    expected_filtered_runs_df = pl.DataFrame({
        "run_id": ["run1"] * 4,
        "embedding_model": ["model1"] * 4,
        "cluster_label": ["cluster_A", "cluster_B", "cluster_C", "cluster_A"],
        "run_length": pl.Series([3, 3, 3, 2], dtype=pl.UInt32),
    })

    # Check columns exist
    expected_columns = ["run_id", "embedding_model", "cluster_label", "run_length"]
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
    # Runs: A(3), O(2), B(3), O(1), C(3), O(1), A(2)
    expected_unfiltered_runs_df = pl.DataFrame({
        "run_id": ["run1"] * 7,
        "embedding_model": ["model1"] * 7,
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

    # Create a synthetic DataFrame with embedding_model, network, and cluster_label columns
    data = {
        "id": [f"00000000-0000-0000-0000-{i:012d}" for i in range(120)],
        "initial_prompt": [f"Test prompt {i % 3 + 1}" for i in range(120)],
        "embedding_model": (["Dummy"] * 60 + ["Dummy2"] * 60),
        "network": (["Network1"] * 30 + ["Network2"] * 30) * 2,
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
