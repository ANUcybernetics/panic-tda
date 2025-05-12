from pathlib import Path

import numpy as np
import polars as pl

from panic_tda.analysis import (
    add_cluster_labels,
    add_semantic_drift_cosine,
    add_semantic_drift_euclid,
    cache_dfs,
    calculate_cosine_distance,
    calculate_euclidean_distances,
    embed_initial_prompts,
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
        "cluster_label",
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

    # Check that cluster labels are integers
    assert df_with_clusters.filter(
        ~pl.col("cluster_label").cast(pl.Int64).is_null()
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

    # Check that the non-null cluster labels are integers
    non_null_count = df_with_clusters.filter(~pl.col("cluster_label").is_null()).height
    assert non_null_count > 0, "Expected some non-null cluster labels"
    assert (
        df_with_clusters.filter(
            ~pl.col("cluster_label").is_null()
            & ~pl.col("cluster_label").cast(pl.Int64).is_null()
        ).height
        == non_null_count
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
