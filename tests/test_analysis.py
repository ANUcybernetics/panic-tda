import numpy as np
import polars as pl

from panic_tda.analysis import (
    load_embeddings_df,
    load_invocations_df,
    load_runs_df,
)
from panic_tda.db import list_runs
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
    df = load_embeddings_df(db_session)

    # Assertions
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 10  # Adjusted to match actual number of embeddings

    # Check column names
    expected_columns = [
        "id",
        "invocation_id",
        "run_id",
        "embedding_model",
        "started_at",
        "completed_at",
        "sequence_number",
        "semantic_drift_overall",
        "semantic_drift_instantaneous",
        "vector_length",
        "initial_prompt",
        "model",
        "cluster_label",  # Ensure cluster_label column is present
    ]
    assert all(col in df.columns for col in expected_columns)
    # Check there are no extraneous columns
    assert set(df.columns) == set(expected_columns)

    # Check values
    assert df.filter(pl.col("embedding_model") == "Dummy").height == 5
    assert df.filter(pl.col("embedding_model") == "Dummy2").height == 5

    # Verify field values using named columns instead of indices
    text_rows = df.filter(pl.col("model") == "DummyI2T")
    assert text_rows.height > 0
    text_row = text_rows.row(0, named=True)
    assert text_row["initial_prompt"] == "test embedding dataframe"
    assert text_row["model"] == "DummyI2T"
    assert text_row["sequence_number"] == 1

    # Check that each embedding has a cluster label assigned
    assert df.filter(pl.col("cluster_label").is_null()).height == 0

    # Check that cluster labels are integers
    assert df.filter(~pl.col("cluster_label").cast(pl.Int64).is_null()).height == len(
        df
    )

    # Check that each embedding model has its own set of clusters
    dummy_clusters = (
        df.filter(pl.col("embedding_model") == "Dummy").select("cluster_label").unique()
    )
    dummy2_clusters = (
        df.filter(pl.col("embedding_model") == "Dummy2")
        .select("cluster_label")
        .unique()
    )

    # Verify that both models have at least one cluster
    assert dummy_clusters.height > 0
    assert dummy2_clusters.height > 0

    # It's expected that cluster labels can be reused across different embedding models
    # So we don't need to check for disjoint cluster labels between different models


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
    assert df.height > 0  # Should have at least one row for birth/death pairs

    # Check column names
    expected_columns = [
        "run_id",
        "experiment_id",
        "network",
        "image_model",
        "text_model",
        "initial_prompt",
        "seed",
        "max_length",
        "num_invocations",
        "stop_reason",
        "loop_length",
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
    assert all(col in df.columns for col in expected_columns)

    # Check experiment_id is correctly stored
    assert df.filter(pl.col("experiment_id") == str(config.id)).height > 0

    # Verify field values that are the same for all features
    first_row = df.row(0, named=True)
    assert first_row["initial_prompt"] == "test runs dataframe"
    assert first_row["seed"] == -1  # Updated to match actual seed value
    assert first_row["network"] == ["DummyT2I", "DummyI2T"]
    assert first_row["max_length"] == 10  # Updated to match configured max_length
    assert first_row["num_invocations"] == 10
    assert first_row["stop_reason"] == "length"
    assert first_row["experiment_id"] == str(config.id)

    # Verify image_model and text_model fields
    assert first_row["image_model"] == "DummyT2I"
    assert first_row["text_model"] == "DummyI2T"

    # Verify persistence diagram related fields
    assert first_row["embedding_model"] == "Dummy"
    assert first_row["persistence_diagram_id"] is not None

    # Check for null values in persistence diagram-related columns
    persistence_diagram_columns = [
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
    ]
    for column in persistence_diagram_columns:
        assert df.filter(pl.col(column).is_null()).height == 0, (
            f"Found null values in {column} column"
        )

    # Verify birth/death pair and homology dimension fields with proper types
    assert "homology_dimension" in first_row
    assert isinstance(first_row["homology_dimension"], int)
    assert isinstance(first_row["feature_id"], int)
    assert isinstance(first_row["birth"], float)
    assert isinstance(first_row["death"], float)
    assert isinstance(first_row["persistence"], float)

    # Check entropy field if available
    if "entropy" in first_row and first_row["entropy"] is not None:
        assert isinstance(first_row["entropy"], float)


def test_load_runs_df_persistence_data(db_session):
    """Test that load_runs_df correctly loads all persistence diagram data."""

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

    # Filter for just this experiment
    exp_df = df.filter(pl.col("experiment_id") == str(config.id))

    # Verify we have rows in the DataFrame
    assert exp_df.height > 0

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
                        # Debug print to help diagnose comparison issues
                        # print(f"Comparing row: birth={row['birth']}, death={row['death']}, hom_dim={row['homology_dimension']} with pair: {bd_pair}, dim={dim}")
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
