import polars as pl

from trajectory_tracer.analysis import load_embeddings_df, load_runs_df
from trajectory_tracer.engine import (
    perform_experiment,
)
from trajectory_tracer.schemas import ExperimentConfig


def test_load_embeddings_df(db_session):
    """Test that load_embeddings_df returns a polars DataFrame with correct data."""

    # Create a simple test configuration
    config = ExperimentConfig(
        networks=[["DummyT2I", "DummyI2T"]],
        seeds=[42],
        prompts=["test embedding dataframe"],
        embedding_models=["Dummy", "Dummy2"],
        max_length=2,
    )

    # Save config to database to get an ID
    db_session.add(config)
    db_session.commit()
    db_session.refresh(config)

    # Run the experiment to populate database
    db_url = str(db_session.get_bind().engine.url)
    perform_experiment(str(config.id), db_url)

    # Call function under test
    df = load_embeddings_df(db_session, use_cache=False)

    # Assertions
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 4  # 2 invocations * 2 embedding models

    # Check column names
    expected_columns = [
        "id",
        "invocation_id",
        "embedding_started_at",
        "embedding_completed_at",
        "invocation_started_at",
        "invocation_completed_at",
        "duration",
        "run_id",
        "experiment_id",
        "type",
        "initial_prompt",
        "seed",
        "model",
        "sequence_number",
        "embedding_model",
    ]
    assert all(col in df.columns for col in expected_columns)
    # Check there are no extraneous columns
    assert set(df.columns) == set(expected_columns)

    # Check values
    assert df.filter(pl.col("embedding_model") == "Dummy").height == 2
    assert df.filter(pl.col("embedding_model") == "Dummy2").height == 2

    # Check experiment_id is correctly stored
    assert df.filter(pl.col("experiment_id") == str(config.id)).height == 4

    # Verify field values using named columns instead of indices
    text_rows = df.filter(pl.col("model") == "DummyT2I")
    assert text_rows.height > 0
    text_row = text_rows.row(0, named=True)
    assert text_row["initial_prompt"] == "test embedding dataframe"
    assert text_row["model"] == "DummyT2I"
    assert text_row["sequence_number"] == 0
    assert text_row["seed"] == 42
    assert text_row["experiment_id"] == str(config.id)

    image_rows = df.filter(pl.col("model") == "DummyI2T")
    assert image_rows.height > 0
    image_row = image_rows.row(0, named=True)
    assert image_row["initial_prompt"] == "test embedding dataframe"
    assert image_row["model"] == "DummyI2T"
    assert image_row["sequence_number"] == 1
    assert image_row["seed"] == 42  # Same seed used for all runs in the config
    assert image_row["experiment_id"] == str(config.id)


def test_load_runs_df(db_session):
    """Test that load_runs_df returns a polars DataFrame with correct data."""

    # Create a simple test configuration
    config = ExperimentConfig(
        networks=[["DummyT2I", "DummyI2T"]],
        seeds=[42],
        prompts=["test runs dataframe"],
        embedding_models=["Dummy"],
        max_length=2,
    )

    # Save config to database to get an ID
    db_session.add(config)
    db_session.commit()
    db_session.refresh(config)

    # Run the experiment to populate database
    db_url = str(db_session.get_bind().engine.url)
    perform_experiment(str(config.id), db_url)

    # Call function under test
    df = load_runs_df(db_session, use_cache=False)

    # Assertions
    assert isinstance(df, pl.DataFrame)
    assert df.height > 0  # Should have at least one row for birth/death pairs

    # Check column names
    expected_columns = [
        "run_id",
        "experiment_id",
        "network",
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
        "entropy"
    ]
    assert all(col in df.columns for col in expected_columns)

    # Check experiment_id is correctly stored
    assert df.filter(pl.col("experiment_id") == str(config.id)).height > 0

    # Verify field values that are the same for all features
    first_row = df.row(0, named=True)
    assert first_row["initial_prompt"] == "test runs dataframe"
    assert first_row["seed"] == 42
    assert first_row["network"] == ["DummyT2I", "DummyI2T"]
    assert first_row["max_length"] == 2
    assert first_row["num_invocations"] == 2
    assert first_row["stop_reason"] == "length"
    assert first_row["experiment_id"] == str(config.id)

    # Verify persistence diagram related fields
    assert first_row["embedding_model"] == "Dummy"
    assert first_row["persistence_diagram_id"] is not None

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
