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

    # Verify field values using named columns instead of indices
    text_rows = df.filter(pl.col("model") == "DummyT2I")
    assert text_rows.height > 0
    text_row = text_rows.row(0, named=True)
    assert text_row["initial_prompt"] == "test embedding dataframe"
    assert text_row["model"] == "DummyT2I"
    assert text_row["sequence_number"] == 0
    assert text_row["seed"] == 42

    image_rows = df.filter(pl.col("model") == "DummyI2T")
    assert image_rows.height > 0
    image_row = image_rows.row(0, named=True)
    assert image_row["initial_prompt"] == "test embedding dataframe"
    assert image_row["model"] == "DummyI2T"
    assert image_row["sequence_number"] == 1
    assert image_row["seed"] == 42  # Same seed used for all runs in the config


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
    assert df.height == 3  # 1 run with 3 homology dimensions (0, 1, 2)

    # Check column names
    expected_columns = [
        "run_id",
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
        "entropy",
        "feature_count",
        "generator_count"
    ]
    assert all(col in df.columns for col in expected_columns)

    # Verify field values that are the same for all homology dimensions
    dim0_row = df.filter(pl.col("homology_dimension") == 0).row(0, named=True)
    assert dim0_row["initial_prompt"] == "test runs dataframe"
    assert dim0_row["seed"] == 42
    assert dim0_row["network"] == ["DummyT2I", "DummyI2T"]
    assert dim0_row["max_length"] == 2
    assert dim0_row["num_invocations"] == 2
    assert dim0_row["stop_reason"] == "length"

    # Verify persistence diagram related fields
    assert dim0_row["embedding_model"] == "Dummy"
    assert dim0_row["persistence_diagram_id"] is not None

    # Verify each dimension has the expected fields with proper types
    for dim in range(3):
        dim_row = df.filter(pl.col("homology_dimension") == dim).row(0, named=True)
        assert dim_row["homology_dimension"] == dim
        assert isinstance(dim_row["feature_count"], int)
        assert isinstance(dim_row["entropy"], float)
        assert isinstance(dim_row["generator_count"], int)
