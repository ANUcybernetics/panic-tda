import polars as pl

from trajectory_tracer.analysis import load_embeddings_df, load_persistence_diagram_df
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
        run_length=2
    )

    # Run the experiment to populate database
    perform_experiment(config, db_session)

    # Call function under test
    df = load_embeddings_df(db_session)

    # Assertions
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 4  # 2 invocations * 2 embedding models

    # Check column names
    expected_columns = ['id', 'invocation_id', 'embedding_started_at', 'embedding_completed_at',
                        'invocation_started_at', 'invocation_completed_at', 'duration', 'run_id',
                        'type', 'initial_prompt', 'seed', 'model', 'sequence_number', 'embedding_model']
    assert all(col in df.columns for col in expected_columns)

    # Check values
    assert df.filter(pl.col('embedding_model') == 'Dummy').height == 2
    assert df.filter(pl.col('embedding_model') == 'Dummy2').height == 2

    # Verify field values
    text_rows = df.filter(pl.col('model') == 'DummyT2I')
    assert text_rows.height > 0
    text_row = text_rows.row(0)
    assert text_row[df.columns.index('initial_prompt')] == "test embedding dataframe"
    assert text_row[df.columns.index('model')] == "DummyT2I"
    assert text_row[df.columns.index('sequence_number')] == 0
    assert text_row[df.columns.index('seed')] == 42

    image_rows = df.filter(pl.col('model') == 'DummyI2T')
    assert image_rows.height > 0
    image_row = image_rows.row(0)
    assert image_row[df.columns.index('initial_prompt')] == "test embedding dataframe"
    assert image_row[df.columns.index('model')] == "DummyI2T"
    assert image_row[df.columns.index('sequence_number')] == 1
    assert image_row[df.columns.index('seed')] == 42  # Same seed used for all runs in the config


def test_load_persistence_diagram_df(db_session):
    """Test that load_persistence_diagram_df returns a polars DataFrame with correct data."""

    # Create a simple test configuration
    config = ExperimentConfig(
        networks=[["DummyT2I", "DummyI2T"]],
        seeds=[42],
        prompts=["test persistence diagram dataframe"],
        embedding_models=["Dummy"],
        run_length=2
    )

    # Run the experiment to populate database
    perform_experiment(config, db_session)

    # Call function under test
    df = load_persistence_diagram_df(db_session)

    # Assertions
    assert isinstance(df, pl.DataFrame)
    assert df.height == 1  # We created one persistence diagram

    # Check column names
    expected_columns = ['id', 'run_id', 'started_at', 'completed_at', 'embedding_model',
                        'num_generators', 'network', 'initial_prompt', 'seed']
    assert all(col in df.columns for col in expected_columns)

    # Verify field values
    row = df.row(0)
    assert row[df.columns.index('embedding_model')] == "Dummy"
    assert row[df.columns.index('initial_prompt')] == "test persistence diagram dataframe"
    assert row[df.columns.index('seed')] == 42
    assert row[df.columns.index('network')] == ["DummyT2I", "DummyI2T"]
    assert isinstance(row[df.columns.index('num_generators')], int)
