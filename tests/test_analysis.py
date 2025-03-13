import numpy as np
import polars as pl

from trajectory_tracer.analysis import load_embeddings_df
from trajectory_tracer.schemas import Embedding, Invocation, InvocationType, Run


def test_load_embeddings_df(db_session):
    """Test that load_embeddings_df returns a polars DataFrame with correct data."""
    # Create a sample run
    sample_run = Run(
        initial_prompt="test embedding dataframe",
        network=["model1"],
        seed=42,
        length=2
    )
    db_session.add(sample_run)
    db_session.flush()

    # Create invocations
    invocation1 = Invocation(
        model="TextModel",
        type=InvocationType.TEXT,
        seed=42,
        run_id=sample_run.id,
        sequence_number=1,
        output_text="First output text"
    )

    invocation2 = Invocation(
        model="ImageModel",
        type=InvocationType.TEXT,
        seed=43,
        run_id=sample_run.id,
        sequence_number=2,
        output_text="Second output text"
    )

    db_session.add(invocation1)
    db_session.add(invocation2)
    db_session.flush()

    # Create embeddings with different models
    embedding1 = Embedding(
        invocation_id=invocation1.id,
        embedding_model="model-A"
    )
    embedding1.vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    embedding2 = Embedding(
        invocation_id=invocation2.id,
        embedding_model="model-B"
    )
    embedding2.vector = np.array([0.4, 0.5, 0.6, 0.7], dtype=np.float32)

    db_session.add(embedding1)
    db_session.add(embedding2)
    db_session.commit()

    # Call function under test
    df = load_embeddings_df(db_session)

    # Assertions
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 2

    # Check column names
    expected_columns = ['id', 'invocation_id', 'run_id', 'type',
                        'content', 'seed', 'model', 'sequence_number', 'embedding_model']
    assert all(col in df.columns for col in expected_columns)

    # Check values
    assert df.filter(pl.col('embedding_model') == 'model-A').height == 1
    assert df.filter(pl.col('embedding_model') == 'model-B').height == 1

    # Verify content and other fields were correctly extracted
    text_row = df.filter(pl.col('model') == 'TextModel').row(0)
    assert text_row[4] == "First output text"  # content field
    assert text_row[6] == "TextModel"  # model field
    assert text_row[7] == 1  # sequence_number field

    image_row = df.filter(pl.col('model') == 'ImageModel').row(0)
    assert image_row[4] == "Second output text"  # content field
    assert image_row[6] == "ImageModel"  # model field
    assert image_row[7] == 2  # sequence_number field
