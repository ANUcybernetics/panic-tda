import polars as pl
from sqlmodel import Session

from trajectory_tracer.db import list_embeddings


def load_embeddings_df(session: Session) -> pl.DataFrame:
    """
    Load all embeddings from the database and flatten them into a polars DataFrame.

    Args:
        session: SQLModel database session

    Returns:
        A polars DataFrame containing all embedding data
    """
    embeddings = list_embeddings(session)

    # Convert the embeddings to a format suitable for a DataFrame
    data = []
    for embedding in embeddings:
        invocation = embedding.invocation
        row = {
            'id': embedding.id,
            'invocation_id': invocation.id,
            'run_id': invocation.run_id,
            'type': invocation.type,
            'content': invocation.output,
            'seed': invocation.run.seed,
            'model': invocation.model,
            'sequence_number': invocation.sequence_number,
            'embedding_model': embedding.embedding_model,
        }
        data.append(row)

    # Create a polars DataFrame
    return pl.DataFrame(data)
