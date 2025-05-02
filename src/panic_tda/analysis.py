import os

import numpy as np
import polars as pl
from numpy.linalg import norm
from sqlmodel import Session

from panic_tda.clustering import hdbscan
from panic_tda.db import list_runs
from panic_tda.genai_models import get_output_type
from panic_tda.schemas import InvocationType, NumpyArrayType


def format_uuid_columns(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """
    Format UUID columns in a polars DataFrame to standard UUID format with hyphens.

    Args:
        df: DataFrame containing UUID columns
        columns: List of column names containing UUIDs to format

    Returns:
        DataFrame with formatted UUID columns
    """
    return df.with_columns([
        pl.col(columns)
        .cast(pl.String)
        .str.slice(0, 8) + "-" +
        pl.col(columns)
        .cast(pl.String)
        .str.slice(8, 4) + "-" +
        pl.col(columns)
        .cast(pl.String)
        .str.slice(12, 4) + "-" +
        pl.col(columns)
        .cast(pl.String)
        .str.slice(16, 4) + "-" +
        pl.col(columns)
        .cast(pl.String)
        .str.slice(20, None)
    ])


def _get_polars_db_uri(session: Session) -> str:
    """
    Get a database URI suitable for pl.read_database_uri.

    Handles the case where a relative path is given for a file-based SQLite
    database, converting it to an absolute path required by the underlying
    connector (e.g., ADBC, ConnectorX).

    Args:
        session: SQLModel database session.

    Returns:
        A database URI string compatible with pl.read_database_uri.
    """
    engine = session.get_bind().engine
    url = engine.url

    # Check if it's a file-based SQLite database
    if url.drivername == "sqlite" and url.database and url.database != ":memory:":
        # Convert relative path to absolute path for Polars
        absolute_path = os.path.abspath(url.database)
        # Polars (or its connector) expects 'sqlite:///<absolute_path>'
        return f"sqlite:///{absolute_path}"
    else:
        # For in-memory SQLite or other databases, the standard URL is fine
        return str(url)


def load_invocations_from_cache() -> pl.DataFrame:
    """
    Load invocations from the cache file.

    Returns:
        A polars DataFrame containing all invocation data from cache
    """
    cache_path = "output/cache/invocations.parquet"
    print(f"Loading invocations from cache: {cache_path}")
    return pl.read_parquet(cache_path)


def load_invocations_df(session: Session) -> pl.DataFrame:
    """
    Load all invocations from the database into a tidy polars DataFrame.

    Args:
        session: SQLModel database session

    Returns:
        A polars DataFrame containing all invocation data
    """
    print("Loading invocations from database...")

    # SQL query to join invocations with runs to get the required data
    query = """
    SELECT
        invocation.id as id,
        invocation.run_id as run_id,
        run.experiment_id as experiment_id,
        invocation.model as model,
        invocation.type as type,
        invocation.sequence_number as sequence_number,
        invocation.started_at as started_at,
        invocation.completed_at as completed_at,
        (invocation.completed_at - invocation.started_at) as duration,
        run.initial_prompt as initial_prompt,
        run.seed as seed
    FROM invocation
    JOIN run ON invocation.run_id = run.id
    """

    # Use polars to read directly from the database, getting the correct URI format
    db_url = _get_polars_db_uri(session)
    df = pl.read_database_uri(query=query, uri=db_url)

    # Format UUID columns using the dedicated function
    df = format_uuid_columns(df, ["id", "run_id", "experiment_id"])

    return df


def load_embeddings_from_cache() -> pl.DataFrame:
    """
    Load embeddings from the cache file.

    Returns:
        A polars DataFrame containing all embedding data from cache
    """
    cache_path = "output/cache/embeddings.parquet"
    print(f"Loading embeddings from cache: {cache_path}")
    return pl.read_parquet(cache_path)


def load_embeddings_df(session: Session) -> pl.DataFrame:
    """
    Load all embeddings from the database into a tidy polars DataFrame.
    Only includes embeddings for text invocations.

    Args:
        session: SQLModel database session

    Returns:
        A polars DataFrame containing all embedding data for text invocations
    """
    print("Loading embeddings from database...")

    # SQL query to join embeddings with invocations and runs
    query = """
    SELECT
        embedding.id AS id,
        embedding.invocation_id AS invocation_id,
        embedding.embedding_model AS embedding_model,
        embedding.started_at AS started_at,
        embedding.completed_at AS completed_at,
        embedding.vector AS vector,
        invocation.run_id AS run_id,
        invocation.sequence_number AS sequence_number,
        invocation.model AS model,
        run.initial_prompt AS initial_prompt
    FROM embedding
    JOIN invocation ON embedding.invocation_id = invocation.id
    JOIN run ON invocation.run_id = run.id
    WHERE invocation.type = 'TEXT'
    ORDER BY run_id, embedding_model, sequence_number
    """

    # Use polars to read directly from the database, getting the correct URI format
    db_url = _get_polars_db_uri(session)
    df = pl.read_database_uri(query=query, uri=db_url)

    # Format UUID columns
    df = format_uuid_columns(df, ["id", "invocation_id", "run_id"])

    # Process the vector column to ensure proper hydration using NumpyArrayType
    # Convert the serialized vector data back to numpy arrays
    numpy_type = NumpyArrayType()

    # Use map_elements to convert each vector value using the NumpyArrayType processor
    df = df.with_columns([
        pl.col("vector").map_elements(
            lambda x: numpy_type.process_result_value(x, None),
            return_dtype=pl.Object
        )
    ])

    return df


def add_cluster_labels(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add cluster labels to the embeddings DataFrame.

    Args:
        df: DataFrame containing embedding data with vectors

    Returns:
        DataFrame with cluster labels added
    """
    print("Starting clustering process...")

    # Create an empty DataFrame to store cluster results
    clusters_df = None

    # Process each embedding model using group_by aggregation
    cluster_results = (
        df.group_by("embedding_model")
        .map_groups(lambda group_df: process_embedding_model(group_df))
    )

    # Join the cluster results with the original DataFrame
    print("Joining cluster labels with main DataFrame...")
    result_df = df.join(cluster_results, on="id", how="left")
    print(f"Clustering complete. DataFrame now has {result_df.shape[0]} rows and {result_df.shape[1]} columns.")

    return result_df


def process_embedding_model(model_df: pl.DataFrame) -> pl.DataFrame:
    """Helper function to process a single embedding model group"""
    embedding_model = model_df["embedding_model"][0]
    print(f"Clustering model: {embedding_model} with {model_df.shape[0]} embeddings")

    # Extract the vector from each embedding to create the ndarray
    # We access the raw vectors directly from the dataframe
    embeddings = np.array([embedding for embedding in model_df["vector"]])

    # Get cluster labels - they are returned in the same order as the input embeddings
    cluster_labels = hdbscan(embeddings)

    # Count unique clusters
    unique_labels = set(cluster_labels)
    print(f"  Found {len(unique_labels)} clusters (including noise)")

    # Create a new dataframe with id and cluster label
    return pl.DataFrame({
        "id": model_df["id"],
        "cluster_label": cluster_labels
    })


def calculate_cosine_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine distance between two vectors."""
    norm_vec1 = norm(vec1)
    norm_vec2 = norm(vec2)

    # Avoid division by zero
    if norm_vec1 > 0 and norm_vec2 > 0:
        cosine_similarity = np.dot(vec1, vec2) / (norm_vec1 * norm_vec2)
        return float(1.0 - cosine_similarity)
    else:
        return 0.0 if np.array_equal(vec1, vec2) else 1.0


def load_runs_from_cache() -> pl.DataFrame:
    """
    Load runs from the cache file.

    Returns:
        A polars DataFrame containing basic run data from cache
    """
    cache_path = "output/cache/runs.parquet"
    print(f"Loading runs from cache: {cache_path}")
    return pl.read_parquet(cache_path)


def load_runs_df(session: Session) -> pl.DataFrame:
    """
    Load all runs from the database into a tidy polars DataFrame.
    Basic run information without persistence diagrams.

    Args:
        session: SQLModel database session

    Returns:
        A polars DataFrame containing basic run data
    """
    print("Loading runs from database...")

    # SQL query to get run data
    query = """
    SELECT
        run.id as run_id,
        run.experiment_id as experiment_id,
        run.network as network,
        run.initial_prompt as initial_prompt,
        run.seed as seed,
        run.max_length as max_length,
        (SELECT COUNT(*) FROM invocation WHERE invocation.run_id = run.id) as num_invocations
    FROM run
    WHERE EXISTS (SELECT 1 FROM invocation WHERE invocation.run_id = run.id)
    """

    # Use polars to read directly from the database, getting the correct URI format
    db_url = _get_polars_db_uri(session)
    df = pl.read_database_uri(query=query, uri=db_url)

    # Format UUID columns
    df = format_uuid_columns(df, ["run_id", "experiment_id"])

    # Parse network from JSON string to List[str]
    df = df.with_columns([
        pl.col("network").str.json_decode().alias("network")
    ])

    # Extract image_model and text_model from network
    def extract_models(network):
        image_model = None
        text_model = None
        for model in network:
            output_type = get_output_type(model)
            if output_type == InvocationType.IMAGE and image_model is None:
                image_model = model
            elif output_type == InvocationType.TEXT and text_model is None:
                text_model = model
            if image_model is not None and text_model is not None:
                break
        return pl.Series([image_model, text_model])

    df = df.with_columns([
        pl.col("network").map_elements(extract_models, return_dtype=pl.List(pl.String)).alias("models")
    ])

    df = df.with_columns([
        pl.col("models").list.get(0).alias("image_model"),
        pl.col("models").list.get(1).alias("text_model")
    ]).drop("models")

    return df


def add_persistence_entropy(df: pl.DataFrame, session: Session) -> pl.DataFrame:
    """
    Add persistence diagram information to a runs DataFrame.

    Args:
        df: DataFrame containing run data
        session: SQLModel database session

    Returns:
        DataFrame with persistence diagram data added
    """
    print("Adding persistence diagram data to runs DataFrame...")

    # Load all runs to get their PDs
    runs = list_runs(session)
    run_map = {str(run.id): run for run in runs}

    data = []
    for run_id in df["run_id"].unique().to_list():
        run = run_map.get(run_id)
        if not run or not run.persistence_diagrams:
            continue

        # Base run information
        base_row = {
            "run_id": run_id,
        }

        for pd in run.persistence_diagrams:
            row = base_row.copy()
            row["persistence_diagram_id"] = str(pd.id)
            row["embedding_model"] = pd.embedding_model
            row["persistence_diagram_started_at"] = pd.started_at
            row["persistence_diagram_completed_at"] = pd.completed_at
            row["persistence_diagram_duration"] = pd.duration

            # Only include persistence diagrams with diagram_data
            if pd.diagram_data and "dgms" in pd.diagram_data:
                # Process each dimension in the diagram data
                for dim, dgm in enumerate(pd.diagram_data["dgms"]):
                    # Add entropy for this dimension if available
                    entropy_value = None
                    if "entropy" in pd.diagram_data and dim < len(
                        pd.diagram_data["entropy"]
                    ):
                        entropy_value = float(pd.diagram_data["entropy"][dim])

                    # Create a row for each birth/death pair in this dimension
                    for i, (birth, death) in enumerate(dgm):
                        feature_row = row.copy()
                        feature_row["homology_dimension"] = dim
                        feature_row["feature_id"] = i
                        feature_row["birth"] = float(birth)
                        feature_row["death"] = float(death)
                        feature_row["persistence"] = float(death - birth)

                        # Add entropy for the dimension
                        if entropy_value is not None:
                            feature_row["entropy"] = entropy_value

                        data.append(feature_row)

    # Create a polars DataFrame with explicit schema for numeric fields
    schema_overrides = {
        "homology_dimension": pl.Int64,
        "feature_id": pl.Int64,
        "birth": pl.Float64,
        "death": pl.Float64,
        "persistence": pl.Float64,
        "entropy": pl.Float64,
    }

    # Only create DataFrame if we have data
    if data:
        pd_df = pl.DataFrame(data, schema_overrides=schema_overrides)
        # Join with the original runs DataFrame
        result_df = df.join(pd_df, on="run_id", how="left")
    else:
        # If no PD data, return the original df to avoid join errors
        # Optionally add empty columns if needed downstream
        result_df = df.with_columns([
            pl.lit(None).cast(pl.String).alias("persistence_diagram_id"),
            pl.lit(None).cast(pl.String).alias("embedding_model"),
            pl.lit(None).cast(pl.Datetime).alias("persistence_diagram_started_at"),
            pl.lit(None).cast(pl.Datetime).alias("persistence_diagram_completed_at"),
            pl.lit(None).cast(pl.Duration).alias("persistence_diagram_duration"),
            pl.lit(None).cast(pl.Int64).alias("homology_dimension"),
            pl.lit(None).cast(pl.Int64).alias("feature_id"),
            pl.lit(None).cast(pl.Float64).alias("birth"),
            pl.lit(None).cast(pl.Float64).alias("death"),
            pl.lit(None).cast(pl.Float64).alias("persistence"),
            pl.lit(None).cast(pl.Float64).alias("entropy"),
        ])


    return result_df


def cache_dfs(
    session: Session,
    runs: bool = True,
    embeddings: bool = True,
    invocations: bool = True,
) -> None:
    """
    Preload and cache dataframes.

    Args:
        session: SQLModel database session
        runs: Whether to cache runs dataframe
        embeddings: Whether to cache embeddings dataframe
        invocations: Whether to cache invocations dataframe

    Returns:
        None
    """
    os.makedirs("output/cache", exist_ok=True) # Ensure cache directory exists

    if runs:
        print("Warming cache for runs dataframe...")
        cache_path = "output/cache/runs.parquet"
        # Remove existing cache file if it exists
        if os.path.exists(cache_path):
            os.remove(cache_path)
        runs_df = load_runs_df(session)
        # Save to cache
        runs_df.write_parquet(cache_path)
        print(f"Saved runs to cache: {cache_path}")

    if embeddings:
        print("Warming cache for embeddings dataframe...")
        cache_path = "output/cache/embeddings.parquet"
        # Remove existing cache file if it exists
        if os.path.exists(cache_path):
            os.remove(cache_path)
        embeddings_df = load_embeddings_df(session)
        # Save to cache
        embeddings_df.write_parquet(cache_path)
        print(f"Saved embeddings to cache: {cache_path}")

    if invocations:
        print("Warming cache for invocations dataframe...")
        cache_path = "output/cache/invocations.parquet"
        # Remove existing cache file if it exists
        if os.path.exists(cache_path):
            os.remove(cache_path)
        invocations_df = load_invocations_df(session)
        # Save to cache
        invocations_df.write_parquet(cache_path)
        print(f"Saved invocations to cache: {cache_path}")

    print("Cache warming complete.")
