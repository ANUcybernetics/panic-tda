import os

import numpy as np
import polars as pl
from numpy.linalg import norm
from sqlmodel import Session

from panic_tda.clustering import hdbscan
from panic_tda.db import list_runs
from panic_tda.genai_models import get_output_type
from panic_tda.schemas import InvocationType


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
        .cast(pl.Utf8)
        .str.slice(0, 8) + "-" +
        pl.col(columns)
        .cast(pl.Utf8)
        .str.slice(8, 4) + "-" +
        pl.col(columns)
        .cast(pl.Utf8)
        .str.slice(12, 4) + "-" +
        pl.col(columns)
        .cast(pl.Utf8)
        .str.slice(16, 4) + "-" +
        pl.col(columns)
        .cast(pl.Utf8)
        .str.slice(20, None)
    ])


def load_invocations_df(session: Session) -> pl.DataFrame:
    """
    Load all invocations from the database into a tidy polars DataFrame.

    Args:
        session: SQLModel database session

    Returns:
        A polars DataFrame containing all invocation data
    """
    cache_path = "output/cache/invocations.parquet"

    # Check if cache exists
    if os.path.exists(cache_path):
        print(f"Loading invocations from cache: {cache_path}")
        return pl.read_parquet(cache_path)

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

    # Use polars to read directly from the database
    db_url = str(session.get_bind().engine.url)
    df = pl.read_database_uri(query=query, uri=db_url)

    # Format UUID columns using the dedicated function
    df = format_uuid_columns(df, ["id", "run_id", "experiment_id"])

    return df


def load_embeddings_df(session: Session) -> pl.DataFrame:
    """
    Load all embeddings from the database into a tidy polars DataFrame.
    Only includes embeddings for text invocations.

    Args:
        session: SQLModel database session

    Returns:
        A polars DataFrame containing all embedding data for text invocations
    """
    cache_path = "output/cache/embeddings.parquet"

    # Check if cache exists
    if os.path.exists(cache_path):
        print(f"Loading embeddings from cache: {cache_path}")
        return pl.read_parquet(cache_path)

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

    # Use polars to read directly from the database
    db_url = str(session.get_bind().engine.url)
    df = pl.read_database_uri(query=query, uri=db_url)

    # Format UUID columns
    df = format_uuid_columns(df, ["id", "invocation_id", "run_id"])

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

    # Group embeddings by model
    embedding_models = df["embedding_model"].unique().to_list()

    all_clusters = []

    # Process each embedding model separately
    for i, embedding_model in enumerate(embedding_models):
        model_df = df.filter(pl.col("embedding_model") == embedding_model)
        embeddings_list = model_df.select(["id", "vector"]).to_dicts()

        print(f"Clustering model {i+1}/{len(embedding_models)}: {embedding_model} with {len(embeddings_list)} embeddings")

        # Get cluster labels using hdbscan
        embeddings_objects = [
            type('obj', (object,), {'id': e['id'], 'vector': e['vector']})
            for e in embeddings_list
        ]

        cluster_labels = hdbscan(embeddings_objects)

        # Count the number of points in each cluster
        unique_labels = set(cluster_labels)
        print(f"  Found {len(unique_labels)} clusters (including noise)")

        # Create a list of dictionaries for the cluster results
        model_clusters = [
            {"id": embedding_obj.id, "cluster_label": label}
            for embedding_obj, label in zip(embeddings_objects, cluster_labels)
        ]

        # Add to clustering results
        all_clusters.extend(model_clusters)
        print(f"  Added {len(model_clusters)} labeled embeddings to results")

    # Convert clustering results to a polars DataFrame
    print("Creating clusters DataFrame...")
    clusters_df = pl.DataFrame(all_clusters)

    # Join the main DataFrame with the clusters DataFrame
    print("Joining cluster labels with main DataFrame...")
    result_df = df.join(clusters_df, on="id", how="left")
    print(f"Clustering complete. DataFrame now has {result_df.shape[0]} rows and {result_df.shape[1]} columns.")

    return result_df

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


def load_runs_df(session: Session) -> pl.DataFrame:
    """
    Load all runs from the database and flatten them into a polars DataFrame.
    Includes persistence diagrams with birth/death pairs for each run.

    Args:
        session: SQLModel database session

    Returns:
        A polars DataFrame containing all run data with persistence diagrams
    """
    cache_path = "output/cache/runs.parquet"

    # Check if cache exists
    if os.path.exists(cache_path):
        print(f"Loading runs from cache: {cache_path}")
        return pl.read_parquet(cache_path)

    print("Loading runs from database...")
    runs = list_runs(session)
    # Print the number of runs
    print(f"Number of runs: {len(runs)}")

    data = []
    for run in runs:
        # temporary hack to only look at 1k runs for SMC paper
        if run.max_length > 1000:
            continue

        # Skip runs with no invocations
        if not run.invocations:
            continue

        # Extract image_model and text_model from network
        image_model = None
        text_model = None
        for model in run.network:
            output_type = get_output_type(model)
            if output_type == InvocationType.IMAGE and image_model is None:
                image_model = model
            elif output_type == InvocationType.TEXT and text_model is None:
                text_model = model

            # If both models have been assigned, we can stop iterating
            if image_model is not None and text_model is not None:
                break

        # Base run information
        base_row = {
            "run_id": str(run.id),
            "experiment_id": str(run.experiment_id),
            "network": run.network,
            "image_model": image_model,
            "text_model": text_model,
            "initial_prompt": run.initial_prompt,
            "seed": run.seed,
            "max_length": run.max_length,
            "num_invocations": len(run.invocations),
        }

        # Only include runs with persistence diagrams
        if run.persistence_diagrams:
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
    df = pl.DataFrame(data, schema_overrides=schema_overrides)

    return df


def warm_caches(
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
    if runs:
        print("Warming cache for runs dataframe...")
        cache_path = "output/cache/runs.parquet"
        # Remove existing cache file if it exists
        if os.path.exists(cache_path):
            os.remove(cache_path)
        runs_df = load_runs_df(session)
        # Save to cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
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
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
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
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        invocations_df.write_parquet(cache_path)
        print(f"Saved invocations to cache: {cache_path}")

    print("Cache warming complete.")
