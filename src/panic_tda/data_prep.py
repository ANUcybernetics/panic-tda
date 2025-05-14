import os
import time
from typing import Dict, Tuple
from uuid import UUID

import numpy as np
import polars as pl
import ray
from sqlmodel import Session

from panic_tda.clustering import hdbscan
from panic_tda.db import list_runs, read_embedding
from panic_tda.embeddings import get_actor_class
from panic_tda.genai_models import get_output_type
from panic_tda.local import prompt_category_mapper
from panic_tda.schemas import InvocationType

# because I like to see the data
pl.Config.set_tbl_rows(1000)


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
        pl.col(columns).cast(pl.String).str.slice(0, 8)
        + "-"
        + pl.col(columns).cast(pl.String).str.slice(8, 4)
        + "-"
        + pl.col(columns).cast(pl.String).str.slice(12, 4)
        + "-"
        + pl.col(columns).cast(pl.String).str.slice(16, 4)
        + "-"
        + pl.col(columns).cast(pl.String).str.slice(20, None)
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


def load_embeddings_from_cache() -> pl.DataFrame:
    """
    Load embeddings from the cache file.

    Returns:
        A polars DataFrame containing all embedding metadata from cache (without vector data)
    """
    cache_path = "output/cache/embeddings.parquet"
    print(f"Loading embeddings from cache: {cache_path}")
    return pl.read_parquet(cache_path)


def fetch_and_cluster_vectors(embedding_ids: pl.Series, session: Session) -> pl.Series:
    """
    Fetch embedding vectors from the database, stack them, and perform clustering.

    Args:
        embedding_ids: A Series containing embedding IDs
        session: SQLModel database session

    Returns:
        A Series of cluster representative texts corresponding to the input embedding IDs
    """
    # Check if we have any embeddings to process
    if len(embedding_ids) == 0:
        return pl.Series([])

    # Fetch embeddings and build a cache of vector -> text
    embeddings = [
        read_embedding(UUID(embedding_id), session) for embedding_id in embedding_ids
    ]
    vectors = [embedding.vector for embedding in embeddings]

    # Create a cache mapping vectors to their corresponding text
    vector_to_text = {}
    for embedding in embeddings:
        # Convert vector to a hashable tuple for dictionary key
        vector_key = tuple(embedding.vector.flatten())
        if embedding.invocation and embedding.invocation.output_text:
            vector_to_text[vector_key] = embedding.invocation.output_text

    # Perform clustering on the vectors
    vectors_array = np.vstack(vectors)
    cluster_result = hdbscan(vectors_array)

    # Create a mapping from label to representative text
    unique_labels = sorted(set(cluster_result["labels"]))
    label_to_text = {}

    for label in unique_labels:
        if label == -1:
            label_to_text[label] = "OUTLIER"
        else:
            # Get the medoid vector for this cluster
            medoid_vector = cluster_result["medoids"][label]

            # Look up the text in our cache
            medoid_key = tuple(medoid_vector.flatten())
            label_to_text[label] = vector_to_text[medoid_key]

    # Create result array with cluster labels
    result = [label_to_text[label] for label in cluster_result["labels"]]

    return pl.Series(result)


def add_cluster_labels(
    df: pl.DataFrame, downsample: int, session: Session
) -> pl.DataFrame:
    """
    Add cluster representative texts to the embeddings DataFrame by fetching vectors on demand.

    Args:
        df: DataFrame containing embedding metadata (without vectors)
        downsample: If > 1, only process every nth embedding (e.g., 2 means every other embedding)
        session: SQLModel database session for fetching vectors

    Returns:
        DataFrame with cluster representative texts added
    """
    # Create a version of the dataframe with only rows to process (downsampled)
    df_downsampled = (
        df.with_row_index("row_idx")
        .filter(pl.col("row_idx") % downsample == 0)
        .drop("row_idx")
    )

    # Process each embedding model group
    cluster_data = []

    for model_name in df_downsampled["embedding_model"].unique():
        model_rows = df_downsampled.filter(pl.col("embedding_model") == model_name)
        embedding_ids = model_rows["id"]

        # Get cluster labels for this model's embeddings
        cluster_labels = fetch_and_cluster_vectors(embedding_ids, session)

        # Create mapping of ID to cluster label
        model_cluster_data = {"id": embedding_ids, "cluster_label": cluster_labels}

        cluster_data.append(pl.DataFrame(model_cluster_data))

    clusters_df = pl.concat(cluster_data)
    # Join the clusters back to the original dataframe
    result_df = df.join(clusters_df, on="id", how="left")

    return result_df


def filter_top_n_clusters(
    df: pl.DataFrame, n: int, group_by_cols: list[str]
) -> pl.DataFrame:
    """
    Filter the embeddings DataFrame to keep only the top n clusters within each
    group defined by group_by_cols.

    Args:
        df: DataFrame containing embeddings with cluster labels
        n: Number of top clusters to keep within each group
        group_by_cols: List of column names to group by (default: ["embedding_model", "initial_prompt"])

    Returns:
        Filtered DataFrame containing only rows from the top n clusters
    """
    # Count occurrences of each cluster within each group
    group_cols = group_by_cols + ["cluster_label"]
    cluster_counts = df.group_by(group_cols).agg(pl.len().alias("count"))

    # Assign rank within each group based on count
    cluster_ranks = cluster_counts.with_columns([
        pl.col("count")
        .rank(method="dense", descending=True)
        .over(group_by_cols)
        .alias("rank")
    ])

    # Filter to keep only the top n clusters
    top_clusters = cluster_ranks.filter(pl.col("rank") <= n)

    # Join with original dataframe to keep only rows from top clusters
    result = df.join(top_clusters.select(group_cols), on=group_cols, how="inner")

    return result


def embed_initial_prompts(session: Session) -> Dict[Tuple[str, str], np.ndarray]:
    """
    Generate embeddings for all unique combinations of initial prompts and embedding models.

    Args:
        session: SQLModel database session

    Returns:
        Dictionary mapping (initial_prompt, embedding_model) tuples to embedding vectors
    """

    # SQL query to get all unique combinations of initial_prompt and embedding_model
    query = """
    SELECT DISTINCT
        run.initial_prompt as initial_prompt,
        embedding.embedding_model as embedding_model
    FROM run
    JOIN invocation ON invocation.run_id = run.id
    JOIN embedding ON embedding.invocation_id = invocation.id
    WHERE run.initial_prompt IS NOT NULL
    """

    # Get the database URI and execute the query
    db_url = _get_polars_db_uri(session)
    df = pl.read_database_uri(query=query, uri=db_url)

    # Create dictionaries to store embedding models and tasks
    embedding_models = {}
    tasks = {}

    # TODO this could be accelerated by grouping by embedding model and using the fact that the embedding infrastructure
    # already works in batch mode (pass a list of str, get a list of embeddings back)
    #
    # Process each combination
    for row in df.iter_rows(named=True):
        initial_prompt = row["initial_prompt"]
        embedding_model_name = row["embedding_model"]
        key = (initial_prompt, embedding_model_name)

        # Get or create the remote embedding model
        if embedding_model_name not in embedding_models:
            model_class = get_actor_class(embedding_model_name)
            embedding_models[embedding_model_name] = model_class.remote()

        # Start the embedding task
        embedding_model = embedding_models[embedding_model_name]
        tasks[key] = embedding_model.embed.remote([initial_prompt])

    # Wait for all tasks to complete and store the results
    embeddings_dict = {key: ray.get(task)[0] for key, task in tasks.items()}

    return embeddings_dict


def calculate_euclidean_distances(
    vectors_array: np.ndarray, reference_vector: np.ndarray
) -> np.ndarray:
    """
    Calculate Euclidean distances between each vector in an array and a reference vector.

    Args:
        vectors_array: 2D array where each row is a vector
        reference_vector: The reference vector to calculate distances from

    Returns:
        1D array of Euclidean distances
    """
    differences = vectors_array - reference_vector
    distances = np.linalg.norm(differences, axis=1)
    return distances


def fetch_and_calculate_drift_euclid(
    embedding_ids: pl.Series,
    initial_prompt_vectors: Dict[Tuple[str, str], np.ndarray],
    session: Session,
) -> pl.Series:
    """
    Fetch embedding vectors from the database and calculate drift from the embedded initial prompt.

    Args:
        embedding_ids: A Series containing embedding IDs
        initial_prompt_vectors: Dictionary mapping (initial_prompt, embedding_model) tuples to vectors
        session: SQLModel database session

    Returns:
        A Series of euclidean distances between each vector and the embedded initial prompt
    """
    # Get vectors from database
    embeddings = [
        read_embedding(UUID(embedding_id), session) for embedding_id in embedding_ids
    ]
    first_embedding = embeddings[0]
    initial_vector = initial_prompt_vectors[
        (first_embedding.invocation.run.initial_prompt, first_embedding.embedding_model)
    ]

    # Stack the vectors and calculate distances
    vectors = np.vstack([embedding.vector for embedding in embeddings])
    distances = calculate_euclidean_distances(vectors, initial_vector)
    return pl.Series(distances)


def add_semantic_drift_euclid(df: pl.DataFrame, session: Session) -> pl.DataFrame:
    """
    Add semantic drift values to the embeddings DataFrame by fetching vectors on demand
    and calculating drift from the embedded initial prompt.

    Args:
        df: DataFrame containing embedding metadata (without vectors)
        session: SQLModel database session for fetching vectors

    Returns:
        DataFrame with semantic drift values added
    """
    initial_prompt_vectors = embed_initial_prompts(session)

    # Add vectors column using map_elements
    df = df.with_columns(
        pl.col("id")
        .map_batches(
            lambda embedding_ids: fetch_and_calculate_drift_euclid(
                embedding_ids, initial_prompt_vectors, session
            ),
        )
        .alias("drift_euclid")
    )

    return df


def calculate_cosine_distance(
    vectors_array: np.ndarray, reference_vector: np.ndarray
) -> np.ndarray:
    """
    Calculate cosine distances between each vector in an array and a reference vector.

    Args:
        vectors_array: 2D array where each row is a vector
        reference_vector: The reference vector to calculate distances from

    Returns:
        1D array of cosine distances
    """
    # Check for vector equality with the reference vector
    equal_vectors = np.all(vectors_array == reference_vector, axis=1)

    # Calculate norms
    vector_norms = np.linalg.norm(vectors_array, axis=1)
    reference_norm = np.linalg.norm(reference_vector)

    # Calculate dot products with the reference vector
    dot_products = np.sum(vectors_array * reference_vector, axis=1)

    # Calculate cosine similarity directly for all vectors (no need to check for zero norms)
    norm_products = vector_norms * reference_norm
    cosine_similarities = dot_products / norm_products
    distances = 1.0 - cosine_similarities

    # Set distance to 0 for identical vectors
    distances[equal_vectors] = 0.0

    return distances


def fetch_and_calculate_drift_cosine(
    embedding_ids: pl.Series,
    initial_prompt_vectors: Dict[Tuple[str, str], np.ndarray],
    session: Session,
) -> pl.Series:
    """
    Fetch embedding vectors from the database and calculate cosine drift from the embedded initial prompt.

    Args:
        embedding_ids: A Series containing embedding IDs
        initial_prompt_vectors: Dictionary mapping (initial_prompt, embedding_model) tuples to vectors
        session: SQLModel database session

    Returns:
        A Series of cosine distances between each vector and the embedded initial prompt
    """
    # Get vectors from database
    embeddings = [
        read_embedding(UUID(embedding_id), session) for embedding_id in embedding_ids
    ]

    # Get the initial vector once from the first embedding
    first_embedding = embeddings[0]
    initial_vector = initial_prompt_vectors[
        (first_embedding.invocation.run.initial_prompt, first_embedding.embedding_model)
    ]

    # Stack the vectors and calculate distances
    vectors = np.vstack([embedding.vector for embedding in embeddings])
    distances = calculate_cosine_distance(vectors, initial_vector)

    return pl.Series(distances)


def add_semantic_drift_cosine(df: pl.DataFrame, session: Session) -> pl.DataFrame:
    """
    Add semantic drift values using cosine distance to the embeddings DataFrame by fetching
    vectors on demand and calculating drift from the embedded initial prompt.

    Args:
        df: DataFrame containing embedding metadata (without vectors)
        session: SQLModel database session for fetching vectors

    Returns:
        DataFrame with semantic drift values added using cosine distance
    """
    initial_prompt_vectors = embed_initial_prompts(session)

    # Add vectors column using map_elements
    df = df.with_columns(
        pl.col("id")
        .map_batches(
            lambda embedding_ids: fetch_and_calculate_drift_cosine(
                embedding_ids, initial_prompt_vectors, session
            ),
        )
        .alias("drift_cosine")
    )
    return df


def load_runs_from_cache() -> pl.DataFrame:
    """
    Load runs from the cache file.

    Returns:
        A polars DataFrame containing basic run data from cache
    """
    cache_path = "output/cache/runs.parquet"
    print(f"Loading runs from cache: {cache_path}")
    return pl.read_parquet(cache_path)


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

    pd_df = pl.DataFrame(data, schema_overrides=schema_overrides)
    # Join with the original runs DataFrame
    result_df = df.join(pd_df, on="run_id", how="left")

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
    os.makedirs("output/cache", exist_ok=True)  # Ensure cache directory exists

    if runs:
        print("Warming cache for runs dataframe...")
        cache_path = "output/cache/runs.parquet"

        start_time = time.time()
        runs_df = load_runs_df(session)
        df_memory_size = runs_df.estimated_size() / (1024 * 1024)  # Convert to MB

        # Save to cache (automatically overwrites if exists)
        runs_df.write_parquet(cache_path)
        elapsed_time = time.time() - start_time

        cache_file_size = os.path.getsize(cache_path) / (1024 * 1024)  # Convert to MB
        print(
            f"Saved runs ({runs_df.shape[0]} rows, {runs_df.shape[1]} columns) to cache: {cache_path}"
        )
        print(
            f"  Memory size: {df_memory_size:.2f} MB, Cache file size: {cache_file_size:.2f} MB"
        )
        print(f"  Time taken: {elapsed_time:.2f} seconds")

    if embeddings:
        print("Warming cache for embeddings dataframe...")
        cache_path = "output/cache/embeddings.parquet"

        start_time = time.time()
        embeddings_df = load_embeddings_df(session)
        embeddings_df = add_cluster_labels(embeddings_df, 10, session)

        df_memory_size = embeddings_df.estimated_size() / (1024 * 1024)  # Convert to MB

        # Save to cache (automatically overwrites if exists)
        embeddings_df.write_parquet(cache_path)
        elapsed_time = time.time() - start_time

        cache_file_size = os.path.getsize(cache_path) / (1024 * 1024)  # Convert to MB
        print(
            f"Saved embeddings ({embeddings_df.shape[0]} rows, {embeddings_df.shape[1]} columns) to cache: {cache_path}"
        )
        print(
            f"  Memory size: {df_memory_size:.2f} MB, Cache file size: {cache_file_size:.2f} MB"
        )
        print(f"  Time taken: {elapsed_time:.2f} seconds")

    if invocations:
        print("Warming cache for invocations dataframe...")
        cache_path = "output/cache/invocations.parquet"

        start_time = time.time()
        invocations_df = load_invocations_df(session)
        df_memory_size = invocations_df.estimated_size() / (
            1024 * 1024
        )  # Convert to MB

        # Save to cache (automatically overwrites if exists)
        invocations_df.write_parquet(cache_path)
        elapsed_time = time.time() - start_time

        cache_file_size = os.path.getsize(cache_path) / (1024 * 1024)  # Convert to MB
        print(
            f"Saved invocations ({invocations_df.shape[0]} rows, {invocations_df.shape[1]} columns) to cache: {cache_path}"
        )
        print(
            f"  Memory size: {df_memory_size:.2f} MB, Cache file size: {cache_file_size:.2f} MB"
        )
        print(f"  Time taken: {elapsed_time:.2f} seconds")

    print("Cache warming complete.")


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
    WHERE run.initial_prompt NOT IN ('yeah', 'nah')
    """

    # Use polars to read directly from the database, getting the correct URI format
    db_url = _get_polars_db_uri(session)
    df = pl.read_database_uri(query=query, uri=db_url)

    # Format UUID columns using the dedicated function
    df = format_uuid_columns(df, ["id", "run_id", "experiment_id"])

    return df


def load_embeddings_df(session: Session) -> pl.DataFrame:
    """
    Load all embeddings metadata from the database into a tidy polars DataFrame.
    Only includes embeddings for text invocations, excludes the actual vector data.

    Args:
        session: SQLModel database session

    Returns:
        A polars DataFrame containing embedding metadata for text invocations
    """
    print("Loading embeddings metadata from database...")

    # Modified SQL query to exclude the vector data column but include output_text
    query = """
    SELECT
        embedding.id AS id,
        embedding.invocation_id AS invocation_id,
        embedding.embedding_model AS embedding_model,
        embedding.started_at AS started_at,
        embedding.completed_at AS completed_at,
        invocation.run_id AS run_id,
        invocation.sequence_number AS sequence_number,
        invocation.model AS text_model,
        invocation.output_text AS text,
        run.initial_prompt AS initial_prompt,
        run.network AS network
    FROM embedding
    JOIN invocation ON embedding.invocation_id = invocation.id
    JOIN run ON invocation.run_id = run.id
    WHERE invocation.type = 'TEXT' AND run.initial_prompt NOT IN ('yeah', 'nah')
    ORDER BY run_id, embedding_model, sequence_number
    """

    # Use polars to read directly from the database
    db_url = _get_polars_db_uri(session)
    df = pl.read_database_uri(query=query, uri=db_url)

    # Format UUID columns
    df = format_uuid_columns(df, ["id", "invocation_id", "run_id"])

    # Parse network from JSON string and create network_path column
    df = df.with_columns([
        pl.col("network").str.json_decode().list.join(" → ").alias("network")
    ])

    return df


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
      AND run.initial_prompt NOT IN ('yeah', 'nah')
    """

    # Use polars to read directly from the database, getting the correct URI format
    db_url = _get_polars_db_uri(session)
    df = pl.read_database_uri(query=query, uri=db_url)

    # Format UUID columns
    df = format_uuid_columns(df, ["run_id", "experiment_id"])

    # Parse network from JSON string to List[str]
    df = df.with_columns([pl.col("network").str.json_decode().alias("network")])

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
        pl.col("network")
        .map_elements(extract_models, return_dtype=pl.List(pl.String))
        .alias("models")
    ])

    df = df.with_columns([
        pl.col("models").list.get(0).alias("image_model"),
        pl.col("models").list.get(1).alias("text_model"),
    ]).drop("models")

    # Add prompt category column
    df = df.with_columns([
        pl.col("initial_prompt")
        .map_elements(prompt_category_mapper, return_dtype=pl.List(pl.String))
        .alias("prompt_category")
    ])

    df = add_persistence_entropy(df, session)

    return df
