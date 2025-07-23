import os
import time
from typing import Any, Dict, Tuple
from uuid import UUID

import numpy as np
import polars as pl
import ray
from humanize.time import naturaldelta
from sqlmodel import Session, select

from panic_tda.clustering import hdbscan
from panic_tda.db import list_runs, read_embedding
from panic_tda.embeddings import get_actor_class
from panic_tda.genai_models import get_output_type
from panic_tda.schemas import ClusteringResult, EmbeddingCluster, InvocationType

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


def load_clusters_from_cache() -> pl.DataFrame:
    """
    Load clustering results from the cache file.

    Returns:
        A polars DataFrame containing all clustering data from cache
    """
    cache_path = "output/cache/clusters.parquet"
    print(f"Loading clusters from cache: {cache_path}")
    return pl.read_parquet(cache_path)


def create_or_get_clustering_result(
    embedding_model: str, algorithm: str, parameters: dict, session: Session
) -> ClusteringResult:
    """
    Create or retrieve a clustering result for the given parameters.
    Does not commit - caller is responsible for transaction management.

    Args:
        embedding_model: The embedding model name
        algorithm: The clustering algorithm name
        parameters: The algorithm parameters
        session: Database session

    Returns:
        ClusteringResult instance
    """
    # Check if we already have this clustering result
    existing = session.exec(
        select(ClusteringResult)
        .where(ClusteringResult.embedding_model == embedding_model)
        .where(ClusteringResult.algorithm == algorithm)
    ).first()

    if existing:
        return existing

    # Create new clustering result
    clustering_result = ClusteringResult(
        embedding_model=embedding_model, algorithm=algorithm, parameters=parameters
    )
    session.add(clustering_result)
    # No commit - let caller handle transaction
    return clustering_result


def fetch_and_cluster_vectors(
    embedding_ids: pl.Series, embedding_model: str, session: Session
) -> Dict[str, Any]:
    """
    Fetch embedding vectors from the database, stack them, and perform global clustering.

    This function clusters embeddings globally across all experiments, storing results
    with experiment_id=NULL in the database.

    Args:
        embedding_ids: A Series containing embedding IDs
        embedding_model: The embedding model name
        session: SQLModel database session

    Returns:
        Dictionary containing:
        - clustering_result_id: UUID of the clustering result
        - assignments: List of tuples (embedding_id, cluster_id, cluster_label)
    """
    # Check if we have any embeddings to process
    if len(embedding_ids) == 0:
        return {"clustering_result_id": None, "assignments": []}

    # Start a transaction for atomic operations
    with session.begin_nested():
        # Get or create global clustering result (experiment_id=None)
        parameters = {"cluster_selection_epsilon": 0.6, "allow_single_cluster": True}
        clustering_result = create_or_get_clustering_result(
            embedding_model, "hdbscan", parameters, session
        )

        # Check if we already have cluster assignments for these embeddings
        existing_assignments = session.exec(
            select(EmbeddingCluster)
            .where(EmbeddingCluster.clustering_result_id == clustering_result.id)
            .where(
                EmbeddingCluster.embedding_id.in_([UUID(eid) for eid in embedding_ids])
            )
        ).all()

        if len(existing_assignments) == len(embedding_ids):
            # We already have all assignments, use them
            assignment_map = {
                str(a.embedding_id): a.cluster_id for a in existing_assignments
            }
            
            # Get cluster texts from clustering_result
            cluster_text_map = {
                c["id"]: c["medoid_text"] for c in clustering_result.clusters
            }
            cluster_text_map[-1] = "OUTLIER"
            
            assignments = [
                (
                    str(eid),
                    assignment_map[str(eid)],
                    cluster_text_map[assignment_map[str(eid)]]
                )
                for eid in embedding_ids
            ]
            
            return {
                "clustering_result_id": str(clustering_result.id),
                "assignments": assignments
            }

        # Fetch embeddings and build a cache of vector -> text
        embeddings = [
            read_embedding(UUID(embedding_id), session)
            for embedding_id in embedding_ids
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

        # Create cluster information
        unique_labels = sorted(set(cluster_result["labels"]))
        clusters_info = []
        label_to_text = {}

        for label in unique_labels:
            if label == -1:
                label_to_text[label] = "OUTLIER"
            else:
                # Get the medoid vector for this cluster
                medoid_vector = cluster_result["medoids"][label]
                medoid_key = tuple(medoid_vector.flatten())
                medoid_text = vector_to_text.get(medoid_key, f"Cluster {label}")

                clusters_info.append({
                    "id": int(label),  # Convert numpy int to Python int
                    "medoid_text": medoid_text,
                })
                label_to_text[label] = medoid_text

        # Update clustering result with clusters
        clustering_result.clusters = clusters_info

        # Create embedding cluster assignments
        for i, (embedding_id, cluster_label) in enumerate(
            zip(embedding_ids, cluster_result["labels"])
        ):
            embedding_cluster = EmbeddingCluster(
                embedding_id=UUID(embedding_id),
                clustering_result_id=clustering_result.id,
                cluster_id=int(cluster_label),
            )
            session.add(embedding_cluster)

        # Transaction will be committed when exiting the with block

    # Create result with assignments
    assignments = [
        (
            str(eid),
            int(cluster_result["labels"][i]),
            label_to_text[cluster_result["labels"][i]]
        )
        for i, eid in enumerate(embedding_ids)
    ]
    
    return {
        "clustering_result_id": str(clustering_result.id),
        "assignments": assignments
    }


def load_clusters_df(session: Session, downsample: int = 10) -> pl.DataFrame:
    """
    Load or create clustering results for all embeddings.

    This function processes embeddings globally across all experiments,
    creating clustering results if they don't exist.

    Args:
        session: SQLModel database session
        downsample: If > 1, only process every nth embedding (e.g., 2 means every other embedding)

    Returns:
        DataFrame containing cluster assignments with columns:
        - clustering_result_id: UUID of the clustering run
        - embedding_id: UUID of the embedding
        - embedding_model: Name of the embedding model
        - cluster_id: Numeric cluster ID (-1 for outliers)
        - cluster_label: Text representation (medoid text or "OUTLIER")
        - run_id, sequence_number, invocation_id: For easy joining
        - algorithm: Clustering algorithm used
        - epsilon: HDBSCAN epsilon parameter
    """
    print("Loading clustering data...")
    
    # First load embeddings metadata
    embeddings_df = load_embeddings_df(session)
    
    # Process each embedding model globally
    cluster_data = []
    
    for model_name in embeddings_df["embedding_model"].unique():
        # Filter to get only rows for this model
        model_df = embeddings_df.filter(pl.col("embedding_model") == model_name)
        
        # Apply downsampling within this model's data
        model_df_downsampled = (
            model_df.with_row_index("row_idx")
            .filter(pl.col("row_idx") % downsample == 0)
            .drop("row_idx")
        )
        
        embedding_ids = model_df_downsampled["id"]
        
        # Log the number of embeddings for this model
        print(
            f"Model {model_name}: {len(model_df)} total, {len(embedding_ids)} after downsampling"
        )
        
        # Skip if too few samples for clustering
        if len(embedding_ids) < 2:
            print(f"  Skipping {model_name} - too few samples for HDBSCAN")
            continue
        
        # Get cluster assignments for this model's embeddings
        cluster_result = fetch_and_cluster_vectors(embedding_ids, model_name, session)
        
        # cluster_result is a dict with clustering_result_id and assignments
        clustering_result_id = cluster_result["clustering_result_id"]
        assignments = cluster_result["assignments"]
        
        # Create dataframe rows with full metadata
        rows = []
        for i, (embedding_id, cluster_id, cluster_label) in enumerate(assignments):
            # Get metadata from the downsampled dataframe
            embedding_row = model_df_downsampled.filter(pl.col("id") == embedding_id).to_dicts()[0]
            
            row = {
                "clustering_result_id": clustering_result_id,
                "embedding_id": embedding_id,
                "embedding_model": model_name,
                "cluster_id": cluster_id,
                "cluster_label": cluster_label,
                "run_id": embedding_row["run_id"],
                "sequence_number": embedding_row["sequence_number"],
                "invocation_id": embedding_row["invocation_id"],
                "algorithm": "hdbscan",
                "epsilon": 0.6,
            }
            rows.append(row)
        
        cluster_data.append(pl.DataFrame(rows))
    
    # Commit all clustering operations atomically
    session.commit()
    
    if cluster_data:
        return pl.concat(cluster_data)
    else:
        # Return empty dataframe with correct schema
        return pl.DataFrame({
            "clustering_result_id": [],
            "embedding_id": [],
            "embedding_model": [],
            "cluster_id": [],
            "cluster_label": [],
            "run_id": [],
            "sequence_number": [],
            "invocation_id": [],
            "algorithm": [],
            "epsilon": [],
        })


def filter_top_n_clusters(
    df: pl.DataFrame, n: int, group_by_cols: list[str] = []
) -> pl.DataFrame:
    """
    Filter a DataFrame to keep only the top n clusters within each
    group defined by group_by_cols.
    
    Note: This function expects a DataFrame that already has cluster labels joined
    (e.g., the result of joining embeddings_df with clusters_df). It does not work
    directly with the separate clusters_df.

    Args:
        df: DataFrame containing data with cluster_label column (e.g., embeddings joined with clusters)
        n: Number of top clusters to keep within each group
        group_by_cols: List of column names to group by (in addition to embedding_model)

    Returns:
        Filtered DataFrame containing only rows from the top n clusters, with count and rank columns
    """
    # Count occurrences of each cluster within each group
    group_cols = ["embedding_model"] + group_by_cols + ["cluster_label"]
    cluster_counts = df.group_by(group_cols).agg(pl.len().alias("count"))

    # Assign rank within each group based on count
    cluster_ranks = cluster_counts.with_columns([
        pl.col("count")
        .rank(method="dense", descending=True)
        .over(["embedding_model"] + group_by_cols)
        .alias("rank")
    ])

    # Filter to keep only the top n clusters
    top_clusters = cluster_ranks.filter(pl.col("rank") <= n)

    # Join with original dataframe to keep only rows from top clusters
    # Include the count and rank columns in the result
    result = df.join(top_clusters, on=group_cols, how="inner")

    return result


def calculate_cluster_transitions(
    df: pl.DataFrame, group_by_cols: list[str], include_outliers: bool = False
) -> pl.DataFrame:
    """
    Calculate transitions between clusters for each group in the DataFrame.

    Args:
        df: DataFrame containing embeddings with cluster labels and sequence_number column
        group_by_cols: List of column names to group by (e.g., ["embedding_model", "run_id"])
        include_outliers: Whether to include "OUTLIER" clusters (default: False)

    Returns:
        DataFrame containing counts of transitions between clusters
    """

    # Create a filtered dataframe based on parameters
    filtered_df = df.filter(pl.col("cluster_label").is_not_null())

    # Additionally filter out outliers if not requested to include them
    if not include_outliers:
        filtered_df = filtered_df.filter(pl.col("cluster_label") != "OUTLIER")

    # Create a shifted version of the cluster_label column to get previous cluster
    transitions_df = filtered_df.select(
        group_by_cols + ["cluster_label", "sequence_number"]
    ).with_columns([
        pl.col("cluster_label").alias("from_cluster"),
        pl.col("cluster_label").shift(-1).over(group_by_cols).alias("to_cluster"),
    ])

    # Count transitions
    transition_counts = (
        transitions_df.group_by(group_by_cols + ["from_cluster", "to_cluster"])
        .agg(pl.len().alias("transition_count"))
        .sort(
            group_by_cols + ["transition_count"],
            descending=[False] * len(group_by_cols) + [True],
        )
    )

    return transition_counts


def calculate_cluster_run_lengths(
    df: pl.DataFrame, include_outliers: bool = False
) -> pl.DataFrame:
    """
    Calculate run lengths of clusters for each run and embedding model.

    The RLE (Run Length Encoding) processing is performed by grouping
    the data by 'run_id' and 'embedding_model', and then sorting by
    'sequence_number' to define the sequences for RLE.

    Args:
        df: DataFrame containing embeddings with 'cluster_label',
            'sequence_number', 'run_id', and 'embedding_model' columns (sequence_number must
            be sorted within run_id).
        include_outliers: Whether to include "OUTLIER" clusters (default: False).

    Returns:
        DataFrame with run length information for each cluster within each
        run/embedding_model group. Columns include 'run_id', 'embedding_model',
        'cluster_label', and 'run_length'.
    """
    # Create a filtered dataframe based on parameters
    filtered_df = df.filter(pl.col("cluster_label").is_not_null())

    # Additionally filter out outliers if not requested to include them
    if not include_outliers:
        filtered_df = filtered_df.filter(pl.col("cluster_label") != "OUTLIER")

    # Define the fixed group columns for RLE processing
    # RLE is applied per run_id and per embedding_model
    group_cols = ["embedding_model", "run_id", "initial_prompt", "network"]

    # Create a dataframe with each group's data and apply RLE
    result_df = (
        filtered_df.group_by(group_cols)
        .agg([pl.col("cluster_label").rle().alias("rle_result")])
        .explode("rle_result")
    )

    # Extract the length and value from the RLE struct
    result_df = result_df.with_columns([
        pl.col("rle_result").struct.field("len").alias("run_length"),
        pl.col("rle_result").struct.field("value").alias("cluster_label"),
    ]).drop("rle_result")

    # Sort by group columns to ensure consistent ordering
    result_df = result_df.sort(group_cols)

    return result_df


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


def calculate_semantic_drift(
    vectors_array: np.ndarray, reference_vector: np.ndarray
) -> np.ndarray:
    """
    Calculate semantic drift (cosine distance) between each vector and a reference vector.
    
    Uses the normalize-then-euclidean trick: for normalized unit vectors,
    euclidean distance is monotonic with cosine distance.

    Args:
        vectors_array: 2D array where each row is a vector
        reference_vector: The reference vector to calculate distances from

    Returns:
        1D array of cosine distances (0 = identical, 2 = opposite)
    """
    # Normalize vectors to unit length
    vectors_normalized = vectors_array / np.linalg.norm(
        vectors_array, axis=1, keepdims=True
    )
    reference_normalized = reference_vector / np.linalg.norm(reference_vector)
    
    # Calculate euclidean distance between normalized vectors
    # For unit vectors: euclidean_dist = sqrt(2 - 2*cos_similarity)
    # So euclidean distance is monotonic with cosine distance
    differences = vectors_normalized - reference_normalized
    distances = np.linalg.norm(differences, axis=1)
    
    return distances


def fetch_and_calculate_drift(
    embedding_ids: pl.Series,
    initial_prompt_vectors: Dict[Tuple[str, str], np.ndarray],
    session: Session,
) -> pl.Series:
    """
    Fetch embedding vectors from the database and calculate semantic drift from the embedded initial prompt.

    Args:
        embedding_ids: A Series containing embedding IDs
        initial_prompt_vectors: Dictionary mapping (initial_prompt, embedding_model) tuples to vectors
        session: SQLModel database session

    Returns:
        A Series of semantic drift values (cosine distances via normalized euclidean)
    """
    # Get vectors from database
    embeddings = [
        read_embedding(UUID(embedding_id), session) for embedding_id in embedding_ids
    ]
    first_embedding = embeddings[0]
    initial_vector = initial_prompt_vectors[
        (first_embedding.invocation.run.initial_prompt, first_embedding.embedding_model)
    ]

    # Stack the vectors and calculate semantic drift
    vectors = np.vstack([embedding.vector for embedding in embeddings])
    distances = calculate_semantic_drift(vectors, initial_vector)
    return pl.Series(distances)


def add_semantic_drift(df: pl.DataFrame, session: Session) -> pl.DataFrame:
    """
    Add semantic drift values to the embeddings DataFrame.
    
    Calculates cosine distance from each embedding to the initial prompt's embedding
    using the normalize-then-euclidean approach for efficiency.

    Args:
        df: DataFrame containing embedding metadata (without vectors)
        session: SQLModel database session for fetching vectors

    Returns:
        DataFrame with semantic drift values added
    """
    initial_prompt_vectors = embed_initial_prompts(session)

    # Add semantic drift column using map_batches
    df = df.with_columns(
        pl.col("id")
        .map_batches(
            lambda embedding_ids: fetch_and_calculate_drift(
                embedding_ids, initial_prompt_vectors, session
            ),
        )
        .alias("semantic_drift")
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
    Add persistence entropy scores to a runs DataFrame.

    Only adds entropy values per homology dimension, without the full birth/death pair data.
    For full persistence diagram data, use load_pd_df() instead.

    Args:
        df: DataFrame containing run data
        session: SQLModel database session

    Returns:
        DataFrame with persistence entropy data added
    """
    print("Adding persistence entropy data to runs DataFrame...")

    # Load all runs to get their PDs
    runs = list_runs(session)
    run_map = {str(run.id): run for run in runs}

    data = []
    for run_id in df["run_id"].unique().to_list():
        run = run_map.get(run_id)
        if not run or not run.persistence_diagrams:
            continue

        for pd in run.persistence_diagrams:
            # Only process if we have entropy data
            if pd.diagram_data and "entropy" in pd.diagram_data:
                # Process each dimension's entropy
                for dim, entropy_value in enumerate(pd.diagram_data["entropy"]):
                    row = {
                        "run_id": run_id,
                        "persistence_diagram_id": str(pd.id),
                        "embedding_model": pd.embedding_model,
                        "homology_dimension": dim,
                        "entropy": float(entropy_value),
                    }
                    data.append(row)

    # Create a polars DataFrame with explicit schema for numeric fields
    schema_overrides = {
        "homology_dimension": pl.Int64,
        "entropy": pl.Float64,
    }

    entropy_df = pl.DataFrame(data, schema_overrides=schema_overrides)

    # Join with the original runs DataFrame
    result_df = df.join(entropy_df, on="run_id", how="left")

    return result_df


def load_pd_from_cache() -> pl.DataFrame:
    """
    Load persistence diagram data from the cache file.

    Returns:
        A polars DataFrame containing all persistence diagram data from cache
    """
    cache_path = "output/cache/persistence_diagrams.parquet"
    print(f"Loading persistence diagrams from cache: {cache_path}")
    return pl.read_parquet(cache_path)


def cache_dfs(
    session: Session,
    runs: bool = True,
    embeddings: bool = True,
    invocations: bool = True,
    persistence_diagrams: bool = True,
    clusters: bool = True,
) -> None:
    """
    Preload and cache dataframes.

    Args:
        session: SQLModel database session
        runs: Whether to cache runs dataframe
        embeddings: Whether to cache embeddings dataframe
        invocations: Whether to cache invocations dataframe
        persistence_diagrams: Whether to cache persistence diagrams dataframe
        clusters: Whether to cache clusters dataframe

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
        print(f"  Time taken: {naturaldelta(elapsed_time)}")

    if embeddings:
        print("Warming cache for embeddings dataframe...")
        cache_path = "output/cache/embeddings.parquet"

        start_time = time.time()
        embeddings_df = load_embeddings_df(session)
        embeddings_df = add_semantic_drift(embeddings_df, session)

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
        print(f"  Time taken: {naturaldelta(elapsed_time)}")

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
        print(f"  Time taken: {naturaldelta(elapsed_time)}")

    if persistence_diagrams:
        print("Warming cache for persistence diagrams dataframe...")
        cache_path = "output/cache/persistence_diagrams.parquet"

        start_time = time.time()
        pd_df = load_pd_df(session)
        df_memory_size = pd_df.estimated_size() / (1024 * 1024)  # Convert to MB

        # Save to cache (automatically overwrites if exists)
        pd_df.write_parquet(cache_path)
        elapsed_time = time.time() - start_time

        cache_file_size = os.path.getsize(cache_path) / (1024 * 1024)  # Convert to MB
        print(
            f"Saved persistence diagrams ({pd_df.shape[0]} rows, {pd_df.shape[1]} columns) to cache: {cache_path}"
        )
        print(
            f"  Memory size: {df_memory_size:.2f} MB, Cache file size: {cache_file_size:.2f} MB"
        )
        print(f"  Time taken: {naturaldelta(elapsed_time)}")

    if clusters:
        print("Warming cache for clusters dataframe...")
        cache_path = "output/cache/clusters.parquet"

        start_time = time.time()
        clusters_df = load_clusters_df(session, downsample=10)
        df_memory_size = clusters_df.estimated_size() / (1024 * 1024)  # Convert to MB

        # Save to cache (automatically overwrites if exists)
        clusters_df.write_parquet(cache_path)
        elapsed_time = time.time() - start_time

        cache_file_size = os.path.getsize(cache_path) / (1024 * 1024)  # Convert to MB
        print(
            f"Saved clusters ({clusters_df.shape[0]} rows, {clusters_df.shape[1]} columns) to cache: {cache_path}"
        )
        print(
            f"  Memory size: {df_memory_size:.2f} MB, Cache file size: {cache_file_size:.2f} MB"
        )
        print(f"  Time taken: {naturaldelta(elapsed_time)}")

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
        run.network AS network,
        run.experiment_id AS experiment_id
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
    df = format_uuid_columns(df, ["id", "invocation_id", "run_id", "experiment_id"])

    # Parse network from JSON string and create network_path column
    df = df.with_columns([
        pl.col("network").str.json_decode().list.join("→").alias("network")
    ])

    return df


def load_pd_df(session: Session) -> pl.DataFrame:
    """
    Load all persistence diagram data from the database into a tidy polars DataFrame.

    Each row represents a single birth/death pair from a persistence diagram,
    with full run context (initial_prompt, network) included via joins.

    Args:
        session: SQLModel database session

    Returns:
        A polars DataFrame containing persistence diagram data with columns:
        - persistence_diagram_id, run_id, embedding_model
        - homology_dimension, birth, death, persistence (calculated)
        - initial_prompt, network (from joined run data)
        - image_model, text_model (extracted from network)
    """
    print("Loading persistence diagram data from database...")

    # SQL query to extract persistence diagram data with run context
    query = """
    SELECT
        pd.id as persistence_diagram_id,
        pd.run_id as run_id,
        pd.embedding_model as embedding_model,
        pd.started_at as started_at,
        pd.completed_at as completed_at,
        (pd.completed_at - pd.started_at) as duration,
        run.initial_prompt as initial_prompt,
        run.network as network,
        run.experiment_id as experiment_id
    FROM persistencediagram pd
    JOIN run ON pd.run_id = run.id
    WHERE run.initial_prompt NOT IN ('yeah', 'nah')
        AND pd.diagram_data IS NOT NULL
    """

    # Use polars to read directly from the database
    db_url = _get_polars_db_uri(session)
    pd_metadata_df = pl.read_database_uri(query=query, uri=db_url)

    # Format UUID columns
    pd_metadata_df = format_uuid_columns(
        pd_metadata_df, ["persistence_diagram_id", "run_id", "experiment_id"]
    )

    # Parse network from JSON string
    pd_metadata_df = pd_metadata_df.with_columns([
        pl.col("network").str.json_decode().list.join("→").alias("network")
    ])

    # Now we need to expand the diagram_data to create rows for each birth/death pair
    # We'll fetch the actual persistence diagrams from the database
    from panic_tda.schemas import PersistenceDiagram
    from sqlmodel import select

    pd_ids = pd_metadata_df["persistence_diagram_id"].unique().to_list()

    # Build expanded data with birth/death pairs
    expanded_data = []

    for pd_id in pd_ids:
        # Get the persistence diagram from database
        pd_obj = session.exec(
            select(PersistenceDiagram).where(PersistenceDiagram.id == UUID(pd_id))
        ).first()

        if not pd_obj or not pd_obj.diagram_data or "dgms" not in pd_obj.diagram_data:
            continue

        # Get the metadata row for this PD
        metadata_row = pd_metadata_df.filter(
            pl.col("persistence_diagram_id") == pd_id
        ).to_dicts()[0]

        # Process each homology dimension
        for dim, dgm in enumerate(pd_obj.diagram_data["dgms"]):
            if not isinstance(dgm, np.ndarray) or len(dgm) == 0:
                continue

            # Create a row for each birth/death pair
            for birth, death in dgm:
                row = metadata_row.copy()
                row["homology_dimension"] = dim
                row["birth"] = float(birth)
                row["death"] = float(death)
                row["persistence"] = float(death - birth)
                expanded_data.append(row)

    # Create the final DataFrame with explicit schema
    schema_overrides = {
        "homology_dimension": pl.Int64,
        "birth": pl.Float64,
        "death": pl.Float64,
        "persistence": pl.Float64,
    }

    pd_df = pl.DataFrame(expanded_data, schema_overrides=schema_overrides)

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

    # Parse network string back to list and extract models
    pd_df = pd_df.with_columns([pl.col("network").str.split("→").alias("network_list")])

    pd_df = pd_df.with_columns([
        pl.col("network_list")
        .map_elements(extract_models, return_dtype=pl.List(pl.String))
        .alias("models")
    ])

    pd_df = pd_df.with_columns([
        pl.col("models").list.get(0).alias("image_model"),
        pl.col("models").list.get(1).alias("text_model"),
    ]).drop(["models", "network_list"])

    return pd_df


def prompt_category_mapper(initial_prompt: str) -> str:
    """
    Map an initial prompt to its category based on the predefined mapping.

    Args:
        initial_prompt: The initial prompt to map

    Returns:
        The category of the prompt or None if not found
    """
    # Mapping from initial prompts to categories derived from prompt-counts.json
    prompt_to_category = {
        "a painting of a man": "people_portraits",
        "a picture of a child": "people_portraits",
        "a picture of a man": "people_portraits",
        "a photorealistic portrait of a child": "people_portraits",
        "a painting of a woman": "people_portraits",
        "a painting of a child": "people_portraits",
        "a photo of a child": "people_portraits",
        "a photo of a man": "people_portraits",
        "a photorealistic portrait of a woman": "people_portraits",
        "a photo of a woman": "people_portraits",
        "a photorealistic portrait of a man": "people_portraits",
        "a picture of a woman": "people_portraits",
        "a photorealistic portrait photo of a child": "people_portraits",
        "a photorealistic portrait photo of a woman": "people_portraits",
        "a photorealistic portrait photo of a man": "people_portraits",
        "nah": "abstract",
        "yeah": "abstract",
        "a giraffe": "animals",
        "a cat": "animals",
        "an elephant": "animals",
        "a hamster": "animals",
        "a rabbit": "animals",
        "a dog": "animals",
        "a lion": "animals",
        "a goldfish": "animals",
        "a red circle on a black background": "geometric_shapes",
        "a red circle on a yellow background": "geometric_shapes",
        "a blue circle on a yellow background": "geometric_shapes",
        "a yellow circle on a blue background": "geometric_shapes",
        "a blue circle on a red background": "geometric_shapes",
        "a yellow circle on a black background": "geometric_shapes",
        "a blue circle on a black background": "geometric_shapes",
        "a red circle on a blue background": "geometric_shapes",
        "a yellow circle on a red background": "geometric_shapes",
        "a pear": "food",
        "a banana": "food",
        "an apple": "food",
        "a boat": "transportation",
        "a train": "transportation",
        "a car": "transportation",
        "orange": "colours",
        "green": "colours",
        "yellow": "colours",
        "red": "colours",
        "blue": "colours",
        "indigo": "colours",
        "violet": "colours",
    }

    return prompt_to_category.get(initial_prompt)


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
