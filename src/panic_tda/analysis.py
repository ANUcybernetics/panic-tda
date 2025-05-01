import os

import numpy as np
import polars as pl
from numpy.linalg import norm
from sqlmodel import Session

from panic_tda.clustering import hdbscan
from panic_tda.db import list_runs
from panic_tda.genai_models import get_output_type
from panic_tda.schemas import InvocationType


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
    runs = list_runs(session)

    data = []
    for run in runs:
        # temporary hack to only look at 1k runs for SMC paper
        if run.max_length > 1000:
            continue

        run_id = str(run.id)

        for invocation in run.invocations:
            row = {
                "id": str(invocation.id),
                "run_id": run_id,
                "experiment_id": str(run.experiment_id) if run.experiment_id else None,
                "model": invocation.model,
                "type": invocation.type.value,
                "sequence_number": invocation.sequence_number,
                "started_at": invocation.started_at,
                "completed_at": invocation.completed_at,
                "duration": invocation.completed_at - invocation.started_at,
                "initial_prompt": run.initial_prompt,
                "seed": run.seed,
            }
            data.append(row)

    # Create a polars DataFrame
    df = pl.DataFrame(data)

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
    runs = list_runs(session)

    # Create a mapping of first embeddings by run_id and embedding_model
    first_embeddings = {}
    data = []

    # Dictionary to collect embeddings by model for clustering
    embeddings_by_model = {}

    # Process each run and its embeddings
    for run in runs:
        # temporary hack to only look at 1k runs for SMC paper
        if run.max_length > 1000:
            continue

        # Get all embeddings for the run
        run_embeddings_dict = run.embeddings
        run_id = str(run.id)

        # Process embeddings for each model
        for embedding_model, embeddings_list in run_embeddings_dict.items():
            # Track first text embedding for each model
            first_text_embedding = None

            # Filter to only keep text invocations
            text_embeddings = []
            for embedding in embeddings_list:
                if embedding.invocation.type == InvocationType.TEXT:
                    text_embeddings.append(embedding)
                    if first_text_embedding is None:
                        first_text_embedding = embedding

                    # Initialize the model's list in the dictionary if it doesn't exist
                    if embedding_model not in embeddings_by_model:
                        embeddings_by_model[embedding_model] = []

                    # Save the embedding for clustering later
                    embeddings_by_model[embedding_model].append(embedding)

            # Store the first text embedding for drift calculations
            if first_text_embedding:
                key = (run_id, embedding_model)
                first_embeddings[key] = first_text_embedding

            # Process all text embeddings for this model
            for i, embedding in enumerate(text_embeddings):
                invocation = embedding.invocation

                # Calculate semantic drift metrics
                semantic_drift_overall = None
                semantic_drift_instantaneous = None
                key = (run_id, embedding_model)
                first_embedding = first_embeddings.get(key)
                current_vector = np.array(embedding.vector)

                # Calculate drift from first embedding (origin)
                if first_embedding:
                    first_vector = np.array(first_embedding.vector)
                    semantic_drift_overall = calculate_cosine_distance(
                        first_vector, current_vector
                    )

                # Calculate drift from previous embedding
                if i > 0 and text_embeddings[i - 1]:
                    prev_vector = np.array(text_embeddings[i - 1].vector)
                    semantic_drift_instantaneous = calculate_cosine_distance(
                        prev_vector, current_vector
                    )

                row = {
                    "id": str(embedding.id),
                    "invocation_id": str(invocation.id),
                    "run_id": run_id,
                    "embedding_model": embedding_model,
                    "started_at": embedding.started_at,
                    "completed_at": embedding.completed_at,
                    "sequence_number": invocation.sequence_number,  # Include sequence_number for easier analysis
                    "semantic_drift_overall": semantic_drift_overall,
                    "semantic_drift_instantaneous": semantic_drift_instantaneous,
                    "vector_length": len(embedding.vector),
                    "initial_prompt": run.initial_prompt,  # Added initial_prompt from run
                    "model": invocation.model,  # Added model from invocation
                }
                data.append(row)

    # Create a polars DataFrame
    df = pl.DataFrame(data)

    # Perform clustering using polars group_by and aggregation
    clustering_results = []

    for embedding_model, embeddings_list in embeddings_by_model.items():
        # Get cluster labels using hdbscan
        cluster_labels = hdbscan(embeddings_list)

        # Create a list of dictionaries for the cluster results
        model_clusters = [
            {"id": str(embedding.id), "cluster_label": label}
            for embedding, label in zip(embeddings_list, cluster_labels)
        ]

        # Add to clustering results
        clustering_results.extend(model_clusters)

    # Convert clustering results to a polars DataFrame
    clusters_df = pl.DataFrame(clustering_results)

    # Join the main DataFrame with the clusters DataFrame
    df = df.join(clusters_df, on="id", how="left")

    return df


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

        # Process stop_reason to separate into reason and loop_length
        stop_reason_value = run.stop_reason
        loop_length = None

        if isinstance(stop_reason_value, tuple) and stop_reason_value[0] == "duplicate":
            stop_reason = "duplicate"
            loop_length = stop_reason_value[1]
        else:
            stop_reason = stop_reason_value

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
            "stop_reason": stop_reason,
            "loop_length": loop_length,
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
        "loop_length": pl.Int64,
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
