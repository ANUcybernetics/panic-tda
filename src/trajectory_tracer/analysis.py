import os

import numpy as np
import polars as pl
from numpy.linalg import norm
from sqlmodel import Session

from trajectory_tracer.db import list_runs
from trajectory_tracer.genai_models import get_output_type
from trajectory_tracer.schemas import InvocationType

## load the DB objects into dataframes


def load_embeddings_df(session: Session, use_cache: bool = False) -> pl.DataFrame:
    """
    Load all embeddings from the database and flatten them into a polars DataFrame.
    Only includes embeddings for text invocations.

    Args:
        session: SQLModel database session
        use_cache: Whether to use cached dataframe if available

    Returns:
        A polars DataFrame containing all embedding data for text invocations
    """
    cache_path = "output/cache/embeddings.parquet"

    # Check if cache exists and should be used
    if use_cache and os.path.exists(cache_path):
        print(f"Loading embeddings from cache: {cache_path}")
        return pl.read_parquet(cache_path)

    print("Loading embeddings from database...")
    runs = list_runs(session)

    # Create a mapping of first embeddings by run_id and embedding_model
    first_embeddings = {}
    data = []

    # Process each run and its embeddings
    for run in runs:
        # Get all embeddings for the run - embeddings is a property that returns a dict
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
                    # Identify first text embedding for this embedding model
                    if first_text_embedding is None:
                        first_text_embedding = embedding

            # Store the first text embedding for drift calculations
            if first_text_embedding:
                key = (run_id, embedding_model)
                first_embeddings[key] = first_text_embedding

            # Process all text embeddings for this model
            for i, embedding in enumerate(text_embeddings):
                invocation = embedding.invocation

                # Calculate semantic drift (distance from first embedding)
                semantic_drift_overall = None
                semantic_drift_instantaneous = (
                    None  # New variable for embedding-to-embedding drift
                )
                key = (run_id, embedding_model)
                first_embedding = first_embeddings.get(key)
                current_vector = np.array(embedding.vector)

                # Calculate drift from first embedding (origin)
                if first_embedding:
                    first_vector = np.array(first_embedding.vector)

                    # Calculate cosine similarity properly, handling non-normalized vectors
                    norm_first = norm(first_vector)
                    norm_current = norm(current_vector)

                    # Avoid division by zero
                    if norm_first > 0 and norm_current > 0:
                        cosine_similarity = np.dot(first_vector, current_vector) / (
                            norm_first * norm_current
                        )
                        semantic_drift_overall = float(1.0 - cosine_similarity)
                    else:
                        semantic_drift_overall = (
                            0.0 if np.array_equal(first_vector, current_vector) else 1.0
                        )

                # Calculate drift from previous embedding
                if i > 0 and text_embeddings[i - 1]:
                    prev_vector = np.array(text_embeddings[i - 1].vector)

                    # Calculate cosine similarity properly, handling non-normalized vectors
                    norm_prev = norm(prev_vector)
                    norm_current = norm(current_vector)

                    # Avoid division by zero
                    if norm_prev > 0 and norm_current > 0:
                        sequential_cosine_similarity = np.dot(
                            prev_vector, current_vector
                        ) / (norm_prev * norm_current)
                        semantic_drift_instantaneous = float(
                            1.0 - sequential_cosine_similarity
                        )
                    else:
                        semantic_drift_instantaneous = (
                            0.0 if np.array_equal(prev_vector, current_vector) else 1.0
                        )

                row = {
                    "id": str(embedding.id),
                    "invocation_id": str(invocation.id),
                    "embedding_model": embedding_model,
                    "embedding_started_at": embedding.started_at,
                    "embedding_completed_at": embedding.completed_at,
                    "invocation_started_at": invocation.started_at,
                    "invocation_completed_at": invocation.completed_at,
                    "run_id": run_id,
                    "experiment_id": str(run.experiment_id)
                    if run.experiment_id
                    else None,
                    "initial_prompt": run.initial_prompt,
                    "seed": run.seed,
                    "model": invocation.model,
                    "sequence_number": invocation.sequence_number,
                    "semantic_drift_overall": semantic_drift_overall,
                    "semantic_drift_instantaneous": semantic_drift_instantaneous,  # Add the new drift metric
                }
                data.append(row)

    # Create a polars DataFrame
    df = pl.DataFrame(data)

    # Only save to cache if use_cache is False
    if not use_cache:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.write_parquet(cache_path)
        print(f"Saved embeddings to cache: {cache_path}")

    return df


def load_runs_df(session: Session, use_cache: bool = False) -> pl.DataFrame:
    """
    Load all runs from the database and flatten them into a polars DataFrame.
    Includes persistence diagrams with birth/death pairs for each run.

    Args:
        session: SQLModel database session
        use_cache: Whether to use cached dataframe if available

    Returns:
        A polars DataFrame containing all run data with persistence diagrams
    """
    cache_path = "output/cache/runs.parquet"

    # Check if cache exists and should be used
    if use_cache and os.path.exists(cache_path):
        print(f"Loading runs from cache: {cache_path}")
        return pl.read_parquet(cache_path)

    print("Loading runs from database...")
    runs = list_runs(session)
    # Print the number of runs
    print(f"Number of runs: {len(runs)}")

    data = []
    for run in runs:
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

    # Only save to cache if use_cache is False
    if not use_cache:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.write_parquet(cache_path)
        print(f"Saved runs to cache: {cache_path}")

    return df


def warm_caches(session: Session, runs: bool = True, embeddings: bool = True) -> None:
    """
    Preload and cache both runs and embeddings dataframes.

    Args:
        session: SQLModel database session

    Returns:
        None
    """
    if runs:
        print("Warming cache for runs dataframe...")
        load_runs_df(session, use_cache=False)

    if embeddings:
        print("Warming cache for embeddings dataframe...")
        load_embeddings_df(session, use_cache=False)

    print("Cache warming complete.")
