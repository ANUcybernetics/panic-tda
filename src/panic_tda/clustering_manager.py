"""
Module for managing clustering operations on embeddings.

This module provides functions to run clustering on all embeddings stored in the database,
monitor clustering progress, and retrieve clustering results.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from uuid import UUID

import polars as pl
from sqlmodel import Session, select, func
from tqdm import tqdm

from panic_tda.data_prep import add_cluster_labels, load_embeddings_df
from panic_tda.schemas import (
    ClusteringResult,
    EmbeddingCluster,
    Embedding,
    ExperimentConfig,
    Invocation,
    Run,
)

logger = logging.getLogger(__name__)


def get_clustering_status(session: Session) -> Dict[UUID, Dict[str, any]]:
    """
    Get clustering status for all experiments in the database.

    Since clustering is now done globally, this provides experiment-level statistics
    about embeddings but not clustering status per experiment.

    Returns:
        Dictionary mapping experiment IDs to their status information
    """
    status = {}

    # Get all experiments
    experiments = session.exec(select(ExperimentConfig)).all()

    # Check if global clustering exists
    global_clustering_results = session.exec(select(ClusteringResult)).all()

    for exp in experiments:
        # Get embedding count for this experiment
        embedding_count = session.exec(
            select(func.count(Embedding.id))
            .join(Invocation)
            .join(Run)
            .where(Run.experiment_id == exp.id)
        ).one()

        # For backward compatibility, return experiment info
        status[exp.id] = {
            "experiment_id": exp.id,
            "run_count": len(exp.runs) if exp.runs else 0,
            "prompt_count": len(exp.prompts) if exp.prompts else 0,
            "network_count": len(exp.networks) if exp.networks else 0,
            "embedding_count": embedding_count,
            "clustered_count": embedding_count if global_clustering_results else 0,
            "clustering_models": [],  # Deprecated
            "total_clusters": 0,  # Deprecated
            "clustering_details": [],  # Deprecated
            "is_fully_clustered": bool(global_clustering_results),
        }

    return status


def get_cluster_details(
    experiment_id: UUID, embedding_model: str, session: Session
) -> Optional[Dict[str, any]]:
    """
    Get detailed cluster information for a specific experiment and embedding model.

    Note: This returns global clustering results filtered to the experiment.

    Args:
        experiment_id: The experiment ID
        embedding_model: The embedding model name
        session: Database session

    Returns:
        Dictionary with cluster details or None if no clustering exists
    """
    # Get global clustering result for this embedding model
    clustering_result = session.exec(
        select(ClusteringResult).where(
            ClusteringResult.embedding_model == embedding_model
        )
    ).first()

    if not clustering_result:
        return None

    # Get cluster assignments for embeddings from this experiment
    assignments = session.exec(
        select(EmbeddingCluster)
        .join(Embedding)
        .join(Invocation)
        .join(Run)
        .where(Run.experiment_id == experiment_id)
        .where(EmbeddingCluster.clustering_result_id == clustering_result.id)
    ).all()

    if not assignments:
        return None

    # Count clusters and build cluster info
    cluster_counts = defaultdict(int)
    for assignment in assignments:
        cluster_counts[assignment.cluster_id] += 1

    # Build cluster details
    clusters = []
    for cluster_id, count in sorted(
        cluster_counts.items(), key=lambda x: x[1], reverse=True
    ):
        # Find cluster info in the global clustering result
        cluster_info = next(
            (c for c in clustering_result.clusters if c["id"] == cluster_id), None
        )
        if cluster_info:
            clusters.append({
                "id": cluster_id,
                "medoid_text": cluster_info.get("medoid_text", f"Cluster {cluster_id}"),
                "size": count,
            })
        elif cluster_id == -1:
            clusters.append({"id": -1, "medoid_text": "OUTLIER", "size": count})

    return {
        "experiment_id": experiment_id,
        "embedding_model": embedding_model,
        "algorithm": clustering_result.algorithm,
        "parameters": clustering_result.parameters,
        "created_at": clustering_result.created_at,
        "total_clusters": len(clusters),
        "total_assignments": len(assignments),
        "clusters": clusters,
    }


def cluster_all_data(
    session: Session, downsample: int = 1, force: bool = False
) -> Dict[str, any]:
    """
    Cluster all embeddings in the database globally.

    This function loads all embeddings across all experiments and performs
    clustering on the entire dataset, storing results with experiment_id=NULL.

    Args:
        session: Database session
        downsample: Downsampling factor (1 = no downsampling)
        force: If True, re-run clustering even if it was done before

    Returns:
        Dictionary with clustering results summary
    """
    logger.info("Starting global clustering on all embeddings in the database")

    # Check if clustering already exists
    if not force:
        existing = session.exec(select(ClusteringResult)).first()

        if existing:
            logger.info(
                "Clustering results already exist. Use force=True to re-cluster."
            )
            return {
                "status": "already_clustered",
                "message": "Clustering results already exist",
            }
    else:
        # Delete existing clustering results if forcing
        for cr in session.exec(select(ClusteringResult)).all():
            session.delete(cr)
        session.commit()
        logger.info("Deleted existing clustering results")

    try:
        # Load all embeddings
        all_embeddings_df = load_embeddings_df(session)

        if len(all_embeddings_df) == 0:
            logger.warning("No embeddings found in the database")
            return {
                "status": "error",
                "message": "No embeddings found in the database",
                "total_embeddings": 0,
                "clustered_embeddings": 0,
                "total_clusters": 0,
                "embedding_models_count": 0,
            }

        total_embeddings = len(all_embeddings_df)
        logger.info(f"Found {total_embeddings:,} total embeddings")

        # Get unique embedding models
        embedding_models = all_embeddings_df["embedding_model"].unique().to_list()
        logger.info(f"Embedding models: {embedding_models}")

        # Add cluster labels with global clustering
        logger.info(f"Running global clustering with downsample factor {downsample}")
        clustered_df = add_cluster_labels(all_embeddings_df, downsample, session)

        # Count results
        clustered_embeddings = clustered_df.filter(
            pl.col("cluster_label").is_not_null()
        ).height

        # Count unique clusters across all models
        total_clusters = 0
        for model in embedding_models:
            model_clusters = (
                clustered_df.filter(pl.col("embedding_model") == model)["cluster_label"]
                .unique()
                .drop_nulls()
            )
            total_clusters += len(model_clusters)

        logger.info(
            f"Global clustering complete: {clustered_embeddings:,}/{total_embeddings:,} embeddings "
            f"clustered into {total_clusters:,} total clusters"
        )

        return {
            "status": "success",
            "total_embeddings": total_embeddings,
            "clustered_embeddings": clustered_embeddings,
            "total_clusters": total_clusters,
            "embedding_models_count": len(embedding_models),
        }

    except Exception as e:
        logger.error(f"Error during global clustering: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "total_embeddings": 0,
            "clustered_embeddings": 0,
            "total_clusters": 0,
            "embedding_models_count": 0,
        }
