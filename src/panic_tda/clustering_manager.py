"""
Module for managing clustering operations on embeddings.

This module provides functions to run clustering on all embeddings stored in the database,
monitor clustering progress, and retrieve clustering results.
"""

import logging
from typing import Dict, List, Optional
from uuid import UUID

from sqlmodel import Session, func, select

from panic_tda.schemas import (
    ClusteringResult,
    Embedding,
    EmbeddingCluster,
    Invocation,
    InvocationType,
)

logger = logging.getLogger(__name__)

# Constants
BATCH_SIZE = 10000
OUTLIER_CLUSTER_ID = -1
OUTLIER_LABEL = "OUTLIER"


def _build_cluster_details(
    clustering_result: ClusteringResult, session: Session, limit: Optional[int] = None
) -> Dict[str, any]:
    """
    Build cluster details dictionary from a clustering result.

    This is a shared helper to avoid code duplication.
    """
    # Get total assignments
    total_assignments = session.exec(
        select(func.count(EmbeddingCluster.id)).where(
            EmbeddingCluster.clustering_result_id == clustering_result.id
        )
    ).one()

    # Count clusters and build cluster info
    cluster_counts_query = (
        select(
            EmbeddingCluster.cluster_id, func.count(EmbeddingCluster.id).label("count")
        )
        .where(EmbeddingCluster.clustering_result_id == clustering_result.id)
        .group_by(EmbeddingCluster.cluster_id)
        .order_by(func.count(EmbeddingCluster.id).desc())
    )

    if limit:
        cluster_counts_query = cluster_counts_query.limit(limit)

    cluster_counts = session.exec(cluster_counts_query).all()

    # Build cluster details
    clusters = []
    cluster_map = {c["id"]: c for c in clustering_result.clusters}

    for cluster_id, count in cluster_counts:
        if cluster_id == OUTLIER_CLUSTER_ID:
            clusters.append({
                "id": OUTLIER_CLUSTER_ID,
                "medoid_text": OUTLIER_LABEL,
                "size": count,
            })
        elif cluster_id in cluster_map:
            cluster_data = cluster_map[cluster_id]
            cluster_info = {
                "id": cluster_id,
                "medoid_text": cluster_data["medoid_text"],
                "medoid_embedding_id": cluster_data["medoid_embedding_id"],
                "size": count,
            }

            # Add invocation details if available
            embedding = session.get(
                Embedding, UUID(cluster_data["medoid_embedding_id"])
            )
            if embedding and embedding.invocation:
                cluster_info["medoid_invocation_id"] = str(embedding.invocation.id)
                cluster_info["medoid_run_id"] = str(embedding.invocation.run_id)

            clusters.append(cluster_info)

    # Count regular clusters (excluding outliers)
    regular_cluster_count = len([
        c for c in clustering_result.clusters if c.get("id") != OUTLIER_CLUSTER_ID
    ])
    has_outliers = any(c["id"] == OUTLIER_CLUSTER_ID for c in clusters)

    return {
        "clustering_id": clustering_result.id,
        "embedding_model": clustering_result.embedding_model,
        "algorithm": clustering_result.algorithm,
        "parameters": clustering_result.parameters,
        "created_at": clustering_result.created_at,
        "started_at": clustering_result.started_at,
        "completed_at": clustering_result.completed_at,
        "duration": clustering_result.duration,
        "total_clusters": regular_cluster_count + (1 if has_outliers else 0),
        "total_assignments": total_assignments,
        "clusters": clusters,
    }


def _is_text_invocation_sampled(sequence_number: int, downsample: int) -> bool:
    """
    Check if a TEXT invocation should be included based on downsampling.

    TEXT invocations have odd sequence numbers (1, 3, 5, ...).
    We downsample by selecting every Nth TEXT invocation.
    """
    if downsample <= 1:
        return True
    # Convert odd sequence number to 0-based index: (1->0, 3->1, 5->2, etc.)
    text_index = (sequence_number - 1) // 2
    return text_index % downsample == 0


def _bulk_insert_with_flush(
    session: Session, objects: List, batch_size: int = BATCH_SIZE
):
    """
    Insert objects in batches with periodic flushes to avoid memory issues.
    """
    batch = []
    for obj in objects:
        batch.append(obj)
        if len(batch) >= batch_size:
            session.bulk_save_objects(batch)
            session.flush()
            batch = []

    # Insert remaining objects
    if batch:
        session.bulk_save_objects(batch)
        session.flush()


def _get_embeddings_query(model_name: str, downsample: int = 1):
    """
    Build a query for embeddings with optional downsampling.
    """
    query = (
        select(Embedding.id, Embedding.vector, Invocation.output_text)
        .select_from(Embedding)
        .join(Invocation, Embedding.invocation_id == Invocation.id)
        .where(Embedding.embedding_model == model_name)
        .where(Invocation.type == InvocationType.TEXT)
    )

    if downsample > 1:
        # Apply downsampling filter
        query = query.where(((Invocation.sequence_number - 1) / 2) % downsample == 0)

    return query.order_by(Invocation.sequence_number)


def get_cluster_details(
    embedding_model: str, session: Session, limit: int = None
) -> Optional[Dict[str, any]]:
    """
    Get detailed cluster information for a specific embedding model globally.

    Args:
        embedding_model: The embedding model name
        session: Database session
        limit: Maximum number of clusters to return (None for all)

    Returns:
        Dictionary with cluster details or None if no clustering exists
    """
    clustering_result = session.exec(
        select(ClusteringResult).where(
            ClusteringResult.embedding_model == embedding_model
        )
    ).first()

    if not clustering_result:
        return None

    return _build_cluster_details(clustering_result, session, limit)


def get_cluster_details_by_id(
    clustering_id: UUID, session: Session, limit: int = None
) -> Optional[Dict[str, any]]:
    """
    Get detailed cluster information for a specific clustering result by ID.

    Args:
        clustering_id: UUID of the clustering result
        session: Database session
        limit: Maximum number of clusters to return (None for all)

    Returns:
        Dictionary with cluster details or None if not found
    """
    clustering_result = session.get(ClusteringResult, clustering_id)
    if not clustering_result:
        return None

    return _build_cluster_details(clustering_result, session, limit)


def cluster_all_data(
    session: Session,
    downsample: int = 1,
    embedding_model_id: str = "all",
    epsilon: float = 0.4,
) -> Dict[str, any]:
    """
    Cluster embeddings in the database globally.

    This function loads embeddings across all experiments and performs
    clustering on the dataset, storing results with experiment_id=NULL.

    Args:
        session: Database session
        downsample: Downsampling factor (1 = no downsampling)
        embedding_model_id: Specific embedding model to cluster, or "all" for all models

    Returns:
        Dictionary with clustering results summary
    """
    scope = (
        "all embeddings"
        if embedding_model_id == "all"
        else f"model: {embedding_model_id}"
    )
    logger.info(f"Starting global clustering on {scope} in the database")

    try:
        # Build base query for counting embeddings
        count_query = (
            select(Embedding.embedding_model, func.count(Embedding.id).label("count"))
            .select_from(Embedding)
            .join(Invocation, Embedding.invocation_id == Invocation.id)
            .where(Invocation.type == InvocationType.TEXT)
        )

        # Apply downsampling and model filters
        if downsample > 1:
            count_query = count_query.where(
                ((Invocation.sequence_number - 1) / 2) % downsample == 0
            )
        if embedding_model_id != "all":
            count_query = count_query.where(
                Embedding.embedding_model == embedding_model_id
            )

        count_query = count_query.group_by(Embedding.embedding_model)
        model_counts = session.exec(count_query).all()

        if not model_counts:
            return {
                "status": "error",
                "message": "No embeddings found in the database",
                "total_embeddings": 0,
                "clustered_embeddings": 0,
                "total_clusters": 0,
                "embedding_models_count": 0,
            }

        # Get total embedding count
        total_query = select(func.count(Embedding.id))
        if embedding_model_id != "all":
            total_query = total_query.select_from(Embedding).where(
                Embedding.embedding_model == embedding_model_id
            )
        total_embeddings = session.exec(total_query).one()

        logger.info(f"Found {total_embeddings:,} total embeddings")
        logger.info(f"Embedding models: {[m[0] for m in model_counts]}")
        logger.info(f"Running clustering with downsample factor {downsample}")

        # Process each embedding model
        total_clusters = 0
        clustered_embeddings = 0

        for model_name, count in model_counts:
            logger.info(f"Model {model_name}: {count} embeddings after downsampling")

            # Skip models with insufficient samples
            if count < 2:
                logger.info(f"  Skipping {model_name} - too few samples for HDBSCAN")
                continue

            # Get embeddings for this model
            embeddings_data = session.exec(
                _get_embeddings_query(model_name, downsample)
            ).all()

            if not embeddings_data:
                logger.warning(f"  No embeddings found for model {model_name}")
                continue

            embedding_ids = [e[0] for e in embeddings_data]
            vectors = [e[1] for e in embeddings_data]
            texts = [e[2] for e in embeddings_data]

            # Import clustering function here to avoid circular imports
            from panic_tda.clustering import hdbscan
            import numpy as np

            # Convert vectors to numpy array for clustering
            vectors_array = np.array(vectors)

            # Perform clustering
            cluster_result = hdbscan(vectors_array, epsilon=epsilon)

            if cluster_result is None:
                logger.warning(f"  Clustering failed for model {model_name}")
                continue

            # Create clustering result record
            from datetime import datetime
            clustering_result = ClusteringResult(
                embedding_model=model_name,
                algorithm="hdbscan",
                started_at=datetime.utcnow(),
                parameters={
                    "cluster_selection_epsilon": epsilon,
                    "allow_single_cluster": True,
                },
                clusters=[],  # Will be populated below
            )
            session.add(clustering_result)
            session.flush()  # Get the ID

            # Build cluster info using medoid indices
            medoid_indices = cluster_result.get("medoid_indices", {})
            clusters_info = []

            for label, medoid_idx in medoid_indices.items():
                if label == OUTLIER_CLUSTER_ID:
                    continue  # Skip outliers in cluster info

                # Direct index lookup - no vector matching needed
                medoid_text = texts[medoid_idx]
                medoid_embedding_id = embedding_ids[medoid_idx]

                clusters_info.append({
                    "id": int(label),
                    "medoid_text": medoid_text,
                    "medoid_embedding_id": str(medoid_embedding_id),
                })

            # Update clustering result
            with session.no_autoflush:
                clustering_result.clusters = clusters_info

            # Create embedding cluster assignments
            assignments = [
                EmbeddingCluster(
                    embedding_id=embedding_id,
                    clustering_result_id=clustering_result.id,
                    cluster_id=int(cluster_label),
                )
                for embedding_id, cluster_label in zip(
                    embedding_ids, cluster_result["labels"]
                )
            ]

            _bulk_insert_with_flush(session, assignments)

            # Mark clustering as completed
            clustering_result.completed_at = datetime.utcnow()

            # Update counts
            clustered_embeddings += len(embedding_ids)
            unique_labels = set(cluster_result["labels"])
            has_outliers = OUTLIER_CLUSTER_ID in unique_labels
            total_clusters += len(clusters_info) + (1 if has_outliers else 0)

            logger.info(
                f"  Clustered {len(embedding_ids)} embeddings into {len(unique_labels)} clusters"
            )

        # Ensure all changes are flushed (commit handled by context manager)
        session.flush()

        logger.info(
            f"Global clustering complete: {clustered_embeddings:,}/{total_embeddings:,} embeddings "
            f"clustered into {total_clusters:,} total clusters"
        )

        return {
            "status": "success",
            "total_embeddings": total_embeddings,
            "clustered_embeddings": clustered_embeddings,
            "total_clusters": total_clusters,
            "embedding_models_count": len(model_counts),
        }

    except Exception as e:
        logger.error(f"Error during global clustering: {str(e)}")
        session.rollback()
        return {
            "status": "error",
            "message": str(e),
            "total_embeddings": 0,
            "clustered_embeddings": 0,
            "total_clusters": 0,
            "embedding_models_count": 0,
        }


def delete_cluster_data(
    session: Session, embedding_model_id: str = "all"
) -> Dict[str, any]:
    """
    Delete clustering results from the database.

    Args:
        session: Database session
        embedding_model_id: Specific embedding model to delete clusters for, or "all" for all models

    Returns:
        Dictionary with deletion results summary
    """
    try:
        # Build query for clustering results to delete
        query = select(ClusteringResult)
        if embedding_model_id != "all":
            query = query.where(ClusteringResult.embedding_model == embedding_model_id)

        results_to_delete = session.exec(query).all()

        if not results_to_delete:
            model_msg = (
                "all models"
                if embedding_model_id == "all"
                else f"model {embedding_model_id}"
            )
            return {
                "status": "not_found",
                "message": model_msg,
            }

        deleted_results = 0
        deleted_assignments = 0

        # Delete associated EmbeddingCluster entries first
        for result in results_to_delete:
            # Count and delete assignments
            assignment_count = session.exec(
                select(func.count(EmbeddingCluster.id)).where(
                    EmbeddingCluster.clustering_result_id == result.id
                )
            ).one()

            deleted_assignments += assignment_count

            # Delete all assignments for this result
            session.exec(
                select(EmbeddingCluster).where(
                    EmbeddingCluster.clustering_result_id == result.id
                )
            )
            for cluster in session.exec(
                select(EmbeddingCluster).where(
                    EmbeddingCluster.clustering_result_id == result.id
                )
            ).all():
                session.delete(cluster)

            # Delete the clustering result
            session.delete(result)
            deleted_results += 1

        # Flush deletions (commit handled by context manager)
        session.flush()

        model_msg = (
            "all models"
            if embedding_model_id == "all"
            else f"model {embedding_model_id}"
        )
        logger.info(
            f"Deleted {deleted_results} clustering result(s) and "
            f"{deleted_assignments} cluster assignments for {model_msg}"
        )

        return {
            "status": "success",
            "deleted_results": deleted_results,
            "deleted_assignments": deleted_assignments,
        }

    except Exception as e:
        logger.error(f"Error deleting cluster data: {str(e)}")
        session.rollback()
        return {
            "status": "error",
            "message": str(e),
            "deleted_results": 0,
            "deleted_assignments": 0,
        }


def delete_single_cluster(clustering_id: UUID, session: Session) -> Dict[str, any]:
    """
    Delete a single clustering result and all its associated data.

    Args:
        clustering_id: UUID of the clustering result to delete
        session: Database session

    Returns:
        Dictionary with deletion status and counts
    """
    try:
        result = session.get(ClusteringResult, clustering_id)
        if not result:
            return {
                "status": "not_found",
                "message": f"Clustering result with ID {clustering_id} not found",
            }

        # Count assignments
        assignments_count = session.exec(
            select(func.count(EmbeddingCluster.id)).where(
                EmbeddingCluster.clustering_result_id == result.id
            )
        ).one()

        # Delete assignments
        for cluster in session.exec(
            select(EmbeddingCluster).where(
                EmbeddingCluster.clustering_result_id == result.id
            )
        ).all():
            session.delete(cluster)

        # Delete the clustering result
        session.delete(result)

        # Flush deletion (commit handled by context manager)
        session.flush()

        logger.info(
            f"Deleted clustering result {clustering_id} and {assignments_count} assignments"
        )

        return {
            "status": "success",
            "deleted_results": 1,
            "deleted_assignments": assignments_count,
        }

    except Exception as e:
        logger.error(f"Error deleting clustering result {clustering_id}: {str(e)}")
        session.rollback()
        return {
            "status": "error",
            "message": str(e),
            "deleted_results": 0,
            "deleted_assignments": 0,
        }


def list_clustering_results(session: Session) -> List[ClusteringResult]:
    """
    List all clustering results in the database.

    Args:
        session: Database session

    Returns:
        List of ClusteringResult objects ordered by creation date
    """
    return session.exec(
        select(ClusteringResult).order_by(ClusteringResult.created_at.desc())
    ).all()


def get_clustering_result_by_id(
    clustering_id: UUID, session: Session
) -> Optional[ClusteringResult]:
    """
    Get a specific clustering result by its ID.

    Args:
        clustering_id: UUID of the clustering result
        session: Database session

    Returns:
        ClusteringResult object or None if not found
    """
    return session.get(ClusteringResult, clustering_id)


def get_latest_clustering_result(session: Session) -> Optional[ClusteringResult]:
    """
    Get the most recently created clustering result.

    Args:
        session: Database session

    Returns:
        ClusteringResult object or None if no clustering exists
    """
    return session.exec(
        select(ClusteringResult).order_by(ClusteringResult.created_at.desc())
    ).first()


def get_medoid_invocation(
    cluster_id: int, clustering_result_id: UUID, session: Session
) -> Optional[Invocation]:
    """
    Get the invocation that produced the medoid for a specific cluster.

    Args:
        cluster_id: The cluster ID
        clustering_result_id: UUID of the clustering result
        session: Database session

    Returns:
        Invocation object for the medoid
    """
    clustering_result = session.get(ClusteringResult, clustering_result_id)
    if not clustering_result:
        return None

    # Find cluster by ID
    cluster_info = next(
        (c for c in clustering_result.clusters if c["id"] == cluster_id), None
    )
    if not cluster_info:
        return None

    # Get the embedding and its invocation
    embedding = session.get(Embedding, UUID(cluster_info["medoid_embedding_id"]))
    return embedding.invocation if embedding else None
