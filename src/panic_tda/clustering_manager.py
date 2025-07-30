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

    # Get unique clusters by medoid_embedding_id with their sizes
    clusters_query = (
        select(
            EmbeddingCluster.medoid_embedding_id,
            func.count(EmbeddingCluster.id).label("size")
        )
        .where(EmbeddingCluster.clustering_result_id == clustering_result.id)
        .group_by(EmbeddingCluster.medoid_embedding_id)
        .order_by(func.count(EmbeddingCluster.id).desc())
    )

    if limit:
        clusters_query = clusters_query.limit(limit)

    cluster_data = session.exec(clusters_query).all()

    # Build cluster details
    clusters = []
    cluster_id = 0  # For display purposes, assign sequential IDs
    
    for medoid_embedding_id, size in cluster_data:
        if medoid_embedding_id is None:
            # This is the outlier cluster
            clusters.append({
                "id": OUTLIER_CLUSTER_ID,
                "medoid_text": OUTLIER_LABEL,
                "size": size,
            })
        else:
            # Get the medoid embedding details
            medoid_embedding = session.get(Embedding, medoid_embedding_id)
            
            cluster_info = {
                "id": cluster_id,
                "medoid_text": "",  # Will be populated below
                "medoid_embedding_id": str(medoid_embedding_id),
                "size": size,
            }
            
            # Get medoid text from the invocation
            if medoid_embedding and medoid_embedding.invocation:
                if medoid_embedding.invocation.output_text:
                    cluster_info["medoid_text"] = medoid_embedding.invocation.output_text
                cluster_info["medoid_invocation_id"] = str(medoid_embedding.invocation.id)
                cluster_info["medoid_run_id"] = str(medoid_embedding.invocation.run_id)
            
            clusters.append(cluster_info)
            cluster_id += 1

    # Count regular clusters (excluding outliers)
    regular_cluster_count = len([c for c in clusters if c["id"] != OUTLIER_CLUSTER_ID])
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
            # Add objects and flush
            for item in batch:
                session.add(item)
            session.flush()
            batch = []

    # Insert remaining objects
    if batch:
        for item in batch:
            session.add(item)
        session.flush()


def _get_embeddings_query(model_name: str, downsample: int = 1):
    """
    Build a query for embeddings with optional downsampling.
    """
    query = (
        select(Embedding.id, Embedding.vector, Invocation.output_text, Invocation.sequence_number)
        .select_from(Embedding)
        .join(Invocation, Embedding.invocation_id == Invocation.id)
        .where(Embedding.embedding_model == model_name)
        .where(Invocation.type == InvocationType.TEXT)
    )

    if downsample > 1:
        # Apply downsampling filter
        query = query.where(((Invocation.sequence_number - 1) / 2) % downsample == 0)

    return query.distinct().order_by(Invocation.sequence_number)


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


def _save_clustering_results(
    session: Session,
    model_name: str,
    embedding_ids: List[UUID],
    cluster_result: dict,
    texts: List[str],
    downsample: int,
    epsilon: float,
) -> Dict[str, any]:
    """
    Save clustering results to the database.

    This is a helper function extracted from cluster_all_data to enable unit testing.

    Args:
        session: Database session
        model_name: Name of the embedding model
        embedding_ids: List of embedding UUIDs
        cluster_result: Dict with 'labels', 'medoids', and 'medoid_indices'
        texts: List of text strings corresponding to embeddings
        downsample: Downsampling factor used
        epsilon: Epsilon value used for clustering

    Returns:
        Dictionary with status and cluster count
    """
    from datetime import datetime

    start_time = datetime.utcnow()

    try:
        # Create clustering result record
        clustering_result = ClusteringResult(
            embedding_model=model_name,
            algorithm="hdbscan",
            started_at=start_time,
            completed_at=datetime.utcnow(),
            parameters={
                "cluster_selection_epsilon": epsilon,
                "allow_single_cluster": True,
            },
        )
        session.add(clustering_result)
        session.flush()  # Get the ID

        # Build cluster info using medoid indices
        medoid_indices = cluster_result.get("medoid_indices", {})
        
        # Map numeric cluster labels to medoid embedding IDs
        cluster_medoid_map = {}  # Maps numeric cluster_label to medoid_embedding_id
        
        # Handle outliers (they have no medoid)
        unique_labels = set(cluster_result["labels"])
        if OUTLIER_CLUSTER_ID in unique_labels:
            cluster_medoid_map[OUTLIER_CLUSTER_ID] = None
            
        # Map regular clusters to their medoid embedding IDs
        for label, medoid_idx in medoid_indices.items():
            if label == OUTLIER_CLUSTER_ID:
                continue  # Already handled above
                
            # Validate medoid index bounds
            if medoid_idx < 0 or medoid_idx >= len(embedding_ids):
                logger.error(
                    f"Invalid medoid index {medoid_idx} for cluster {label} (out of bounds for {len(embedding_ids)} embeddings)"
                )
                continue
                
            # Direct index lookup - no vector matching needed
            medoid_embedding_id = embedding_ids[medoid_idx]
            
            # Validate medoid_embedding_id is a proper UUID
            if not isinstance(medoid_embedding_id, UUID):
                logger.error(
                    f"Invalid medoid_embedding_id type for cluster {label}: {type(medoid_embedding_id)}, value: {medoid_embedding_id}"
                )
                continue
                
            cluster_medoid_map[label] = medoid_embedding_id

        # Create embedding cluster assignments using medoid embedding IDs
        assignments = []
        seen_embeddings = set()  # Track which embeddings we've already assigned
        
        for embedding_id, cluster_label in zip(embedding_ids, cluster_result["labels"]):
            # Ensure all IDs are proper UUID objects
            if not isinstance(embedding_id, UUID):
                logger.error(
                    f"Invalid embedding_id type: {type(embedding_id)}, value: {embedding_id}"
                )
                continue
                
            # Skip if we've already seen this embedding
            if embedding_id in seen_embeddings:
                logger.warning(f"Duplicate embedding_id {embedding_id} found, skipping")
                continue
            seen_embeddings.add(embedding_id)
            
            # Get the medoid embedding ID for this cluster
            medoid_embedding_id = cluster_medoid_map.get(int(cluster_label))
            
            # Skip if we don't have a mapping for this label
            if int(cluster_label) not in cluster_medoid_map:
                logger.error(
                    f"No medoid mapping found for cluster label {cluster_label}"
                )
                continue
                
            if not isinstance(clustering_result.id, UUID):
                logger.error(
                    f"Invalid clustering_result.id type: {type(clustering_result.id)}"
                )
                continue
                
            # Create the assignment with validated UUIDs
            assignment = EmbeddingCluster(
                embedding_id=embedding_id,
                clustering_result_id=clustering_result.id,
                medoid_embedding_id=medoid_embedding_id,  # None for outliers
            )
            assignments.append(assignment)

        # Bulk insert assignments
        _bulk_insert_with_flush(session, assignments)

        # Commit the entire write phase
        session.commit()

        # Count unique clusters (including outliers)
        unique_clusters = len(set(cluster_medoid_map.values()))
        
        return {
            "status": "success",
            "clusters_created": unique_clusters,
        }

    except Exception as e:
        logger.error(f"Error saving clustering results: {str(e)}")
        session.rollback()
        return {
            "status": "error",
            "message": str(e),
            "clusters_created": 0,
        }


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

        # Import necessary modules
        from panic_tda.clustering import hdbscan
        import numpy as np
        from datetime import datetime

        # Process each embedding model
        total_clusters = 0
        clustered_embeddings = 0

        for model_name, count in model_counts:
            logger.info(f"Model {model_name}: {count} embeddings after downsampling")

            # Skip models with insufficient samples
            if count < 2:
                logger.info(f"  Skipping {model_name} - too few samples for HDBSCAN")
                continue

            # PHASE 1: LOAD (read-only)
            logger.info(f"  Phase 1: Loading embeddings for {model_name}...")
            embeddings_data = session.exec(
                _get_embeddings_query(model_name, downsample)
            ).all()

            if not embeddings_data:
                logger.warning(f"  No embeddings found for model {model_name}")
                continue

            # Extract and validate data
            embedding_ids = []
            vectors = []
            texts = []

            for e in embeddings_data:
                embedding_id, vector, text, seq_num = e

                # Validate embedding_id - it should be a UUID object or convertible to one
                if not isinstance(embedding_id, UUID):
                    if isinstance(embedding_id, str) and len(embedding_id) == 36:
                        try:
                            embedding_id = UUID(embedding_id)
                        except ValueError:
                            logger.warning(
                                f"  Skipping invalid embedding ID: {embedding_id} (not a valid UUID)"
                            )
                            continue
                    else:
                        logger.warning(
                            f"  Skipping invalid embedding ID: {embedding_id} (type: {type(embedding_id)})"
                        )
                        continue

                embedding_ids.append(embedding_id)
                vectors.append(vector)
                texts.append(text)

            if not embedding_ids:
                logger.warning(f"  No valid embeddings found for model {model_name}")
                continue

            # Convert vectors to numpy array for clustering
            vectors_array = np.array(vectors)
            logger.info(f"  Loaded {len(embedding_ids)} embeddings")

            # PHASE 2: COMPUTE (no DB access)
            logger.info("  Phase 2: Running HDBSCAN clustering...")
            start_time = datetime.utcnow()

            # Perform clustering
            cluster_result = hdbscan(vectors_array, epsilon=epsilon)

            if cluster_result is None:
                logger.warning(f"  Clustering failed for model {model_name}")
                continue

            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"  Clustering completed in {duration:.1f} seconds")

            # PHASE 3: LOOKUP (read-only, if needed)
            # In this implementation, we already have all the data we need from Phase 1
            # (embedding IDs and texts), so no additional lookups are required

            # PHASE 4: WRITE (single transaction)
            logger.info("  Phase 4: Writing clustering results...")

            # Use the extracted save function
            save_result = _save_clustering_results(
                session=session,
                model_name=model_name,
                embedding_ids=embedding_ids,
                cluster_result=cluster_result,
                texts=texts,
                downsample=downsample,
                epsilon=epsilon,
            )

            if save_result["status"] != "success":
                logger.warning(
                    f"  Failed to save clustering results for {model_name}: {save_result.get('message', 'Unknown error')}"
                )
                continue

            logger.info(f"  Successfully saved clustering results for {model_name}")

            # Update counts
            clustered_embeddings += len(embedding_ids)
            total_clusters += save_result.get("clusters_created", 0)

            logger.info(
                f"  Clustered {len(embedding_ids)} embeddings into {save_result.get('clusters_created', 0)} clusters"
            )

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
        from sqlmodel import delete

        # Count clustering results before deletion
        count_query = select(func.count(ClusteringResult.id))
        if embedding_model_id != "all":
            count_query = count_query.where(
                ClusteringResult.embedding_model == embedding_model_id
            )
        deleted_results = session.exec(count_query).one()

        if deleted_results == 0:
            model_msg = (
                "all models"
                if embedding_model_id == "all"
                else f"model {embedding_model_id}"
            )
            return {
                "status": "not_found",
                "message": f"No clustering results found for {model_msg}",
            }

        # Count total assignments before deletion
        assignment_count_query = select(func.count(EmbeddingCluster.id)).join(
            ClusteringResult,
            EmbeddingCluster.clustering_result_id == ClusteringResult.id,
        )
        if embedding_model_id != "all":
            assignment_count_query = assignment_count_query.where(
                ClusteringResult.embedding_model == embedding_model_id
            )
        total_assignments = session.exec(assignment_count_query).one()

        # Get clustering results to delete
        if embedding_model_id == "all":
            results_to_delete = session.exec(select(ClusteringResult)).all()
        else:
            results_to_delete = session.exec(
                select(ClusteringResult).where(
                    ClusteringResult.embedding_model == embedding_model_id
                )
            ).all()

        # Delete in reverse dependency order
        for result in results_to_delete:
            # Delete embedding cluster assignments using a delete statement
            delete_stmt = delete(EmbeddingCluster).where(
                EmbeddingCluster.clustering_result_id == result.id
            )
            session.exec(delete_stmt)

            # No need to delete clusters anymore - they're gone!

            # Delete the clustering result itself
            session.delete(result)

        # Commit changes
        session.commit()

        model_msg = (
            "all models"
            if embedding_model_id == "all"
            else f"model {embedding_model_id}"
        )
        logger.info(
            f"Deleted {deleted_results} clustering result(s) and "
            f"{total_assignments} cluster assignments for {model_msg}"
        )

        return {
            "status": "success",
            "deleted_results": deleted_results,
            "deleted_assignments": total_assignments,
        }

    except Exception as e:
        import traceback

        logger.error(f"Error deleting cluster data: {str(e)}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
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
        from sqlmodel import delete

        # Get the clustering result
        result = session.get(ClusteringResult, clustering_id)
        if not result:
            return {
                "status": "not_found",
                "message": f"Clustering result with ID {clustering_id} not found",
            }

        # Count assignments before deletion
        assignments_count = session.exec(
            select(func.count(EmbeddingCluster.id)).where(
                EmbeddingCluster.clustering_result_id == clustering_id
            )
        ).one()

        # Delete embedding cluster assignments using a delete statement
        delete_stmt = delete(EmbeddingCluster).where(
            EmbeddingCluster.clustering_result_id == result.id
        )
        session.exec(delete_stmt)

        # No need to delete clusters anymore - they're gone!

        # Delete the clustering result itself
        session.delete(result)

        # Commit changes
        session.commit()

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
        cluster_id: The cluster ID (display ID, not medoid embedding ID)
        clustering_result_id: UUID of the clustering result
        session: Database session

    Returns:
        Invocation object for the medoid
    """
    # For the new schema, we need to look up clusters by their position in the sorted list
    # Get all clusters sorted by size (descending) to match the display order
    clusters_query = (
        select(
            EmbeddingCluster.medoid_embedding_id,
            func.count(EmbeddingCluster.id).label("size")
        )
        .where(EmbeddingCluster.clustering_result_id == clustering_result_id)
        .where(EmbeddingCluster.medoid_embedding_id.is_not(None))  # Exclude outliers
        .group_by(EmbeddingCluster.medoid_embedding_id)
        .order_by(func.count(EmbeddingCluster.id).desc())
    )
    
    clusters = session.exec(clusters_query).all()
    
    # Find the cluster at the requested position
    if cluster_id < 0 or cluster_id >= len(clusters):
        return None
        
    medoid_embedding_id = clusters[cluster_id][0]
    
    # Get the medoid embedding and its invocation
    medoid_embedding = session.get(Embedding, medoid_embedding_id)
    if not medoid_embedding:
        return None
        
    return medoid_embedding.invocation
