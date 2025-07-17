"""
Module for managing clustering operations on embeddings.

This module provides functions to run clustering on all embeddings stored in the database,
monitor clustering progress, and retrieve clustering results.
"""

import logging
from collections import defaultdict
from typing import Dict, Optional
from uuid import UUID

from sqlmodel import Session, select, func

# Removed dataframe-based imports
from panic_tda.schemas import (
    ClusteringResult,
    EmbeddingCluster,
    Embedding,
    ExperimentConfig,
    Invocation,
    InvocationType,
    Run,
)

logger = logging.getLogger(__name__)


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
        # Query to get embedding counts per model with downsampling
        # Note: TEXT invocations have odd sequence numbers (1, 3, 5, ...)
        # We downsample by selecting every Nth TEXT invocation
        # Since we're dealing with odd numbers, we use: ((sequence_number - 1) / 2) % downsample == 0
        count_query = (
            select(Embedding.embedding_model, func.count(Embedding.id).label("count"))
            .select_from(Embedding)
            .join(Invocation, Embedding.invocation_id == Invocation.id)
            .where(Invocation.type == InvocationType.TEXT)
            .where(((Invocation.sequence_number - 1) / 2) % downsample == 0)
            .group_by(Embedding.embedding_model)
        )

        model_counts = session.exec(count_query).all()

        if not model_counts:
            logger.warning("No embeddings found in the database")
            return {
                "status": "error",
                "message": "No embeddings found in the database",
                "total_embeddings": 0,
                "clustered_embeddings": 0,
                "total_clusters": 0,
                "embedding_models_count": 0,
            }

        # Get total embedding count
        total_embeddings_query = select(func.count(Embedding.id))
        total_embeddings = session.exec(total_embeddings_query).one()
        logger.info(f"Found {total_embeddings:,} total embeddings")

        embedding_models = [m[0] for m in model_counts]
        logger.info(f"Embedding models: {embedding_models}")
        logger.info(f"Running global clustering with downsample factor {downsample}")

        # Process each embedding model
        total_clusters = 0
        clustered_embeddings = 0

        for model_name, count in model_counts:
            logger.info(f"Model {model_name}: {count} embeddings after downsampling")

            # Skip models with insufficient samples
            if count < 2:
                logger.info(f"  Skipping {model_name} - too few samples for HDBSCAN")
                continue

            # Query embeddings for this model with downsampling
            embeddings_query = (
                select(Embedding.id, Embedding.vector, Invocation.output_text)
                .select_from(Embedding)
                .join(Invocation, Embedding.invocation_id == Invocation.id)
                .where(Embedding.embedding_model == model_name)
                .where(Invocation.type == InvocationType.TEXT)
                .where(((Invocation.sequence_number - 1) / 2) % downsample == 0)
                .order_by(Invocation.sequence_number)
            )

            # Fetch embeddings and cluster them
            from panic_tda.data_prep import create_or_get_clustering_result
            from panic_tda.clustering import hdbscan
            import numpy as np

            # Get or create clustering result
            parameters = {
                "cluster_selection_epsilon": 0.6,
                "allow_single_cluster": True,
            }
            clustering_result = create_or_get_clustering_result(
                model_name, "hdbscan", parameters, session
            )

            # Fetch embeddings
            embeddings_data = session.exec(embeddings_query).all()
            embedding_ids = [e[0] for e in embeddings_data]
            vectors = [e[1] for e in embeddings_data]
            output_texts = [e[2] for e in embeddings_data]

            # Stack vectors and convert to float32 for memory efficiency
            vectors_array = np.vstack(vectors).astype(np.float32)
            cluster_result = hdbscan(vectors_array)

            # Create cluster information
            unique_labels = sorted(set(cluster_result["labels"]))
            clusters_info = []

            # Create mapping for medoid text (using float32 for consistency)
            vector_to_text = {}
            for i, (vec, text) in enumerate(zip(vectors, output_texts)):
                if text:
                    vector_to_text[tuple(vec.astype(np.float32).flatten())] = text

            for label in unique_labels:
                if label == -1:
                    continue  # Skip outliers in cluster info
                else:
                    # Get the medoid vector for this cluster
                    if label < len(cluster_result["medoids"]):
                        medoid_vector = cluster_result["medoids"][label]
                        medoid_key = tuple(medoid_vector.astype(np.float32).flatten())
                        medoid_text = vector_to_text.get(medoid_key, f"Cluster {label}")
                    else:
                        medoid_text = f"Cluster {label}"

                    clusters_info.append({
                        "id": int(label),
                        "medoid_text": medoid_text,
                    })

            # Update clustering result with clusters
            clustering_result.clusters = clusters_info

            # Create embedding cluster assignments
            for embedding_id, cluster_label in zip(
                embedding_ids, cluster_result["labels"]
            ):
                embedding_cluster = EmbeddingCluster(
                    embedding_id=embedding_id,
                    clustering_result_id=clustering_result.id,
                    cluster_id=int(cluster_label),
                )
                session.add(embedding_cluster)

            # Commit this model's clustering
            session.commit()

            # Update counts
            clustered_embeddings += len(embedding_ids)
            total_clusters += len(clusters_info) + (1 if -1 in unique_labels else 0)

            logger.info(
                f"  Clustered {len(embedding_ids)} embeddings into {len(unique_labels)} clusters"
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
