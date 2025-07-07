"""
Module for managing clustering operations on embeddings.

This module provides functions to run clustering on embeddings stored in the database,
monitor clustering progress, and retrieve clustering results.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from uuid import UUID

import polars as pl
from sqlmodel import Session, select
from tqdm import tqdm

from panic_tda.data_prep import add_cluster_labels, load_embeddings_df
from panic_tda.schemas import ClusteringResult, EmbeddingCluster, Embedding, ExperimentConfig, Invocation, Run

logger = logging.getLogger(__name__)


def get_clustering_status(session: Session) -> Dict[UUID, Dict[str, any]]:
    """
    Get clustering status for all experiments in the database.
    
    Returns:
        Dictionary mapping experiment IDs to their clustering status information
    """
    status = {}
    
    # Get all experiments
    experiments = session.exec(select(ExperimentConfig)).all()
    
    for exp in experiments:
        # Get embedding count for this experiment
        embeddings = session.exec(
            select(Embedding)
            .join(Invocation)
            .join(Run)
            .where(Run.experiment_id == exp.id)
        ).all()
        embedding_count = len(embeddings)
        
        # Get clustering results
        clustering_results = session.exec(
            select(ClusteringResult)
            .where(ClusteringResult.experiment_id == exp.id)
        ).all()
        
        # Count clustered embeddings
        clustered_count = 0
        models_with_clusters = []
        total_clusters = 0
        clustering_details = []
        
        for cr in clustering_results:
            # Count assignments for this clustering result
            assignments = session.exec(
                select(EmbeddingCluster)
                .where(EmbeddingCluster.clustering_result_id == cr.id)
            ).all()
            assignment_count = len(assignments)
            
            clustered_count += assignment_count
            models_with_clusters.append(cr.embedding_model)
            total_clusters += len(cr.clusters)
            
            clustering_details.append({
                "id": cr.id,
                "embedding_model": cr.embedding_model,
                "algorithm": cr.algorithm,
                "parameters": cr.parameters,
                "cluster_count": len(cr.clusters),
                "assignment_count": assignment_count,
                "created_at": cr.created_at
            })
        
        # Remove duplicates from clustered count
        if len(models_with_clusters) > 0:
            clustered_count = clustered_count // len(set(models_with_clusters))
        
        status[exp.id] = {
            "experiment_id": exp.id,
            "run_count": len(exp.runs) if exp.runs else 0,
            "prompt_count": len(exp.prompts) if exp.prompts else 0,
            "network_count": len(exp.networks) if exp.networks else 0,
            "embedding_count": embedding_count,
            "clustered_count": clustered_count,
            "clustering_models": sorted(set(models_with_clusters)),
            "total_clusters": total_clusters,
            "clustering_details": clustering_details,
            "is_fully_clustered": clustered_count >= embedding_count if embedding_count > 0 else True
        }
    
    return status


def get_experiments_without_clustering(session: Session) -> List[ExperimentConfig]:
    """Get all experiments that don't have clustering results yet."""
    all_experiments = session.exec(select(ExperimentConfig)).all()
    
    experiments_without_clustering = []
    for exp in all_experiments:
        clustering_results = session.exec(
            select(ClusteringResult).where(ClusteringResult.experiment_id == exp.id)
        ).first()
        
        if not clustering_results:
            experiments_without_clustering.append(exp)
    
    return experiments_without_clustering


def cluster_experiment(
    experiment_id: UUID,
    session: Session,
    downsample: int = 1,
    force: bool = False
) -> Dict[str, any]:
    """
    Cluster all embeddings for a given experiment.
    
    Args:
        experiment_id: The experiment to process
        session: Database session
        downsample: Downsampling factor (1 = no downsampling)
        force: If True, re-cluster even if clustering already exists
        
    Returns:
        Dictionary with clustering results summary
    """
    logger.info(f"Processing experiment {experiment_id}")
    
    # Check if clustering already exists
    if not force:
        existing = session.exec(
            select(ClusteringResult)
            .where(ClusteringResult.experiment_id == experiment_id)
        ).first()
        
        if existing:
            logger.info(f"Experiment {experiment_id} already has clustering results. Use force=True to re-cluster.")
            return {
                "status": "already_clustered",
                "experiment_id": experiment_id,
                "message": "Experiment already has clustering results"
            }
    else:
        # Delete existing clustering results if forcing
        for cr in session.exec(
            select(ClusteringResult)
            .where(ClusteringResult.experiment_id == experiment_id)
        ).all():
            session.delete(cr)
        session.commit()
        logger.info(f"Deleted existing clustering results for experiment {experiment_id}")
    
    # Load all embeddings
    all_embeddings_df = load_embeddings_df(session)
    
    # Filter to just this experiment
    exp_embeddings = all_embeddings_df.filter(
        pl.col("experiment_id") == str(experiment_id)
    )
    
    if len(exp_embeddings) == 0:
        logger.warning(f"No embeddings found for experiment {experiment_id}")
        return {
            "status": "no_embeddings",
            "experiment_id": experiment_id,
            "message": "No embeddings found for this experiment"
        }
    
    logger.info(f"Found {len(exp_embeddings)} embeddings for experiment {experiment_id}")
    
    # Get unique embedding models
    embedding_models = exp_embeddings["embedding_model"].unique().to_list()
    logger.info(f"Embedding models: {embedding_models}")
    
    # Add cluster labels
    logger.info(f"Running clustering with downsample={downsample}...")
    clustered_df = add_cluster_labels(exp_embeddings, downsample=downsample, session=session)
    
    # Gather statistics
    non_null_clusters = clustered_df.filter(pl.col("cluster_label").is_not_null())
    clustered_count = len(non_null_clusters)
    
    # Get cluster distribution
    cluster_distribution = (
        non_null_clusters
        .group_by(["embedding_model", "cluster_label"])
        .agg(pl.len().alias("count"))
        .sort(["embedding_model", "count"], descending=[False, True])
    )
    
    return {
        "status": "success",
        "experiment_id": experiment_id,
        "total_embeddings": len(exp_embeddings),
        "clustered_embeddings": clustered_count,
        "embedding_models": embedding_models,
        "downsample_factor": downsample,
        "cluster_distribution": cluster_distribution.to_dicts()
    }


def cluster_all_experiments(
    session: Session,
    downsample: int = 1,
    force: bool = False
) -> List[Dict[str, any]]:
    """
    Cluster embeddings for all experiments without clustering.
    
    Args:
        session: Database session
        downsample: Downsampling factor (1 = no downsampling)
        force: If True, re-cluster all experiments
        
    Returns:
        List of clustering results for each experiment
    """
    if force:
        experiments = session.exec(select(ExperimentConfig)).all()
    else:
        experiments = get_experiments_without_clustering(session)
    
    if not experiments:
        logger.info("All experiments already have clustering results!")
        return []
    
    logger.info(f"Found {len(experiments)} experiments to cluster")
    
    results = []
    for exp in tqdm(experiments, desc="Clustering experiments"):
        try:
            result = cluster_experiment(exp.id, session, downsample, force)
            results.append(result)
        except Exception as e:
            logger.error(f"Error clustering experiment {exp.id}: {e}")
            results.append({
                "status": "error",
                "experiment_id": exp.id,
                "error": str(e)
            })
    
    return results


def get_cluster_details(
    experiment_id: UUID,
    embedding_model: str,
    session: Session
) -> Optional[Dict[str, any]]:
    """
    Get detailed information about clusters for a specific experiment and embedding model.
    
    Args:
        experiment_id: The experiment ID
        embedding_model: The embedding model name
        session: Database session
        
    Returns:
        Dictionary with cluster details or None if not found
    """
    # Get clustering result
    clustering_result = session.exec(
        select(ClusteringResult)
        .where(ClusteringResult.experiment_id == experiment_id)
        .where(ClusteringResult.embedding_model == embedding_model)
    ).first()
    
    if not clustering_result:
        return None
    
    # Get cluster sizes
    cluster_sizes = defaultdict(int)
    assignments = session.exec(
        select(EmbeddingCluster)
        .where(EmbeddingCluster.clustering_result_id == clustering_result.id)
    ).all()
    
    for assignment in assignments:
        cluster_sizes[assignment.cluster_id] += 1
    
    # Build cluster details
    clusters = []
    for cluster_info in clustering_result.clusters:
        cluster_id = cluster_info["id"]
        clusters.append({
            "id": cluster_id,
            "medoid_text": cluster_info["medoid_text"],
            "size": cluster_sizes.get(cluster_id, 0)
        })
    
    # Add outliers if any
    if -1 in cluster_sizes:
        clusters.append({
            "id": -1,
            "medoid_text": "OUTLIER",
            "size": cluster_sizes[-1]
        })
    
    # Sort by size
    clusters.sort(key=lambda x: x["size"], reverse=True)
    
    return {
        "experiment_id": experiment_id,
        "embedding_model": embedding_model,
        "algorithm": clustering_result.algorithm,
        "parameters": clustering_result.parameters,
        "created_at": clustering_result.created_at,
        "total_clusters": len(clusters),
        "total_assignments": len(assignments),
        "clusters": clusters
    }