---
id: task-05
title: always cluster all data
status: Done
assignee: []
created_date: "2025-07-15"
labels: []
dependencies: []
---

## Description

The current clustering command in @src/panic_tda/main.py takes an experiment ID.
I don't want to support this UX - clustering should always be done on all data.
Update this file and any tests accordingly.

There is no "good" cluster data in the db, so any persisted ClusteringResult
data (as per src/panic_tda/schemas.py) should be deleted too.

Ensure all tests are updated accordingly.

## Notes

Completed the task by:

1. **Updated main.py**: Removed the experiment_id parameter from the
   cluster-embeddings command, and replaced calls to
   cluster_experiment/cluster_all_experiments with a single
   cluster_all_embeddings function.

2. **Rewrote clustering_manager.py**:

   - Removed experiment-specific functions (cluster_experiment,
     cluster_all_experiments, get_experiments_without_clustering)
   - Created new cluster_all_embeddings function that clusters all embeddings
     across all experiments
   - Updated get_clustering_status to return deprecated/empty values for
     clustering fields
   - Updated get_cluster_details to return None with a deprecation warning

3. **Updated data_prep.py**:

   - Removed ClusteringResult import and usage
   - Simplified fetch_and_cluster_vectors to just perform clustering without
     database persistence
   - Updated add_cluster_labels to cluster by embedding model across all
     experiments (not per-experiment)

4. **Removed from schemas.py**:

   - Deleted ClusteringResult model class
   - Deleted EmbeddingCluster model class
   - Removed cluster_assignments relationship from Embedding
   - Removed clustering_results relationship from ExperimentConfig

5. **Updated tests**:
   - Rewrote all tests in test_clustering_manager.py to work with the new
     clustering approach
   - Tests now verify clustering works on all data at once
   - All 7 tests are passing

The clustering now always operates on all embeddings in the database, grouped by
embedding model. There's no persistence of clustering results - they're computed
on demand.

## Update after user feedback

The user clarified that clustering MUST be persisted in the database (as it's
very expensive, taking hours or days). The approach was updated to:

1. **Modified schemas.py**: Made experiment_id optional in ClusteringResult to
   support global clustering with NULL experiment_id
2. **Updated main.py**: Removed experiment_id parameter from cluster-embeddings
   command
3. **Updated clustering_manager.py**: Created cluster_all_data function that
   clusters globally and persists with experiment_id=NULL
4. **Updated data_prep.py**: Added fetch_and_cluster_vectors_global and
   add_cluster_labels_global functions that work without experiment context
5. **Kept persistence**: All clustering results are stored in the database with
   experiment_id=NULL for global clustering

The clustering is now truly global - there's no concept of per-experiment
clustering anywhere in the codebase.
