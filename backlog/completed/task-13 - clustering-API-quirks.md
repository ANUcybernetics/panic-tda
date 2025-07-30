---
id: task-13
title: clustering API quirks
status: Done
assignee: []
created_date: "2025-07-21"
labels: []
dependencies: []
---

## Description

The "clustering" API (in @src/panic_tda/main.py) has some quirks compared to
e.g. the "experiments" stuff.

- there's no `list-clusters` command (analogous to `list-experiments`)
- there's no `delete-cluster` command (analogous to `delete-experiment`); there
  is a `delete-clusters` which drops all of them, but I think it should be
  `delete-cluster` which drops a single cluster (and perhaps an `all` command to
  drop all clusters---with confirmation, of course)
- cluster-details should be `cluster-status` and take a ClusteringResult ID

Does this sound like an improvement to the overall ergonomics of the CLI? What
changes would be required to the codebase and/or test suite?

## Analysis

Yes, these changes would improve the ergonomics and consistency of the CLI. Here's what would be needed:

### 1. Add `list-clusters` command
- Lists all ClusteringResult entries in the database
- Shows ID, embedding model, algorithm, creation date, and cluster count
- Supports --verbose flag for detailed output

### 2. Rename `delete-clusters` to `delete-cluster` with single ID
- Takes a ClusteringResult ID as argument
- Deletes a single clustering result and its assignments
- Add support for "all" to delete all clusters (with confirmation)
- Keep the --force/-f flag to skip confirmation

### 3. Rename `cluster-details` to `cluster-status`
- Takes a ClusteringResult ID instead of embedding model name
- Shows the same detailed information about clusters
- Supports "latest" to show the most recent clustering result

### Required changes:
1. Add helper functions in clustering_manager.py:
   - `list_clustering_results()` - returns all ClusteringResult entries
   - `get_clustering_result_by_id()` - fetch by UUID
   - `delete_single_cluster()` - delete one ClusteringResult
   - Modify `get_cluster_details()` to accept ClusteringResult ID

2. Update main.py commands to match the new API

3. Update tests if any exist for these commands

## Implementation Plan

1. Add helper functions to clustering_manager.py
2. Implement list-clusters command
3. Refactor delete-clusters to delete-cluster
4. Refactor cluster-details to cluster-status
5. Test all commands

## Completed Implementation

All changes have been successfully implemented:

1. **Added helper functions to clustering_manager.py**:
   - `list_clustering_results()` - Lists all ClusteringResult entries
   - `get_clustering_result_by_id()` - Fetches by UUID
   - `get_latest_clustering_result()` - Gets most recent clustering
   - `delete_single_cluster()` - Deletes one ClusteringResult
   - `get_cluster_details_by_id()` - Modified version that accepts UUID

2. **Added `list-clusters` command**:
   - Lists all clustering results with ID, model, algorithm, clusters, assignments
   - Supports --verbose flag for detailed output

3. **Renamed `delete-clusters` to `delete-cluster`**:
   - Takes a ClusteringResult ID as argument
   - Supports "all" to delete all clusters (with confirmation)
   - Shows clustering details before deletion
   - Keeps the --force/-f flag to skip confirmation

4. **Renamed `cluster-details` to `cluster-status`**:
   - Takes a ClusteringResult ID instead of embedding model name
   - Supports "latest" to show the most recent clustering result
   - Shows the same detailed information about clusters
   - Maintains the outlier percentage display added earlier

The clustering API is now consistent with the experiments API pattern.
