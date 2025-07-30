---
id: task-20
title: Investigate duplicate rows in clusters dataframe
status: Done
assignee: []
created_date: "2025-07-24"
labels: []
dependencies: []
---

## Description

The clusters.parquet file contains duplicate rows (10,960 duplicate key
combinations found for embedding_id, clustering_result_id, epsilon). Need to
trace through the data pipeline from load_clusters_df() to understand where
duplicates are introduced. The embeddingcluster table itself has no duplicates,
so the issue must be in the SQL join or post-processing steps.

## Investigation Findings

After thorough investigation, I discovered that:

1. **The embeddingcluster table DOES contain duplicates** - 32,880 duplicate
   combinations of (embedding_id, clustering_result_id) were found
2. **Root cause**: The `EmbeddingCluster` table lacks a unique constraint on the
   combination of `embedding_id` and `clustering_result_id`
3. **How duplicates occur**: The same embedding can be assigned to the same
   clustering result multiple times, creating duplicate entries

### Evidence:

- Database query confirmed 32,880 duplicate (embedding_id, clustering_result_id)
  combinations in the embeddingcluster table
- Example: embedding_id `067efb8691a97ae6b0a815590dcfa857` appears twice for
  clustering_result_id `06880c9a677f7724ba6eaf94074b198d`, both times with
  cluster_id = -1
- The clusters.parquet file accurately reflects these duplicates from the
  database

### Impact:

- The duplicates inflate cluster sizes and assignment counts
- Analysis results may be skewed by counting the same embedding multiple times
  per cluster
- The issue is in the database layer, not in the data loading pipeline

## Proposed Solution

1. **Add a unique constraint** to the `EmbeddingCluster` table:

   ```python
   # In schemas.py, add to the EmbeddingCluster class:
   __table_args__ = (
       UniqueConstraint("embedding_id", "clustering_result_id",
                       name="unique_embedding_clustering"),
   )
   ```

2. **Clean up existing duplicates** in the database:

   - Identify all duplicate entries
   - Keep only one entry per (embedding_id, clustering_result_id) combination
   - Delete the redundant entries

3. **Update clustering code** to handle constraint violations gracefully:

   - Check for existing assignments before inserting
   - Or use INSERT ... ON CONFLICT DO NOTHING pattern

4. **Regenerate the cache** after cleaning up the database
