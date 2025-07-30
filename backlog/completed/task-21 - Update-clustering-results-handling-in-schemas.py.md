---
id: task-21
title: Update clustering results handling in schemas.py
status: Done
assignee: []
created_date: "2025-07-25"
labels: []
dependencies: []
---

## Description

Improve how clustering results are stored and managed, including FK references
to medoid Embedding and other best practices

## Problem Statement

Currently, there's messy and sometimes lossy work to keep track of what cluster
labels map to what cluster medoids in several places:

- In the database (ClusteringResult stores clusters as JSON)
- In the "cluster df" in data_prep.py
- In code that uses these (e.g., datavis.py)

This causes issues when trying to access the medoid's associated Invocation
text/image and maintain data integrity.

Additional challenges:

- Many embeddings don't get clustered (downsampling factor of 10 is common)
- Outlier clusters (stored as integer -1) don't have associated medoid
  Embeddings

## Success Criteria

1. **Direct FK references**: Ability to directly reference medoid Embeddings
   from cluster records
2. **Cleaner data access**: Eliminate the need for complex JSON parsing and
   mapping in data_prep.py
3. **Better querying**: Ability to easily query:
   - All embeddings that are medoids across different clusterings
   - Get medoid text/image directly through relationships
   - Find all clusters and their properties without JSON extraction
4. **Handle edge cases properly**:
   - Outlier clusters (-1) that have no medoid
   - Embeddings that weren't included in clustering (due to downsampling)

## Technical Approach

### Option 1: Separate Cluster Table

Create a new `Cluster` table with:

- `id`: Primary key
- `clustering_result_id`: FK to ClusteringResult
- `cluster_id`: The numeric cluster identifier (including -1 for outliers)
- `medoid_embedding_id`: Optional FK to Embedding (null for outliers)
- `size`: Number of members in the cluster
- `properties`: JSON field for additional metadata (e.g., density, stability)

Benefits:

- Clean relational structure
- Direct FK to medoid Embedding
- Can store per-cluster metadata
- EmbeddingCluster can reference Cluster.id instead of just cluster_id

### Option 2: Enhance Current Structure

Keep current structure but add:

- `medoid_embeddings` relationship table mapping ClusteringResult to Embedding
- Better indexing on the JSON clusters field
- Helper methods for cleaner access

Benefits:

- Less migration work
- Maintains current API

### Recommended: Option 1

The separate Cluster table provides better normalization and cleaner access
patterns.

## Implementation Steps

1. **Schema Changes**:

   - Create new `Cluster` model
   - Update `EmbeddingCluster` to reference `Cluster` instead of just cluster_id
   - Add proper indexes and constraints

2. **Migration**:

   - Script to migrate existing JSON cluster data to new structure
   - Handle existing outlier clusters appropriately

3. **Update Data Access**:

   - Simplify `load_clusters_df()` to use proper joins instead of JSON parsing
   - Update visualization code to use cleaner relationships

4. **Testing**:
   - Ensure all existing functionality works
   - Verify performance improvements
   - Test edge cases (outliers, non-clustered embeddings)

## Out of Scope

- Changing clustering algorithms themselves
- Modifying how embeddings are stored
- Changing the relationship between runs and clustering results
- Backwards compatibility (okay to delete existing clustering data)
