---
id: task-29
title: >-
  Refactor clustering schema to remove deprecated code and add medoid embeddings
  FK
status: To Do
assignee: []
created_date: '2025-07-28'
labels: [database, refactoring, clustering]
dependencies: []
---

## Description

Refactor the database schema for clustering results to simplify the structure, remove all deprecated code, and properly store medoid embeddings with foreign keys to the Embedding table.

## Current State

The clustering schema in `src/panic_tda/schemas.py` has deprecated code:
- `ClusteringResult.clusters` field (line 611-615) is marked as DEPRECATED
- The comment says it's "kept for migration compatibility"
- Current schema uses `Cluster` and `EmbeddingCluster` tables for proper relational structure
- `Cluster` table already has `medoid_embedding_id` FK field (line 663-668)

## Requirements

1. **Remove deprecated code**:
   - Remove the `clusters` field from `ClusteringResult` (lines 611-615)
   - No data migration needed - user is okay with dropping existing cluster data

2. **Verify medoid embedding storage**:
   - `Cluster.medoid_embedding_id` already exists as an FK to `Embedding` table
   - Ensure clustering implementation properly sets this field when creating clusters

3. **Schema simplification**:
   - Review if any other simplifications can be made to the clustering schema
   - Ensure all relationships are properly defined

4. **Update tests**:
   - Update any tests that reference the deprecated `clusters` field
   - Ensure tests verify medoid embeddings are properly stored

## Implementation Plan

1. Remove deprecated `clusters` field from `ClusteringResult`
2. Search codebase for any references to `ClusteringResult.clusters`
3. Update clustering implementation to ensure `medoid_embedding_id` is properly set
4. Update or add tests to verify:
   - Clustering results are stored correctly
   - Medoid embeddings are properly linked
   - No references to deprecated fields remain
5. Drop and recreate database tables (user is okay with data loss)

## Success Criteria

- All deprecated code removed from schemas.py
- Clustering results properly store medoid embedding references
- All tests pass
- No references to deprecated fields in codebase
