# Task 22: Refactor cluster embeddings command for long-running operations

**Status**: "In Progress" **Created**: 2025-07-26

## Problem

The current clustering implementation holds a database transaction open for the
entire clustering operation, which can take 5+ hours for large datasets. This
causes SQLite transaction timeouts (default 5 minutes), resulting in lost work
where clustering completes but results fail to persist.

## Current Implementation Issues

1. `cluster_all_data()` processes all embedding models in a single transaction
2. Each model's clustering can take 1.5-2+ hours for large datasets
3. SQLite times out after 5 minutes, rolling back all work
4. The temporary fix (commit after each model) still keeps transactions open
   during the long HDBSCAN computation

## Proposed Solution

Refactor the `cluster embeddings` command to separate read-only operations from
write operations:

### For each embedding model:

1. **Load Phase** (read-only)

   - Query all embeddings for the model from the database
   - Load embedding vectors into a single numpy ndarray
   - Store embedding IDs in a parallel array for later reference

2. **Compute Phase** (no DB access)

   - Run HDBSCAN clustering on the ndarray (in `src/panic_tda/clustering.py`)
   - This is the time-consuming part (hours) but requires no DB access
   - Returns cluster labels and medoid indices

3. **Lookup Phase** (read-only, if needed)

   - Look up Embedding objects for medoids using stored embedding IDs
   - Gather any other required reference data

4. **Write Phase** (single transaction)
   - Create ClusteringResult object
   - Create all Cluster objects with medoid references
   - Create all EmbeddingCluster assignment objects
   - Commit everything in a single, fast transaction
   - Use bulk inserts for EmbeddingCluster objects

## Implementation Details

- Keep the existing `hdbscan()` function in `clustering.py` as-is
- Modify `cluster_all_data()` in `clustering_manager.py` to follow the new
  pattern
- No DB reads or writes during the compute phase
- All writes happen in a short transaction at the end
- Use `_bulk_insert_with_flush()` for EmbeddingCluster assignments

## Benefits

- No transaction timeouts regardless of clustering duration
- Clean separation of compute from I/O
- Progress is saved after each model completes
- Failed models don't affect completed ones
- Simpler code flow

## Notes

- No need for backwards compatibility
- Plan to drop all existing clustering data and re-run
- Keep the implementation simple and straightforward

## Progress

2025-07-26: Refactored `cluster_all_data()` function to implement the 4-phase
approach:

- Phase 1: Load embeddings (read-only)
- Phase 2: Compute clustering with HDBSCAN (no DB access)
- Phase 3: Lookup phase (not needed - we already have all data from Phase 1)
- Phase 4: Write all results in a single fast transaction

Key improvements:

- No DB access during the long-running HDBSCAN computation
- Added clearer logging for each phase
- Transaction is only held open during the fast write phase
- Commit happens immediately after writing each model's results
