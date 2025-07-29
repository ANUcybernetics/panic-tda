---
id: task-34
title: unique constraint error on embeddingcluster.id during clustering
status: To Do
assignee: []
created_date: "2025-07-29"
labels: []
dependencies: []
---

## Description

Clustering fails intermittently with UNIQUE constraint failed:
embeddingcluster.id error

## Problem Details

When running clustering on embeddings, the process sometimes fails with:

```
ERROR - Error saving clustering results: (sqlite3.IntegrityError) UNIQUE constraint failed: embeddingcluster.id
```

This is different from the original issue in task-33 which was about the unique
constraint on `(embedding_id, clustering_result_id)`.

## Observed Behavior

1. The error occurs during the bulk insert of EmbeddingCluster records
2. It's intermittent - in one clustering run:
   - Nomic model: FAILED with this error
   - STSBMpnet model: SUCCEEDED
   - STSBRoberta model: FAILED with this error
3. The IDs that conflict all share similar timestamp prefixes (e.g.,
   `06888680-50ce-7...`), suggesting they were generated in rapid succession
4. The clustering result that would have contained these records doesn't exist
   in the database (transaction was rolled back)

## Investigation Results

1. **UUID Generation**: The EmbeddingCluster model uses
   `id: UUID = Field(default_factory=uuid7, primary_key=True)` which should
   auto-generate unique IDs
2. **No Manual ID Setting**: The code doesn't manually set IDs - just creates
   objects like:
   ```python
   assignment = EmbeddingCluster(
       embedding_id=embedding_id,
       clustering_result_id=clustering_result.id,
       cluster_id=cluster_uuid,
   )
   ```
3. **Batch Processing**: Objects are created in a loop, appended to a list, then
   bulk inserted with periodic flushes
4. **UUID7 Testing**: Direct testing shows uuid7 generates unique IDs correctly

## Hypothesis

The issue might be related to:

1. SQLAlchemy/SQLModel's handling of default_factory when objects are created in
   rapid succession
2. Some SQLite-specific behavior with UUID primary keys
3. Possible issue with uuid7 library when called many times in quick succession
   within SQLAlchemy's context
4. Objects being reused after a failed transaction (though the code doesn't seem
   to do this)

## Example Error

From logs/clustering_2025-07-29_15-52-54.log:

```
2025-07-29 16:19:51,154 - ERROR - Error saving clustering results: (sqlite3.IntegrityError) UNIQUE constraint failed: embeddingcluster.id
[SQL: INSERT INTO embeddingcluster (id, embedding_id, clustering_result_id, cluster_id) VALUES (?, ?, ?, ?)]
[parameters: [('0688868050ce750f978308d76d36c9f5', '067efbb192227a168a0e8982cbd84bcb', '068886804d957d13ba8a6bcdcb9a8dcd', '068886804db87214af448cb79ee8dde6'), ...]]
```

## Proposed Investigation Approach

To systematically debug this issue:

1. **Find a reproducible test case**:
   - Identify a combination of embedding model, epsilon, and downsample factor that deterministically triggers the bug
   - Use the highest possible downsample factor that still reproduces the issue (for faster feedback during testing)
   - Document the exact parameters and data that trigger the failure

2. **Add comprehensive checks to pinpoint the duplicate source**:
   - Check for duplicates in the initial "get embeddings to cluster" query results
   - Track UUID generation throughout the clustering process
   - Verify uniqueness in the clustering algorithm outputs
   - Monitor the batched write process for any ID reuse or duplication
   - Log all generated IDs before attempting database insertion

3. **Narrow down the root cause**:
   - Determine if duplicates originate from:
     - The embedding retrieval stage
     - The clustering algorithm itself
     - The batch processing/writing logic
     - UUID generation timing issues
     - SQLAlchemy/SQLModel object handling

## Workarounds to Consider

1. Use database-generated IDs instead of application-generated UUIDs
2. Switch from uuid7 to uuid4 (fully random instead of time-based)
3. Add explicit ID generation with collision checking
4. Process inserts one at a time instead of bulk
5. Add retry logic with fresh object creation on failure
