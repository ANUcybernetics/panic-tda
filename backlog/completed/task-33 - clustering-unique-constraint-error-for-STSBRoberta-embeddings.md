---
id: task-33
title: clustering unique constraint error for STSBRoberta embeddings
status: Done
assignee: []
created_date: "2025-07-29"
labels: []
dependencies: []
---

## Description

There's a persistent error that happens when clustering the STSBRoberta
embeddings. Here's an example log file for one of the failed runs
@logs/clustering_2025-07-29_10-36-36.log (there are many like it, and it's
always the STSBRoberta model that triggers the failure).

What's the specific error that's being caused? Is it because the clustering is
returning the same embedding ID as medoid for multiple different clusters - and
if so why is this happening? Or if not, what's the issue and how do we fix it?

## Investigation

The error is a UNIQUE constraint violation on the EmbeddingCluster table, which
has a unique constraint on (embedding_id, clustering_result_id). This means the
same embedding is being assigned to a clustering result multiple times.

## Root Causes Identified

1. **Query Issue**: The `_get_embeddings_query` function wasn't including
   `sequence_number` in the SELECT, which could cause issues with the DISTINCT
   and ORDER BY clauses working together properly.

2. **Pre-existing Data**: If a clustering result already exists for a model and
   clustering is run again, it could cause conflicts.

## Fixes Applied

1. **Updated Query**: Added `Invocation.sequence_number` to the SELECT clause in
   `_get_embeddings_query` to ensure the DISTINCT operation works correctly with
   the ORDER BY.

2. **Added Duplicate Detection**: Enhanced logging to detect and report any
   duplicate embedding IDs in the query results.

3. **Clean Up Existing Results**: Added code to check for and delete any
   existing clustering results for a model before creating new ones, preventing
   conflicts from pre-existing data.

4. **Improved Error Handling**: The existing duplicate detection in
   `_save_clustering_results` now logs warnings when duplicates are found.

## Testing Results

- Verified no duplicate embeddings exist in the database
- Successfully ran clustering on STSBRoberta with downsample factor 100 (14,160
  embeddings) - completed in 53.9 seconds
- The fix ensures that any existing clustering results are deleted before
  creating new ones, preventing unique constraint violations
- Updated unit tests to reflect the new behavior where clustering results are
  replaced rather than accumulated

## Conclusion

After further analysis based on user feedback, the initial "fix" of deleting
existing clustering results was incorrect. The design intentionally allows
multiple ClusteringResult records for the same embedding model with different
parameters.

The actual fix was simpler:

1. Removed the unnecessary `.distinct()` from the query - since we're selecting
   by embedding ID which is already unique, distinct was not needed
2. Kept the duplicate detection logging for debugging purposes

The unique constraint `(embedding_id, clustering_result_id)` is correct and
ensures that within a single clustering result, each embedding is only assigned
to one cluster. Multiple clustering results with different parameters are
allowed and expected.
