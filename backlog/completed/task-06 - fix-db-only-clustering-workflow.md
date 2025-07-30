---
id: task-06
title: fix db-only clustering workflow
status: Done
assignee: []
created_date: "2025-07-16"
labels: []
dependencies: []
---

## Description

Implement clustering that works directly with database queries, not dataframes.
The current implementation in `cluster_all_data` uses dataframes which causes
issues with downsampling logic.

## Desired Workflow

The clustering should work as follows:

1. **Query embeddings from database** with downsampling filter:

   - Use SQL query with
     `WHERE invocation.sequence_number % downsample_factor = 0`
   - Group by `embedding_model`
   - Join with invocation table to access sequence_number

2. **For each embedding_model group**:

   - Fetch the actual embedding vectors from the database
   - Stack vectors into numpy ndarray
   - Run HDBSCAN clustering on the vectors
   - Store the cluster assignments

3. **Save results to database**:
   - Create/update `ClusteringResult` objects
   - Create `EmbeddingCluster` objects for each embedding
   - Commit all changes atomically

## Implementation Details

### Key Issues Found During Investigation

1. **Downsampling bug**: The current implementation applies downsampling
   globally before grouping by model, causing models with few embeddings (like
   "Dummy" with only 25 total) to get insufficient samples for HDBSCAN.

2. **Minimum samples**: HDBSCAN requires at least 2 samples to run. Need to
   check sample count after downsampling and skip models with < 2 samples.

3. **Embedding distribution** (from testing):
   - Total embeddings: 3,288,025
   - Models: STSBMpnet, STSBRoberta, Dummy, Nomic
   - "Dummy" model has very few embeddings (~25)

### Proposed Implementation

1. **Update `cluster_all_data` in clustering_manager.py**:

   ```python
   def cluster_all_data(session: Session, downsample: int = 1, force: bool = False):
       # Query to get embedding counts per model with downsampling
       count_query = """
       SELECT
           embedding.embedding_model,
           COUNT(*) as count
       FROM embedding
       JOIN invocation ON embedding.invocation_id = invocation.id
       WHERE invocation.sequence_number % :downsample = 0
       GROUP BY embedding.embedding_model
       """

       # For each model with sufficient samples:
       embeddings_query = """
       SELECT
           embedding.id,
           embedding.vector,
           invocation.output_text
       FROM embedding
       JOIN invocation ON embedding.invocation_id = invocation.id
       WHERE embedding.embedding_model = :model_name
         AND invocation.sequence_number % :downsample = 0
       ORDER BY invocation.sequence_number
       """
   ```

2. **Handle the clustering results properly**:

   - Don't use dataframe joins
   - Create EmbeddingCluster records directly from the clustering results
   - Ensure atomic commits

3. **Add proper logging**:
   - Log sample counts per model before and after downsampling
   - Log when models are skipped due to insufficient samples
   - Log clustering progress

## Testing Strategy

1. Test with downsample=10000 (should work, ~329 samples)
2. Test with downsample=1000 (should skip Dummy model, cluster others)
3. Test with downsample=100 (should cluster all models with sufficient samples)
4. Test with downsample=10 (monitor for performance/hanging)

## Files to Modify

- `src/panic_tda/clustering_manager.py` - Main changes to `cluster_all_data()`
- Remove any dataframe-based clustering from `data_prep.py`
- Ensure `perform-clustering.sh` continues to work with the new implementation

## Implementation Notes

### Issue Discovered

The main issue was that TEXT invocations (which have embeddings) are only on odd
sequence numbers (1, 3, 5, ...), while IMAGE invocations are on even sequence
numbers (0, 2, 4, ...). When using `sequence_number % downsample == 0`, we were
only matching IMAGE invocations, which have no embeddings.

### Solution Implemented

1. **Updated clustering queries** to:

   - Filter for TEXT invocations explicitly using
     `Invocation.type == InvocationType.TEXT`
   - Adjust downsampling logic to work with odd sequence numbers:
     `((sequence_number - 1) / 2) % downsample == 0`
   - This transforms the odd sequence (1, 3, 5, 7...) to (0, 1, 2, 3...) for
     proper downsampling

2. **Removed dataframe dependencies**:

   - Removed imports of `load_embeddings_df` and `add_cluster_labels` from
     `data_prep.py`
   - Implemented direct database queries using SQLModel

3. **Added proper logging**:
   - Log sample counts per model after downsampling
   - Log when models are skipped due to insufficient samples
   - Log clustering progress for each model

### Testing Results

- Tested with downsample=10: Successfully clustered embeddings
- Tested with downsample=100: Successfully clustered 42,480 embeddings into 71
  clusters
- Tested with downsample=1000: Successfully clustered 7,728 embeddings into 287
  clusters
- Dummy model correctly skipped when it has < 2 samples after downsampling
