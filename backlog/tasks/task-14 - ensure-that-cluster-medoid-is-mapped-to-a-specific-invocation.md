---
id: task-14
title: ensure that cluster medoid is mapped to a specific invocation
status: Done
assignee: []
created_date: "2025-07-22"
labels: []
dependencies: []
---

## Description

Currently, the clustering process returns the medoids (because sklearn is called
with the relevant argument set to True). However, finding the specific Embedding
(and therefore the Invocation) that generated the medoid is not straightforward.
This task aims to ensure that the cluster medoid is mapped to a specific
invocation, making it easier to identify the source of the medoid.

What's the best solution here? Can a relevant foreign key be added to the
ClusteringResult schema (or something similar)? And is there a nicer way to do
this than the "un-normalise and find the matching embedding" process (e.g. by
using the index back into the ndarray passed to the clustering algorithm) to
retrieve the specific Invocation? Or does it not matter, because medoid is based
on the text embedding, but multiple image Invocations may have that text output
(and therefore the same embedding)?

## Analysis

After analyzing the current implementation:

1. **Current approach**: The clustering process returns medoid indices from HDBSCAN, which are then matched back to embeddings by finding the closest normalized vector. The medoid text is stored but the specific embedding ID is lost.

2. **The issue**: While we store `medoid_text` in the `clusters` JSON field, we don't store which specific `Embedding` (and therefore which `Invocation`) was selected as the medoid.

3. **Why it matters**: Even though multiple invocations might produce the same text (and therefore same embedding), it would be valuable to know which specific invocation was chosen as the medoid for reproducibility and deeper analysis.

## Proposed Solution

The best approach is to store the medoid embedding ID in the cluster information. This requires:

1. Modifying the clustering process to track which embedding index corresponds to each medoid
2. Storing the embedding ID alongside the medoid text in the `clusters` JSON field
3. Updating the cluster details retrieval to include invocation information

This approach:
- Maintains backward compatibility (existing `medoid_text` field remains)
- Provides full traceability from cluster → medoid embedding → invocation
- Enables richer analysis (e.g., viewing the actual image that produced the medoid text)

## Implementation

The solution has been implemented with the following changes:

1. **Modified clustering process** (`clustering_manager.py:314-344`):
   - Added `vector_to_embedding_id` mapping to track embedding IDs
   - Store `medoid_embedding_id` in the cluster info JSON

2. **Enhanced cluster details retrieval** (`clustering_manager.py:83-101`):
   - `_build_cluster_details` now includes `medoid_embedding_id` in cluster info
   - Also attempts to include `medoid_invocation_id` and `medoid_run_id` for full traceability

3. **Added helper function** (`clustering_manager.py:610-658`):
   - `get_medoid_invocation()` function to retrieve the full invocation object for a cluster's medoid

## Benefits

- **Full traceability**: Can now trace from cluster → medoid embedding → invocation → run
- **Backward compatible**: Existing code continues to work with `medoid_text`
- **Richer analysis**: Can retrieve the actual image/prompt that generated the medoid text
- **Minimal overhead**: Only stores an additional UUID per cluster

## Usage Example

```python
# Get the invocation that produced a cluster's medoid
medoid_invocation = get_medoid_invocation(
    cluster_id=5,
    clustering_result_id=clustering_result.id,
    session=session
)

if medoid_invocation:
    print(f"Medoid image path: {medoid_invocation.output_path}")
    print(f"Medoid prompt: {medoid_invocation.input_text}")
```

## Final Implementation

All backward compatibility has been removed as requested:

1. **Deleted all existing clustering data** to avoid legacy data issues
2. **Made medoid_embedding_id required** - clustering will error if medoid cannot be mapped
3. **Removed all fallback code** - no more optional checks for medoid_embedding_id
4. **Added test coverage** to ensure medoid_embedding_id is always present

The implementation is now cleaner and guarantees full traceability from cluster to invocation.

## Further Simplifications Made

1. **Enhanced clustering function** to return medoid indices directly:
   - `hdbscan()` now returns `medoid_indices` dict mapping cluster_id → embedding_index
   - Eliminates fragile floating-point vector matching

2. **Simplified medoid mapping** in `cluster_all_data()`:
   - Direct index lookup: `medoid_text = texts[medoid_idx]`
   - No more vector-to-text/id dictionaries
   - No more float32 tuple keys

3. **Streamlined `get_medoid_invocation()`**:
   - Reduced from 25 lines to 8 lines
   - Uses list comprehension and direct session.get()
   - Cleaner one-liner return

4. **Simplified `_build_cluster_details()`**:
   - Direct access to cluster data without .get() fallbacks
   - Cleaner structure with less nesting

The code is now much simpler, more maintainable, and less error-prone by eliminating the complex vector matching approach.
