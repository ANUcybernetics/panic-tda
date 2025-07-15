---
id: task-02
title: use cosine distance for clustering
status: In Progress
assignee: []
created_date: "2025-07-15"
labels: []
dependencies: []
---

## Description

The clustering in @src/panic_tda/clustering.py (hdbscan is the one I'm currently
using, although there's an optics function in there as well) is done by sklearn.
Currently it uses the euclidean distance, but we should switch to cosine
distance. Sklearn provides this functionality somehow, but we may need to look
at the docs to see the best way to do it.

It's crucial that any tests are updated as well (e.g.
@tests/test_clustering.py).

## Implementation Notes

### Research Findings
- sklearn's HDBSCAN supports the `metric` parameter which accepts any metric supported by `pairwise_distances`
- However, sklearn.cluster.HDBSCAN does NOT support cosine distance directly (only sklearn-contrib/hdbscan does)
- Must use `metric="precomputed"` with a precomputed cosine distance matrix

### Changes Made
1. Updated `hdbscan()` function in src/panic_tda/clustering.py to:
   - Import `cosine_distances` from sklearn
   - Compute cosine distance matrix before clustering
   - Use `metric="precomputed"` with the distance matrix
   - Manually compute medoids since precomputed distances don't support `store_centers="medoid"`
   - Adjusted `cluster_selection_epsilon` from 0.6 to 0.3 (cosine distance ranges 0-2 vs euclidean)

### Test Updates
The existing tests were designed for euclidean distance and needed updates:
- Test data originally used magnitude-based clusters ([-10,-10,...], [10,10,...], [0,0,...])
- Updated tests to use direction-based clusters:
  - Cluster 1: First half dims positive, second half negative
  - Cluster 2: First half dims negative, second half positive (opposite of cluster 1)
  - Cluster 3: Alternating positive/negative pattern
- All vectors are normalized to unit length for stable cosine distance
- All tests now pass successfully

The clustering function is used on embedding vectors in data_prep.py, where cosine distance is appropriate since embeddings typically care more about semantic direction than magnitude.
