---
id: task-18
title: debug semantic drift (cosine) calculation
status: Done
assignee: []
created_date: "2025-07-23"
labels: []
dependencies: []
---

## Description

I want to:

- remove the cosine AND euclidean "semantic drift" calculations (in
  data_prep.py, but referenced in other places); replace it with a single one
  (that's cosine, but that doesn't need to be in the name anymore)
- comprehensively test the "add semantic drift" calculation, which takes an
  embeddings_df and
  - splits it by run + embedding_model
  - re-computes the embedding for the initial prompt
  - calculates the cosine distance from the initial prompt's embedding to each
    successive embedding in the run (well, the text ones anyway)
- for that calculation, it probably makes sense to use the "normalise, then use
  euclidean" trick that we use in clustering.py

The most important thing is that this is tested (ideally with known values).

In principle this functionality is already there, but I want to refactor it and
convince myself that it's correct.

## Implementation Notes

Completed the refactoring:

1. **Removed euclidean semantic drift**: Deleted `calculate_euclidean_distances`, `fetch_and_calculate_drift_euclid`, and `add_semantic_drift_euclid` functions.

2. **Unified implementation**: Created a single `calculate_semantic_drift` function that uses the normalize-then-euclidean approach:
   - Normalizes vectors to unit length
   - Calculates euclidean distance between normalized vectors
   - This is mathematically equivalent to cosine distance for normalized vectors
   - For unit vectors: euclidean_dist = sqrt(2 - 2*cos_similarity)

3. **Renamed functions**:
   - `add_semantic_drift_cosine` → `add_semantic_drift`
   - `drift_cosine` column → `semantic_drift` column
   - Updated all references across the codebase

4. **Added comprehensive tests**:
   - `test_calculate_semantic_drift`: Tests the core calculation with known cosine relationships
   - `test_semantic_drift_with_known_values`: Tests specific cases including identical, scaled, orthogonal, and opposite vectors
   - Verified that normalized euclidean distances match expected cosine distance values

5. **Updated all references**:
   - data_prep.py: Updated cache_dfs to use add_semantic_drift
   - datavis.py: Updated plot_semantic_drift to use semantic_drift column
   - local.py: Updated import (though function wasn't actually used)
   - test_datavis.py: Updated test data to use semantic_drift column

All tests pass successfully.
