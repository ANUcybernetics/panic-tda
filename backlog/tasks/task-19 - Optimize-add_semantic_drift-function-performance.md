---
id: task-19
title: Optimize add_semantic_drift function performance
status: In Progress
assignee: []
created_date: '2025-07-24'
labels: []
dependencies: []
---

## Description

The add_semantic_drift function in src/panic_tda/data_prep.py is very slow. Need to optimize by: 1) Batch fetching embedding vectors from DB, 2) Using vectorized numpy operations for cosine distance calculations, 3) Caching initial prompt embeddings since there are only a few unique prompts

## Current Implementation Analysis

The current implementation (lines 638-665) has several performance issues:

1. **Sequential DB fetches**: The `fetch_and_calculate_drift` function fetches embeddings one by one from the database (line 624-626)
2. **Inefficient grouping**: The function uses `map_batches` but still processes embeddings sequentially within each batch
3. **Redundant initial prompt embeddings**: Each batch re-embeds the same initial prompts even though there are only a few unique ones

## Optimization Strategy

1. **Batch fetch all embeddings for a run**: Instead of fetching one by one, fetch all embeddings for a run in a single DB query
2. **Vectorized operations**: Use numpy broadcasting to calculate all cosine distances at once
3. **Cache initial prompt embeddings**: The `embed_initial_prompts` function already generates all unique initial prompt embeddings - we should leverage this better
4. **Group by run_id for efficient processing**: Process all embeddings within a run together to minimize DB queries

## Implementation Complete

### Changes Made:

1. **Created `batch_fetch_embeddings` function**: Fetches all embeddings for a list of IDs in a single database query using SQLModel's `in_` operator
2. **Created `calculate_semantic_drift_batch` function**: Processes all embeddings from a single run together, using vectorized numpy operations
3. **Refactored `add_semantic_drift` function**: Now groups by `run_id` and `embedding_model` to process entire runs at once
4. **Maintained compatibility**: The semantic drift values are still cached via parquet when `cache_dfs` is called

### Performance Improvements:

- **Reduced DB queries**: From N queries (one per embedding) to 1 query per run
- **Vectorized calculations**: All cosine distances for a run are calculated in a single numpy operation
- **Better caching**: Initial prompt embeddings are computed once and reused across all runs

### Testing:

All tests pass successfully:
- `test_add_semantic_drift`: Verifies semantic drift column is added correctly
- `test_calculate_semantic_drift`: Tests the cosine distance calculation
- `test_semantic_drift_with_known_values`: Tests edge cases
- All other data_prep tests continue to pass

The optimization maintains backward compatibility while significantly improving performance for large datasets.
