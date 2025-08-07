---
id: task-46
title: Fix doctor command embedding null vector check and output format
status: In Progress
assignee: []
created_date: '2025-08-07 22:44'
labels: []
dependencies: []
---

## Description

The doctor command has two issues: 1) It incorrectly reports all embeddings as having null vectors because it uses SQL IS NULL on a BLOB field instead of checking the deserialized numpy array. 2) The embedding issues output doesn't include embedding_id which would be helpful for debugging.

## Problem Analysis

The doctor command in `src/panic_tda/doctor.py` has critical issues:

### 1. False positive null vector detection

The function `check_experiment_embeddings()` at line 324-333 uses:
```python
null_vectors = session.exec(
    select(func.count())
    .select_from(Embedding)
    .where(
        Embedding.invocation_id == invocation.id,
        Embedding.embedding_model == embedding_model,
        Embedding.vector.is_(None),
    )
).one()
```

This SQL query checks if the BLOB field in the database is NULL. However, the `NumpyArrayType` custom type (defined in `schemas.py`) stores numpy arrays as binary data (bytes). Valid numpy arrays are stored as non-NULL BLOB fields, so `vector.is_(None)` will return false positives.

Testing confirmed this: when retrieving NomicVision embeddings, they have valid numpy arrays with actual data (shape=(768,), dtype=float32, with min/max values), but the SQL IS NULL check reports 1,379,100 out of 3,245,100 embeddings as having NULL vectors - clearly incorrect.

### 2. Missing embedding_id in output

The `add_embedding_issue()` method doesn't include the embedding ID, only the invocation ID. This makes it harder to debug specific embedding issues. The embedding_id should be included for better traceability.

### 3. Unnecessary run/invocation IDs in output

The JSON output includes experiment_id and invocation_id for each embedding issue, but doesn't include the more useful embedding_id. For debugging, we need the embedding_id, and the run/invocation IDs can be looked up from the database if needed.

## Solution

1. Fix the null vector check to actually retrieve embeddings and check if the deserialized vector is None (not using SQL IS NULL)
2. Include embedding_id in the embedding issues output  
3. Consider removing redundant IDs from the output or making them optional

## Implementation Notes

### Changes Made

1. **Fixed null vector check in `check_experiment_embeddings()`** (src/panic_tda/doctor.py:324-345):
   - Changed from using SQL `IS NULL` check on the BLOB field to retrieving actual embeddings
   - Now correctly checks if the deserialized numpy array is None
   - This fixes the false positive issue where valid numpy arrays stored as non-NULL BLOBs were incorrectly reported as null

2. **Added embedding IDs to issues output**:
   - Added `embedding_ids` field to the embedding issue dictionary
   - This provides specific embedding IDs for debugging purposes

### Testing

- Manual testing with script() function confirmed:
  - The old SQL IS NULL method incorrectly reported millions of null vectors
  - The new method correctly identifies actual null vectors by checking the deserialized numpy arrays
  - Real null vectors do exist in the database (confirmed by checking specific IDs)
  - The doctor command now correctly reports these issues with embedding IDs included

### Note on Test Suite

The full test suite is currently failing due to an unrelated Ray package size issue (exceeds 512MB limit). This is not related to our changes to the doctor command.
