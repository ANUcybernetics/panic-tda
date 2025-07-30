---
id: task-24
title: Fix UUID type error when deleting clustering results
status: Done
assignee: []
created_date: "2025-07-26"
labels: []
dependencies: []
---

## Description

SQLAlchemy is encountering float values where UUID objects are expected during
cascade deletion of clustering results

## Problem Details

When attempting to delete clustering results using
`panic-tda cluster reset --force` or `panic-tda cluster delete <id>`, the
operation fails with:

```
AttributeError: 'float' object has no attribute 'replace'
```

This occurs in the UUID parsing code when SQLAlchemy tries to load related
records during cascade deletion.

### Full Error Traceback

```
File "/usr/lib/python3.12/uuid.py", line 175, in __init__
    hex = hex.replace('urn:', '').replace('uuid:', '')
AttributeError: 'float' object has no attribute 'replace'
```

The error occurs during lazy loading of relationships when SQLAlchemy tries to
cascade delete related records.

## Root Cause Analysis

The issue appears to be that somewhere in the clustering process, non-UUID
values (floats or integers) are being stored in UUID fields in the database.
This could happen in several places:

1. **medoid_embedding_id in Cluster table**: Despite validation, invalid values
   might still be inserted
2. **cluster_id in EmbeddingCluster table**: Should reference Cluster.id (UUID)
   but might contain integer cluster labels
3. **embedding_id in EmbeddingCluster table**: Should reference Embedding.id
   (UUID)

## Investigation Steps Taken

1. Added comprehensive unit tests to validate UUID types during clustering
2. Added bounds checking for medoid indices in clustering_manager.py
3. Added UUID type validation before inserting records
4. Tests pass but the issue persists in production data

## Potential Solutions

### 1. Direct SQL Cleanup (Immediate Fix)

Create a script to identify and remove invalid records before deletion:

```python
# Identify rows with non-text UUID fields
SELECT * FROM cluster WHERE typeof(medoid_embedding_id) != 'text' AND medoid_embedding_id IS NOT NULL;
SELECT * FROM embeddingcluster WHERE typeof(cluster_id) != 'text';
SELECT * FROM embeddingcluster WHERE typeof(embedding_id) != 'text';
```

However, this fix is NOT SUFFICIENT.

### 2. Bypass Cascade Deletion (Workaround)

Delete records manually in the correct order without relying on SQLAlchemy
cascades:

```python
# Delete in reverse dependency order
session.exec(text("DELETE FROM embeddingcluster"))
session.exec(text("DELETE FROM cluster"))
session.exec(text("DELETE FROM clusteringresult"))
```

### 3. Fix Data Insertion (Long-term Solution)

Despite existing validation, somewhere the wrong data types are being inserted.
Potential areas:

- Check if SQLite is coercing types during bulk inserts
- Verify the \_bulk_insert_with_flush function preserves types correctly
- Add database-level constraints or CHECK constraints to prevent invalid data

### 4. Add Type Decorators (Defensive)

Create custom SQLAlchemy type decorators that enforce UUID validation at the
database level:

```python
class StrictUUID(TypeDecorator):
    impl = UUID
    def process_bind_param(self, value, dialect):
        if value is not None and not isinstance(value, UUID):
            raise ValueError(f"Expected UUID, got {type(value)}")
        return value
```

## Recommended Action Plan

Option 3 is the best plan. Option 4 shouldn't be necessary - we let the db
itself enforce FK validity.

## Related Code Locations

- `/src/panic_tda/clustering_manager.py`: Lines 412-446 (cluster creation)
- `/src/panic_tda/clustering_manager.py`: Lines 449-471 (embedding assignments)
- `/src/panic_tda/schemas.py`: Cluster and EmbeddingCluster model definitions
- `/src/panic_tda/main.py`: Lines 802-906 (delete commands)
