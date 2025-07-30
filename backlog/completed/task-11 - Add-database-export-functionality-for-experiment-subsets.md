---
id: task-11
title: Add database export functionality for experiment subsets
status: Done
assignee: []
created_date: '2025-07-21'
labels: []
dependencies: []
---

## Description

Add a function to db.py that exports a subset of the database containing specific experiments with all their related data, maintaining referential integrity

## Specification

### Function Signature
```python
def export_experiments(
    source_db_str: str,
    target_db_str: str, 
    experiment_ids: List[str],
    timeout: int = 30
) -> None:
```

### Requirements
1. Export all data for specified experiments across all tables
2. Maintain referential integrity (no FK constraint violations)
3. Create target database with same schema if it doesn't exist
4. Include all related data:
   - ExperimentConfig records for given IDs
   - All Run records for those experiments
   - All Invocation records for those runs
   - All Embedding records for those invocations
   - All PersistenceDiagram records for those runs

### Implementation Approach
1. Create target database and tables if needed
2. Query and copy data in correct order to respect FK constraints:
   - ExperimentConfig first
   - Run records (depends on ExperimentConfig)
   - Invocation records (depends on Run)
   - Embedding records (depends on Invocation)
   - PersistenceDiagram records (depends on Run)

## Notes

Implementation completed. Added `export_experiments()` function to db.py that:
- Takes source DB, target DB, and list of experiment IDs
- Creates target database and schema if needed  
- Copies all related data maintaining FK constraints
- Handles data in correct order: ExperimentConfig → Run → Invocation → Embedding/PersistenceDiagram

Added test `test_export_experiments` to tests/test_db.py that comprehensively tests the functionality including:
- Exporting a single experiment with all its related data
- Exporting multiple experiments
- Verifying referential integrity is maintained
- Testing error handling for non-existent experiments

Usage example:
```python
from panic_tda.db import export_experiments

export_experiments(
    "sqlite:///source.db",
    "sqlite:///target.db", 
    ["experiment-uuid-1", "experiment-uuid-2"]
)
