---
id: task-40
title: add missing db constraints
status: To Do
assignee: []
created_date: "2025-08-05 05:04"
labels: []
dependencies: []
---

## Description

From a data model perspective (i.e. in @src/panic_tda/schemas.py) the following
things should **always** be true:

- every invocation should have an associated run
- every run should have an associated experiment
- in a given run, there should only be one invocation with a given
  sequence_number
- every embedding should have an associated invocation
- for a given invocation + embedding_model there should only be one embedding
- for a given run + embedding_model there should only be one persistence diagram

Looking at the sqlite db schema, are any of those constraints not enforced at a
db level?

## Analysis

Analyzed the current database schema and found that the following constraints
were **already present**:

- ✅ invocation -> run foreign key
- ✅ run -> experimentconfig foreign key
- ✅ embedding -> invocation foreign key
- ✅ unique(invocation_id, embedding_model) on embedding table

The following constraints were **missing**:

- ❌ unique(run_id, sequence_number) on invocation table
- ❌ unique(run_id, embedding_model) on persistencediagram table

## Solution

Created alembic migration `caedeefdeae7_add_missing_db_constraints.py` that adds
the two missing unique constraints:

1. `unique_run_sequence_number` on invocation table columns (run_id,
   sequence_number)
2. `unique_run_embedding_model` on persistencediagram table columns (run_id,
   embedding_model)

To apply the migration:

```bash
uv run alembic upgrade head
```
