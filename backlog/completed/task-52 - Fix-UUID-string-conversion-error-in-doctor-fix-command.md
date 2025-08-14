---
id: task-52
title: Fix UUID string conversion error in doctor --fix command
status: Done
assignee: []
created_date: '2025-08-13 05:50'
labels: []
dependencies: []
---

## Description

The doctor --fix command crashes with AttributeError: 'str' object has no attribute 'hex' when processing persistence diagram issues. The root cause is unnecessary string conversion of UUID values that are later used in database queries.

## Error Details

When running `panic-tda doctor --fix --yes` on the production database, the command crashes with:

```
AttributeError: 'str' object has no attribute 'hex'
```

The full traceback shows the error occurs when SQLAlchemy tries to process a UUID parameter:

- Error location: `/home/ben/Code/panic_tda/src/panic_tda/doctor.py:850`
- The query `session.exec(select(Run).where(Run.id == run_id)).first()` fails
- SQLAlchemy's UUID bind processor expects a UUID object but receives a string

## Root Cause Analysis

In `fix_persistence_diagrams()` function:

1. At line 827, run_id is converted to string: `str(issue["run_id"])`
2. This string is added to `pd_pairs_to_recompute` set
3. At line 850, the string run_id is used directly in a SQLModel query
4. SQLModel/SQLAlchemy expects a UUID object, not a string
5. The UUID type's bind processor tries to call `.hex` on the string, causing the crash

The issue dictionary contains actual UUID objects (from `run.id` at lines 445 and 477 in `check_experiment_persistence_diagrams`), so the string conversion is unnecessary and causes the type mismatch.

## Fix Required

Remove the unnecessary string conversion in `src/panic_tda/doctor.py`:

Line 827, change from:
```python
pd_pairs_to_recompute.add((
    str(issue["run_id"]),  # This converts UUID to string
    issue["embedding_model"],
))
```

To:
```python
pd_pairs_to_recompute.add((
    issue["run_id"],  # Keep as UUID object
    issue["embedding_model"],
))
```

This will ensure the run_id remains a UUID object throughout the process and can be properly used in database queries.

## Testing

After applying the fix:
1. Run `uv run pytest -n 8` to ensure all tests pass
2. Test on production database: `panic-tda doctor --fix --yes`
3. Verify no AttributeError occurs and PD issues are properly fixed

## Solution Applied

Fixed the issue by converting the string `run_id` back to UUID when querying the database at line 850 in `src/panic_tda/doctor.py`:

Changed from:
```python
run = session.exec(select(Run).where(Run.id == run_id)).first()
```

To:
```python
run = session.exec(select(Run).where(Run.id == UUID(run_id))).first()
```

The root cause was that Ray remote functions (`compute_persistence_diagram`) expect string UUIDs as parameters (and convert them internally), so the string conversion at line 827 is correct. However, when using these string UUIDs in SQLModel queries, they must be converted back to UUID objects.

All tests pass successfully after this fix.
