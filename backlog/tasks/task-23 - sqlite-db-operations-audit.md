---
id: task-23
title: sqlite db operations audit
status: To Do
assignee: []
created_date: "2025-07-26"
labels: []
dependencies: []
---

## Description

Perform an audit of all the databse operations in this application. Pay
attention to:

- use sqlmodel (rather than sqlalchemy) functionality
- use sensible, modern defaults (e.g. sqlite WAL mode) but keep the code
  readable and maintainable---no need to over-engineer for performance reasons
- ensure the use of db session objects are properly and efficiently managed and
  closed
- push filtering/sorting/matching to the db (rather than doing it in application
  code) whenever possible
- ensure the main database code paths are tested using the pytest infrastructure
  in this project

## Audit Findings

### 1. SQLModel vs SQLAlchemy Usage

**Finding**: The codebase has mixed usage of SQLModel and SQLAlchemy:
- ✅ Models are properly defined using SQLModel (src/panic_tda/schemas.py)
- ✅ Most queries use SQLModel's `select()` and `session.exec()`
- ⚠️ Some instances still use SQLAlchemy directly:
  - `src/panic_tda/local.py:38` - Uses `session.query(Run)` instead of SQLModel
  - `src/panic_tda/clustering_manager.py:130` - Imports `IntegrityError` from SQLAlchemy
  - `src/panic_tda/main.py:654` - Uses `sqlalchemy.text()` for raw SQL

**Recommendations**:
- Replace `session.query()` with SQLModel's `select()` pattern
- Import exceptions from SQLModel where available
- Consider using SQLModel's approach for raw SQL if needed

### 2. SQLite Configuration

**Finding**: ✅ Excellent SQLite configuration in `src/panic_tda/db.py:40-46`:
- WAL mode enabled for better concurrency
- Connection pooling with QueuePool
- Appropriate cache size (10000 pages)
- PRAGMA synchronous=NORMAL for balanced performance/durability
- Configurable timeout (default 30s)

### 3. Session Management

**Finding**: ✅ Well-structured session management:
- Context manager pattern in `get_session_from_connection_string()` (src/panic_tda/db.py:52-68)
- Proper commit/rollback/close handling
- Sessions are properly scoped and closed
- Good use of `session.flush()` in bulk operations (clustering_manager.py)

**Minor Issue**:
- `clustering_manager.py:137-167` - Uses `session.bulk_save_objects()` which is SQLAlchemy-specific

### 4. Filtering/Sorting in Database vs Application

**Finding**: Mixed approach with room for improvement:
- ✅ Most filtering done in database:
  - `db.py` - All queries use `.where()` clauses
  - Good use of `.order_by()` and `.limit()`
- ⚠️ Some post-processing in application code:
  - `data_prep.py:263` - Polars DataFrame filtering after DB query
  - `data_prep.py:392` - Sorting in Polars after fetching from DB
  - `schemas.py:503-506` - Sorting invocations in Python

**Recommendations**:
- Consider moving more complex filtering/sorting to database queries where possible
- Document when application-level processing is intentionally chosen for performance

### 5. Test Coverage

**Finding**: ✅ Comprehensive test coverage in `tests/test_db.py`:
- Tests for all CRUD operations
- Session management tests
- Relationship and cascade delete tests
- Edge cases (non-existent IDs, empty results)
- Complex queries (export_experiments, find_embedding_for_vector)
- Integration tests with multiple models

### Additional Observations

**Strengths**:
- Consistent use of UUID v7 for primary keys
- Good use of indexes on foreign keys
- Proper relationship definitions with cascade options
- Type hints throughout

**Areas for Improvement**:
1. **Bulk Operations**: The `bulk_save_objects()` in clustering_manager could use SQLModel's approach
2. **Raw SQL**: Consider abstracting the raw SQL in main.py into proper queries
3. **Query Optimization**: Some complex queries could benefit from joins vs multiple queries

## Summary

The database operations are generally well-implemented with proper session management, good SQLite configuration, and comprehensive test coverage. The main recommendation is to complete the migration from SQLAlchemy to SQLModel for consistency, particularly in the areas identified above.
