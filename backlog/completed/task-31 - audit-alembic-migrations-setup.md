---
id: task-31
title: audit alembic migrations setup
status: Done
assignee: []
created_date: "2025-07-28"
labels: []
dependencies: []
---

## Description

This project uses sqlmodel for database operations, but `alembic` is used for
managing database migrations. Alembic in particular was added much later on in
the project, and I'm not 100% sure that it is set up in a best-practices way.

Perform an audit to ensure that the alembic setup (including any associated
documentation) is set up in a best-practices way.

## Audit Findings

### Current Setup

1. **Configuration Files**:

   - `alembic.ini` is properly configured with standard settings
   - Database URL is hardcoded to `sqlite:///db/trajectory_data.sqlite`
   - Logging configuration is set up appropriately

2. **env.py**:

   - Uses SQLModel.metadata for autogenerate support
   - Has proper offline and online migration functions
   - **Issue**: Models are not imported, preventing autogenerate from working

3. **Migration Files**:

   - Proper naming convention with revision IDs and descriptive slugs
   - Clean upgrade/downgrade functions
   - Initial migration is empty (likely because DB existed before alembic)

4. **Database Integration**:

   - Project uses SQLModel with models in `src/panic_tda/schemas.py`
   - Database setup in `src/panic_tda/db.py` includes WAL mode and other SQLite
     optimizations

5. **Documentation**:
   - **Issue**: No documentation exists for alembic usage in README or other
     docs

### Issues Identified

1. **Critical Issue - Models Not Imported in env.py**:

   - The `env.py` file imports SQLModel but not the actual model classes
   - This prevents autogenerate from detecting schema changes
   - Need to add: `from panic_tda.schemas import *` or import specific models

2. **Hardcoded Database URL**:

   - The database URL is hardcoded in `alembic.ini`
   - Should use environment variables or configuration for flexibility

3. **Missing Documentation**:

   - No documentation on how to create/apply migrations
   - No mention of alembic in README or other project docs

4. **Initial Migration is Empty**:

   - The initial migration doesn't create the schema
   - This could cause issues for new developers or fresh deployments

5. **No Migration Helpers**:
   - No Makefile targets or scripts for common migration tasks
   - Developers need to remember alembic commands

### Recommendations

1. **Fix env.py to Import Models**:

   ```python
   # Add after line 9 in env.py
   from panic_tda import schemas  # Import all models
   ```

2. **Use Environment Variables for Database URL**:

   ```python
   # In env.py, override the URL from environment if available
   import os
   db_url = os.getenv("DATABASE_URL", config.get_main_option("sqlalchemy.url"))
   ```

3. **Add Migration Documentation**:

   - Create a section in README or separate MIGRATIONS.md
   - Document common commands:
     - `alembic upgrade head` - Apply all migrations
     - `alembic revision --autogenerate -m "description"` - Create new migration
     - `alembic downgrade -1` - Rollback one migration

4. **Add Helper Commands**:

   - Consider adding commands to the CLI tool or Makefile:
     - `uv run panic-tda db migrate` - Create migration
     - `uv run panic-tda db upgrade` - Apply migrations

5. **Consider Generating Full Initial Migration**:
   - Create a proper initial migration that creates all tables
   - This helps with fresh deployments and testing
