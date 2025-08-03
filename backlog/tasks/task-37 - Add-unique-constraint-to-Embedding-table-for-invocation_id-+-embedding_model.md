---
id: task-37
title: Add unique constraint to Embedding table for invocation_id + embedding_model
status: To Do
assignee: []
created_date: '2025-08-03'
labels: [database, migration, constraint]
dependencies: []
---

## Description

Add a database constraint to prevent duplicate embeddings for the same invocation and embedding model combination. This ensures data integrity by guaranteeing that each invocation can only have one embedding per embedding model.

## Implementation Plan

### 1. Wait for current computation to complete
- Do not start this task until any long-running embedding computations are finished
- This prevents disruption to ongoing database operations

### 2. Check for existing duplicates
Before creating the migration, run this query to identify any duplicate embeddings:
```sql
SELECT invocation_id, embedding_model, COUNT(*) as count
FROM embedding 
GROUP BY invocation_id, embedding_model 
HAVING COUNT(*) > 1;
```

### 3. Create Alembic migration
Generate a new migration file:
```bash
uv run alembic revision -m "add_unique_constraint_embedding_invocation_model"
```

### 4. Write migration script
The migration should:
1. Remove duplicate embeddings (keeping only one per invocation_id + embedding_model pair)
2. Add the unique constraint

Example migration content:
```python
"""add unique constraint embedding invocation model

Revision ID: [auto-generated]
Revises: [previous-revision]
Create Date: [auto-generated]

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '[auto-generated]'
down_revision: Union[str, None] = '[previous-revision]'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # First, remove duplicates keeping only the first embedding for each invocation_id + embedding_model pair
    # This keeps the embedding with the smallest ID (assumed to be the oldest/first created)
    op.execute("""
        DELETE FROM embedding e1
        WHERE EXISTS (
            SELECT 1 FROM embedding e2
            WHERE e2.invocation_id = e1.invocation_id
            AND e2.embedding_model = e1.embedding_model
            AND e2.id < e1.id
        )
    """)
    
    # Now add the unique constraint
    op.create_unique_constraint(
        'unique_invocation_embedding_model',
        'embedding',
        ['invocation_id', 'embedding_model']
    )


def downgrade() -> None:
    # Remove the unique constraint
    op.drop_constraint('unique_invocation_embedding_model', 'embedding', type_='unique')
```

### 5. Update the SQLModel schema
Add the unique constraint to the Embedding class in `src/panic_tda/schemas.py`:
```python
class Embedding(SQLModel, table=True):
    """..."""
    
    __table_args__ = (
        UniqueConstraint(
            "invocation_id", "embedding_model", name="unique_invocation_embedding_model"
        ),
    )
    
    # ... rest of the class
```

### 6. Test the migration
1. Backup the database before running the migration
2. Run the migration in a test environment first if possible
3. Execute the migration:
   ```bash
   uv run alembic upgrade head
   ```
4. Verify the constraint is in place and working correctly

### 7. Verify post-migration
After migration, run verification queries:
```sql
-- Check no duplicates remain
SELECT invocation_id, embedding_model, COUNT(*) as count
FROM embedding 
GROUP BY invocation_id, embedding_model 
HAVING COUNT(*) > 1;

-- Verify constraint exists (PostgreSQL)
SELECT conname FROM pg_constraint 
WHERE conname = 'unique_invocation_embedding_model';
```

## Success Criteria
- [ ] No duplicate (invocation_id, embedding_model) pairs exist in the database
- [ ] Unique constraint is successfully added to the embedding table
- [ ] Embedding model in schemas.py reflects the constraint
- [ ] Future attempts to insert duplicate embeddings are rejected by the database
- [ ] Existing application code continues to work correctly

## Notes
- The duplicate removal strategy keeps the embedding with the smallest ID (oldest)
- Alternative strategies could keep the most recent or compare vector values
- Consider the impact on any code that might expect multiple embeddings per invocation/model pair

## Related
- File: `src/panic_tda/schemas.py` (Embedding class definition)
- Alembic migrations directory: `alembic/versions/`