"""add_unique_constraint_embedding_invocation_model

Revision ID: 9c62d27ee223
Revises: 83fb17def420
Create Date: 2025-08-04 13:58:08.513174

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9c62d27ee223'
down_revision: Union[str, Sequence[str], None] = '83fb17def420'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # First, remove duplicates keeping only the first embedding for each invocation_id + embedding_model pair
    # This keeps the embedding with the smallest ID (assumed to be the oldest/first created)
    # SQLite doesn't support table aliases in DELETE, so use rowid approach
    op.execute("""
        DELETE FROM embedding
        WHERE rowid NOT IN (
            SELECT MIN(rowid)
            FROM embedding
            GROUP BY invocation_id, embedding_model
        )
    """)
    
    # For SQLite with a large table, we'll create a unique index instead of a constraint
    # This achieves the same effect but is more efficient for large tables
    op.create_index(
        'unique_invocation_embedding_model',
        'embedding',
        ['invocation_id', 'embedding_model'],
        unique=True
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Remove the unique index
    op.drop_index('unique_invocation_embedding_model', table_name='embedding')
