"""remove_legacy_clusters_column

Revision ID: f0ec8e9694a0
Revises: f102157468d4
Create Date: 2025-07-29 08:37:42.806316

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f0ec8e9694a0'
down_revision: Union[str, Sequence[str], None] = 'f102157468d4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Remove the legacy clusters column from clusteringresult table."""
    # Drop the clusters column
    op.drop_column('clusteringresult', 'clusters')


def downgrade() -> None:
    """Re-add the clusters column for rollback."""
    # Add back the clusters column with NOT NULL constraint
    op.add_column(
        'clusteringresult', 
        sa.Column('clusters', sa.JSON(), nullable=False, server_default='[]')
    )
