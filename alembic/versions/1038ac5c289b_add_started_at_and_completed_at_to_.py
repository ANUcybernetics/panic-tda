"""Add started_at and completed_at to ClusteringResult

Revision ID: 1038ac5c289b
Revises: d5147590cfc2
Create Date: 2025-07-24 12:40:01.503107

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "1038ac5c289b"
down_revision: Union[str, Sequence[str], None] = "d5147590cfc2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add started_at and completed_at columns to clusteringresult table."""
    op.add_column(
        "clusteringresult", sa.Column("started_at", sa.DateTime(), nullable=True)
    )
    op.add_column(
        "clusteringresult", sa.Column("completed_at", sa.DateTime(), nullable=True)
    )


def downgrade() -> None:
    """Remove started_at and completed_at columns from clusteringresult table."""
    op.drop_column("clusteringresult", "completed_at")
    op.drop_column("clusteringresult", "started_at")
