"""Make required fields non-optional in schemas

Revision ID: c9611d0e807d
Revises: caedeefdeae7
Create Date: 2025-08-14 13:45:15.571936

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

# revision identifiers, used by Alembic.
revision: str = "c9611d0e807d"
down_revision: Union[str, Sequence[str], None] = "caedeefdeae7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Make required fields non-optional.

    IMPORTANT: Run 'panic-tda doctor --fix' before applying this migration
    to properly handle any NULL values. This migration will DELETE any
    remaining records with NULL values in required fields.
    """
    # SQLite doesn't support ALTER COLUMN, so we need to use batch operations

    # Update Invocation table - make started_at NOT NULL
    with op.batch_alter_table("invocation") as batch_op:
        # Delete any invocations with NULL started_at
        # (doctor --fix should have handled these already)
        batch_op.execute(text("DELETE FROM invocation WHERE started_at IS NULL"))
        # Now alter the column to be NOT NULL with a default for future records
        batch_op.alter_column(
            "started_at",
            existing_type=sa.DATETIME(),
            nullable=False,
            server_default=text("(datetime('now'))"),
        )

    # Update Run table - make experiment_id NOT NULL
    with op.batch_alter_table("run") as batch_op:
        # Delete any orphaned runs without experiment_id
        batch_op.execute(text("DELETE FROM run WHERE experiment_id IS NULL"))
        # Now make experiment_id NOT NULL
        batch_op.alter_column(
            "experiment_id", existing_type=sa.CHAR(length=32), nullable=False
        )

    # Update Embedding table - make started_at NOT NULL
    with op.batch_alter_table("embedding") as batch_op:
        # Delete any embeddings with NULL started_at
        # (doctor --fix should have handled these already)
        batch_op.execute(text("DELETE FROM embedding WHERE started_at IS NULL"))
        # Now alter the column to be NOT NULL with a default
        batch_op.alter_column(
            "started_at",
            existing_type=sa.DATETIME(),
            nullable=False,
            server_default=text("(datetime('now'))"),
        )

        # Delete any embeddings with NULL vectors first
        batch_op.execute(text("DELETE FROM embedding WHERE vector IS NULL"))
        # Now make vector NOT NULL
        batch_op.alter_column("vector", existing_type=sa.BLOB(), nullable=False)

    # Update ClusteringResult table - make started_at NOT NULL
    with op.batch_alter_table("clusteringresult") as batch_op:
        # Delete any clustering results with NULL started_at
        # (doctor --fix should have handled these already)
        batch_op.execute(text("DELETE FROM clusteringresult WHERE started_at IS NULL"))
        # Now alter the column to be NOT NULL with a default
        batch_op.alter_column(
            "started_at",
            existing_type=sa.DATETIME(),
            nullable=False,
            server_default=text("(datetime('now'))"),
        )

    # Update PersistenceDiagram table - make started_at NOT NULL
    # Note: We do NOT make diagram_data NOT NULL because of legitimate OOM cases
    # in experiment 067efc98-c179-7da1-9e25-07bf296960e1 with length=5000 runs
    with op.batch_alter_table("persistencediagram") as batch_op:
        # Delete any persistence diagrams with NULL started_at
        # (doctor --fix should have handled these already)
        batch_op.execute(
            text("DELETE FROM persistencediagram WHERE started_at IS NULL")
        )
        # Now alter the column to be NOT NULL
        batch_op.alter_column(
            "started_at",
            existing_type=sa.DATETIME(),
            nullable=False,
            server_default=text("(datetime('now'))"),
        )

        # For diagram_data: Delete NULL values EXCEPT for the special OOM cases
        # from experiment 067efc98-c179-7da1-9e25-07bf296960e1
        batch_op.execute(
            text(
                """DELETE FROM persistencediagram 
            WHERE diagram_data IS NULL 
            AND run_id NOT IN (
                SELECT id FROM run 
                WHERE experiment_id = '067efc98c1797da19e2507bf296960e1'
            )"""
            )
        )
        # NOTE: We intentionally do NOT make diagram_data NOT NULL to preserve
        # the legitimate OOM cases from experiment 067efc98-c179-7da1-9e25-07bf296960e1


def downgrade() -> None:
    """Revert to optional fields."""
    # Revert Invocation table
    with op.batch_alter_table("invocation") as batch_op:
        batch_op.alter_column(
            "started_at",
            existing_type=sa.DATETIME(),
            nullable=True,
            server_default=None,
        )

    # Revert Run table
    with op.batch_alter_table("run") as batch_op:
        batch_op.alter_column(
            "experiment_id", existing_type=sa.CHAR(length=32), nullable=True
        )

    # Revert Embedding table
    with op.batch_alter_table("embedding") as batch_op:
        batch_op.alter_column(
            "started_at",
            existing_type=sa.DATETIME(),
            nullable=True,
            server_default=None,
        )
        batch_op.alter_column("vector", existing_type=sa.BLOB(), nullable=True)

    # Revert ClusteringResult table
    with op.batch_alter_table("clusteringresult") as batch_op:
        batch_op.alter_column(
            "started_at",
            existing_type=sa.DATETIME(),
            nullable=True,
            server_default=None,
        )

    # Revert PersistenceDiagram table
    with op.batch_alter_table("persistencediagram") as batch_op:
        batch_op.alter_column(
            "started_at",
            existing_type=sa.DATETIME(),
            nullable=True,
            server_default=None,
        )
        # Note: diagram_data remains nullable (wasn't changed in upgrade)
