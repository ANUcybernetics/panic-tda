"""Add missing db constraints

Revision ID: caedeefdeae7
Revises: 9c62d27ee223
Create Date: 2025-08-05 15:10:54.170135

"""

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "caedeefdeae7"
down_revision: Union[str, Sequence[str], None] = "9c62d27ee223"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add unique constraint on (run_id, sequence_number) in invocation table
    with op.batch_alter_table("invocation", schema=None) as batch_op:
        batch_op.create_unique_constraint(
            "unique_run_sequence_number", ["run_id", "sequence_number"]
        )

    # Add unique constraint on (run_id, embedding_model) in persistencediagram table
    with op.batch_alter_table("persistencediagram", schema=None) as batch_op:
        batch_op.create_unique_constraint(
            "unique_run_embedding_model", ["run_id", "embedding_model"]
        )


def downgrade() -> None:
    """Downgrade schema."""
    # Remove unique constraint from persistencediagram table
    with op.batch_alter_table("persistencediagram", schema=None) as batch_op:
        batch_op.drop_constraint("unique_run_embedding_model", type_="unique")

    # Remove unique constraint from invocation table
    with op.batch_alter_table("invocation", schema=None) as batch_op:
        batch_op.drop_constraint("unique_run_sequence_number", type_="unique")
