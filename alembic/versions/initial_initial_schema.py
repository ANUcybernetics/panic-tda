"""Initial schema

Revision ID: initial
Revises:
Create Date: 2025-07-16 13:07:03.103438

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "initial"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Initial migration - database already exists
    pass


def downgrade() -> None:
    """Downgrade schema."""
    # Initial migration - nothing to downgrade
    pass
