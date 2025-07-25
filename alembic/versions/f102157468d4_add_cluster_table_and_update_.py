"""add_cluster_table_and_update_embeddingcluster

Revision ID: f102157468d4
Revises: 1038ac5c289b
Create Date: 2025-07-25 14:26:54.934018

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "f102157468d4"
down_revision: Union[str, Sequence[str], None] = "1038ac5c289b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create the Cluster table
    op.create_table(
        "cluster",
        sa.Column("id", sa.UUID(), nullable=False, primary_key=True),
        sa.Column("clustering_result_id", sa.UUID(), nullable=False),
        sa.Column("cluster_id", sa.Integer(), nullable=False),
        sa.Column("medoid_embedding_id", sa.UUID(), nullable=True),
        sa.Column("size", sa.Integer(), nullable=False, default=0),
        sa.Column("properties", sa.JSON(), nullable=False, default={}),
        sa.ForeignKeyConstraint(["clustering_result_id"], ["clusteringresult.id"]),
        sa.ForeignKeyConstraint(["medoid_embedding_id"], ["embedding.id"]),
        sa.UniqueConstraint(
            "clustering_result_id", "cluster_id", name="unique_cluster_per_result"
        ),
    )

    # Create indexes
    op.create_index(
        "ix_cluster_clustering_result_id", "cluster", ["clustering_result_id"]
    )
    op.create_index(
        "ix_cluster_medoid_embedding_id", "cluster", ["medoid_embedding_id"]
    )

    # Create temporary table for new EmbeddingCluster structure
    op.create_table(
        "embeddingcluster_new",
        sa.Column("id", sa.UUID(), nullable=False, primary_key=True),
        sa.Column("embedding_id", sa.UUID(), nullable=False),
        sa.Column("clustering_result_id", sa.UUID(), nullable=False),
        sa.Column("cluster_id", sa.UUID(), nullable=False),  # Now references Cluster.id
        sa.ForeignKeyConstraint(["embedding_id"], ["embedding.id"]),
        sa.ForeignKeyConstraint(["clustering_result_id"], ["clusteringresult.id"]),
        sa.ForeignKeyConstraint(["cluster_id"], ["cluster.id"]),
        sa.UniqueConstraint(
            "embedding_id", "clustering_result_id", name="unique_embedding_clustering"
        ),
    )

    # Create indexes on new table
    op.create_index(
        "ix_embeddingcluster_new_embedding_id", "embeddingcluster_new", ["embedding_id"]
    )
    op.create_index(
        "ix_embeddingcluster_new_clustering_result_id",
        "embeddingcluster_new",
        ["clustering_result_id"],
    )
    op.create_index(
        "ix_embeddingcluster_new_cluster_id", "embeddingcluster_new", ["cluster_id"]
    )

    # Drop old EmbeddingCluster table
    op.drop_table("embeddingcluster")

    # Rename new table to EmbeddingCluster
    op.rename_table("embeddingcluster_new", "embeddingcluster")


def downgrade() -> None:
    """Downgrade schema."""
    # Create temporary table with old structure
    op.create_table(
        "embeddingcluster_old",
        sa.Column("id", sa.UUID(), nullable=False, primary_key=True),
        sa.Column("embedding_id", sa.UUID(), nullable=False),
        sa.Column("clustering_result_id", sa.UUID(), nullable=False),
        sa.Column("cluster_id", sa.Integer(), nullable=False),  # Back to integer
        sa.ForeignKeyConstraint(["embedding_id"], ["embedding.id"]),
        sa.ForeignKeyConstraint(["clustering_result_id"], ["clusteringresult.id"]),
        sa.UniqueConstraint(
            "embedding_id", "clustering_result_id", name="unique_embedding_clustering"
        ),
    )

    # Drop current EmbeddingCluster table
    op.drop_table("embeddingcluster")

    # Rename old structure back
    op.rename_table("embeddingcluster_old", "embeddingcluster")

    # Drop Cluster table
    op.drop_table("cluster")
