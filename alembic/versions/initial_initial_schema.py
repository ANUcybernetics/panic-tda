"""Initial schema

Revision ID: initial
Revises:
Create Date: 2025-07-16 13:07:03.103438

"""

from typing import Sequence, Union


# revision identifiers, used by Alembic.
revision: str = "initial"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create all tables for fresh database installations."""
    import sqlalchemy as sa
    from alembic import op

    # Only create tables if they don't exist (for new installations)
    # This allows the migration to be idempotent for existing databases

    # Check if experimentconfig table exists (as a proxy for the schema existing)
    conn = op.get_bind()

    # Skip inspection in offline mode (SQL generation)
    if hasattr(conn, "connection"):
        # We're in online mode, check if tables exist
        inspector = sa.inspect(conn)
        existing_tables = inspector.get_table_names()

        if "experimentconfig" in existing_tables:
            # Tables already exist, skip creation
            return

    # Create all tables
    op.create_table(
        "clusteringresult",
        sa.Column("id", sa.CHAR(32), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("embedding_model", sa.String(), nullable=False),
        sa.Column("algorithm", sa.String(), nullable=False),
        sa.Column("parameters", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "experimentconfig",
        sa.Column("id", sa.CHAR(32), nullable=False),
        sa.Column("networks", sa.JSON(), nullable=False),
        sa.Column("seeds", sa.JSON(), nullable=False),
        sa.Column("prompts", sa.JSON(), nullable=False),
        sa.Column("embedding_models", sa.JSON(), nullable=False),
        sa.Column("max_length", sa.Integer(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "run",
        sa.Column("id", sa.CHAR(32), nullable=False),
        sa.Column("network", sa.JSON(), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=False),
        sa.Column("max_length", sa.Integer(), nullable=False),
        sa.Column("initial_prompt", sa.String(), nullable=False),
        sa.Column("experiment_id", sa.CHAR(32), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["experiment_id"], ["experimentconfig.id"]),
    )

    op.create_table(
        "invocation",
        sa.Column("id", sa.CHAR(32), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("model", sa.String(), nullable=False),
        sa.Column("type", sa.String(5), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.CHAR(32), nullable=False),
        sa.Column("sequence_number", sa.Integer(), nullable=False),
        sa.Column("input_invocation_id", sa.CHAR(32), nullable=True),
        sa.Column("output_text", sa.String(), nullable=True),
        sa.Column("output_image_data", sa.LargeBinary(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["run_id"], ["run.id"]),
        sa.ForeignKeyConstraint(["input_invocation_id"], ["invocation.id"]),
    )

    op.create_table(
        "persistencediagram",
        sa.Column("id", sa.CHAR(32), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("diagram_data", sa.LargeBinary(), nullable=True),
        sa.Column("run_id", sa.CHAR(32), nullable=False),
        sa.Column("embedding_model", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["run_id"], ["run.id"]),
    )

    op.create_table(
        "embedding",
        sa.Column("id", sa.CHAR(32), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("invocation_id", sa.CHAR(32), nullable=False),
        sa.Column("embedding_model", sa.String(), nullable=False),
        sa.Column("vector", sa.LargeBinary(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["invocation_id"], ["invocation.id"]),
    )

    op.create_table(
        "cluster",
        sa.Column("id", sa.CHAR(32), nullable=False),
        sa.Column("clustering_result_id", sa.CHAR(32), nullable=False),
        sa.Column("cluster_id", sa.Integer(), nullable=False),
        sa.Column("medoid_embedding_id", sa.CHAR(32), nullable=True),
        sa.Column("size", sa.Integer(), nullable=False),
        sa.Column("properties", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "clustering_result_id", "cluster_id", name="unique_cluster_per_result"
        ),
        sa.ForeignKeyConstraint(["clustering_result_id"], ["clusteringresult.id"]),
        sa.ForeignKeyConstraint(["medoid_embedding_id"], ["embedding.id"]),
    )

    op.create_table(
        "embeddingcluster",
        sa.Column("id", sa.CHAR(32), nullable=False),
        sa.Column("embedding_id", sa.CHAR(32), nullable=False),
        sa.Column("clustering_result_id", sa.CHAR(32), nullable=False),
        sa.Column("cluster_id", sa.CHAR(32), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "embedding_id", "clustering_result_id", name="unique_embedding_clustering"
        ),
        sa.ForeignKeyConstraint(["embedding_id"], ["embedding.id"]),
        sa.ForeignKeyConstraint(["clustering_result_id"], ["clusteringresult.id"]),
        sa.ForeignKeyConstraint(["cluster_id"], ["cluster.id"]),
    )

    # Create indexes
    op.create_index(
        "ix_embeddingcluster_cluster_id", "embeddingcluster", ["cluster_id"]
    )
    op.create_index(
        "ix_embeddingcluster_clustering_result_id",
        "embeddingcluster",
        ["clustering_result_id"],
    )
    op.create_index(
        "ix_embeddingcluster_embedding_id", "embeddingcluster", ["embedding_id"]
    )


def downgrade() -> None:
    """Drop all tables."""
    from alembic import op

    # Drop indexes first
    op.drop_index("ix_embeddingcluster_embedding_id", "embeddingcluster")
    op.drop_index("ix_embeddingcluster_clustering_result_id", "embeddingcluster")
    op.drop_index("ix_embeddingcluster_cluster_id", "embeddingcluster")

    # Drop tables in reverse order of dependencies
    op.drop_table("embeddingcluster")
    op.drop_table("cluster")
    op.drop_table("embedding")
    op.drop_table("persistencediagram")
    op.drop_table("invocation")
    op.drop_table("run")
    op.drop_table("experimentconfig")
    op.drop_table("clusteringresult")
