"""Persistent snapshots of normalized source evidence."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


revision = "0002_evidence_snapshots"
down_revision = "0001_course_workflow"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "evidence_snapshots",
        sa.Column("key", sa.String(64), primary_key=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("payload", sa.JSON(none_as_null=True).with_variant(JSONB(none_as_null=True), "postgresql")),
    )
    op.create_index("ix_evidence_snapshots_expires_at", "evidence_snapshots", ["expires_at"])


def downgrade() -> None:
    raise RuntimeError("Миграция additive. Удаление данных выполняется отдельной согласованной миграцией.")
