"""Журнал вызовов модели."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


revision = "0003_model_calls"
down_revision = "0002_evidence_snapshots"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if sa.inspect(op.get_bind()).has_table("model_calls"):
        return
    op.create_table(
        "model_calls",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("course_id", sa.String(36), nullable=True),
        sa.Column("run_id", sa.String(100), nullable=True),
        sa.Column("stage", sa.String(60), nullable=False),
        sa.Column("subject", sa.String(300), nullable=True),
        sa.Column("output", sa.String(80), nullable=False),
        sa.Column("model", sa.String(300), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("outcome", sa.String(30), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("parameters", sa.JSON().with_variant(JSONB(), "postgresql"), nullable=True),
    )
    op.create_index("ix_model_calls_course_id", "model_calls", ["course_id"])
    op.create_index("ix_model_calls_outcome", "model_calls", ["outcome"])


def downgrade() -> None:
    op.drop_index("ix_model_calls_outcome", "model_calls")
    op.drop_index("ix_model_calls_course_id", "model_calls")
    op.drop_table("model_calls")
