"""Persist queued retry deadlines and attempt counts."""

from alembic import op
import sqlalchemy as sa


revision = "0003_workflow_retries"
down_revision = "0002_evidence_snapshots"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("workflow_commands", sa.Column("attempt_count", sa.Integer(), nullable=False,
                                                 server_default=sa.text("0")))
    op.add_column("workflow_commands", sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("workflow_commands", sa.Column("last_error_code", sa.String(40), nullable=True))
    op.create_index("ix_workflow_commands_due", "workflow_commands", ["status", "next_attempt_at", "created_at"])


def downgrade() -> None:
    raise RuntimeError("Миграция additive. Удаление данных выполняется отдельной согласованной миграцией.")
