"""Добавление таблиц проектирования курсов без изменения пользователей."""
from alembic import op
from sqlalchemy import Column, DateTime, ForeignKey, JSON, String, inspect
from sqlalchemy.dialects.postgresql import JSONB

from storage.models import Base

revision = "0001_course_workflow"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    collisions = set(inspect(bind).get_table_names()) & set(Base.metadata.tables)
    if collisions:
        raise RuntimeError(f"Таблицы уже существуют; требуется адаптер: {sorted(collisions)}")
    # Keep the original queue schema here. Later revisions upgrade both new and existing databases.
    tables = [table for table in Base.metadata.sorted_tables if table.name != "workflow_commands"]
    Base.metadata.create_all(bind=bind, tables=tables, checkfirst=False)
    op.create_table(
        "workflow_commands",
        Column("id", String(100), primary_key=True),
        Column("course_id", String(36), ForeignKey("courses.id"), nullable=False),
        Column("kind", String(30), nullable=False),
        Column("payload", JSON().with_variant(JSONB(), "postgresql"), nullable=False),
        Column("status", String(20), nullable=False),
        Column("created_at", DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_workflow_commands_course_id", "workflow_commands", ["course_id"])


def downgrade() -> None:
    raise RuntimeError("Миграция additive. Удаление данных выполняется отдельной согласованной миграцией.")
