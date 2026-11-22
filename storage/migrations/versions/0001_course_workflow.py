"""Добавление таблиц проектирования курсов без изменения пользователей."""
from alembic import op
from sqlalchemy import inspect

from storage.activity_models import ACTIVITY_TABLES
from storage.models import Base

revision = "0001_course_workflow"
down_revision = None
branch_labels = None
depends_on = None


# Таблицы, которые добавляют следующие миграции.
LATER_TABLES = frozenset({"model_calls", "profession_boundaries", *ACTIVITY_TABLES})


def upgrade() -> None:
    bind = op.get_bind()
    tables = [table for name, table in Base.metadata.tables.items() if name not in LATER_TABLES]
    collisions = set(inspect(bind).get_table_names()) & {table.name for table in tables}
    if collisions:
        raise RuntimeError(f"Таблицы уже существуют; требуется адаптер: {sorted(collisions)}")
    Base.metadata.create_all(bind=bind, tables=tables, checkfirst=False)


def downgrade() -> None:
    raise RuntimeError("Миграция additive. Удаление данных выполняется отдельной согласованной миграцией.")
