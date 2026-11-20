"""Добавление таблиц проектирования курсов без изменения пользователей."""
from alembic import op
from sqlalchemy import inspect

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
    Base.metadata.create_all(bind=bind, checkfirst=False)


def downgrade() -> None:
    raise RuntimeError("Миграция additive. Удаление данных выполняется отдельной согласованной миграцией.")
