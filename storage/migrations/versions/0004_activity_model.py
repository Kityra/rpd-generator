"""Модель деятельности профессии и таблицы программы."""

from alembic import op
import sqlalchemy as sa


revision = "0004_activity_model"
down_revision = "0003_model_calls"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Tables come from storage.activity_models so the migration and the ORM match (hackathon assumption:
    # later changes to these tables go into separate migrations with op.add_column etc.).
    from storage.activity_models import ACTIVITY_TABLES
    from storage.models import Base

    bind = op.get_bind()
    existing = set(sa.inspect(bind).get_table_names())
    Base.metadata.create_all(bind, tables=[Base.metadata.tables[name] for name in ACTIVITY_TABLES
                                           if name not in existing])


def downgrade() -> None:
    raise RuntimeError("Миграция additive. Удаление данных выполняется отдельной согласованной миграцией.")
