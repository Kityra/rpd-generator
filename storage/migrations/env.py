import os

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import text

from storage.database import DEFAULT_APP_DATABASE_URL, make_engine, run_async
from storage.models import Base


load_dotenv(encoding="utf-8-sig", interpolate=False)
url = context.config.attributes.get("database_url") or os.environ.get("APP_DATABASE_URL", DEFAULT_APP_DATABASE_URL)


def configure(connection) -> None:
    context.configure(connection=connection, target_metadata=Base.metadata,
                      render_as_batch=connection.dialect.name == "sqlite",
                      include_name=lambda name, kind, parent: kind != "table" or name in Base.metadata.tables)
    with context.begin_transaction():
        context.run_migrations()


async def online() -> None:
    engine = make_engine(url)
    try:
        async with engine.begin() as connection:
            if engine.dialect.name == "postgresql":
                await connection.execute(text("SELECT pg_advisory_xact_lock(625149383)"))
            await connection.run_sync(configure)
    finally:
        await engine.dispose()


if context.is_offline_mode():
    raise RuntimeError("Миграция проверяет коллизии существующих таблиц и требует подключения к БД.")
else:
    run_async(online())
