import asyncio
import os

from alembic import context
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine

from storage.database import prepare_sqlite_folder
from storage.models import Base


load_dotenv(encoding="utf-8-sig", interpolate=False)
url = context.config.attributes.get("database_url") or os.environ.get(
    "APP_DATABASE_URL", "sqlite+aiosqlite:///./data/application.sqlite")


def configure(connection) -> None:
    context.configure(connection=connection, target_metadata=Base.metadata,
                      include_name=lambda name, kind, parent: kind != "table" or name in Base.metadata.tables)
    with context.begin_transaction():
        context.run_migrations()


async def online() -> None:
    prepare_sqlite_folder(url)
    engine = create_async_engine(url)
    async with engine.connect() as connection:
        await connection.run_sync(configure)
    await engine.dispose()


if context.is_offline_mode():
    raise RuntimeError("Миграция проверяет коллизии существующих таблиц и требует подключения к БД.")
else:
    asyncio.run(online())
