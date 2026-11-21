import asyncio
import os
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from collections.abc import AsyncIterator

from sqlalchemy import inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine


MIGRATIONS = Path(__file__).resolve().parent / "migrations"
_MIGRATION_THREAD_LOCK = threading.Lock()


class DatabaseSetupError(RuntimeError):
    """База приложения не готова, и автоматически подготовить её нельзя."""


def sqlite_file(url: str) -> Path | None:
    """Путь к файлу SQLite из URL или None для других баз и базы в памяти."""
    parsed = make_url(url)
    database = parsed.database or ""
    if parsed.get_backend_name() != "sqlite" or database in ("", ":memory:") or database.startswith("file:"):
        return None
    return Path(database).expanduser()


def prepare_sqlite_folder(url: str) -> None:
    """Создаёт папку для файла SQLite, указанного в URL."""
    path = sqlite_file(url)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def migration_lock(url: str, timeout: float = 120):
    """Не даёт нескольким потокам и процессам одновременно создавать таблицы."""
    with _MIGRATION_THREAD_LOCK:
        path = sqlite_file(url)
        if path is None:
            yield
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + timeout
        with path.with_suffix(".migrate.lock").open("a+b") as handle:
            while True:
                handle.seek(0)
                try:
                    if os.name == "nt":
                        import msvcrt
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise DatabaseSetupError("База приложения занята другим процессом. Повторите запуск позже.")
                    time.sleep(0.2)
            try:
                yield
            finally:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle, fcntl.LOCK_UN)


def make_engine(url: str) -> AsyncEngine:
    prepare_sqlite_folder(url)
    return create_async_engine(url, pool_pre_ping=True)


async def table_state(url: str) -> tuple[set[str], set[str]]:
    """Возвращает отсутствующие и уже существующие таблицы приложения."""
    from storage.models import Base, WorkflowCommand

    def inspect_schema(sync):
        inspector = inspect(sync)
        existing = set(inspector.get_table_names())
        if "workflow_commands" in existing and set(Base.metadata.tables) <= existing:
            columns = {column["name"] for column in inspector.get_columns("workflow_commands")}
            if set(WorkflowCommand.__table__.columns.keys()) - columns:
                raise DatabaseSetupError("Примените миграции базы приложения: python -m alembic upgrade head.")
        return existing

    engine = make_engine(url)
    try:
        async with engine.connect() as connection:
            existing = await connection.run_sync(inspect_schema)
    finally:
        await engine.dispose()
    expected = set(Base.metadata.tables)
    return expected - existing, expected & existing


def upgrade_schema(url: str) -> None:
    """Применяет миграции Alembic к указанной базе независимо от текущей папки."""
    from alembic import command
    from alembic.config import Config

    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS).replace("%", "%%"))
    config.attributes["database_url"] = url
    command.upgrade(config, "head")


def ensure_schema(url: str) -> None:
    """Создаёт таблицы приложения в пустой базе.

    Базу, где таблицы уже есть, функция не изменяет. Вызывать вне работающего цикла asyncio.
    """
    if not asyncio.run(table_state(url))[0]:
        return
    with migration_lock(url):
        missing, present = asyncio.run(table_state(url))
        if not missing:
            return
        if present:
            raise DatabaseSetupError(
                "В базе приложения есть только часть таблиц (нет: " + ", ".join(sorted(missing)) + "). "
                "Укажите в APP_DATABASE_URL новую базу или восстановите эту из резервной копии."
            )
        try:
            upgrade_schema(url)
        except Exception as error:
            # Другой процесс с иной базой блокировок (например, PostgreSQL) мог успеть раньше.
            if asyncio.run(table_state(url))[0]:
                raise DatabaseSetupError(
                    "Не удалось создать таблицы базы приложения. Проверьте APP_DATABASE_URL в .env "
                    "и права на запись в папку базы."
                ) from error


@asynccontextmanager
async def write_transaction(factory: async_sessionmaker[AsyncSession]) -> AsyncIterator[AsyncSession]:
    async with factory() as db:
        try:
            if db.bind.dialect.name == "sqlite":
                await db.execute(text("BEGIN IMMEDIATE"))
            yield db
            await db.commit()
        except BaseException:
            await db.rollback()
            raise
