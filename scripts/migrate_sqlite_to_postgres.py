"""Copy application rows and LangGraph history from trusted SQLite backups to empty PostgreSQL."""
import argparse
from contextlib import closing
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory

from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver
import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb
from sqlalchemy import Boolean, DateTime, JSON, URL
from sqlalchemy.engine import make_url

from orchestration.checkpointer import CHECKPOINT_SETUP_LOCK
from storage.database import DEFAULT_APP_DATABASE_URL, postgres_dsn, upgrade_schema
from storage.evidence_snapshots import SNAPSHOTS
from storage.models import Base


def open_sqlite(path: Path):
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)


def snapshot(source: Path, target: Path) -> None:
    with closing(open_sqlite(source)) as original, closing(sqlite3.connect(target)) as copy:
        original.backup(copy)
        if copy.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise ValueError(f"SQLite не прошёл проверку целостности: {source.name}.")


def convert_column(value, column):
    if value is None:
        return None
    if isinstance(column.type, JSON):
        return Jsonb(json.loads(value))
    if isinstance(column.type, Boolean):
        return bool(value)
    if isinstance(column.type, DateTime):
        timestamp = datetime.fromisoformat(value)
        if column.type.timezone and timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return timestamp
    return value


def count_rows(cursor, table: str) -> int:
    cursor.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table)))
    return cursor.fetchone()[0]


def copy_table(source, target, table) -> int:
    columns = list(table.columns)
    name = sql.Identifier(table.name)
    names = sql.SQL(", ").join(sql.Identifier(column.name) for column in columns)
    rows = source.execute(sql.SQL("SELECT {} FROM {}").format(names, name).as_string())
    statement = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
        name, names, sql.SQL(", ").join(sql.Placeholder() for _ in columns),
    )
    target.executemany(statement, (tuple(convert_column(value, column) for value, column in zip(row, columns, strict=True))
                                   for row in rows))
    actual = count_rows(target, table.name)
    if actual != source.execute(sql.SQL("SELECT count(*) FROM {}").format(name).as_string()).fetchone()[0]:
        raise ValueError(f"Не совпало количество строк в {table.name}.")
    if (column := table.autoincrement_column) is not None and actual:
        target.execute("SELECT pg_get_serial_sequence(%s, %s)", (name.as_string(), column.name))
        if (sequence := target.fetchone()[0]) is not None:
            target.execute(sql.SQL("SELECT setval(%s, (SELECT max({}) FROM {}), true)").format(
                sql.Identifier(column.name), name), (sequence,))
    return actual


def copy_checkpoints(source, target, saver: PostgresSaver) -> dict[str, int]:
    rows = source.execute("SELECT thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, "
                          "metadata FROM checkpoints ORDER BY thread_id, checkpoint_ns, checkpoint_id")
    for thread_id, namespace, checkpoint_id, parent_id, kind, data, metadata_bytes in rows:
        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": namespace, "checkpoint_id": parent_id}}
        checkpoint = saver.serde.loads_typed((kind, data))
        if checkpoint["id"] != checkpoint_id:
            raise ValueError("Идентификатор контрольной точки не совпадает с сохранённым ключом.")
        metadata = json.loads(metadata_bytes) if metadata_bytes else {}
        saver.put(config, checkpoint, metadata, checkpoint["channel_versions"])
        if "writes" in metadata:
            target.execute("UPDATE checkpoints SET metadata = %s WHERE thread_id = %s AND checkpoint_ns = %s "
                           "AND checkpoint_id = %s", (Jsonb(metadata), thread_id, namespace, checkpoint_id))

    # Передаём сериализованные записи побайтно, включая индексы interrupt/resume и путь задачи.
    has_task_path = any(column[1] == "task_path" for column in source.execute("PRAGMA table_info(writes)"))
    path_column = "task_path" if has_task_path else "''"
    writes = source.execute("SELECT thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value, "
                            f"{path_column} FROM writes")
    target.executemany("INSERT INTO checkpoint_writes "
                       "(thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, blob, task_path) "
                       "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)", writes)
    counts = {table: count_rows(target, table) for table in ("checkpoints", "checkpoint_writes")}
    if (counts["checkpoints"] != source.execute("SELECT count(*) FROM checkpoints").fetchone()[0]
            or counts["checkpoint_writes"] != source.execute("SELECT count(*) FROM writes").fetchone()[0]):
        raise ValueError("Не совпало количество контрольных точек или записей LangGraph.")
    return counts


def migrate(application: Path, checkpoints: Path, target_url: str) -> dict[str, int]:
    if make_url(target_url).get_backend_name() not in {"postgres", "postgresql"}:
        raise ValueError("Задайте PostgreSQL в APP_DATABASE_URL.")
    dsn = postgres_dsn(target_url)
    if not application.is_file() or not checkpoints.is_file():
        raise ValueError("Нужны оба исходных файла SQLite. Проверьте --application и --checkpoints.")
    tables = [*Base.metadata.sorted_tables, SNAPSHOTS]
    with TemporaryDirectory(prefix="rpd-sqlite-export-") as folder:
        source_app = Path(folder) / "application.sqlite"
        source_graph = Path(folder) / "checkpoints.sqlite"
        snapshot(application, source_app)
        snapshot(checkpoints, source_graph)
        with closing(open_sqlite(source_app)) as source:
            names = {row[0] for row in source.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        if unknown := names - {table.name for table in tables} - {"alembic_version", "sqlite_sequence"}:
            raise ValueError("В SQLite есть таблицы вне схемы приложения: " + ", ".join(sorted(unknown)))
        source_url = URL.create("sqlite+aiosqlite", database=str(source_app)).render_as_string(hide_password=False)
        upgrade_schema(source_url)
        upgrade_schema(dsn)
        with PostgresSaver.from_conn_string(dsn) as setup_saver:
            setup_saver.conn.execute("SELECT pg_advisory_lock(%s)", (CHECKPOINT_SETUP_LOCK,))
            setup_saver.setup()
        target_tables = [table.name for table in tables] + ["checkpoints", "checkpoint_writes", "checkpoint_blobs"]
        with (closing(open_sqlite(source_app)) as app, closing(open_sqlite(source_graph)) as graph,
              psycopg.connect(dsn) as connection, connection.cursor() as target):
            target.execute(sql.SQL("LOCK TABLE {} IN SHARE ROW EXCLUSIVE MODE NOWAIT").format(
                sql.SQL(", ").join(map(sql.Identifier, sorted(target_tables))),
            ))
            if any(count_rows(target, table) for table in target_tables):
                raise ValueError("PostgreSQL уже содержит данные проекта. Укажите пустую базу для переноса.")
            counts = {table.name: copy_table(app, target, table) for table in tables}
            counts.update(copy_checkpoints(graph, target, PostgresSaver(connection)))
        return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--application", type=Path, default=Path("data/application.sqlite"))
    parser.add_argument("--checkpoints", type=Path, default=Path("data/checkpoints.sqlite"))
    args = parser.parse_args()
    load_dotenv(encoding="utf-8-sig", interpolate=False)
    try:
        counts = migrate(args.application, args.checkpoints, os.getenv("APP_DATABASE_URL", DEFAULT_APP_DATABASE_URL))
    except ValueError as error:
        parser.exit(1, f"Перенос прерван. {error}\n")
    except Exception as error:
        code = getattr(error, "sqlstate", None)
        parser.exit(1, f"Перенос прерван ({type(error).__name__}{' / ' + code if code else ''}). "
                    "Проверьте доступ к PostgreSQL и схему исходных файлов.\n")
    for table, count in sorted(counts.items()):
        print(f"{table}: {count}")


if __name__ == "__main__":
    main()
