"""Optional persistent cache in the application database."""

from datetime import datetime, timedelta, timezone
from functools import lru_cache
import os

from dotenv import dotenv_values
from sqlalchemy import Column, DateTime, JSON, MetaData, String, Table, create_engine, delete, select
from sqlalchemy.dialects.postgresql import JSONB, insert as postgres_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import make_url
from sqlalchemy.pool import NullPool

from storage.database import DEFAULT_APP_DATABASE_URL, postgres_dsn


# This cache can be unavailable while core course tables remain usable. Alembic
# manages it separately from the core schema checked by the app at startup.
SNAPSHOTS = Table(
    "evidence_snapshots", MetaData(),
    Column("key", String(64), primary_key=True),
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("retrieved_at", DateTime(timezone=True), nullable=False),
    Column("expires_at", DateTime(timezone=True), nullable=False, index=True),
    Column("payload", JSON(none_as_null=True).with_variant(JSONB(none_as_null=True), "postgresql")),
)


def _utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


class EvidenceSnapshotStore:
    def __init__(self, url: str):
        target = make_url(url)
        if target.get_backend_name() == "sqlite":
            if target.database in (None, "", ":memory:"):
                raise ValueError("Снимкам нужен файловый SQLite или PostgreSQL.")
            self.engine = create_engine(target.set(drivername="sqlite+pysqlite"), poolclass=NullPool,
                                        connect_args={"timeout": 0.5})
            self.insert = sqlite_insert
        elif target.get_backend_name() in {"postgres", "postgresql"}:
            target = make_url(postgres_dsn(url))
            options = target.query.get("options", "")
            self.engine = create_engine(target.set(drivername="postgresql+psycopg"), poolclass=NullPool,
                                        connect_args={"connect_timeout": 2,
                                                      "options": f"{options} -c statement_timeout=1000 -c lock_timeout=500".strip()})
            self.insert = postgres_insert
        else:
            raise ValueError("Снимкам нужен SQLite или PostgreSQL.")

    def read(self, key: str, access_key: str, at: datetime, max_age: timedelta
             ) -> tuple[list[dict], datetime] | None:
        with self.engine.connect() as connection:
            query = select(SNAPSHOTS).where(SNAPSHOTS.c.key.in_((key, access_key)), SNAPSHOTS.c.expires_at > at)
            rows = {row.key: row for row in connection.execute(query)}
        snapshot = rows.get(key)
        denied = rows.get(access_key)
        if snapshot is None or (denied is not None and denied.started_at >= snapshot.started_at):
            return None
        retrieved_at = _utc(snapshot.retrieved_at)
        if not at - max_age <= retrieved_at <= at:
            return None
        return snapshot.payload, retrieved_at

    def write(self, key: str, started_at: datetime, received_at: datetime, expires_at: datetime,
              payload: list[dict] | None) -> None:
        statement = self.insert(SNAPSHOTS).values(key=key, started_at=started_at, retrieved_at=received_at,
                                                 expires_at=expires_at, payload=payload)
        statement = statement.on_conflict_do_update(
            index_elements=[SNAPSHOTS.c.key],
            set_={name: statement.excluded[name] for name in ("started_at", "retrieved_at", "expires_at", "payload")},
            where=SNAPSHOTS.c.started_at < statement.excluded.started_at,
        )
        with self.engine.begin() as connection:
            connection.execute(statement)
            expired = (select(SNAPSHOTS.c.key).where(SNAPSHOTS.c.expires_at <= received_at)
                       .order_by(SNAPSHOTS.c.expires_at).limit(128))
            connection.execute(delete(SNAPSHOTS).where(SNAPSHOTS.c.key.in_(expired),
                                                       SNAPSHOTS.c.expires_at <= received_at))


@lru_cache(maxsize=4)
def _snapshot_store(url: str) -> EvidenceSnapshotStore:
    return EvidenceSnapshotStore(url)


def snapshot_store() -> EvidenceSnapshotStore:
    url = os.environ.get("APP_DATABASE_URL") or dotenv_values(encoding="utf-8-sig", interpolate=False).get("APP_DATABASE_URL")
    return _snapshot_store(url or DEFAULT_APP_DATABASE_URL)
