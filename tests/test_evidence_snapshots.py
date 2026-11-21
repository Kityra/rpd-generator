from datetime import datetime, timedelta, timezone
from hashlib import sha256
from importlib import import_module
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.operations import Operations
import httpx
from openai import AuthenticationError
from sqlalchemy import select, text

from orchestration import evidence_cache
from orchestration.evidence_cache import SnapshotEvidence, search_evidence, source_record
from orchestration.resilience import ExternalUnavailable
from storage.evidence_snapshots import EvidenceSnapshotStore, SNAPSHOTS


ROOT = Path(__file__).resolve().parents[1]
SCOPE = ("Yandex", "vector_search", "service-account", "RPD", "index")
QUERY = ("owner", "Test university", "Databases", "2026/2027")
TTL = timedelta(days=7)


class PersistentSnapshotTests(unittest.TestCase):
    def setUp(self):
        path = Path(self.enterContext(TemporaryDirectory())) / "courses.sqlite"
        self.url = "sqlite+aiosqlite:///" + path.as_posix()
        self.store = EvidenceSnapshotStore(self.url)
        self.time = datetime(2026, 11, 21, 9, tzinfo=timezone.utc)
        self.enterContext(patch.dict(os.environ, {"APP_DATABASE_URL": self.url}))
        self.enterContext(patch.object(evidence_cache, "_now", side_effect=lambda: self.time))

    def create_table(self):
        with self.store.engine.begin() as connection:
            SNAPSHOTS.create(connection)

    def record(self, content="Approved study plan"):
        return SnapshotEvidence(source=source_record("RPD", "https://example.org/plan", "Plan", content))

    def lookup(self, fetch, query=QUERY):
        return search_evidence(SCOPE, query, fetch, ttl=TTL)

    def test_migrate_existing_database_and_read_after_process_restart(self):
        config = Config(str(ROOT / "alembic.ini"))
        config.set_main_option("script_location", str(ROOT / "storage" / "migrations"))
        command.upgrade(config, "0001_course_workflow")
        with self.store.engine.begin() as connection:
            connection.execute(text("CREATE TABLE existing_marker (value TEXT)"))
            connection.execute(text("INSERT INTO existing_marker VALUES ('keep')"))
        command.upgrade(config, "head")
        with self.store.engine.connect() as connection:
            self.assertEqual(connection.scalar(text("SELECT value FROM existing_marker")), "keep")
        expected = self.record()
        self.lookup(lambda: (expected,))
        with self.store.engine.connect() as connection:
            row = connection.execute(select(SNAPSHOTS)).one()
            self.assertEqual(row.payload[0]["source"]["source_id"], expected.source.source_id)
            self.assertEqual(row.payload[0]["source"]["raw_hash"], sha256(expected.source.content.encode()).hexdigest())
            self.assertNotIn("Test university", str(row.payload))

        script = """
import json
from datetime import datetime, timedelta
from unittest.mock import patch
from orchestration.evidence_cache import search_evidence
from orchestration.resilience import ExternalUnavailable
def unavailable():
    raise ExternalUnavailable()
with patch("orchestration.evidence_cache._now", return_value=datetime.fromisoformat("2026-11-21T09:01:00+00:00")):
    records, date = search_evidence(("Yandex", "vector_search", "service-account", "RPD", "index"),
        ("owner", "Test university", "Databases", "2026/2027"), unavailable, ttl=timedelta(days=7))
print(json.dumps([records[0].source.model_dump(mode="json"), date.isoformat()]))
"""
        process = subprocess.run([sys.executable, "-c", script], cwd=ROOT, text=True,
                                 capture_output=True, check=True, timeout=15)
        self.assertEqual(json.loads(process.stdout), [expected.source.model_dump(mode="json"), self.time.isoformat()])

    def test_expiry_and_successful_empty_replaces_previous_snapshot(self):
        self.create_table()
        fetch = MagicMock(return_value=(self.record(),))
        self.lookup(fetch)
        self.time += timedelta(days=1)
        fetch.side_effect = ExternalUnavailable()
        self.assertEqual(len(self.lookup(fetch)[0]), 1)
        self.time += timedelta(days=6)
        with self.assertRaises(ExternalUnavailable):
            self.lookup(fetch)

        fetch.side_effect = None
        self.lookup(fetch)
        self.time += timedelta(seconds=1)
        fetch.return_value = ()
        self.assertEqual(self.lookup(fetch), ((), None))
        fetch.side_effect = ExternalUnavailable()
        self.assertEqual(self.lookup(fetch), ((), self.time))

    def test_auth_failure_blocks_old_snapshots_for_same_index(self):
        self.create_table()
        other_query = (*QUERY[:-1], "2027/2028")
        for query in (QUERY, other_query):
            self.lookup(lambda: (self.record(),), query)
        self.time += timedelta(seconds=1)
        sdk_response = httpx.Response(401, request=httpx.Request("POST", "https://example.org"))
        forbidden = MagicMock(side_effect=AuthenticationError("unauthorized", response=sdk_response, body=None))
        with self.assertRaises(AuthenticationError):
            self.lookup(forbidden)
        unavailable = MagicMock(side_effect=ExternalUnavailable())
        for query in (QUERY, other_query):
            with self.assertRaises(ExternalUnavailable):
                self.lookup(unavailable, query)
        self.time += timedelta(seconds=1)
        current = self.record("Current version")
        self.lookup(lambda: (current,))
        self.assertEqual(self.lookup(unavailable)[0], (current,))

    def test_late_or_invalid_result_does_not_replace_valid_snapshot(self):
        self.create_table()
        started = self.time
        valid = self.record("Current version")
        self.lookup(lambda: (valid,))
        with self.store.engine.connect() as connection:
            key = connection.execute(select(SNAPSHOTS.c.key)).scalar_one()
        self.time += timedelta(seconds=1)
        older = self.record("Old version")
        self.store.write(key, started - timedelta(seconds=1), self.time, self.time + TTL,
                         [older.model_dump(mode="json")])
        with self.assertRaises(ValueError):
            self.lookup(MagicMock(side_effect=ValueError("Malformed response")))
        self.assertEqual(self.lookup(MagicMock(side_effect=ExternalUnavailable()))[0], (valid,))

    def test_live_result_remains_available_without_cache_table(self):
        record = self.record()
        self.assertEqual(self.lookup(lambda: (record,)), ((record,), None))
        failure = ExternalUnavailable("Original upstream error")
        with self.assertRaises(ExternalUnavailable) as raised:
            self.lookup(MagicMock(side_effect=failure))
        self.assertIs(raised.exception, failure)


class PostgresMigrationTests(unittest.TestCase):
    def test_postgresql_sql_uses_jsonb_and_an_expiry_index(self):
        buffer = StringIO()
        context = MigrationContext.configure(dialect_name="postgresql", opts={"as_sql": True, "output_buffer": buffer})
        migration = import_module("storage.migrations.versions.0002_evidence_snapshots")
        with Operations.context(context):
            migration.upgrade()
        self.assertIn("payload JSONB", buffer.getvalue())
        self.assertIn("CREATE INDEX ix_evidence_snapshots_expires_at", buffer.getvalue())
