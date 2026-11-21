from datetime import datetime, timedelta, timezone
from importlib import import_module
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import AsyncMock, patch

from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.operations import Operations
import httpx
from openai import AuthenticationError
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import async_sessionmaker

from auth.session import aware
from auth.user_repository import CurrentUser
from domain.workflow_queue import MAX_ATTEMPTS
from orchestration.resilience import ExternalUnavailable
from run_rpd import YandexModeratedResponse
from storage.database import DatabaseSetupError, ensure_schema, make_engine
from storage.models import Base, Course, WorkflowCommand
from storage.portal import create_course, retry_command
from workers import course_worker
from workers.retry_policy import classify_failure


ROOT = Path(__file__).resolve().parents[1]


class QueueRetryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        folder = self.enterContext(TemporaryDirectory())
        self.url = "sqlite+aiosqlite:///" + (Path(folder) / "application.sqlite").as_posix()
        self.engine = make_engine(self.url)
        self.addAsyncCleanup(self.engine.dispose)
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        self.factory = async_sessionmaker(self.engine, expire_on_commit=False)
        self.teacher = CurrentUser(id="teacher", email="test@example.org", display_name="Teacher", role="teacher")
        self.clock = datetime(2026, 11, 21, 10, tzinfo=timezone.utc)
        self.enterContext(patch.object(course_worker, "now", side_effect=lambda: self.clock))
        self.enterContext(patch("workers.retry_policy.uniform", side_effect=lambda first, last: first))
        self.execute = self.enterContext(patch.object(course_worker, "execute", new_callable=AsyncMock))

    async def enqueue(self, course_id):
        await create_course(self.factory, self.teacher, course_id, {"title": course_id}, [])

    async def read(self, course_id):
        async with self.factory() as db:
            return await db.get(WorkflowCommand, f"create:{course_id}"), await db.get(Course, course_id)

    async def process(self, engine=None):
        factory = async_sessionmaker(engine, expire_on_commit=False) if engine is not None else self.factory
        return await course_worker.process_next(engine or self.engine, factory, None)

    async def test_delayed_command_survives_restart_and_other_course_runs(self):
        await self.enqueue("first")
        await self.enqueue("second")
        self.execute.side_effect = [ExternalUnavailable(), {"status": "waiting_for_review"},
                                    {"status": "waiting_for_expert"}]
        self.assertTrue(await self.process())
        first, course = await self.read("first")
        self.assertEqual((first.status, first.attempt_count, first.last_error_code, course.status),
                         ("pending", 1, "upstream_unavailable", "queued"))
        self.assertEqual(aware(first.next_attempt_at) - self.clock, timedelta(seconds=30))
        self.assertTrue(await self.process())
        self.assertEqual((await self.read("second"))[0].status, "done")
        self.assertFalse(await self.process())
        self.clock = aware(first.next_attempt_at)
        new_engine = make_engine(self.url)
        self.addAsyncCleanup(new_engine.dispose)
        self.assertTrue(await self.process(new_engine))
        retried, course = await self.read("first")
        self.assertEqual(retried.id, first.id)
        self.assertEqual((retried.status, retried.attempt_count, retried.next_attempt_at, course.status),
                         ("done", 2, None, "waiting_for_expert"))
        self.assertEqual([call.args[2].course_id for call in self.execute.await_args_list], ["first", "second", "first"])

    async def test_limit_and_manual_retry_reset_the_same_command(self):
        await self.enqueue("course")
        self.execute.side_effect = TimeoutError
        for expected in range(1, MAX_ATTEMPTS + 1):
            self.assertTrue(await self.process())
            cmd, course = await self.read("course")
            self.assertEqual(cmd.attempt_count, expected)
            self.assertEqual(cmd.last_error_code, "timeout")
            if expected < MAX_ATTEMPTS:
                self.assertEqual(course.status, "queued")
                self.clock = aware(cmd.next_attempt_at)
        self.assertEqual((cmd.status, course.status, cmd.next_attempt_at), ("failed", "failed", None))
        self.assertFalse(await self.process())
        await retry_command(self.factory, self.teacher, "course")
        reset, _ = await self.read("course")
        self.assertEqual((reset.id, reset.attempt_count, reset.last_error_code), (cmd.id, 0, None))
        self.execute.side_effect = None
        self.execute.return_value = {"status": "waiting_for_review"}
        self.assertTrue(await self.process())
        self.assertEqual((await self.read("course"))[0].attempt_count, 1)

    async def test_permanent_failure_is_not_rescheduled(self):
        await self.enqueue("course")
        response = httpx.Response(401, request=httpx.Request("POST", "https://example.org"))
        self.execute.side_effect = AuthenticationError("private", response=response, body=None)
        self.assertTrue(await self.process())
        cmd, course = await self.read("course")
        self.assertEqual((cmd.status, cmd.attempt_count, cmd.last_error_code, cmd.next_attempt_at, course.status),
                         ("failed", 1, "authorization", None, "failed"))
        self.assertFalse(await self.process())

    async def test_crash_leaves_a_reserved_attempt_that_the_worker_can_resume(self):
        await self.enqueue("course")
        with patch.object(course_worker, "finish_command", side_effect=RuntimeError("interruption")):
            with self.assertRaisesRegex(RuntimeError, "interruption"):
                await self.process()
        reserved, _ = await self.read("course")
        self.assertEqual((reserved.status, reserved.attempt_count), ("running", 1))
        self.assertTrue(await self.process())
        actual, _ = await self.read("course")
        self.assertEqual((actual.status, actual.attempt_count), ("done", 1))


class RetryPolicyTests(unittest.TestCase):
    def test_queue_honours_rate_limit_and_rejects_moderation(self):
        response = httpx.Response(429, headers={"Retry-After": "75"},
                                  request=httpx.Request("POST", "https://example.org"))
        with self.assertRaises(httpx.HTTPStatusError) as http:
            response.raise_for_status()
        failure = classify_failure(ExternalUnavailable())  # Open circuit has no HTTP cause.
        self.assertTrue(failure.retryable)
        wrapped = ExternalUnavailable()
        wrapped.__cause__ = http.exception
        rate = classify_failure(wrapped)
        self.assertEqual(rate.code, "rate_limited")
        self.assertGreaterEqual(rate.delay(1), timedelta(seconds=75))
        self.assertFalse(classify_failure(YandexModeratedResponse()).retryable)
        self.assertFalse(classify_failure(ValueError()).retryable)


class RetryMigrationTests(unittest.TestCase):
    def test_upgrade_existing_pending_command(self):
        with TemporaryDirectory() as folder:
            filename = (Path(folder) / "retry.sqlite").as_posix()
            url = "sqlite+aiosqlite:///" + filename
            cfg = Config()
            cfg.set_main_option("script_location", (ROOT / "storage" / "migrations").as_posix())
            cfg.attributes["database_url"] = url
            command.upgrade(cfg, "0002_evidence_snapshots")
            engine = create_engine("sqlite:///" + filename)
            try:
                with engine.begin() as db:
                    db.execute(text("INSERT INTO workflow_commands VALUES ('existing', 'course', 'create', '{}', 'pending', CURRENT_TIMESTAMP)"))
                with self.assertRaisesRegex(DatabaseSetupError, "alembic upgrade head"):
                    ensure_schema(url)
                command.upgrade(cfg, "head")
                ensure_schema(url)
                with engine.connect() as db:
                    row = db.execute(text("SELECT status, attempt_count, next_attempt_at, last_error_code "
                                          "FROM workflow_commands WHERE id='existing'")).one()
                    self.assertEqual(tuple(row), ("pending", 0, None, None))
            finally:
                engine.dispose()

    def test_postgresql_migration_types(self):
        output = StringIO()
        ctx = MigrationContext.configure(dialect_name="postgresql", opts={"as_sql": True, "output_buffer": output})
        with Operations.context(ctx):
            import_module("storage.migrations.versions.0003_workflow_retries").upgrade()
        self.assertIn("attempt_count INTEGER DEFAULT 0 NOT NULL", output.getvalue())
        self.assertIn("next_attempt_at TIMESTAMP WITH TIME ZONE", output.getvalue())
