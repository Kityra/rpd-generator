import base64
import hashlib
import importlib.util
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import bcrypt
from argon2 import PasswordHasher
from langgraph.types import Command
from pydantic import ValidationError
from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from auth.password_verifier import ExistingPasswordVerifier
from auth.session import AuthService
from auth.settings import Settings
from auth.user_repository import SqlUserRepository
from domain.schemas import CourseProgramModel, ExpertClaimVerification, InterviewTurn, SourceRecord, SubjectRationale
from orchestration.checkpointer import open_checkpointer
from orchestration.clients import YandexEvidenceClient
from orchestration.graph import build_course_graph
from orchestration.resilience import ExternalUnavailable
from storage.models import Base, Course, CourseVersion, WorkflowCommand
from storage.workflow import SqlWorkflowStore


def records() -> dict:
    return {kind: SourceRecord(source_id=kind, source_type=kind, url=f"https://example.org/{kind}",
                              title=kind, content="Дейт К. Введение в системы баз данных. Реляционные модели данных и запросы SQL",
                              raw_hash="a" * 64, retrieved_at=datetime.now(timezone.utc))
            for kind in ("HH", "Crossref", "Books", "RPD", "Interview")}


def block(key: str, title: str) -> dict:
    return {"id": key, "title": title, "change_status": "added", "previous_value": None,
            "is_fundamental": True, "locked": True, "source_ids": ["Books"],
            "justification": {"reason_type": "fundamental_core", "evidence_summary": "Базовые модели данных", "source_ids": ["Books"]},
            "learning_references": [{"source_ids": ["Books"], "source_id": "Books", "locator": "Реляционные модели",
                                     "bibliographic_excerpt": "Дейт К. Введение в системы баз данных."}]}


def program() -> dict:
    assignment = {**block("assignment", "Практика SQL"), "instructions": "Составьте запрос",
                  "expected_artifacts": ["SQL"], "quality_criteria": ["Корректность", "Время"]}
    topic = {**block("topic", "SQL"), "activity_task_ids": ["task0"], "assignments": [assignment]}
    subject = {**block("subject", "Базы данных"), "origin": "manual", "status": "locked",
               "conflict_with": None, "topics": [topic]}
    return {"title": "Курс", "source_ids": ["Books"], "subjects": [subject], "business_audit": None}


def rationale_batch(request: dict) -> dict:
    items = []
    for subject in request["disciplines"]:
        choices = [item["excerpt_id"] for item in subject["fragments"] if item["index"] == "Books"]
        supported = choices and subject["title"] == "Базы данных"
        items.append({"subject_id": subject["subject_id"],
                      "explanation": ("Дисциплина даёт основу для составления запросов SQL к реляционным моделям данных."
                                      if supported else None),
                      "excerpt_ids": choices[:1] if supported else []})
    return {"items": items}


class SchemaTests(unittest.TestCase):
    def test_discipline_rationale_rejects_external_source_and_invented_quote(self):
        ctx = {"sources": records()}
        for source_id, excerpt in (("Crossref", "Реляционные модели данных и запросы SQL"),
                                   ("Books", "Дословная цитата, которой нет в книге")):
            with self.subTest(source=source_id), self.assertRaises(ValidationError):
                SubjectRationale.model_validate({
                    "explanation": "Изучаем реляционные модели данных для составления запросов.",
                    "source_ids": [source_id], "evidence": [{"source_id": source_id, "excerpt": excerpt}],
                }, context=ctx)

    def test_unconfirmed_claim_without_citation_requires_clarification(self):
        claim = {"claim": "Рабочий процесс", "explanation": "Нет подтверждения", "source_ids": [],
                 "verification_status": "needs_clarification", "clarification_question": "Как проверили результат?"}
        context = {"sources": records()}
        ExpertClaimVerification.model_validate(claim, context=context)
        with self.assertRaises(ValidationError):
            ExpertClaimVerification.model_validate(dict(claim, verification_status="verified"), context=context)

    @patch("orchestration.clients.open_yandex")
    @patch("orchestration.clients.search_fragments")
    def test_factcheck_continues_if_one_index_fails(self, search, open_client):
        sources = records()
        search.side_effect = [ExternalUnavailable("HTTP 500"), ((sources["HH"],), None), ((sources["Books"],), None)]
        settings = {f"YANDEX_VECTOR_STORE_ID_{name}": name for name in ("PRIKLAD", "HH", "BOOKS")}
        with self.assertLogs("orchestration.clients", level="WARNING"):
            found = YandexEvidenceClient(settings)._search("factcheck", "ML-инженерия", "teacher")
        self.assertEqual([source.source_type for source in found], ["HH", "Books"])
        self.assertEqual(found.notices, ["Индекс Priklad недоступен."])

    def test_supported_password_formats(self):
        password = "test-password"
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), b"salt", 1000)
        hashes = [PasswordHasher().hash(password), bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode(),
                  "pbkdf2_sha256$1000$salt$" + base64.b64encode(digest).decode()]
        for stored in hashes:
            self.assertTrue(ExistingPasswordVerifier().verify(password, stored))
            self.assertFalse(ExistingPasswordVerifier().verify("wrong", stored))

    def test_unknown_hash_fails_closed(self):
        with self.assertRaises(NotImplementedError):
            ExistingPasswordVerifier().verify("password", "unknown")

    def test_interviewer_cannot_return_generation_instructions(self):
        with self.assertRaises(ValidationError):
            InterviewTurn(prompt="Верни JSON по gap_report?")

    def test_invented_source_rejected(self):
        value = program()
        value["subjects"][0]["source_ids"] = ["invented"]
        with self.assertRaises(ValidationError):
            CourseProgramModel.model_validate(value, context={"sources": records()})

    def test_fabricated_learning_locator_rejected(self):
        value = program()
        value["subjects"][0]["learning_references"][0]["locator"] = "Вымышленный учебник"
        with self.assertRaises(ValidationError):
            CourseProgramModel.model_validate(value, context={"sources": records()})

    def test_protected_core_cannot_be_removed(self):
        ctx = {"sources": records()}
        baseline = CourseProgramModel.model_validate(program(), context=ctx)
        value = program()
        value["subjects"][0]["topics"][0]["assignments"] = []
        with self.assertRaises(ValidationError):
            CourseProgramModel.model_validate(value, context={**ctx, "baseline": baseline})

    def test_manual_subject_preserved(self):
        with self.assertRaises(ValidationError):
            CourseProgramModel.model_validate(program(), context={"sources": records(), "manual_subjects": ["Сети"]})


class DatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.url = "sqlite+aiosqlite:///" + str(Path(self.directory.name) / "application.sqlite")
        self.engine = create_async_engine(self.url)
        self.factory = async_sessionmaker(self.engine, expire_on_commit=False)
        self.settings = Settings(_env_file=None, session_signing_key="test-signing-key-with-32-characters", user_database_url=self.url)
        async with self.engine.begin() as connection:
            await connection.execute(text("CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT, password_hash TEXT, role TEXT, display_name TEXT)"))
            await connection.execute(text("INSERT INTO users VALUES (1, 'teacher@example.org', :hash, 'teacher', 'Teacher')"),
                                     {"hash": PasswordHasher().hash("test-password")})
            await connection.run_sync(Base.metadata.create_all)
        self.users = SqlUserRepository(self.engine, self.settings)
        await self.users.initialize()

    async def asyncTearDown(self):
        await self.engine.dispose()

    async def test_existing_user_read_only(self):
        user = await self.users.get_by_email("teacher@example.org")
        self.assertEqual(user.id, "users:1")
        async with self.engine.connect() as connection:
            self.assertEqual(await connection.scalar(text("SELECT COUNT(*) FROM users")), 1)

    async def test_login_logout_and_no_plaintext_storage(self):
        service = AuthService(self.settings, self.users, ExistingPasswordVerifier(), self.factory)
        token = await service.login("teacher@example.org", "test-password", "127.0.0.1", "test")
        self.assertEqual((await service.authenticate(token)).role, "teacher")
        async with self.engine.connect() as connection:
            rows = str((await connection.execute(text("SELECT * FROM sessions"))).all())
            self.assertNotIn("test-password", rows)
            self.assertNotIn(token, rows)
        await service.logout(token, "127.0.0.1", "test")
        self.assertIsNone(await service.authenticate(token))

    async def test_five_failures_lock_login(self):
        service = AuthService(self.settings, self.users, ExistingPasswordVerifier(), self.factory)
        for _ in range(5):
            with self.assertRaises(PermissionError):
                await service.login("teacher@example.org", "wrong", "127.0.0.1", "test")
        with self.assertRaises(PermissionError):
            await service.login("teacher@example.org", "test-password", "127.0.0.1", "test")

    async def test_migration_keeps_existing_user_schema(self):
        from alembic.migration import MigrationContext
        from alembic.operations import Operations
        other = create_async_engine("sqlite+aiosqlite:///:memory:")
        try:
            async with other.begin() as connection:
                await connection.execute(text("CREATE TABLE users (id INTEGER PRIMARY KEY, login TEXT)"))
                def migrate(sync):
                    before = inspect(sync).get_columns("users")
                    path = Path(__file__).resolve().parents[1] / "storage/migrations/versions/0001_course_workflow.py"
                    spec = importlib.util.spec_from_file_location("migration", path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    module.op = Operations(MigrationContext.configure(sync))
                    module.upgrade()
                    after = inspect(sync).get_columns("users")
                    self.assertEqual([{**item, "type": str(item["type"])} for item in before],
                                     [{**item, "type": str(item["type"])} for item in after])
                await connection.run_sync(migrate)
        finally:
            await other.dispose()

    async def test_invitation_is_scoped_and_submission_idempotent(self):
        async with self.engine.begin() as connection:
            await connection.execute(text("INSERT INTO users VALUES (2, 'expert@example.org', :hash, 'expert', 'Expert')"),
                                     {"hash": PasswordHasher().hash("expert-password")})
        async with self.factory.begin() as db:
            db.add(Course(id="interview-course", owner_user_id="users:1", thread_id="interview-course", course_meta={}))
        store = SqlWorkflowStore(self.factory)
        interview = await store.prepare_interview("interview-course", "interrupt", {"prompt": "Вопрос"})
        teacher = await self.users.get_by_id("users:1")
        expert = await self.users.get_by_id("users:2")
        token = await store.invite(interview, expert.id, teacher, self.users)
        with self.assertRaises(PermissionError):
            await store.submit_interview(token, teacher, "interrupt", "submission", "Ответ", [])
        await store.submit_interview(token, expert, "interrupt", "submission", "Ответ", [])
        await store.submit_interview(token, expert, "interrupt", "submission", "Ответ", [])
        with self.assertRaises(ValueError):
            await store.submit_interview(token, expert, "interrupt", "submission", "Другой ответ", [])
        async with self.factory() as db:
            self.assertIsNotNone(await db.get(WorkflowCommand, f"interview:{interview}:interrupt"))

    async def test_graph_persists_interrupt_and_resumes_without_agents_repeated(self):
        source_map = records()
        evidence = AsyncMock()
        evidence.search.side_effect = lambda role, query, **kwargs: list(source_map.values())
        llm = AsyncMock()
        activity_calls = 0
        audit_calls = 0
        async def generate(output, prompt, payload, context, flash=False):
            nonlocal activity_calls, audit_calls
            if output.__name__ == "AgentReport":
                value = {"findings": []}
            elif output.__name__ == "AppliedActivityModel":
                activity_calls += 1
                tasks = [{"id": f"task{i}", "title": f"Задача {i}", "stack": ["SQL"], "artifacts": ["Запрос"],
                          "quality_criteria": ["Корректность", "Время"], "typical_errors": ["Ошибка"],
                          "source_ids": ["HH", "Crossref"]} for i in range(4 if activity_calls == 1 else 5)]
                value = {"tasks": tasks, "processes": [{"id": str(i), "title": "Процесс", "task_ids": ["task0"],
                         "source_ids": ["HH"]} for i in range(3)], "aspects": [], "source_ids": ["HH"]}
            elif output.__name__ == "SubjectPlan":
                value = {"subjects": [{"name": "Базы данных", "is_fundamental": True, "conflict_with": None, "source_ids": ["Books"]}]}
            elif output.__name__ == "InterviewTurn":
                value = {"prompt": "В чём состоит пятая рабочая задача?"}
            elif output.__name__ == "VerificationReport":
                value = {"claims": [{"claim": "Проверка SQL", "verification_status": "verified", "explanation": "Подтверждено",
                         "clarification_question": None, "source_ids": ["HH"]}]}
            elif output.__name__ == "CourseProgramModel":
                value = program()
            elif output.__name__ == "SubjectRationaleBatch":
                value = rationale_batch(payload)
            else:
                audit_calls += 1
                value = {"source_ids": ["HH"], "market_fit_score": 0.7, "metrics": [], "unavailable_metrics": ["active_vacancies"],
                         "evidence_summary": "Практика", "suggested_changes": [
                             {"reason_type": "market_demand", "evidence_summary": "Практика SQL", "source_ids": ["HH"]}
                         ] if audit_calls == 1 else []}
            return output.model_validate(value, context=context)
        llm.generate.side_effect = generate
        async with self.factory.begin() as db:
            db.add(Course(id="course", owner_user_id="users:1", thread_id="course", course_meta={"title": "Курс"}))
        store = SqlWorkflowStore(self.factory)
        config = {"configurable": {"thread_id": "course"}}
        checkpoint_path = str(Path(self.directory.name) / "checkpoints.sqlite")
        async with open_checkpointer(checkpoint_path) as saver:
            graph = build_course_graph(self.settings, llm, evidence, store, saver)
            first = await graph.ainvoke({"request_id": "course", "owner_user_id": "users:1", "run_id": "run1", "mode": "create",
                                        "course_meta": {"title": "Курс"}, "manual_subjects": ["Базы данных"]}, config)
            self.assertEqual(first["status"], "waiting_for_expert")
        async with open_checkpointer(checkpoint_path) as saver:
            graph = build_course_graph(self.settings, llm, evidence, store, saver)
            state = await graph.aget_state(config)
            answer = {"interrupt_id": state.values["interrupt_id"], "content": "Пятая задача", "attachments": []}
            second = await graph.ainvoke(Command(resume=answer), config)
            self.assertEqual(second["status"], "waiting_for_review")
            self.assertEqual(audit_calls, 2)
            self.assertEqual(sum(call.args[0] == "business" for call in evidence.search.call_args_list), 1)
            await graph.ainvoke(Command(resume={"actor_user_id": "users:1", "version_id": second["version_id"], "action": "approve"}), config)
        async with self.factory() as db:
            course = await db.get(Course, "course")
            self.assertEqual(course.status, "approved")
            self.assertIsNotNone(course.next_audit_at)
            self.assertEqual((await db.get(CourseVersion, course.active_version_id)).status, "approved")
