import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from zipfile import ZipFile

import httpx
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from auth.settings import Settings
from domain.export import OUTLINE_LABEL, LEGACY_OUTLINE_SUMMARY, export_course, generation_summary, subject_origin
from domain.schemas import CourseProgramModel
from orchestration.call_log import (CALL_SCOPE, CallScope, RecordingModelClient, classify_outcome, sanitize_error)
from orchestration.checkpointer import open_checkpointer
from orchestration.clients import YandexModelClient, grounded_schema
from orchestration.graph import build_course_graph
from orchestration.llm_judge import JudgeVerdicts, judge_batch
from orchestration.resilience import ExternalUnavailable, Resilience
from run_rpd import YandexModeratedResponse
from storage.database import ensure_schema, make_engine
from storage.models import Base, Course, CourseVersion, Interview, ModelCall, SourceRow
from storage.portal import create_course, decide, finish_without_answer
from storage.workflow import SqlWorkflowStore, model_call_failures
from test_course_architecture import program, records
from test_portals import Model, user
from workers.course_worker import process_next


def timeout_error() -> ExternalUnavailable:
    try:
        try:
            raise TimeoutError("read timeout")
        except TimeoutError as cause:
            raise ExternalUnavailable("Yandex/responses: временный сбой внешнего сервиса.") from cause
    except ExternalUnavailable as error:
        return error


def batch_program(names: list[str]) -> dict:
    value = program()
    template = value["subjects"][0]
    subjects = []
    for number, name in enumerate(names):
        subject = json.loads(json.dumps(template))
        subject.update(id=f"s{number}", title=name, origin="auto", is_fundamental=False, locked=False, status="suggested")
        subject["justification"]["reason_type"] = "market_demand"
        for block in [*subject["topics"], *(a for t in subject["topics"] for a in t["assignments"])]:
            block.update(id=f"{block['id']}{number}", is_fundamental=False, locked=False)
        subjects.append(subject)
    return {**value, "subjects": subjects}


class OutcomeTests(unittest.TestCase):
    def test_outcomes_are_classified(self):
        try:
            CourseProgramModel.model_validate({})
        except ValidationError as error:
            validation = error
        cases = {
            "ok": None, "validation_error": validation, "timeout": timeout_error(),
            "circuit_open": ExternalUnavailable("Yandex/responses: запросы временно приостановлены."),
            "moderated": YandexModeratedResponse("Yandex AI Studio ограничила ответ модели."),
            "truncated": RuntimeError("Модель не вернула полный ответ (статус incomplete, max_output_tokens)."),
            "unavailable": ExternalUnavailable("HTTP 503"), "error": ValueError("other"),
        }
        for expected, error in cases.items():
            with self.subTest(expected=expected):
                self.assertEqual(classify_outcome(error), expected)

    def test_error_text_hides_secrets(self):
        text = sanitize_error(RuntimeError("Api-Key AQVN1234567890abcdefghijklmnopqrstuv failed; token=abc"))
        self.assertNotIn("AQVN1234567890abcdefghijklmnopqrstuv", text)
        self.assertNotIn("abc", text.split("token")[-1])
        self.assertIn("RuntimeError", text)

    def test_server_only_provenance_is_hidden_from_model_schema(self):
        schema = grounded_schema(CourseProgramModel, {"sources": records()})
        subject = schema["$defs"]["CourseSubject"]
        self.assertNotIn("generated_by", subject["properties"])
        self.assertNotIn("generation_note", subject["required"])

    def test_legacy_outline_is_recognised_and_old_version_has_unknown_origin(self):
        old = program()["subjects"][0]
        self.assertIsNone(subject_origin(old))
        legacy = {**old, "justification": {**old["justification"], "evidence_summary": LEGACY_OUTLINE_SUMMARY}}
        self.assertEqual(subject_origin(legacy), "outline")
        summary = generation_summary({"subjects": [old, legacy]})
        self.assertEqual((summary["outline"], summary["unknown"]), (1, 1))
        self.assertIn("причина не сохранена", summary["reasons"][0][0])


class RecordingClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_judge_transport_attempts_once_and_logs_failures(self):
        settings = {"YANDEX_MODEL_URI": "gpt://folder/yandexgpt", "YANDEX_FOLDER_ID": "folder"}
        request = [{"pair_id": "1", "claim": "Проверяет SQL", "source_type": "Books", "evidence": "Проверка SQL"}]
        cases = ((httpx.ReadTimeout("URL с приватными данными"), "timeout"),
                 (httpx.ConnectError("URL с приватными данными"), "network_error"),
                 (YandexModeratedResponse("Приватный фрагмент"), "moderated"),
                 (ExternalUnavailable("Запросы временно приостановлены."), "circuit_open"),
                 (None, "validation_error"))
        for error, outcome in cases:
            with self.subTest(outcome=outcome):
                sdk, policy = MagicMock(), Resilience(invoke=lambda call, _: call(), wait=MagicMock())
                sdk.responses.create.side_effect = error
                sdk.responses.create.return_value = SimpleNamespace(status="completed", output=[], output_text="{broken",
                                                                     incomplete_details=None, usage=None)
                entries = []

                async def record(entry):
                    entries.append(entry)

                token = CALL_SCOPE.set(CallScope(course_id="c", run_id="run", stage="subject_rationale"))
                try:
                    with patch("orchestration.clients.open_yandex") as open_client, patch("orchestration.yandex_calls.EXTERNAL", policy):
                        open_client.return_value.__enter__.return_value = sdk
                        result = await judge_batch(RecordingModelClient(YandexModelClient(settings), record),
                                                   JudgeVerdicts, request, subject_id="s")
                finally:
                    CALL_SCOPE.reset(token)
                self.assertIsNone(result)
                sdk.responses.create.assert_called_once()
                policy.wait.assert_not_called()
                self.assertEqual(len(entries), 1)
                entry = entries[0]
                self.assertEqual((entry["stage"], entry["subject_id"], entry["outcome"]), ("llm_judge", "s", outcome))
                self.assertNotIn("приватными", entry["error"])

    @patch("orchestration.clients.open_yandex")
    @patch("orchestration.clients.model_response")
    async def test_yandex_client_reports_model_and_tokens(self, response, open_client):
        open_client.return_value.__enter__.return_value = MagicMock()
        value = program()
        response.return_value = SimpleNamespace(
            usage=SimpleNamespace(input_tokens=1200, output_tokens=340), status="completed",
            incomplete_details=None, output=[], output_text=json.dumps(value, ensure_ascii=False))
        entries = []

        async def record(entry):
            entries.append(entry)

        client = RecordingModelClient(YandexModelClient({"YANDEX_MODEL_URI": "gpt://folder/yandexgpt",
                                                         "YANDEX_FOLDER_ID": "folder"}), record)
        CALL_SCOPE.set(CallScope(course_id="c1", stage="design"))
        await client.generate(CourseProgramModel, "prompt", {}, {"sources": records()})
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual((entry["stage"], entry["model"], entry["outcome"]), ("design", "gpt://folder/yandexgpt", "ok"))
        self.assertEqual((entry["input_tokens"], entry["output_tokens"]), (1200, 340))

    async def test_failed_call_is_recorded_and_reraised(self):
        entries = []

        async def record(entry):
            entries.append(entry)

        inner = MagicMock()

        async def fail(*args, **kwargs):
            raise timeout_error()

        inner.generate = fail
        CALL_SCOPE.set(CallScope(course_id="c1", stage="design", subject="Регрессия"))
        with self.assertRaises(ExternalUnavailable):
            await RecordingModelClient(inner, record).generate(CourseProgramModel, "p", {}, {})
        self.assertEqual(entries[0]["outcome"], "timeout")
        self.assertEqual(entries[0]["subject"], "Регрессия")
        self.assertIn("ExternalUnavailable", entries[0]["error"])


class OutlineProvenanceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        path = Path(self.tmp.name)
        self.engine = make_engine("sqlite+aiosqlite:///" + (path / "courses.db").as_posix())
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        self.factory = async_sessionmaker(self.engine, expire_on_commit=False)
        self.checkpoint = str(path / "checkpoints.db")
        self.store = SqlWorkflowStore(self.factory)
        self.settings = Settings(_env_file=None, session_signing_key="s" * 40, draft_retry_delays="30,120,300")
        self.delays = []
        self.evidence = MagicMock()

        async def search(*args, **kwargs):
            return list(records().values())

        self.evidence.search = search
        self.teacher = user("teacher")
        await create_course(self.factory, self.teacher, "course", {"title": "Курс", "institution": "Вуз", "year": "2026"}, [])

    async def asyncTearDown(self):
        await self.engine.dispose()
        self.tmp.cleanup()

    async def sleep(self, seconds):
        self.delays.append(seconds)

    async def process(self, model):
        async with open_checkpointer(self.checkpoint) as saver:
            graph = build_course_graph(self.settings, model, self.evidence, self.store, saver, sleep=self.sleep)
            self.assertTrue(await process_next(self.engine, self.factory, graph))
        async with self.factory() as db:
            return await db.get(Course, "course")

    async def test_failed_model_marks_outline_and_logs_reason(self):
        model = Model(gap=True)
        original = model.generate
        names = [f"Дисциплина {i}" for i in range(6)]

        async def generate(output, prompt, payload, context, flash=False, **options):
            if output.__name__ == "SubjectPlan":
                return output.model_validate({"subjects": [
                    {"name": name, "is_fundamental": False, "conflict_with": None, "source_ids": ["Books"]}
                    for name in names]}, context=context)
            if output.__name__ == "CourseProgramModel":
                if "Дисциплина 0" in payload["batch_subjects"]:
                    raise timeout_error()
                return output.model_validate(batch_program(payload["batch_subjects"]), context=context)
            return await original(output, prompt, payload, context, flash, **options)

        model.generate = generate
        course = await self.process(model)
        async with self.factory() as db:
            interview = await db.scalar(select(Interview))
        await finish_without_answer(self.factory, self.teacher, course.id, interview.id)
        course = await self.process(model)
        self.assertEqual(course.status, "waiting_for_review")
        async with self.factory() as db:
            version = await db.get(CourseVersion, course.draft_version_id)
            calls = (await db.scalars(select(ModelCall).where(ModelCall.stage == "design"))).all()
        subjects = {subject["title"]: subject for subject in version.content["program"]["subjects"]}
        outlined = [title for title, subject in subjects.items() if subject["generated_by"] == "outline"]
        self.assertEqual(outlined, ["Дисциплина 0"])
        self.assertIn("timeout", subjects["Дисциплина 0"]["generation_note"])
        self.assertIn("Повторов — 3", subjects["Дисциплина 0"]["generation_note"])
        self.assertEqual(self.delays, [30, 120, 300])
        self.assertTrue(all(subjects[title]["generated_by"] == "model" for title in names[1:]))
        self.assertEqual(sorted(call.outcome for call in calls), ["ok"] * 5 + ["timeout"] * 4)
        self.assertEqual(await model_call_failures(self.factory, course.id), [("design", "timeout", 4)])
        self.assertTrue(any("1 из 6 дисциплин собраны шаблоном" in note for note in version.content["source_notices"]))
        await decide(self.factory, self.teacher, course.id, version.id, "approve")
        course = await self.process(model)
        async with self.factory() as db:
            active = await db.get(CourseVersion, course.active_version_id)
            sources = {row.source_id: row.record for row in (await db.scalars(
                select(SourceRow).where(SourceRow.course_id == course.id))).all()}
        with ZipFile(BytesIO(export_course(course.course_meta, active, sources))) as package:
            documents = [json.loads(package.read(f"rpd/{index:02}/rpd.json")) for index in range(1, 7)]
            report = package.read("course-report.html").decode()
            html = package.read("rpd/01/rpd.html").decode()
        marked = [doc for doc in documents if doc.get("generationWarning") == OUTLINE_LABEL]
        self.assertEqual([doc["title"] for doc in marked], ["Дисциплина 0"])
        self.assertIn("timeout", marked[0]["generationNote"])
        self.assertIn("Собрано моделью — 5, шаблоном — 1", report)
        self.assertIn(OUTLINE_LABEL, html)


class MigrationTests(unittest.TestCase):
    def test_new_database_reaches_head_with_call_log(self):
        with tempfile.TemporaryDirectory() as directory:
            url = "sqlite+aiosqlite:///" + (Path(directory) / "app.sqlite").as_posix()
            ensure_schema(url)
            import sqlite3
            connection = sqlite3.connect(Path(directory) / "app.sqlite")
            try:
                tables = {row[0] for row in connection.execute("select name from sqlite_master where type='table'")}
                version = connection.execute("select version_num from alembic_version").fetchone()[0]
            finally:
                connection.close()
            self.assertTrue({"model_calls", "courses", "evidence_snapshots"} <= tables)
            self.assertEqual(version, "0006_profession_boundaries")

    def test_existing_database_gets_call_log_table(self):
        with tempfile.TemporaryDirectory() as directory:
            url = "sqlite+aiosqlite:///" + (Path(directory) / "app.sqlite").as_posix()
            from alembic import command
            from alembic.config import Config
            from storage.database import MIGRATIONS
            config = Config()
            config.set_main_option("script_location", str(MIGRATIONS))
            config.attributes["database_url"] = url
            command.upgrade(config, "0002_evidence_snapshots")
            ensure_schema(url)
            import sqlite3
            connection = sqlite3.connect(Path(directory) / "app.sqlite")
            try:
                tables = {row[0] for row in connection.execute("select name from sqlite_master where type='table'")}
                version = connection.execute("select version_num from alembic_version").fetchone()[0]
            finally:
                connection.close()
            self.assertIn("model_calls", tables)
            self.assertEqual(version, "0006_profession_boundaries")


if __name__ == "__main__":
    unittest.main()
