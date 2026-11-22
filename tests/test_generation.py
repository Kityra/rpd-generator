"""Этап 2: надёжная генерация черновика — по одной дисциплине, повторы, обрезанные ответы, цитаты."""
import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
from openai import InternalServerError
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from auth.settings import Settings
from domain.schemas import CourseProgramModel, SourceRecord
from orchestration import yandex_calls
from orchestration.call_log import GENERATION_NOTES
from orchestration.checkpointer import open_checkpointer
from orchestration.clients import YandexModelClient, request_budget
from orchestration.draft import make_outline
from orchestration.graph import build_course_graph
from orchestration.resilience import ExternalUnavailable, Resilience
from storage.database import make_engine
from storage.models import Base, Course, CourseVersion, Interview, ModelCall
from storage.portal import create_course, finish_without_answer
from storage.workflow import SqlWorkflowStore
from test_course_architecture import program, records
from test_model_calls import batch_program, timeout_error
from test_portals import Model, user
from workers.course_worker import process_next


def reply(text: str, tokens=(10, 20)):
    return SimpleNamespace(incomplete_details=None, output=[], status="completed", output_text=text,
                           usage=SimpleNamespace(input_tokens=tokens[0], output_tokens=tokens[1]))


def truncated_json_error() -> ValidationError:
    try:
        CourseProgramModel.model_validate_json('{"title": "Курс", "subjects": [', context={"sources": {}})
    except ValidationError as error:
        return error


class BreakerTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.waits = []
        self.policy = Resilience(clock=lambda: self.now, wait=self.advance, jitter=lambda low, high: low,
                                 invoke=lambda request, remaining: request())

    def advance(self, seconds):
        self.waits.append(seconds)
        self.now += seconds

    def fail(self, timeout):
        response = httpx.Response(503, request=httpx.Request("POST", "https://example.org"))
        raise InternalServerError("error", response=response, body=None)

    def test_one_failing_request_does_not_stop_parallel_disciplines(self):
        service = ("Yandex", "responses", "scope", "model")
        for _ in range(4):
            with self.assertRaises(ExternalUnavailable):
                self.policy.call(service, self.fail, budget=8, read=3, request_key="subject-1")
        # Другая дисциплина продолжает работу: сбои одного запроса считаются один раз.
        self.assertEqual(self.policy.call(service, lambda timeout: "ok", budget=8, read=3, request_key="subject-2"), "ok")

    def test_breaker_opens_after_distinct_requests_and_waits_softly(self):
        service = ("Yandex", "responses", "scope", "model")
        for key in ("a", "b", "c"):
            with self.assertRaises(ExternalUnavailable):
                self.policy.call(service, self.fail, budget=8, read=3, request_key=key)
        with self.assertRaises(ExternalUnavailable) as raised:
            self.policy.call(service, lambda timeout: "ok", budget=8, read=3, request_key="d")
        self.assertIn("приостановлены", str(raised.exception))
        self.waits.clear()
        result = self.policy.call(service, lambda timeout: "ok", budget=60, read=3, request_key="e",
                                  cooldown=10, wait_if_open=True)
        self.assertEqual(result, "ok")
        self.assertEqual(len(self.waits), 1)

    def test_own_timeout_of_long_answer_does_not_open_breaker(self):
        service = ("Yandex", "responses", "scope", "model")

        def slow(timeout):
            raise TimeoutError("read timeout")

        for key in ("a", "b", "c", "d"):
            with self.assertRaises(ExternalUnavailable):
                self.policy.call(service, slow, budget=8, read=3, request_key=key, count_timeouts=False)
        self.assertEqual(self.policy.call(service, lambda timeout: "ok", budget=8, read=3, request_key="e"), "ok")


class ClientTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.settings = {"YANDEX_FOLDER_ID": "folder", "YANDEX_API_KEY": "fixture", "YANDEX_MODEL_URI": "model",
                         "YANDEX_TEMPERATURE": "0.1", "YANDEX_DRAFT_MAX_OUTPUT_TOKENS": "12000"}
        self.client = MagicMock()
        patcher = patch.object(yandex_calls, "EXTERNAL", Resilience(wait=lambda delay: None))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_budget_grows_with_request_size(self):
        self.assertLess(request_budget(2000, 4096), request_budget(80000, 12000))
        self.assertLessEqual(request_budget(10 ** 7, 10 ** 6), 600)

    async def test_draft_call_uses_fixed_temperature_larger_limit_and_stream(self):
        value = program()
        events = [SimpleNamespace(type="response.output_text.delta"),
                  SimpleNamespace(type="response.completed", response=reply(json.dumps(value, ensure_ascii=False)))]
        self.client.responses.create.return_value = iter(events)
        model = YandexModelClient(self.settings)
        with patch("orchestration.clients.open_yandex", return_value=nullcontext(self.client)):
            await model.generate(CourseProgramModel, "prompt", {"batch_subjects": ["Базы данных"]},
                                 {"sources": records()})
        arguments = self.client.responses.create.call_args.kwargs
        self.assertEqual(arguments["temperature"], 0.1)
        self.assertEqual(arguments["max_output_tokens"], 12000)
        self.assertTrue(arguments["stream"])
        self.assertEqual(model.describe()["temperature"], 0.1)

    async def test_failed_quote_drops_reference_and_keeps_discipline(self):
        value = program()
        topic = value["subjects"][0]["topics"][0]
        topic["activity_task_ids"] = []
        topic["learning_references"][0]["locator"] = "Цитата, которой нет в источнике"
        self.client.responses.create.return_value = reply(json.dumps(value, ensure_ascii=False))
        notes = []
        GENERATION_NOTES.set(notes)
        model = YandexModelClient(self.settings)
        with patch("orchestration.clients.open_yandex", return_value=nullcontext(self.client)):
            result = await model.generate(CourseProgramModel, "prompt", {}, {
                "sources": records(), "allow_incomplete_activity": True})
        self.assertEqual(result.subjects[0].topics[0].learning_references, [])
        self.assertEqual(len(result.subjects[0].learning_references), 1)
        self.assertEqual(self.client.responses.create.call_count, 1)
        self.assertTrue(notes and "неподтверждённые учебные ссылки — 1" in notes[0])

    async def test_other_validation_errors_still_go_to_repair(self):
        value = program()
        value["subjects"][0]["source_ids"] = ["invented"]
        self.client.responses.create.side_effect = [reply(json.dumps(value, ensure_ascii=False)),
                                                    reply(json.dumps(program(), ensure_ascii=False))]
        with patch("orchestration.clients.open_yandex", return_value=nullcontext(self.client)):
            await YandexModelClient(self.settings).generate(CourseProgramModel, "prompt", {}, {"sources": records()})
        self.assertEqual(self.client.responses.create.call_count, 2)


class OutlineTests(unittest.TestCase):
    def test_outline_uses_matching_sources_and_marks_fundamental(self):
        sources = records()
        ctx = {"sources": sources, "manual_subjects": [], "allow_incomplete_activity": True}
        result = make_outline("Курс", ["Базы данных"], [
            {"name": "Базы данных", "is_fundamental": True, "conflict_with": None, "source_ids": ["Books"]}], ctx)
        subject = result.subjects[0]
        self.assertTrue(subject.is_fundamental and subject.locked)
        self.assertIn("Books", subject.source_ids)
        self.assertEqual(subject.topics[0].title, "Основные понятия дисциплины «Базы данных»")

    def test_outline_without_materials_fails_loudly(self):
        sources = {key: value for key, value in records().items() if key == "Interview"}
        with self.assertRaises(ValueError):
            make_outline("Курс", ["Квантовая химия"], [], {"sources": sources, "manual_subjects": []})


class DesignBatchedTests(unittest.IsolatedAsyncioTestCase):
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

    async def run_draft(self, design):
        model = Model(gap=True)
        original = model.generate
        names = [f"Дисциплина {i}" for i in range(3)]

        async def generate(output, prompt, payload, context, flash=False, **options):
            if output.__name__ == "SubjectPlan":
                return output.model_validate({"subjects": [
                    {"name": name, "is_fundamental": False, "conflict_with": None, "source_ids": ["Books"]}
                    for name in names]}, context=context)
            if output.__name__ in {"CourseProgramModel", "TopicAssignments"}:
                return await design(output, payload, context)
            return await original(output, prompt, payload, context, flash, **options)

        model.generate = generate
        for _ in range(2):
            async with open_checkpointer(self.checkpoint) as saver:
                graph = build_course_graph(self.settings, model, self.evidence, self.store, saver, sleep=self.sleep)
                self.assertTrue(await process_next(self.engine, self.factory, graph))
            async with self.factory() as db:
                course = await db.get(Course, "course")
                interview = await db.scalar(select(Interview))
            if course.status == "waiting_for_expert":
                await finish_without_answer(self.factory, self.teacher, course.id, interview.id)
        self.assertEqual(course.status, "waiting_for_review")
        async with self.factory() as db:
            version = await db.get(CourseVersion, course.draft_version_id)
        return {subject["title"]: subject for subject in version.content["program"]["subjects"]}

    async def test_each_discipline_is_a_separate_call_and_timeout_is_retried(self):
        calls = []

        async def design(output, payload, context):
            name = payload["batch_subjects"][0]
            calls.append(tuple(payload["batch_subjects"]))
            if name == "Дисциплина 1" and calls.count(("Дисциплина 1",)) <= 2:
                raise timeout_error()
            if name == "Дисциплина 2":
                raise ExternalUnavailable("Yandex/responses: запросы временно приостановлены.")
            return output.model_validate(batch_program(payload["batch_subjects"]), context=context)

        subjects = await self.run_draft(design)
        self.assertTrue(all(len(call) == 1 for call in calls))
        self.assertEqual(subjects["Дисциплина 0"]["generated_by"], "model")
        self.assertEqual(subjects["Дисциплина 1"]["generated_by"], "model")
        self.assertEqual(subjects["Дисциплина 2"]["generated_by"], "outline")
        self.assertIn("circuit_open", subjects["Дисциплина 2"]["generation_note"])
        self.assertEqual(sorted(self.delays), sorted([30, 120] + [30, 120, 300]))

    async def test_truncated_answer_is_split_into_topics_and_assignments(self):
        calls = []

        async def design(output, payload, context):
            calls.append((output.__name__, payload.get("assignments_separately", False)))
            if output.__name__ == "TopicAssignments":
                assignment = batch_program(payload["batch_subjects"])["subjects"][0]["topics"][0]["assignments"][0]
                return output.model_validate({"assignments": [assignment]}, context=context)
            if not payload.get("assignments_separately"):
                raise truncated_json_error()
            value = batch_program(payload["batch_subjects"])
            for topic in value["subjects"][0]["topics"]:
                topic["assignments"] = []
            return output.model_validate(value, context=context)

        subjects = await self.run_draft(design)
        for subject in subjects.values():
            self.assertEqual(subject["generated_by"], "model")
            self.assertEqual(len(subject["topics"][0]["assignments"]), 1)
        self.assertEqual(calls.count(("TopicAssignments", False)), 3)
        self.assertEqual(calls.count(("CourseProgramModel", True)), 3)
        self.assertEqual(self.delays, [])

    async def test_moderated_discipline_goes_to_outline_without_waiting(self):
        from run_rpd import YandexModeratedResponse

        async def design(output, payload, context):
            if payload["batch_subjects"] == ["Дисциплина 0"]:
                raise YandexModeratedResponse("Yandex AI Studio ограничила ответ модели.")
            return output.model_validate(batch_program(payload["batch_subjects"]), context=context)

        subjects = await self.run_draft(design)
        self.assertEqual(subjects["Дисциплина 0"]["generated_by"], "outline")
        self.assertIn("moderated", subjects["Дисциплина 0"]["generation_note"])
        self.assertEqual(self.delays, [])
        async with self.factory() as db:
            outcomes = [row.outcome for row in (await db.scalars(select(ModelCall).where(ModelCall.stage == "design"))).all()]
        self.assertEqual(sorted(outcomes), ["moderated", "ok", "ok"])


class ContextLimitTests(unittest.TestCase):
    """Реальная причина «0 из 11 моделью»: запрос ≈64 тыс. токенов при лимите модели 32 768."""

    def test_big_request_is_shrunk_to_model_limit(self):
        from orchestration.clients import estimate_tokens, fit_request
        sources = [{"source_id": f"s{i}", "content": "Текст источника. " * 400} for i in range(60)]
        request = {"sources": sources, "course_program": {"title": "Курс", "subjects": [
            {"title": f"Д{i}", "topics": [{"title": "Тема", "assignments": [{"instructions": "x" * 3000}]}]}
            for i in range(11)]}}
        fitted, steps = fit_request(request, "инструкция", {"type": "object"}, 29000)
        self.assertTrue(steps)
        self.assertLessEqual(estimate_tokens("инструкция", '{"type": "object"}', json.dumps(fitted, ensure_ascii=False)),
                             29000)
        self.assertEqual(fitted["course_program"]["subjects"][0], {"title": "Д0", "topics": ["Тема"]})

    def test_hopeless_request_raises_clear_error(self):
        from orchestration.clients import InputTooLarge, fit_request
        with self.assertRaises(InputTooLarge):
            fit_request({"text": "x" * 200000}, "", {}, 1000)

    def test_failed_status_with_token_limit_is_input_too_large(self):
        from orchestration.call_log import classify_outcome
        from orchestration.clients import InputTooLarge
        failed = SimpleNamespace(status="failed", incomplete_details=None, output=[], output_text="", usage=None,
                                 error=SimpleNamespace(message="number of input tokens must be no more than 32768, "
                                                               "got 64270"))
        client = MagicMock()
        client.responses.create.return_value = failed
        settings = {"YANDEX_FOLDER_ID": "f", "YANDEX_API_KEY": "k", "YANDEX_MODEL_URI": "m",
                    "YANDEX_STREAM_LONG_RESPONSES": "0"}
        with patch.object(yandex_calls, "EXTERNAL", Resilience(wait=lambda delay: None)), \
             patch("orchestration.clients.open_yandex", return_value=nullcontext(client)):
            with self.assertRaises(InputTooLarge) as raised:
                YandexModelClient(settings)._generate(CourseProgramModel, "p", {}, {"sources": records()}, False)
        self.assertEqual(classify_outcome(raised.exception), "input_too_large")

    def test_fundamental_block_is_locked_automatically_in_first_draft(self):
        from orchestration.clients import parse_output
        value = program()
        assignment = value["subjects"][0]["topics"][0]["assignments"][0]
        assignment.update(locked=False, change_status="modified", previous_value="старое")
        result = parse_output(CourseProgramModel, json.dumps(value, ensure_ascii=False), {"sources": records()})
        block = result.subjects[0].topics[0].assignments[0]
        self.assertTrue(block.locked)
        self.assertEqual((block.change_status, block.previous_value), ("added", None))


class SubjectRequestTests(unittest.TestCase):
    def test_request_has_only_related_tasks_sources_and_outline_hint(self):
        from orchestration.graph import subject_request
        sources = records()
        for index in range(40):
            extra = SourceRecord.model_validate({**sources["Books"].model_dump(), "source_id": f"x{index}",
                                                 "content": "Посторонний текст о геологии. " * 200})
            sources[extra.source_id] = extra
        tasks = [{"id": "task0", "title": "Составление запросов SQL к базам данных", "stack": ["SQL"],
                  "artifacts": ["Запрос"], "quality_criteria": ["a", "b"], "source_ids": ["HH"]},
                 {"id": "task1", "title": "Полевая геология", "stack": ["Компас"], "artifacts": ["Карта"],
                  "quality_criteria": ["a", "b"], "source_ids": ["HH"]}]
        state = {"course_meta": {"title": "Аналитик", "institution": "Вуз"}, "applied_activity": {"tasks": tasks},
                 "gap_report": {"gap_types": ["missing_tasks"]}}
        suggestions = [{"name": "Базы данных", "is_fundamental": False, "conflict_with": None, "source_ids": ["Books"]}]
        ctx = {"sources": sources, "manual_subjects": [], "allow_incomplete_activity": True}
        request, allowed = subject_request(state, ["Базы данных"], suggestions, ctx)
        self.assertEqual([task["id"] for task in request["applied_activity"]["tasks"]], ["task0"])
        self.assertLessEqual(len(request["sources"]), 8)
        self.assertIn("Books", allowed)
        self.assertNotIn("Interview", allowed)
        self.assertIn("outline_hint", request)
        self.assertNotIn("institution", request["course"])
        self.assertLess(len(json.dumps(request, ensure_ascii=False)), 20000)


if __name__ == "__main__":
    unittest.main()
