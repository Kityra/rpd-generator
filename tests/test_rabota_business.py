from concurrent.futures import Future
from contextlib import asynccontextmanager
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from langgraph.checkpoint.memory import InMemorySaver

from orchestration.clients import source_record
from orchestration.rabota_business import BusinessEvidenceClient, RabotaBusiness
from rabota_it import VacancyStore, import_export
import run_rpd
from workers import course_worker


VACANCY_URL = "https://www.rabota.ru/vacancy/101"
VACANCY = {
    "id": 101, "title": "Python разработчик", "url": VACANCY_URL,
    "requirements": "Поддерживать сервисы на FastAPI", "region": "Москва",
    "published_at": "2026-11-20", "skills": ["Python"],
}


def create_snapshot(directory: Path) -> Path:
    export = directory / "approved-vacancies.json"
    export.write_text(json.dumps([VACANCY], ensure_ascii=False), encoding="utf-8")
    path = directory / "rabota.sqlite"
    import_export(VacancyStore(path), export)
    return path


def evidence_client():
    hh = source_record("HH", "https://hh.ru/vacancy/1", "HH", "Прежний источник")
    base = AsyncMock()
    base.search.side_effect = lambda role, query, **kwargs: [hh] if role == "business" else []
    return base


class RpdBusinessTests(unittest.TestCase):
    def test_rpd_business_gets_vacancies_alongside_hh_and_priklad(self):
        with TemporaryDirectory() as path:
            self._assert_rpd(create_snapshot(Path(path)), ["Priklad", "HH", "RabotaRu"])

    def test_missing_snapshot_preserves_rpd_business_and_does_not_create_db(self):
        with TemporaryDirectory() as path:
            missing = Path(path) / "missing.sqlite"
            with self.assertLogs("orchestration.rabota_business", level="WARNING") as logs:
                self._assert_rpd(missing, ["Priklad", "HH"])
            self.assertFalse(missing.exists())
            self.assertNotIn(str(missing), "\n".join(logs.output))

    def test_disabled_integration_preserves_rpd_business(self):
        self._assert_rpd("", ["Priklad", "HH"])

    def _assert_rpd(self, database, expected_sources):
        settings = {
            "YANDEX_FOLDER_ID": "fixture", "YANDEX_MODEL_URI": "fixture",
            "YANDEX_VECTOR_STORE_ID_HH": "hh", "YANDEX_VECTOR_STORE_ID_PRIKLAD": "priklad",
            "YANDEX_VECTOR_STORE_ID_RPD": "rpd",
        }
        client = MagicMock()
        hit = SimpleNamespace(file_id="existing", filename="existing",
                              content=[SimpleNamespace(type="text", text="Рабочие задачи")])
        client.vector_stores.search.return_value = SimpleNamespace(data=[hit])
        calls = {}

        def reply(**request):
            name = request["text"]["format"]["name"]
            calls[name] = request
            if name == "rpd_business":
                value = {"findings": [{"fragment": len(expected_sources), "summary": "Рабочие задачи"}]}
            else:
                value = {"document": {
                    "programs": [{"programCode": "01.03.02"}],
                    "purposeAndTasks": {"goals": ["Освоить Python"], "tasks": ["Написать сервис"]},
                    "competencies": [{"code": "ПК-1", "description": "Разрабатывает программы"}],
                    "workload": {"totalHours": 108, "totalCredits": 3},
                    "curriculum": [{"semester": 3, "topics": [{"title": "Python"}]}],
                }, "questions": []}
            return SimpleNamespace(status="completed", incomplete_details=None, output=[], output_text=json.dumps(value))

        client.responses.create.side_effect = reply
        with patch.dict("os.environ", {"RABOTA_IT_DATABASE": str(database)}), \
             patch("orchestration.evidence_cache.snapshot_store"), \
             patch.object(run_rpd, "collect_recent_articles", return_value=[]):
            graph = run_rpd.create_rpd_graph(client, settings, draft_first=True)
            outcome = graph.invoke(
                {"request": {"discipline": "Python", "institution": "Вуз", "academic_year": "2026/27"}},
                config={"configurable": {"thread_id": self.id()}},
            )
        self.assertTrue(outcome["ready"])
        request = calls["rpd_business"]
        fragments = json.loads(request["input"])["fragments"]
        self.assertEqual([item["index"] for item in fragments], expected_sources)
        if "RabotaRu" in expected_sources:
            self.assertIn(VACANCY_URL, fragments[-1]["text"])
            self.assertIn("FastAPI", fragments[-1]["text"])
            self.assertIn("RabotaRu", request["instructions"])
            self.assertIn(VACANCY_URL, calls["rpd_draft"]["input"])
        else:
            self.assertNotIn("RabotaRu", request["instructions"])


class CourseBusinessTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.path = create_snapshot(self.directory)
        self.rabota = RabotaBusiness(self.path)
        self.base = evidence_client()
        self.evidence = BusinessEvidenceClient(self.base, self.rabota, source_record)

    async def test_worker_connects_snapshot_to_langgraph_business_and_formalization(self):
        llm = AsyncMock()
        seen = {}

        async def generate(output, instructions, payload, context, **kwargs):
            seen[output.__name__] = (instructions, payload, context)
            if output.__name__ == "SubjectPlan":
                return output.model_validate({"subjects": []})
            ids = [item["source_id"] for item in payload["sources"] if item["source_type"] == "RabotaRu"]
            if output.__name__ == "AppliedActivityModel":
                value = {"source_ids": ids, "processes": [], "aspects": [], "tasks": [{
                    "source_ids": ids, "id": "task", "title": "Поддерживать сервисы",
                    "stack": ["FastAPI"], "artifacts": ["Сервис"], "quality_criteria": [],
                    "typical_errors": [],
                }]}
            else:
                value = {"findings": [{"summary": "Работа с FastAPI", "source_ids": ids}]}
            return output.model_validate(value, context=context)

        llm.generate.side_effect = generate
        store = AsyncMock()
        engine = AsyncMock()
        @asynccontextmanager
        async def open_saver(url):
            yield InMemorySaver()

        async def process(engine, factory, graph, evidence):
            state = await graph.ainvoke(
                {"request_id": "course", "run_id": "run", "owner_user_id": "owner", "mode": "create",
                 "course_meta": {"title": "Python", "institution": "Частные данные курса"}},
                config={"configurable": {"thread_id": "course"}}, interrupt_before=["prepare_interview"],
            )
            self.assertEqual(state["status"], "gap_analysis")
            self.assertEqual(state["gap_report"]["business_coverage"], 1.0)
            self.assertEqual(state["gap_report"]["science_coverage"], 0.0)

        settings = SimpleNamespace(app_database_url="unused", checkpoint_url="unused",
                                   business_coverage_threshold=0.6, science_coverage_threshold=0.6)
        with patch.dict("os.environ", {"RABOTA_IT_DATABASE": str(self.path)}), \
             patch.object(course_worker, "Settings", return_value=settings), \
             patch.object(course_worker, "yandex_settings", return_value={}), \
             patch.object(course_worker, "make_engine", return_value=engine), \
             patch.object(course_worker, "open_checkpointer", open_saver), \
             patch.object(course_worker, "SqlWorkflowStore", return_value=store), \
             patch.object(course_worker, "YandexEvidenceClient", return_value=self.base), \
             patch.object(course_worker, "YandexModelClient", return_value=llm), \
             patch.object(course_worker, "process_next", new=process):
            await course_worker.run(once=True)
        for name in ("AgentReport", "AppliedActivityModel"):
            instructions, payload, context = seen[name]
            self.assertIn("RabotaRu", instructions)
            self.assertEqual([card["source_type"] for card in payload["sources"]], ["HH", "RabotaRu"])
            record = next(item for item in context["sources"].values() if item.source_type == "RabotaRu")
            self.assertEqual(record.url, VACANCY_URL)
            self.assertIn("FastAPI", record.content)
        engine.dispose.assert_awaited_once()

    async def test_bad_query_other_role_and_corrupt_snapshot_keep_original_result(self):
        with patch.object(self.rabota, "_read", wraps=self.rabota._read) as read:
            self.assertEqual(await self.evidence.search("science", "Python"), [])
            result = await self.evidence.search("business", "{ invalid json")
            self.assertEqual([item.source_type for item in result], ["HH"])
            read.assert_not_called()
        corrupt = self.directory / "corrupt.sqlite"
        corrupt.write_text("not a database", encoding="utf-8")
        self.rabota.path = corrupt
        with self.assertLogs("orchestration.rabota_business", level="WARNING"):
            result = await self.evidence.search("business", "Python")
        self.assertEqual([item.source_type for item in result], ["HH"])

    async def test_timeout_preserves_primary_source(self):
        pending = Future()
        self.rabota.timeout = 0.02
        with patch.object(self.rabota, "_submit", return_value=pending), \
             self.assertLogs("orchestration.rabota_business", level="WARNING"):
            result = await self.evidence.search("business", "Python")
        self.assertEqual([item.source_type for item in result], ["HH"])
        self.assertTrue(pending.cancelled())


if __name__ == "__main__":
    unittest.main()
