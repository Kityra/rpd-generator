import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from zipfile import ZipFile

from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from auth.settings import Settings
from domain.export import PROFESSION_REVIEW_LABEL, export_course, profession_profile_html
from orchestration.call_log import CALL_SCOPE, CallScope, RecordingModelClient
from orchestration.evidence_cache import source_record
from orchestration.graph import build_course_graph
from orchestration.interview_skills import SKILL_FILES, load_interview_skills, select_interview_method
from orchestration.profession_pipeline import activity_for_design, apply_expert_turn, initial_profile, market_evidence
from orchestration.profession_profile import EvidenceRef, pending_claim
from orchestration.profession_verifier import ProfessionEvidenceVerdicts, verify_profession_profile
from storage.database import make_engine
from storage.models import Base, Course, CourseVersion, Interview, ModelCall
from storage.portal import create_course, finish_without_answer
from storage.workflow import SqlWorkflowStore
from test_course_architecture import program, records
from test_portals import Model, user
from workers.course_worker import process_next


def claim(key, text, source_id="chunk", *, kind="rag_chunk", excerpt=None):
    return pending_claim(key, "work_task", text, [
        EvidenceRef(source_type=kind, source_id=source_id, quote_or_metric=excerpt or text)])


def profile(*claims):
    return initial_profile("Инженер данных", {}, {}, 0, {}).model_copy(update={"claims": list(claims)})


class SkillTests(unittest.TestCase):
    def test_loads_all_four_and_warns_when_unavailable(self):
        path = Path(__file__).resolve().parents[1] / "skills" / "interview_methods"
        skills = load_interview_skills(path)
        self.assertEqual(set(skills), set(SKILL_FILES))
        self.assertTrue(all(skill.max_turns > 0 and skill.instructions for skill in skills.values()))
        with tempfile.TemporaryDirectory() as folder, self.assertLogs("orchestration.interview_skills", "WARNING") as logs:
            self.assertEqual(load_interview_skills(Path(folder) / "missing"), {})
        self.assertEqual(logs.records[0].event, "interview_skills_unavailable")

    def test_route_priority(self):
        base = dict(sample_size=0, rag_count=0, has_unverified=False, missing_lab_tasks=False, missing_slots=["tools"])
        cases = [({}, "dacum_task_analysis"),
                 ({"sample_size": 60, "has_unverified": True}, "member_checking_validation"),
                 ({"rag_count": 4, "missing_lab_tasks": True}, "critical_incident_technique"),
                 ({"rag_count": 4}, "gap_focused_slot_filling")]
        for overrides, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(select_interview_method(**(base | overrides)), expected)

    def test_market_counts_only_real_vacancy_cards(self):
        hh = source_record("HH", "https://hh.ru/vacancy/123", "Инженер данных", "Работа с Python и PostgreSQL.")
        csv = source_record("HH", "https://ai.api.cloud.yandex.net/v1/files/hh", "HH.csv", "Вакансии Python")
        size, market = market_evidence({}, {source.source_id: source for source in (hh, csv)}, [])
        self.assertEqual(size, 1)
        self.assertEqual(market["skill:Python"]["metric"], "1/1 (100%)")
        self.assertEqual(market["skill:Python"]["provider"], "HH")


class VerificationTests(unittest.IsolatedAsyncioTestCase):
    async def test_strips_fabricated_id_quote_and_expert_status(self):
        text = "Внедряет транзакции PostgreSQL"
        forged = claim("forgery", text, "turn:1", kind="expert_interview")
        forged.evidence[0].expert_status = "confirmed_by_expert"
        value = profile(claim("fake", text, "fake_999"),
                        claim("wrong-quote", text, excerpt="Несуществующая цитата PostgreSQL"),
                        claim("no-overlap", "Верстает интерфейс", excerpt=text), forged)
        judge = AsyncMock()
        with self.assertLogs("orchestration.profession_verifier", "WARNING") as logs:
            result = await verify_profession_profile(value, {"chunk": text}, {}, {}, judge)
        self.assertEqual(len(logs.records), 4)
        self.assertTrue(all(not item.evidence and item.verification_status == "needs_review" for item in result.claims))
        judge.generate.assert_not_awaited()

    async def test_expert_can_confirm_edit_and_reject(self):
        value = profile(claim("work_task:A", "Проектирует схемы PostgreSQL"),
                        claim("work_task:B", "Сохраняет сырые данные"),
                        claim("work_task:C", "Прячет сбои выгрузки"))
        value, turn = apply_expert_turn(value, [
            {"claim_id": "work_task:A", "action": "confirm"},
            {"claim_id": "work_task:B", "action": "edit", "statement": "Валидирует схему Avro"},
            {"claim_id": "work_task:C", "action": "reject"},
        ], "turn:2")
        judge = AsyncMock()
        result = await verify_profession_profile(value, {"chunk": "Проектирует схемы PostgreSQL"}, {}, {"turn:2": turn}, judge)
        self.assertEqual([item.verification_status for item in result.claims], ["verified", "verified", "rejected"])
        self.assertEqual(result.claims[1].statement, "Валидирует схему Avro")
        design, _ = activity_for_design(result, {}, ["work_task:C"], "https://example.org/course", {})
        self.assertEqual([task["id"] for task in design["tasks"]], ["A", "B"])
        judge.generate.assert_not_awaited()

    async def test_one_batch_with_all_verdicts_and_real_sql_log(self):
        judge = AsyncMock()

        async def decide(output, prompt, request, context, **options):
            self.assertIs(output, ProfessionEvidenceVerdicts)
            self.assertEqual(options["temperature"], 0.0)
            self.assertEqual(len(request["pairs"]), 4)
            return output(items=[{"pair_id": row["pair_id"], "verdict": verdict}
                                 for row, verdict in zip(request["pairs"], ["supports", "weak", "unrelated", "supports"])])

        judge.generate.side_effect = decide
        quote = "Оптимизирует SQL-запросы в PostgreSQL."
        value = profile(claim("a", quote), claim("b", "Проверяет запросы SQL", excerpt=quote),
                        claim("c", "Удаляет PostgreSQL", excerpt=quote),
                        claim("market", "SQL указан в вакансиях", "sql", kind="market_cluster", excerpt="42/60 (70%)"),
                        claim("invented", "SQL указан в вакансиях", "sql", kind="market_cluster", excerpt="41/60 (99%)"))
        with tempfile.TemporaryDirectory() as tmp:
            engine = make_engine("sqlite+aiosqlite:///" + (Path(tmp) / "log.sqlite").as_posix())
            try:
                async with engine.begin() as connection:
                    await connection.run_sync(Base.metadata.create_all)
                factory = async_sessionmaker(engine)
                await create_course(factory, user("teacher"), "course", {"title": "SQL"}, [])
                store = SqlWorkflowStore(factory)
                scope = CALL_SCOPE.set(CallScope(course_id="course", run_id="run", stage="profession_verification"))
                try:
                    result = await verify_profession_profile(value, {"chunk": quote},
                        {"sql": {"skill": "SQL", "metric": "42/60 (70%)"}}, {},
                        RecordingModelClient(judge, store.record_model_call))
                finally:
                    CALL_SCOPE.reset(scope)
                async with factory() as db:
                    logged = (await db.scalars(select(ModelCall))).one()
                    self.assertEqual(logged.output, "ProfessionEvidenceVerdicts")
                    self.assertEqual(logged.outcome, "ok")
            finally:
                await engine.dispose()
        self.assertEqual([item.verification_status for item in result.claims],
                         ["verified", "needs_review", "needs_review", "verified", "needs_review"])
        self.assertEqual(result.claims[-1].evidence, [])
        judge.generate.assert_awaited_once()

    async def test_invalid_judge_reply_fails_closed(self):
        judge = AsyncMock()
        judge.generate.return_value = ProfessionEvidenceVerdicts(items=[{"pair_id": "fake", "verdict": "supports"}])
        text = "Проектирует схемы PostgreSQL"
        with self.assertLogs("orchestration.profession_verifier", "WARNING"):
            result = await verify_profession_profile(profile(claim("work_task:x", text)), {"chunk": text}, {}, {}, judge)
        self.assertEqual(result.claims[0].verification_status, "needs_review")
        self.assertIn("недоступна", result.claims[0].verification_reason)

    async def test_zip_and_portal_share_badges_and_remove_rejected_text(self):
        example = "Оптимизирует SQL-запросы"
        value, turn = apply_expert_turn(profile(claim("rag", example),
            claim("market", "SQL указан в вакансиях", "sql", kind="market_cluster", excerpt="42/60 (70%)"),
            claim("manual", "Проектирует API", "fake"), claim("a", "Обучает модель"),
            claim("b", "СКРЫТЫЙ ОТКЛОНЁННЫЙ ТЕКСТ")), [
                {"claim_id": "a", "action": "confirm"}, {"claim_id": "b", "action": "reject"}], "turn:1")
        judge = AsyncMock()
        judge.generate.return_value = ProfessionEvidenceVerdicts(items=[
            {"pair_id": "1", "verdict": "supports"}, {"pair_id": "2", "verdict": "supports"}])
        market = {"sql": {"skill": "SQL", "provider": "HH", "metric": "42/60 (70%)"}}
        sources = {key: val.model_dump(mode="json") for key, val in records().items()}
        sources["chunk"] = sources["HH"] | {"content": example}
        result = await verify_profession_profile(value, {"chunk": example}, market, {"turn:1": turn}, judge)
        data = result.model_dump(mode="json")
        version = SimpleNamespace(status="approved", revision=1, content={"program": program(),
                                  "profession_profile": data, "profession_market": market})
        ui = profession_profile_html(data, market, sources)
        with ZipFile(BytesIO(export_course({"institution": "Вуз", "year": "2026"}, version, sources))) as archive:
            report = archive.read("course-report.html").decode()
            saved = json.loads(archive.read("course.json"))
            manifest = json.loads(archive.read("sources.json"))
        for badge in (PROFESSION_REVIEW_LABEL, "[Подтверждено экспертом, реплика #1]",
                      "[Рынок HH: 42/60 вакансий (70%), подтверждено]", "[Подтверждено источником RAG]"):
            self.assertIn(badge, report)
            self.assertIn(badge, ui)
        self.assertIn("chunk", manifest)
        self.assertEqual(saved["profession_profile"]["counts"], {"verified": 3, "needs_review": 1, "rejected": 1})
        self.assertNotIn("СКРЫТЫЙ ОТКЛОНЁННЫЙ ТЕКСТ", report)
        self.assertNotIn("СКРЫТЫЙ ОТКЛОНЁННЫЙ ТЕКСТ", json.dumps(saved, ensure_ascii=False))


class ColdStartTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_sources_require_manual_tasks_before_generating(self):
        teacher = user("teacher")
        evidence = AsyncMock()
        evidence.search.return_value = []
        model = Model(gap=False)
        original = model.generate
        requested = []

        async def generate(output, prompt, payload, context, **options):
            requested.append(output.__name__)
            if output.__name__ in {"AppliedActivityModel", "ProfessionEvidenceVerdicts"}:
                self.fail("Нет данных для кандидатов или проверки источников")
            if output.__name__ == "SubjectPlan":
                return output.model_validate({"subjects": []})
            if output.__name__ == "CourseProgramModel":
                self.assertEqual([task.title for task in context["activity"].tasks], ["Собирает отчёт о загрузке"])
                self.assertEqual(payload["profession_basis"][0]["verification_status"], "needs_review")
                self.assertNotIn("ФГОС", json.dumps(payload, ensure_ascii=False))
                value = program()
                value["subjects"][0]["origin"] = "manual"
                value["source_ids"] = context["activity"].source_ids
                def cite(block):
                    block["source_ids"] = context["activity"].source_ids
                    block["learning_references"] = []
                    if "justification" in block:
                        block["justification"]["source_ids"] = context["activity"].source_ids
                for block in [*value["subjects"], *value["subjects"][0]["topics"],
                              *value["subjects"][0]["topics"][0]["assignments"]]:
                    cite(block)
                value["subjects"][0]["topics"][0]["activity_task_ids"] = [context["activity"].tasks[0].id]
                return output.model_validate(value, context=context)
            if output.__name__ == "BusinessCourseAudit":
                raise RuntimeError("Нет рыночных источников")
            return await original(output, prompt, payload, context, **options)

        model.generate = generate
        with tempfile.TemporaryDirectory() as tmp:
            engine = make_engine("sqlite+aiosqlite:///" + (Path(tmp) / "cold.sqlite").as_posix())
            try:
                async with engine.begin() as connection:
                    await connection.run_sync(Base.metadata.create_all)
                factory = async_sessionmaker(engine, expire_on_commit=False)
                await create_course(factory, teacher, "cold", {"title": "Инженер данных", "institution": "Вуз",
                                                              "year": "2026"}, ["Базы данных"])
                graph = build_course_graph(Settings(_env_file=None, session_signing_key="x" * 40), model, evidence,
                                           SqlWorkflowStore(factory), InMemorySaver())
                self.assertTrue(await process_next(engine, factory, graph))
                async with factory() as db:
                    course = await db.get(Course, "cold")
                    interview = (await db.scalars(select(Interview))).one()
                self.assertEqual(course.status, "waiting_for_expert")
                self.assertEqual(interview.gap_report["profession_profile"]["sample_adequacy"], "empty_external_data")
                self.assertFalse(interview.gap_report["profession_profile"]["claims"])
                self.assertNotIn("CourseProgramModel", requested)
                await finish_without_answer(factory, teacher, course.id, interview.id,
                                            manual_tasks=["Собирает отчёт о загрузке"])
                self.assertTrue(await process_next(engine, factory, graph))
                async with factory() as db:
                    course = await db.get(Course, "cold")
                    version = await db.get(CourseVersion, course.draft_version_id)
                self.assertEqual(course.status, "waiting_for_review")
                persisted = version.content["profession_profile"]
                self.assertEqual(persisted["sample_adequacy"], "empty_external_data")
                self.assertEqual([item["statement"] for item in persisted["claims"]], ["Собирает отчёт о загрузке"])
            finally:
                await engine.dispose()
