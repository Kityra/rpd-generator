import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from zipfile import ZipFile

from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from apps.teacher_app import report_panel
from auth.settings import Settings
from domain.export import export_course
from domain.schemas import MarketDataset, MarketVacancy
from orchestration.call_log import CALL_SCOPE, CallScope, RecordingModelClient
from orchestration.evidence_cache import source_record
from orchestration.graph import build_course_graph
from orchestration.interview_skills import SKILL_FILES, load_interview_skills, select_interview_method
from orchestration.interviewer_turn import (InterviewerRequest, classify_user_turn, extract_interview_claims,
                                            interview_result)
from orchestration.market_analysis import analyse_market, with_competences
from orchestration.profession_pipeline import activity_for_design, apply_expert_turn, initial_profile, market_evidence
from orchestration.profession_profile import EvidenceRef, pending_claim
from orchestration.profession_verifier import ProfessionEvidenceVerdicts, verify_profession_profile
from storage.database import make_engine
from storage.models import Base, Course, CourseVersion, Interview, ModelCall
from storage.portal import assigned_interviews, create_course, finish_without_answer
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
    async def test_blacklists_noise_even_with_a_valid_turn_entry(self):
        text = "Оптимизирует SQL-запросы"
        value, turn = apply_expert_turn(profile(claim("work_task:A", text)), [
            {"claim_id": "work_task:A", "action": "confirm"}], "turn:1")
        judge = AsyncMock()
        with self.assertLogs("orchestration.profession_verifier", "WARNING"):
            result = await verify_profession_profile(value, {}, {}, {"turn:1": turn}, judge, rejected_noise_turns=[1])
        self.assertEqual(result.rejected_noise_turns, [1])
        self.assertEqual(result.claims[0].evidence, [])
        self.assertEqual(result.claims[0].verification_status, "needs_review")
        judge.generate.assert_not_awaited()

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
        self.assertEqual(len(logs.records), 3)
        self.assertTrue(all(item.verification_status == "needs_review" for item in result.claims))
        self.assertEqual([item.claim_id for item in result.claims if not item.evidence], ["fake", "wrong-quote", "forgery"])
        judge.generate.assert_not_awaited()

    async def test_judge_prefilter_zero_overlap_skips_llm(self):
        sql = "Оптимизирует запросы PostgreSQL"
        service = "Трудоёмкость работы с PostgreSQL (час) 302, контактная работа (час) 22"
        value = profile(claim("unrelated", "Верстает интерфейс", excerpt=sql),
                        claim("service", sql, "service", excerpt=service))
        judge = AsyncMock()
        result = await verify_profession_profile(value, {"chunk": sql, "service": service}, {}, {}, judge)
        self.assertEqual([item.evidence[0].judge_verdict for item in result.claims], ["unrelated", "unrelated"])
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

    async def test_judge_single_batched_call(self):
        judge = AsyncMock()

        async def decide(output, prompt, request, context, **options):
            self.assertIs(output, ProfessionEvidenceVerdicts)
            self.assertEqual(options["temperature"], 0.0)
            self.assertEqual(len(request["pairs"]), 5)
            self.assertEqual(set(request), {"pairs"})
            self.assertEqual(context, {})
            self.assertTrue(all(set(pair) == {"pair_id", "claim", "source_type", "evidence"}
                                for pair in request["pairs"]))
            return output(items=[{"pair_id": row["pair_id"], "verdict": verdict}
                                 for row, verdict in zip(request["pairs"],
                                                         ["supports", "weak", "unrelated", "weak", "supports"], strict=True)])

        judge.generate.side_effect = decide
        quote = "Оптимизирует SQL-запросы в PostgreSQL."
        title = "SQL указан в вакансиях"
        a = claim("a", quote)
        b = claim("b", "Проверяет запросы SQL", excerpt=quote)
        c = claim("market", title, "sql", kind="market_cluster", excerpt="42/60 (70%)")
        for item in (a, b):
            item.evidence.append(EvidenceRef(source_type="rag_chunk", source_id="second", quote_or_metric=quote))
        c.evidence.append(EvidenceRef(source_type="market_cluster", source_id="sql", quote_or_metric="41/60 (99%)"))
        value = profile(a, b, c)
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
                    result = await verify_profession_profile(value, {"chunk": quote, "second": quote},
                        {"sql": {"skill": "SQL", "metric": "42/60 (70%)"}}, {},
                        RecordingModelClient(judge, store.record_model_call))
                finally:
                    CALL_SCOPE.reset(scope)
                async with factory() as db:
                    logged = (await db.scalars(select(ModelCall))).one()
                    self.assertEqual(logged.output, "ProfessionEvidenceVerdicts")
                    self.assertEqual((logged.course_id, logged.stage, logged.subject_id), ("course", "llm_judge", None))
                    self.assertEqual(logged.outcome, "ok")
            finally:
                await engine.dispose()
        self.assertEqual([item.verification_status for item in result.claims],
                         ["verified", "needs_review", "verified"])
        self.assertEqual(len(result.claims[-1].evidence), 1)
        judge.generate.assert_awaited_once()

    async def test_judge_missing_pair_id_handled_safely(self):
        judge = AsyncMock()
        judge.generate.return_value = ProfessionEvidenceVerdicts(items=[
            {"pair_id": "2", "verdict": "supports"}, {"pair_id": "fake", "verdict": "supports"}])
        text = "Проектирует схемы PostgreSQL"
        with self.assertLogs("orchestration.llm_judge", "WARNING"):
            result = await verify_profession_profile(profile(claim("work_task:x", text), claim("work_task:y", text)),
                                                      {"chunk": text}, {}, {}, judge)
        self.assertEqual(result.claims[0].verification_status, "needs_review")
        self.assertEqual(result.claims[0].evidence[0].judge_verdict, "weak")
        self.assertEqual(result.claims[1].verification_status, "verified")
        self.assertFalse(result.judge_unavailable)
        judge.generate.assert_awaited_once()

    async def test_zip_and_portal_show_market_table_without_profile_claim_list(self):
        example = "Оптимизирует SQL-запросы"
        value, turn = apply_expert_turn(profile(claim("rag", example),
            claim("market", "В выборке вакансий указан навык SQL.", "sql", kind="market_cluster", excerpt="42/60 (70%)"),
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
        data["unresolved_slots"] = ["work_tasks"]
        version = SimpleNamespace(status="approved", revision=1, content={"program": program(),
                                  "profession_profile": data, "profession_market": market})
        dataset = MarketDataset(source="HH", vacancies=[MarketVacancy(title="Аналитик", skills=["SQL"])])
        analysis = with_competences(analyse_market(version.content["program"], dataset), {})
        with patch("apps.teacher_app.st") as ui:
            report_panel(version, sources, [], analysis)
            ui.dataframe.assert_called_once()
            self.assertEqual(ui.dataframe.call_args.args[0][0]["Компетенция"], "Проектирование и работа с базами данных")
            self.assertEqual(ui.dataframe.call_args.args[0][0]["Инструменты"], "SQL")
            self.assertEqual(ui.dataframe.call_args.args[0][0]["Вакансий"], 1)
        package = export_course({"institution": "Вуз", "year": "2026"}, version, sources,
                                analysis.model_dump(mode="json"))
        with ZipFile(BytesIO(package)) as archive:
            report = archive.read("course-report.html").decode()
            saved = json.loads(archive.read("course.json"))
            manifest = json.loads(archive.read("sources.json"))
        self.assertIn("<td>SQL</td><td>100%</td><td>1</td>", report)
        for removed in ("Поля интервью для проверки методистом", "В выборке вакансий указан навык",
                        "<summary>Основание</summary>"):
            self.assertNotIn(removed, report)
            self.assertNotIn(removed, str(ui.html.call_args_list))
        self.assertIn("chunk", manifest)
        self.assertEqual(saved["profession_profile"]["counts"], {"verified": 3, "needs_review": 1, "rejected": 1})
        self.assertTrue(any(item["claim_id"] == "market" for item in saved["profession_profile"]["claims"]))
        self.assertNotIn("СКРЫТЫЙ ОТКЛОНЁННЫЙ ТЕКСТ", report)
        self.assertNotIn("СКРЫТЫЙ ОТКЛОНЁННЫЙ ТЕКСТ", json.dumps(saved, ensure_ascii=False))

    def test_legacy_user_input_serializes_as_methodologist_input(self):
        ref = EvidenceRef.model_validate({"source_type": "user_input", "source_id": "manual",
                                         "quote_or_metric": "Проверяет отчёт"})
        self.assertEqual(ref.model_dump()["source_type"], "methodologist_input")


class InterviewTurnTests(unittest.IsolatedAsyncioTestCase):
    async def test_extraction_filters_slots_files_and_fabricated_quotes(self):
        request = InterviewerRequest(requested_slots=["work_tasks"], question="Что вы делаете?")
        answer = "Сверяет миграции PostgreSQL. Использует Python."
        llm = AsyncMock()
        llm.generate.return_value = {"work_tasks": ["Сверяет миграции PostgreSQL.", "Выдуманная задача"],
                                     "tool_stack": ["Использует Python."], "extra_file": "report.docx",
                                     "unrequested_subjects": ["Информатика"],
                                     "extracted_claims": [{"slot_name": "tool_stack", "statement": "Использует Python.",
                                                           "file": "unrequested.docx"}]}
        extracted = await extract_interview_claims(request, answer, llm)
        result = interview_result(request, "valid_answer", extracted, answer)
        self.assertEqual(result.filled_slots, {"work_tasks": ["Сверяет миграции PostgreSQL."]})
        self.assertEqual([item.slot_name for item in result.extracted_claims], ["work_tasks"])
        self.assertEqual(result.unresolved_slots, [])
        self.assertNotIn(".docx", result.model_dump_json())

    async def test_filters_local_noise_and_checks_semantic_relevance(self):
        request = InterviewerRequest(requested_slots=["work_tasks"], question="Что вы делаете?")
        llm = AsyncMock()
        for text in ("бла-бла-бла", "asdfgh", "сделайте красиво не знаю что", ""):
            self.assertEqual(await classify_user_turn(request, text, llm), "nonsense_or_offtopic")
        llm.generate.assert_not_awaited()
        llm.generate.return_value = {"classification": "nonsense_or_offtopic"}
        self.assertEqual(await classify_user_turn(request, "В выходные солнечная погода.", llm), "nonsense_or_offtopic")
        llm.generate.assert_awaited_once()

    async def test_noise_turn_redirects_and_valid_second_turn_binds_new_claim(self):
        teacher, expert = user("teacher"), user("expert")
        model, evidence, users = Model(gap=True), AsyncMock(), AsyncMock()
        users.get_by_id.return_value = expert
        evidence.search.return_value = list(records().values())
        statement = "Сверяет миграции PostgreSQL по контрольной схеме."
        original = model.generate

        async def generate(output, prompt, payload, context, **options):
            if output.__name__ == "InterviewerExtraction":
                self.assertEqual(payload["requested_slots"], ["work_tasks"])
                self.assertEqual(payload["answer"], statement)
                return {"work_tasks": [statement], "extra_file": "report.docx"}
            return await original(output, prompt, payload, context, **options)

        model.generate = AsyncMock(side_effect=generate)
        with tempfile.TemporaryDirectory() as tmp:
            engine = make_engine("sqlite+aiosqlite:///" + (Path(tmp) / "interview.sqlite").as_posix())
            try:
                async with engine.begin() as connection:
                    await connection.run_sync(Base.metadata.create_all)
                factory = async_sessionmaker(engine, expire_on_commit=False)
                store = SqlWorkflowStore(factory)
                await create_course(factory, teacher, "course",
                                    {"title": "Базы данных", "institution": "Вуз", "year": "2026"}, [])
                graph = build_course_graph(Settings(_env_file=None, session_signing_key="x" * 40), model, evidence,
                                           store, InMemorySaver())
                await process_next(engine, factory, graph)
                async with factory() as db:
                    first = (await db.scalars(select(Interview))).one()
                token = await store.invite(first.id, expert.id, teacher, users)
                await store.submit_interview(token, expert, first.interrupt_id, "one", "бла-бла-бла", [])
                await process_next(engine, factory, graph)
                async with factory() as db:
                    first = await db.get(Interview, first.id)
                    invitation, second, _ = next(row for row in (await db.execute(assigned_interviews(expert))).all()
                                                 if row[1].id != first.id)
                refused = first.gap_report["verification"]["interviewer_result"]
                self.assertEqual(refused["extracted_claims"], [])
                self.assertEqual(refused["filled_slots"], {})
                self.assertIn("Пожалуйста", second.gap_report["prompt"])
                self.assertEqual(second.gap_report["interview_turn"], 2)
                self.assertEqual(second.gap_report["profession_profile"]["rejected_noise_turns"], [1])
                self.assertNotIn("InterviewerExtraction", [call.args[0].__name__ for call in model.generate.call_args_list])
                await store.submit_interview("", expert, second.interrupt_id, "two", statement, [],
                                             invitation_id=invitation.id)
                await process_next(engine, factory, graph)
                async with factory() as db:
                    course = await db.get(Course, "course")
                    self.assertEqual(course.status, "waiting_for_review")
                    version = await db.get(CourseVersion, course.draft_version_id)
                persisted = version.content["profession_profile"]
                accepted = next(item for item in persisted["claims"] if item["statement"] == statement)
                self.assertEqual(accepted["verification_status"], "verified")
                self.assertEqual(accepted["evidence"][0]["source_id"], "turn:2")
                self.assertEqual(persisted["rejected_noise_turns"], [1])
                self.assertNotIn("бла-бла-бла", json.dumps(persisted, ensure_ascii=False))
            finally:
                await engine.dispose()

    async def test_redirect_limit_passes_empty_profile_to_methodologist(self):
        teacher, expert = user("teacher"), user("expert")
        model, evidence, users = Model(gap=True), AsyncMock(), AsyncMock()
        users.get_by_id.return_value = expert
        evidence.search.return_value = []
        with tempfile.TemporaryDirectory() as tmp:
            engine = make_engine("sqlite+aiosqlite:///" + (Path(tmp) / "handoff.sqlite").as_posix())
            try:
                async with engine.begin() as connection:
                    await connection.run_sync(Base.metadata.create_all)
                factory = async_sessionmaker(engine, expire_on_commit=False)
                store = SqlWorkflowStore(factory)
                await create_course(factory, teacher, "course", {"title": "Инженер данных"}, [])
                settings = Settings(_env_file=None, session_signing_key="x" * 40, max_invalid_redirects=0)
                graph = build_course_graph(settings, model, evidence, store, InMemorySaver())
                await process_next(engine, factory, graph)
                async with factory() as db:
                    first = (await db.scalars(select(Interview))).one()
                token = await store.invite(first.id, expert.id, teacher, users)
                await store.submit_interview(token, expert, first.interrupt_id, "one", "asdfgh", [])
                await process_next(engine, factory, graph)
                async with factory() as db:
                    handoff = (await db.scalars(select(Interview).where(Interview.id != first.id))).one()
                    invites = (await db.execute(assigned_interviews(expert))).all()
                    self.assertEqual(len(invites), 1)
                self.assertTrue(handoff.gap_report["methodologist_handoff"])
                saved = handoff.gap_report["profession_profile"]
                self.assertEqual(saved["rejected_noise_turns"], [1])
                self.assertIn("work_tasks", saved["unresolved_slots"])
                self.assertFalse(saved["claims"])
                with self.assertRaisesRegex(ValueError, "методисту"):
                    await store.invite(handoff.id, expert.id, teacher, users)
            finally:
                await engine.dispose()


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
