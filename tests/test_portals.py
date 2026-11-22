import hashlib
import json
import os
import tempfile
import unittest
from datetime import datetime, time, timedelta, timezone
from io import BytesIO
from functools import partial
from pathlib import Path
from unittest.mock import AsyncMock, patch
from zipfile import ZipFile

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker
from streamlit.testing.v1 import AppTest

from auth.password_verifier import ExistingPasswordVerifier
from auth.settings import Settings
from auth.user_repository import CurrentUser
from auth.yandex_users import accounts_from_text
from domain.export import block_html, export_course, rationale_report_html, source_html
from domain.schemas import CourseLifespanAssessment, EvidenceResults, block_value
from domain.skill_mapper import ML_COMPETENCE
from literature_parser import LiteratureSource, TopicLiterature, enrich_course_program
from orchestration.checkpointer import open_checkpointer
from orchestration.course_lifespan import INSUFFICIENT_EVIDENCE, SUPPORT_FEED_UNAVAILABLE, SupportDeadline
from orchestration.evidence_cache import source_record
from orchestration.graph import build_course_graph
from orchestration.market_analysis import course_market
from orchestration.subject_rationale import explain_subjects
from run_rpd import YandexModeratedResponse
from storage.database import make_engine
from storage.models import (Base, Course, CourseVersion, Interview, ModelCall, Notification, ProfessionBoundary,
                            SourceRow, WorkflowCommand, now)
from storage.portal import (acknowledge, assigned_interviews, create_course, decide, delete_course, finish_without_answer,
                            rebuild_course, schedule_due)
from storage.workflow import SqlWorkflowStore
from test_course_architecture import program, rationale_batch, records
from workers.course_worker import process_next


LIFESPAN_POINTS = ["В вакансиях встречаются новые инструменты.", "Версии библиотек регулярно обновляются.",
                   "Рабочие задачи включают проверку результатов."]


def lifespan_data(years):
    return {"estimated_lifespan_years": years, "is_short_lived": years <= 5,
            "hr_digest_points": LIFESPAN_POINTS, "source_ids": ["HH"]}


class LifespanAssessmentTests(unittest.TestCase):
    def test_horizon_boundaries_and_strict_flag(self):
        for years, short in ((0, True), (5, True), (5.1, False)):
            with self.subTest(years=years):
                data = lifespan_data(years)
                context = {"sources": records()}
                model = CourseLifespanAssessment.model_validate(data, context=context)
                self.assertIs(model.is_short_lived, short)
                with self.assertRaises(ValidationError):
                    CourseLifespanAssessment.model_validate({**data, "is_short_lived": not short}, context=context)
        for years in (-0.1, 15.1):
            with self.assertRaises(ValidationError):
                CourseLifespanAssessment.model_validate(lifespan_data(years), context={"sources": records()})

    def test_grounding_and_digest_contract(self):
        invalid = ({"source_ids": ["invented"]}, {"source_ids": []}, {"extra_field": True},
                   {"estimated_lifespan_years": True}, {"estimated_lifespan_years": "5"},
                   {"hr_digest_points": ["Один тезис"]},
                   {"hr_digest_points": [*LIFESPAN_POINTS[:2], "Запуск нецелесообразен: не рекомендуется запускать."]})
        for override in invalid:
            with self.subTest(override=override), self.assertRaises(ValidationError):
                CourseLifespanAssessment.model_validate({**lifespan_data(5), **override}, context={"sources": records()})


class SourceDisplayTests(unittest.TestCase):
    def test_private_file_opens_escaped_excerpt_and_public_article_opens_link(self):
        source = {"source_type": "RPD", "title": "Учебный материал", "content": "<script>text</script>",
                  "url": "https://ai.api.cloud.yandex.net/v1/files/test"}
        html = source_html(source)
        self.assertIn("<details><summary", html)
        self.assertIn("&lt;script&gt;text&lt;/script&gt;", html)
        self.assertNotIn(source["url"], html)
        public = dict(source, source_type="Crossref", url="https://doi.org/10.1234/test")
        self.assertIn(f'href="{public["url"]}"', source_html(public))

    def test_discipline_lists_file_once_and_keeps_distinct_excerpts(self):
        subject = program()["subjects"][0]
        topic = subject["topics"][0]
        subject["justification"]["source_ids"] = ["Books", "duplicate"]
        topic["justification"]["source_ids"] = ["duplicate", "other_excerpt"]
        book = {"source_type": "Books", "title": "Мл4.pdf", "content": "Первый текст.",
                "url": "https://ai.api.cloud.yandex.net/v1/files/entry1"}
        sources = {"Books": book,
                   "duplicate": dict(book, url="https://ai.api.cloud.yandex.net/v1/files/entry2"),
                   "other_excerpt": dict(book, content="Другой фрагмент.")}
        html = block_html(subject, sources)
        self.assertEqual(1, html.count("Мл4.pdf</summary>"))
        self.assertEqual(1, html.count("Первый текст."))
        self.assertEqual(1, html.count("Другой фрагмент."))
        self.assertIn("Источник 1 —", html)

    def test_job_export_cannot_appear_in_discipline_materials(self):
        subject = program()["subjects"][0]
        subject["justification"]["source_ids"].append("HH")
        sources = {key: record.model_dump(mode="json") for key, record in records().items()}
        sources["HH"]["title"] = "jobix_hh_only_part_0002.csv"
        html = block_html(subject, sources)
        materials, supporting = html.split("<h3>Источники для обоснований</h3>", 1)
        self.assertIn("Материалы дисциплины", materials)
        self.assertNotIn("jobix_hh_only_part_0002.csv", materials)
        self.assertIn("jobix_hh_only_part_0002.csv", supporting)


def user(role, suffix=""):
    return CurrentUser(id=role + suffix, email=f"{role}{suffix}@example.org", display_name=role, role=role)


class Model:
    def __init__(self, gap):
        self.gap, self.relevance = gap, True
        self.claim_status = "verified"

    async def generate(self, output, prompt, payload, context, flash=False, **options):
        name = output.__name__
        if name == "AgentReport":
            value = {"findings": []}
        elif name == "ProfessionEvidenceVerdicts":
            value = {"items": [{"pair_id": pair["pair_id"], "verdict": "supports"} for pair in payload["pairs"]]}
        elif name == "InterviewerTurnQuality":
            value = {"classification": "valid_answer"}
        elif name == "InterviewerExtraction":
            value = {"items": []}
        elif name == "AppliedActivityModel":
            count = 4 if self.gap and not any(s.source_type == "expert" for s in context["sources"].values()) else 5
            value = {"tasks": [{"id": f"task{i}", "title": f"Проверяет запрос SQL на этапе {i}", "stack": ["SQL"], "artifacts": ["Отчёт"],
                                "quality_criteria": ["Время", "Корректность"], "typical_errors": [], "source_ids": ["Crossref", "HH"]}
                               for i in range(count)], "processes": [{"id": str(i), "title": "Сбор", "task_ids": ["task0"],
                               "source_ids": ["HH"]} for i in range(3)], "aspects": [], "source_ids": ["HH"]}
        elif name == "SubjectPlan":
            value = {"subjects": [{"name": "Базы данных", "is_fundamental": True, "conflict_with": None, "source_ids": ["Books"]}]}
        elif name == "InterviewTurn":
            value = {"prompt": "Как вы проверяете отчёт?"}
        elif name == "VerificationReport":
            value = {"claims": [{"claim": "Сверка", "verification_status": self.claim_status, "explanation": "Есть источник",
                                 "clarification_question": None, "source_ids": ["HH"]}]}
        elif name == "CourseProgramModel":
            value = program()
        elif name == "SubjectRationaleBatch":
            assert all(source["index"] in {"Books", "RPD", "Priklad", "HH"}
                       for subject in payload["disciplines"] for source in subject["fragments"])
            value = rationale_batch(payload)
        elif name == "RelevanceDecision":
            value = {"requires_update": self.relevance, "explanation": "Изменение практики", "source_ids": ["HH"]}
        elif name == "RationaleVerdicts":
            value = {"items": [{"pair_id": row["pair_id"], "verdict": "supports"} for row in payload["pairs"]]}
        else:
            value = {"source_ids": ["HH"], "market_fit_score": .8, "metrics": [], "unavailable_metrics": [],
                     "evidence_summary": "SQL", "suggested_changes": []}
        return output.model_validate(value, context=context)


class PortalTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        path = Path(self.tmp.name)
        self.engine = make_engine("sqlite+aiosqlite:///" + (path / "courses.db").as_posix())
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        self.factory = async_sessionmaker(self.engine, expire_on_commit=False)
        self.checkpoint = str(path / "checkpoints.db")
        self.store = SqlWorkflowStore(self.factory)
        self.settings = Settings(_env_file=None, session_signing_key="s" * 40)
        self.model = Model(gap=False)
        self.university_literature_enricher = None
        self.evidence = AsyncMock()
        self.evidence.search.return_value = list(records().values())
        self.teacher = user("teacher")
        await create_course(self.factory, self.teacher, "course", {"title": "Курс", "institution": "Вуз", "year": "2026"}, [])

    async def asyncTearDown(self):
        await self.engine.dispose()
        self.tmp.cleanup()

    async def process(self):
        # Новый saver и новый граф имитируют смену процесса между всеми действиями.
        async with open_checkpointer(self.checkpoint) as saver:
            graph = build_course_graph(self.settings, self.model, self.evidence, self.store, saver,
                                       university_literature_enricher=self.university_literature_enricher)
            self.assertTrue(await process_next(self.engine, self.factory, graph))
        async with self.factory() as db:
            return await db.get(Course, "course")

    async def test_lifespan_card_replaces_sidebar_market_controls_and_archives(self):
        doc = source_record("documentation", "https://example.org/schedule", "График поддержки", "Окончание поддержки")
        deadline = SupportDeadline("Инструмент 1", "Базы данных", now().date() + timedelta(days=730), doc)
        with patch("orchestration.graph.fetch_support_deadlines", return_value=([deadline], False)):
            course = await self.process()
        self.assertEqual(course.status, "waiting_for_review")
        async with self.factory() as db:
            draft = await db.get(CourseVersion, course.draft_version_id)
            calculated = draft.content["lifespan_assessment"]
            self.assertEqual(calculated["estimated_lifespan_years"], 2)
            self.assertEqual(calculated["review_by"], deadline.end.isoformat())
            self.assertEqual(calculated["source_ids"], [doc.source_id])
            self.assertIsNotNone(await db.get(SourceRow, (course.id, doc.source_id)))
        await decide(self.factory, self.teacher, course.id, draft.id, "approve")
        course = await self.process()

        script = """from apps.teacher_app import render
from auth.settings import Settings
from auth.user_repository import CurrentUser
from sqlalchemy.ext.asyncio import async_sessionmaker
from storage.database import make_engine, run_async
async def open_course():
    settings = Settings()
    engine = make_engine(settings.app_database_url)
    try:
        actor = CurrentUser(id='teacher', email='teacher@example.org', display_name='Методист', role='teacher')
        await render(actor, settings, async_sessionmaker(engine, expire_on_commit=False), None)
    finally:
        await engine.dispose()
run_async(open_course())
"""
        settings = {"APP_DATABASE_URL": self.engine.url.render_as_string(), "SESSION_SIGNING_KEY": "s" * 40}
        cases = (
            (None, None, "Эта версия сохранена до появления прогноза актуальности."),
            (None, "Оценка актуальности курса недоступна: нет рыночных или научных источников.",
             "Для расчёта не нашлось рыночных или научных источников по курсу."),
            (None, "Оценка актуальности курса недоступна (circuit_open).",
             "Прогноз недоступен. Запросы к модели временно приостановлены."),
            (None, INSUFFICIENT_EVIDENCE, "Горизонт пока нельзя рассчитать. Нужна дата окончания поддержки версии "
             "из программы или заметное изменение доли навыка в сопоставимых выборках вакансий."),
            (lifespan_data(5.1), None, None),
            (calculated, None, None),
        )
        for assessment, notice, message in cases:
            async with self.factory.begin() as db:
                active = await db.get(CourseVersion, course.active_version_id)
                content = dict(active.content)
                content["source_notices"] = [notice] if notice else []
                if assessment is None and notice is None:
                    content.pop("lifespan_assessment", None)
                else:
                    content["lifespan_assessment"] = assessment
                active.content = content
            with self.subTest(assessment=assessment, notice=notice), patch.dict(os.environ, settings):
                app = AppTest.from_string(script, default_timeout=30)
                app.query_params["course"] = course.id
                app.run()
                self.assertFalse(app.exception)
                self.assertNotIn("Рыночные данные курса", [item.label for item in app.expander])
                self.assertIn("Загрузка рыночных данных и настройка соответствий",
                              [item.label for item in app.tabs[1].expander])
                self.assertIn("Прогноз актуальности курса", [item.value for item in app.subheader])
                self.assertEqual(app.button(key=f"delete-{course.id}").label, "Удалить курс")
                if assessment:
                    self.assertIn("Основания прогноза", [item.label for item in app.expander])
                    self.assertTrue(any(assessment["hr_digest_points"][0] in item.proto.body for item in app.get("html")))
                    expected = f"{assessment['estimated_lifespan_years']:g}".replace(".", ",")
                    self.assertIn(expected, [metric.value for metric in app.metric if metric.label.startswith("Горизонт")])
                    if assessment.get("calculation_method"):
                        self.assertIn(f"Проверить к {deadline.end:%d.%m.%Y}.", [item.value for item in app.caption])
                else:
                    captions = [item.value for item in app.caption]
                    self.assertIn(message, captions)
                    if notice:
                        self.assertNotIn(notice, captions)
                if assessment and assessment["is_short_lived"]:
                    app.button(key=f"delete-{course.id}").click().run()
                    async with self.factory() as db:
                        self.assertEqual((await db.get(Course, course.id)).status, "approved")
                    app.button(key=f"delete-confirm-{course.id}").click().run()
                    self.assertFalse(app.exception)
                    self.assertIn("Ваши курсы", [item.value for item in app.header])
                    self.assertNotIn("selected_course", app.session_state)
        async with self.factory() as db:
            self.assertEqual((await db.get(Course, course.id)).status, "cancelled")
            self.assertEqual((await db.get(CourseVersion, draft.id)).status, "archived")
            self.assertIsNotNone(await db.get(SourceRow, (course.id, "HH")))

    async def test_lifespan_feed_failure_leaves_reviewable_draft(self):
        with patch("orchestration.graph.fetch_support_deadlines", return_value=([], True)):
            course = await self.process()
        self.assertEqual(course.status, "waiting_for_review")
        async with self.factory() as db:
            version = await db.get(CourseVersion, course.draft_version_id)
            self.assertIsNone(version.content["lifespan_assessment"])
            self.assertEqual(version.content["lifespan_note"], SUPPORT_FEED_UNAVAILABLE)
            self.assertIsNone(await db.scalar(select(ModelCall).where(ModelCall.stage == "lifespan_assessment")))

    async def test_lifespan_compares_saved_vacancy_samples_and_keeps_source_ids(self):
        today = now().date()

        def vacancies(collected, count):
            return [source_record("Rostrud", f"https://example.org/{collected}/{i}", "Инженер",
                                  f"Опубликована: {collected.isoformat()}\nНавыки: {'SQL' if i < count else 'работа в команде'}",
                                  retrieved_at=datetime.combine(collected, time(), timezone.utc)) for i in range(20)]

        old = vacancies(today - timedelta(days=60), 15)
        new = vacancies(today, 5)
        cards = old
        self.evidence.search.side_effect = lambda role, *args, **kwargs: EvidenceResults(
            records().values(), market_records=cards if role == "business" else [])
        course = await self.process()
        async with self.factory() as db:
            first = await db.get(CourseVersion, course.draft_version_id)
        self.assertIsNone(first.content["lifespan_assessment"])
        self.assertEqual(len(first.content["lifespan_market_sample"]["source_ids"]), 20)
        cards = new
        await rebuild_course(self.factory, self.teacher, course.id)
        with patch("orchestration.graph.fetch_support_deadlines", return_value=([], True)):
            course = await self.process()
        self.assertEqual(course.status, "waiting_for_review")
        async with self.factory() as db:
            draft = await db.get(CourseVersion, course.draft_version_id)
            assessment = draft.content["lifespan_assessment"]
            self.assertEqual(assessment["calculation_method"], "market_shift")
            self.assertEqual(assessment["estimated_lifespan_years"], 0)
            self.assertEqual(set(assessment["source_ids"]), {row.source_id for row in old + new})
            self.assertIsNotNone(await db.get(SourceRow, (course.id, old[0].source_id)))

    async def test_delete_requires_owner_and_waits_for_pending_commands(self):
        for actor in (user("teacher", "-other"), user("expert")):
            with self.assertRaises(PermissionError):
                await delete_course(self.factory, "course", actor)
        with self.assertRaisesRegex(ValueError, "завершения обработки"):
            await delete_course(self.factory, "course", self.teacher)
        course = await self.process()
        await delete_course(self.factory, course.id, self.teacher)
        await delete_course(self.factory, course.id, self.teacher)
        self.assertEqual(await schedule_due(self.factory, now() + timedelta(days=275)), 0)

    async def test_new_market_tags_are_normalized_and_saved_once_per_version(self):
        vacancy = source_record("HH", "https://hh.ru/vacancy/123", "Инженер",
                                "Навыки: SQL, NeoTool, OtherTool")
        self.evidence.search.side_effect = lambda role, *args, **kwargs: [
            *records().values(), *([vacancy] if role == "business" else [])]
        base = self.model.generate
        calls = []

        async def generate(output, prompt, payload, context, **options):
            if output.__name__ == "SkillMappingBatch":
                calls.append(payload["raw_skills"])
                return output.model_validate({"items": [
                    {"raw_skill": tag, "competence_name": "Автоматизация рабочих процессов", "tool_name": tag}
                    for tag in payload["raw_skills"]]})
            return await base(output, prompt, payload, context, **options)

        self.model.generate = generate
        course = await self.process()
        self.assertEqual(course.status, "waiting_for_review")
        self.assertEqual(len(calls), 1)
        self.assertCountEqual(calls[0], ["NeoTool", "OtherTool"])
        async with self.factory() as db:
            draft = await db.get(CourseVersion, course.draft_version_id)
            self.assertEqual(draft.content["skill_mappings"]["neotool"]["tool_name"], "NeoTool")
        await decide(self.factory, self.teacher, course.id, draft.id, "refresh_materials")
        self.assertEqual((await self.process()).status, "waiting_for_review")
        self.assertEqual(len(calls), 1)

    async def test_rebuild_creates_model_draft_and_keeps_approved_version(self):
        course = await self.process()
        await decide(self.factory, self.teacher, course.id, course.draft_version_id, "approve")
        course = await self.process()
        approved = course.active_version_id
        await rebuild_course(self.factory, self.teacher, course.id)
        with self.assertRaises(ValueError):
            await rebuild_course(self.factory, self.teacher, course.id)  # уже в очереди
        course = await self.process()
        self.assertEqual(course.status, "waiting_for_review")
        self.assertEqual(course.active_version_id, approved)
        async with self.factory() as db:
            draft = await db.get(CourseVersion, course.draft_version_id)
        self.assertEqual(draft.revision, 2)
        self.assertTrue(all(subject["generated_by"] == "model" for subject in draft.content["program"]["subjects"]))

    async def test_verified_boundary_reaches_generator_and_flags_existing_subject_without_fallback(self):
        async with self.factory.begin() as db:
            db.add(ProfessionBoundary(course_id="course", primary_profession="Курс", adjacent_profession="BI-аналитик",
                skill_jaccard_overlap=0.5, exclusive_to_adjacent_json=["SQL"], verified_by_methodologist=True,
                source_interview_id="boundary-interview"))
            db.add(ProfessionBoundary(course_id="course", primary_profession="Курс", adjacent_profession="Аналитик",
                skill_jaccard_overlap=0.4, exclusive_to_adjacent_json=["Python"]))
        original = self.model.generate
        contexts = []

        async def generate(output, prompt, payload, context, **options):
            if output.__name__ in {"SubjectPlan", "CourseProgramModel"}:
                exclusions = payload["adjacent_profession_exclusions"]
                self.assertEqual([item["exclusive_item"] for item in exclusions], ["SQL"])
                contexts.append(output.__name__)
            return await original(output, prompt, payload, context, **options)

        self.model.generate = generate
        with patch("orchestration.graph.make_outline", side_effect=AssertionError("Пересечение не повод для каркаса")) as outline:
            course = await self.process()
        outline.assert_not_called()
        self.assertEqual(course.status, "waiting_for_review")
        self.assertEqual(set(contexts), {"SubjectPlan", "CourseProgramModel"})
        async with self.factory() as db:
            draft = await db.get(CourseVersion, course.draft_version_id)
        subject, = draft.content["program"]["subjects"]
        self.assertTrue(subject["needs_review"])
        self.assertEqual(subject["title"], program()["subjects"][0]["title"])
        self.assertEqual({item["source_interview_id"] for item in subject["boundary_warnings"]}, {"boundary-interview"})

    async def test_judge_timeout_falls_back_to_needs_review_not_outline(self):
        original = self.model.generate
        names = ("Базы данных", "SQL-запросы")

        async def generate(output, prompt, payload, context, **options):
            if output.__name__ in {"ProfessionEvidenceVerdicts", "RationaleVerdicts"}:
                raise TimeoutError("token=SECRET; private input")
            if output.__name__ == "SubjectPlan":
                return output.model_validate({"subjects": [
                    {"name": name, "is_fundamental": False, "conflict_with": None, "source_ids": ["Books"]}
                    for name in names]}, context=context)
            if output.__name__ == "CourseProgramModel":
                value = program()
                subject = value["subjects"][0]
                subject.update(title=payload["batch_subjects"][0], origin="auto", status="suggested",
                               is_fundamental=False, locked=False)
                subject["topics"][0]["activity_task_ids"] = []
                return output.model_validate(value, context=context)
            return await original(output, prompt, payload, context, **options)

        self.model.generate = generate
        with patch("orchestration.graph.make_outline", side_effect=AssertionError("Не строить каркас")) as outline:
            course = await self.process()
        outline.assert_not_called()
        self.assertEqual(course.status, "waiting_for_review")
        async with self.factory() as db:
            draft = await db.get(CourseVersion, course.draft_version_id)
            calls = (await db.scalars(select(ModelCall).where(ModelCall.stage == "llm_judge"))).all()
        claims = draft.content["profession_profile"]["claims"]
        self.assertTrue(claims and all(claim["verification_status"] == "needs_review" for claim in claims))
        self.assertEqual([subject["title"] for subject in draft.content["program"]["subjects"]], list(names))
        self.assertTrue(all(subject["generated_by"] == "model" for subject in draft.content["program"]["subjects"]))
        profile_calls = [call for call in calls if call.output == "ProfessionEvidenceVerdicts"]
        self.assertEqual(len(profile_calls), 1)
        self.assertEqual(profile_calls[0].outcome, "timeout")
        self.assertNotIn("SECRET", profile_calls[0].error)
        rationale_calls = [call for call in calls if call.output == "RationaleVerdicts"]
        subjects_with_rationale = [subject for subject in draft.content["program"]["subjects"]
                                   if subject.get("inclusion_rationale")]
        self.assertTrue(subjects_with_rationale)
        self.assertEqual({call.subject_id for call in rationale_calls}, {subject["id"] for subject in subjects_with_rationale})
        self.assertTrue(all(subject["rationale_check"]["verification_status"] == "needs_review"
                            for subject in subjects_with_rationale))

    async def test_filtered_books_summary_and_market_audit_do_not_stop_the_course(self):
        filtered = source_record("Books", "https://example.org/filtered", "Подборка книг", "Текст для теста")
        self.evidence.search.side_effect = lambda role, *_args, **_kwargs: (
            [filtered] if role == "books" else list(records().values()))
        original = self.model.generate

        async def generate(output, prompt, payload, context, flash=False, **options):
            if output.__name__ == "BusinessCourseAudit" or filtered.source_id in context.get("sources", {}):
                raise YandexModeratedResponse("Провайдер ограничил ответ.")
            return await original(output, prompt, payload, context, flash, **options)

        self.model.generate = generate
        course = await self.process()
        self.assertEqual(course.status, "waiting_for_review")
        async with self.factory() as db:
            version = await db.get(CourseVersion, course.draft_version_id)
            self.assertIn("books: Yandex AI Studio ограничила анализ этой подборки источников.",
                          version.content["source_notices"])
            self.assertIsNone(version.content["program"]["business_audit"])
            self.assertIsNone(await db.get(SourceRow, (course.id, filtered.source_id)))
            self.assertTrue(version.content["program"]["subjects"][0]["inclusion_rationale"])

    async def test_filtered_interview_question_falls_back_and_allows_draft(self):
        self.model.gap = True
        original = self.model.generate

        async def generate(output, prompt, payload, context, flash=False, **options):
            if output.__name__ == "InterviewTurn":
                raise YandexModeratedResponse("Провайдер ограничил ответ.")
            return await original(output, prompt, payload, context, flash, **options)

        self.model.generate = generate
        course = await self.process()
        self.assertEqual(course.status, "waiting_for_expert")
        async with self.factory() as db:
            interview = await db.scalar(select(Interview).where(Interview.course_id == course.id))
            self.assertEqual(interview.gap_report["question_origin"], "gap_report")
            self.assertIn(interview.gap_report["prompt"], interview.gap_report["questions"])

        await finish_without_answer(self.factory, self.teacher, course.id, interview.id)
        course = await self.process()
        self.assertEqual(course.status, "waiting_for_review")
        async with self.factory() as db:
            draft = await db.get(CourseVersion, course.draft_version_id)
            self.assertIn("Интервью: использован резервный вопрос по выявленному пробелу.",
                          draft.content["source_notices"])

    async def test_legacy_workload_does_not_block_review_or_leak_into_export(self):
        async with self.factory.begin() as db:
            stored = await db.get(Course, "course")
            stored.course_meta = dict(stored.course_meta, workload={"total_credits": 30, "hours_per_credit": 36})
        course = await self.process()
        async with self.factory.begin() as db:
            draft = await db.get(CourseVersion, course.draft_version_id)
            self.assertNotIn("workload_proposal", draft.content)
            # Черновик и команда могли остаться от приложения до удаления расчёта.
            draft.content = {**draft.content, "workload_proposal": {"total_hours": 1080},
                             "workloads": {"subject": {"lectures": 18}}}
        await decide(self.factory, self.teacher, course.id, draft.id, "approve")
        async with self.factory.begin() as db:
            command = await db.get(WorkflowCommand, f"review:{draft.id}")
            command.payload = {**command.payload, "workloads": draft.content["workloads"]}
        await decide(self.factory, self.teacher, course.id, draft.id, "approve")
        course = await self.process()
        self.assertEqual(course.status, "approved")
        async with self.factory() as db:
            active = await db.get(CourseVersion, course.active_version_id)
            self.assertEqual(active.content["workloads"], draft.content["workloads"])
            sources = {row.source_id: row.record for row in (await db.scalars(
                select(SourceRow).where(SourceRow.course_id == course.id))).all()}
            archive = export_course(course.course_meta, active, sources)
        with ZipFile(BytesIO(archive)) as package:
            for field in ("workload_proposal", "workloads"):
                self.assertNotIn(field, json.loads(package.read("course.json")))
            for field in ("workload", "workloadInputs"):
                self.assertNotIn(field, json.loads(package.read("rpd/01/rpd.json")))
            self.assertNotIn("Нагрузка", package.read("rpd/01/rpd.html").decode())

    async def test_mixed_revision_updates_course_content_without_workload(self):
        course = await self.process()
        original = self.model.generate

        async def generate(output, prompt, payload, context, flash=False, **options):
            if output.__name__ == "CourseProgramModel":
                revised = program()
                assignment = revised["subjects"][0]["topics"][0]["assignments"][0]
                old = context["baseline"].subjects[0].topics[0].assignments[0]
                assignment.update(instructions="Составьте SQL-запрос и проверьте работу индекса.",
                                  change_status="modified", previous_value=block_value(old))
                return output.model_validate(revised, context=context)
            return await original(output, prompt, payload, context, flash, **options)

        self.model.generate = generate
        await decide(self.factory, self.teacher, course.id, course.draft_version_id, "revise",
                     "Установи 216 академических часов на курс и добавь в задание SQL проверку индекса.")
        course = await self.process()
        self.assertEqual(course.status, "waiting_for_review")
        async with self.factory() as db:
            draft = await db.get(CourseVersion, course.draft_version_id)
            self.assertNotIn("workload_proposal", draft.content)
            assignment = draft.content["program"]["subjects"][0]["topics"][0]["assignments"][0]
            self.assertEqual(assignment["instructions"], "Составьте SQL-запрос и проверьте работу индекса.")

    async def test_refresh_keeps_discipline_ids_and_adds_evidence_to_existing_draft(self):
        model = self.model.generate

        async def old_model(output, prompt, payload, context, flash=False, **options):
            if output.__name__ == "SubjectRationaleBatch":
                return output.model_validate({"items": []}, context=context)
            return await model(output, prompt, payload, context, flash, **options)

        self.model.generate = old_model
        course = await self.process()
        original_id = course.draft_version_id
        async with self.factory() as db:
            original = await db.get(CourseVersion, original_id)
            before = original.content["program"]["subjects"][0]
            self.assertIsNone(before["inclusion_rationale"])

        source = LiteratureSource("Практическое введение в SQL", ["А. Иванов"], "textbook", 2024,
                                  "Издательство", "Каталог", None, "https://example.org/sql", 0.95)
        parser = AsyncMock()
        parser.find_many.return_value = [TopicLiterature("Базы данных", "SQL", [], [source])]
        self.university_literature_enricher = partial(enrich_course_program, parser=parser)
        self.model.generate = AsyncMock(side_effect=model)
        await decide(self.factory, self.teacher, course.id, original_id, "refresh_materials")
        course = await self.process()
        self.assertEqual(course.status, "waiting_for_review")
        async with self.factory() as db:
            revised = await db.get(CourseVersion, course.draft_version_id)
            after = revised.content["program"]["subjects"][0]
            self.assertEqual(after["id"], before["id"])
            self.assertTrue(after["inclusion_rationale"])
            self.assertTrue(any(ref["source_id"].startswith("Literature-") for ref in after["topics"][0]["learning_references"]))
            self.assertEqual((await db.get(CourseVersion, original_id)).status, "archived")
        self.assertEqual({call.args[0].__name__ for call in self.model.generate.call_args_list},
                         {"SubjectRationaleBatch", "RationaleVerdicts"})

    async def test_weak_and_strong_ai_rationale_excludes_weak_learners(self):
        subject = program()["subjects"][0]
        subject["title"] = "Теория слабого и сильного ИИ"
        bad = source_record("Books", "https://example.org/boosting", "Бустинг",
                            "Слабые классификаторы объединяются в сильную модель с помощью бустинга.")
        good = source_record("RPD", "https://example.org/ai", "AI methods and models",
                             "The objectives include acquaintance with the theory of weak (narrow) and strong AI.")
        self.evidence.search.return_value = [bad, good]

        async def generate(output, _prompt, payload, context, flash=False):
            row = payload["disciplines"][0]
            self.assertEqual([item["title"] for item in row["fragments"]], [good.title])
            return output.model_validate({"items": [{"subject_id": row["subject_id"],
                "explanation": "Дисциплина знакомит с теорией слабого и сильного искусственного интеллекта.",
                "excerpt_ids": [row["fragments"][0]["excerpt_id"]]}]}, context=context)

        llm = AsyncMock()
        llm.generate.side_effect = generate
        result, added = await explain_subjects({"title": "ML-инженер", "subjects": [subject]}, {},
                                               self.evidence, llm, "teacher")
        self.assertEqual(result["subjects"][0]["inclusion_rationale"]["source_ids"], [good.source_id])
        self.assertEqual(added, [good])

    async def test_direct_course_review_export_and_audit(self):
        notes = ["Books: использован сохранённый снимок."]
        vacancy = source_record("Rostrud", "https://trudvsem.ru/vacancy/card/1/2", "Аналитик данных",
                                "Работодатель: Компания\nОпубликована: 2026-11-20\nОбязанности: Сбор данных")
        self.evidence.search.side_effect = lambda role, *_args, **_kwargs: EvidenceResults(
            [*records().values(), *([vacancy] if role == "business" else [])], notes)
        course = await self.process()
        self.assertEqual(course.status, "waiting_for_review")
        args = (self.factory, self.teacher, course.id, course.draft_version_id)
        with self.assertRaises(PermissionError):
            await decide(self.factory, user("teacher", "other"), *args[2:], "approve")
        await decide(*args, "approve")
        await decide(*args, "approve")
        course = await self.process()
        self.assertEqual(course.status, "approved")
        active = course.active_version_id
        async with self.factory() as db:
            version = await db.get(CourseVersion, active)
            sources = {k: v.model_dump(mode="json") for k, v in records().items()}
            sources["Books"]["url"] = "https://ai.api.cloud.yandex.net/v1/files/test-book"
            self.assertEqual(version.content["vacancy_source_ids"], [vacancy.source_id])
            stored = await db.scalar(select(SourceRow).where(SourceRow.course_id == course.id,
                                                             SourceRow.source_id == vacancy.source_id))
            sources[vacancy.source_id] = stored.record
            archive = export_course(course.course_meta, version, sources)
        with ZipFile(BytesIO(archive)) as package:
            rpd = json.loads(package.read("rpd/01/rpd.json"))
            self.assertNotIn("workload", rpd)
            self.assertTrue(rpd["assessmentMaterials"])
            self.assertEqual(rpd["sourceNotes"], notes)
            self.assertEqual(json.loads(package.read("course.json"))["source_notices"], notes)
            self.assertEqual(json.loads(package.read("sources.json"))[vacancy.source_id]["url"], vacancy.url)
            report = package.read("course-report.html").decode()
            self.assertIn("Базы данных", report)
            self.assertIn("Дисциплина даёт основу для составления запросов SQL", report)
            self.assertIn("Реляционные модели данных и запросы SQL", report)
            self.assertIn("Оценка курса индустрией", report)
            self.assertIn(vacancy.url, report)
            self.assertNotIn("ai.api.cloud.yandex.net", report)
            html = package.read("rpd/01/rpd.html").decode()
            self.assertIn("color:", html)
            self.assertIn("<details><summary", html)
            self.assertNotIn("url", rpd["readingList"][0])
        self.assertEqual(await schedule_due(self.factory), 0)
        for needs_update in (False, True):
            self.model.relevance = needs_update
            self.assertEqual(await schedule_due(self.factory, now() + timedelta(days=271)), 1)
            self.assertEqual(await schedule_due(self.factory, now() + timedelta(days=271)), 0)
            course = await self.process()
            self.assertEqual(course.status, "pending_update_review" if needs_update else "approved")
        async with self.factory() as db:
            self.assertEqual((await db.scalar(select(Notification))).user_id, self.teacher.id)
        await acknowledge(self.factory, self.teacher, course.id)
        await decide(self.factory, self.teacher, course.id, course.draft_version_id, "reject")
        course = await self.process()
        self.assertEqual(course.active_version_id, active)
        self.assertIsNone(course.draft_version_id)

    async def test_university_adds_literature_and_preserves_business_evidence(self):
        study = LiteratureSource("Исследование обработки запросов SQL", ["А. Иванов"], "article", 2024,
                                 "Труды конференции", "DOI", None, "https://doi.org/10.1234/sql", 0.95,
                                 doi="10.1234/sql")
        parser = AsyncMock()
        parser.find_many.return_value = [TopicLiterature("Базы данных", "SQL", ["SQL"], [study])]
        self.university_literature_enricher = partial(enrich_course_program, parser=parser)
        course = await self.process()
        self.assertEqual(course.status, "waiting_for_review")
        parser.find_many.assert_awaited_once_with([("SQL", "Базы данных")])
        async with self.factory() as db:
            version = await db.get(CourseVersion, course.draft_version_id)
            records = (await db.scalars(select(SourceRow))).all()
        corpus = {row.source_id: row.record for row in records}
        topic = version.content["program"]["subjects"][0]["topics"][0]
        citation = topic["learning_references"][0]
        self.assertEqual(corpus[citation["source_id"]]["url"], study.source_url)
        self.assertIn("А. Иванов", citation["bibliographic_excerpt"])
        self.assertEqual(corpus["HH"]["source_type"], "HH")
        self.assertEqual(version.content["program"]["business_audit"]["source_ids"], ["HH"])
        await decide(self.factory, self.teacher, course.id, version.id, "approve")
        await self.process()
        async with self.factory() as db:
            approved = await db.get(CourseVersion, version.id)
        archive = export_course(course.course_meta, approved, corpus)
        with ZipFile(BytesIO(archive)) as package:
            subject = json.loads(package.read("rpd/01/rpd.json"))
            self.assertIn(study.source_url, {item.get("url") for item in subject["readingList"]})
            self.assertIn(study.source_url, package.read("rpd/01/rpd.html").decode())

    async def test_expert_tool_link_populates_market_table_without_adding_vacancies(self):
        self.model.gap = True
        vacancy = source_record("HH", "https://hh.ru/vacancy/123", "Инженер", "Навыки: машинное обучение")
        self.evidence.search.side_effect = lambda role, *args, **kwargs: [
            *records().values(), *([vacancy] if role == "business" else [])]
        quote = "В PyTorch обучаю классификатор."
        rejected = "В FictionTool проверяю прогноз."
        original = self.model.generate

        async def generate(output, prompt, payload, context, **options):
            if output.__name__ == "InterviewerExtraction":
                self.assertEqual(payload["tool_competence"], ML_COMPETENCE)
                return output.model_validate({"items": [
                    {"slot_name": "tool_stack", "statement": quote, "tool_name": "PyTorch", "operation": "обучаю классификатор"},
                    {"slot_name": "tool_stack", "statement": rejected, "tool_name": "FictionTool", "operation": "проверяю прогноз"},
                    {"slot_name": "tool_stack", "statement": quote, "tool_name": "Keras", "operation": "обучаю классификатор"},
                ]})
            if output.__name__ == "VerificationReport":
                return output.model_validate({"claims": [
                    {"claim": statement, "verification_status": status, "explanation": "Результат сверки",
                     "clarification_question": None, "source_ids": ["HH"]}
                    for statement, status in [(quote, "verified"), (rejected, "rejected_hallucination")]]}, context=context)
            return await original(output, prompt, payload, context, **options)

        self.model.generate = generate
        self.assertEqual((await self.process()).status, "waiting_for_expert")
        async with self.factory() as db:
            interview = (await db.scalars(select(Interview))).one()
        self.assertIn(ML_COMPETENCE, interview.gap_report["prompt"])
        self.assertIn("tool_stack", interview.gap_report["requested_slots"])
        expert = user("expert")
        users = AsyncMock()
        users.get_by_id.return_value = expert
        token = await self.store.invite(interview.id, expert.id, self.teacher, users)
        await self.store.submit_interview(token, expert, interview.interrupt_id, "tools", f"{quote} {rejected}", [])
        course = await self.process()
        self.assertEqual(course.status, "waiting_for_review")
        async with self.factory() as db:
            version = await db.get(CourseVersion, course.draft_version_id)
            corpus = {row.source_id: row.record for row in (await db.scalars(select(SourceRow))).all()}
        content = version.content
        links = content["profession_profile"]["expert_tool_links"]
        self.assertEqual([(link["tool_name"], link["quote"], link["interview_id"]) for link in links],
                         [("PyTorch", quote, interview.id)])
        report = course_market(course.course_meta, content["program"], corpus, content["vacancy_source_ids"],
                               content["activity"], profile=content["profession_profile"])
        ml = next(row for row in report.competences if row.competence_name == ML_COMPETENCE)
        self.assertEqual((ml.tools, ml.expert_tools, ml.vacancies, ml.share), ([], ["PyTorch"], 1, 1.0))
        await decide(self.factory, self.teacher, course.id, version.id, "approve")
        await self.process()
        async with self.factory() as db:
            approved = await db.get(CourseVersion, version.id)
        with ZipFile(BytesIO(export_course(course.course_meta, approved, corpus))) as archive:
            self.assertIn("PyTorch (эксперт)", archive.read("course-report.html").decode())

    async def test_expert_then_teacher_revision(self):
        self.model.gap = True
        course = await self.process()
        self.assertEqual(course.status, "waiting_for_expert")
        async with self.factory() as db:
            interview = await db.scalar(select(Interview))
        expert = user("expert")
        users = AsyncMock()
        users.get_by_id.return_value = expert
        token = await self.store.invite(interview.id, expert.id, self.teacher, users)
        async with self.factory() as db:
            self.assertIsNone((await db.execute(assigned_interviews(user("expert", "other")))).first())
            inv, _, _ = (await db.execute(assigned_interviews(expert, token_hash=hashlib.sha256(token.encode()).hexdigest()))).one()
        files = [{"name": "task.txt", "text": "Сверка отчёта"}]
        await self.store.submit_interview("", expert, interview.interrupt_id, inv.id, "Сверка", files, invitation_id=inv.id)
        await self.store.submit_interview("", expert, interview.interrupt_id, inv.id, "Сверка", files, invitation_id=inv.id)
        course = await self.process()
        self.assertEqual(course.status, "waiting_for_review")
        initial = course.draft_version_id
        factcheck_query = next(call.args[1] for call in self.evidence.search.call_args_list if call.args[0] == "factcheck")
        self.assertTrue(factcheck_query.startswith("Курс:"))
        self.assertNotIn("Сверка", factcheck_query)
        await decide(self.factory, self.teacher, course.id, initial, "revise", "", files)
        course = await self.process()
        self.assertNotEqual(initial, course.draft_version_id)
        self.assertEqual(course.status, "waiting_for_review")
        async with self.factory() as db:
            self.assertTrue((await db.get(Interview, interview.id)).gap_report["verification"]["claims"])
            self.assertEqual((await db.get(CourseVersion, initial)).status, "archived")
            cmd = await db.get(WorkflowCommand, f"review:{initial}")
            cmd.status = "pending"
            await db.commit()
        self.assertEqual((await self.process()).draft_version_id, course.draft_version_id)
        self.assertEqual(sum(call.args[0] == "business" for call in self.evidence.search.call_args_list), 1)

    async def test_rejected_expert_claim_still_produces_reviewable_draft(self):
        self.model.gap = True
        self.model.claim_status = "rejected_hallucination"
        course = await self.process()
        async with self.factory() as db:
            interview = await db.scalar(select(Interview))
        expert = user("expert")
        users = AsyncMock()
        users.get_by_id.return_value = expert
        token = await self.store.invite(interview.id, expert.id, self.teacher, users)
        await self.store.submit_interview(token, expert, interview.interrupt_id, "answer", "Сверка", [])
        course = await self.process()
        self.assertEqual(course.status, "waiting_for_review")
        async with self.factory() as db:
            self.assertEqual(len((await db.scalars(select(Interview))).all()), 1)
            stored = await db.get(Interview, interview.id)
            self.assertEqual(stored.gap_report["verification"]["claims"][0]["verification_status"], "rejected_hallucination")
            version = await db.get(CourseVersion, course.draft_version_id)
            self.assertIn("missing_tasks", version.content["evidence_gaps"]["gap_types"])
            sources = (await db.scalars(select(SourceRow))).all()
            self.assertFalse(any(source.record["source_type"] == "expert" for source in sources))

    async def test_member_checking_keeps_turns_and_drafts_only_confirmed_work(self):
        self.model.gap = True
        original = self.model.generate

        async def generate(output, prompt, payload, context, **options):
            if output.__name__ == "ProfessionEvidenceVerdicts":
                return output.model_validate({"items": [
                    {"pair_id": pair["pair_id"], "verdict": "weak" if pair["claim"].endswith(("этапе 0", "этапе 1"))
                     else "supports"} for pair in payload["pairs"]]}, context=context)
            if output.__name__ == "CourseProgramModel":
                tasks = {task.id: task.title for task in context["activity"].tasks}
                self.assertIn("task0", tasks)
                self.assertNotIn("task1", tasks)
                self.assertIn("Сверяет миграции PostgreSQL", tasks.values())
            return await original(output, prompt, payload, context, **options)

        self.model.generate = generate
        extra = source_record("Priklad", "https://example.org/work", "Работа с PostgreSQL", "Проверка миграций PostgreSQL.")
        self.evidence.search.return_value = [*records().values(), extra]
        await self.process()
        async with self.factory() as db:
            first = (await db.scalars(select(Interview))).one()
        self.assertEqual(first.gap_report["profession_profile"]["interview_method_used"], "member_checking_validation")
        self.assertIn("work_task:task0", first.gap_report["review_candidate_ids"])
        expert = user("expert")
        users = AsyncMock()
        users.get_by_id.return_value = expert
        token = await self.store.invite(first.id, expert.id, self.teacher, users)
        await self.store.submit_interview(token, expert, first.interrupt_id, "one", "Проверка задач", [],
                                          profession_decisions=[{"claim_id": "work_task:task0", "action": "confirm"},
                                                                {"claim_id": "work_task:task1", "action": "reject"}],
                                          continue_interview=True)
        self.assertEqual((await self.process()).status, "waiting_for_expert")
        async with self.factory() as db:
            invitation, second, _ = next(entry for entry in (await db.execute(assigned_interviews(expert))).all()
                                         if entry[1].id != first.id)
        self.assertEqual(second.gap_report["interview_turn"], 2)
        await self.store.submit_interview("", expert, second.interrupt_id, "two", "Новый рабочий кейс", [],
                                          invitation_id=invitation.id,
                                          profession_decisions=[{"claim_id": "new", "action": "add",
                                                                "statement": "Сверяет миграции PostgreSQL"}])
        course = await self.process()
        self.assertEqual(course.status, "waiting_for_review")
        async with self.factory() as db:
            version = await db.get(CourseVersion, course.draft_version_id)
        claims = {item["claim_id"]: item for item in version.content["profession_profile"]["claims"]}
        self.assertEqual(claims["work_task:task0"]["evidence"][-1]["source_id"], "turn:1")
        self.assertEqual(claims["work_task:task1"]["verification_status"], "rejected")
        self.assertEqual(claims["expert:turn:2:new"]["verification_status"], "verified")

    async def test_teacher_can_finish_existing_interview_and_open_draft(self):
        self.model.gap = True
        course = await self.process()
        async with self.factory() as db:
            interview = await db.scalar(select(Interview))
        with self.assertRaises(PermissionError):
            await finish_without_answer(self.factory, user("teacher", "other"), course.id, interview.id)
        await finish_without_answer(self.factory, self.teacher, course.id, interview.id)
        await finish_without_answer(self.factory, self.teacher, course.id, interview.id)
        self.assertEqual((await self.process()).status, "waiting_for_review")
        async with self.factory() as db:
            self.assertEqual((await db.get(Interview, interview.id)).status, "closed")

    async def test_twelve_auto_subjects_plus_manual_survive_batch_timeout(self):
        self.model.gap = True
        original = self.model.generate

        async def generate(output, prompt, payload, context, flash=False, **options):
            if output.__name__ == "SubjectPlan":
                return output.model_validate({"subjects": [
                    {"name": f"Дисциплина {i}", "is_fundamental": True, "conflict_with": None, "source_ids": ["Books"]}
                    for i in range(14)]}, context=context)
            if output.__name__ == "CourseProgramModel":
                self.assertLessEqual(len(payload["batch_subjects"]), 4)
                raise RuntimeError("Внешняя модель не ответила")
            return await original(output, prompt, payload, context, flash, **options)

        self.model.generate = AsyncMock(side_effect=generate)
        async with self.factory.begin() as db:
            course = await db.get(Course, "course")
            course.course_meta = dict(course.course_meta, manual_subjects=["Ручной предмет"])
        course = await self.process()
        async with self.factory() as db:
            interview = await db.scalar(select(Interview))
        await finish_without_answer(self.factory, self.teacher, course.id, interview.id)
        course = await self.process()
        self.assertEqual(course.status, "waiting_for_review")
        async with self.factory() as db:
            version = await db.get(CourseVersion, course.draft_version_id)
            subjects = version.content["program"]["subjects"]
            self.assertEqual(sum(subject["origin"] == "auto" for subject in subjects), 12)
            self.assertEqual([s["title"] for s in subjects if s["origin"] == "manual"], ["Ручной предмет"])
            self.assertTrue(version.content["source_notices"])
            self.assertTrue(all(subject["inclusion_rationale"] is None for subject in subjects))
            self.assertEqual(rationale_report_html(version.content["program"], {}).count(
                "В базе не подтверждено обоснование"), len(subjects))


class DirectoryTests(unittest.TestCase):
    def test_actual_text_layout_and_stable_lookup(self):
        items = list(accounts_from_text("Заголовок\n\nЛогин: Test@example.org\nПароль: xyz:123\n", "expert"))
        self.assertEqual(items[0].user.email, "test@example.org")
        self.assertTrue(ExistingPasswordVerifier().verify("xyz:123", items[0].password_hash))
        self.assertNotIn("xyz:123", repr(items[0]))
