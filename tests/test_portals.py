import hashlib
import json
import tempfile
import unittest
from datetime import timedelta
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock
from zipfile import ZipFile

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from auth.password_verifier import ExistingPasswordVerifier
from auth.settings import Settings
from auth.user_repository import CurrentUser
from auth.yandex_users import accounts_from_text
from domain.export import export_course
from domain.schemas import EvidenceResults
from orchestration.checkpointer import open_checkpointer
from orchestration.graph import build_course_graph
from storage.database import make_engine
from storage.models import Base, Course, CourseVersion, ExpertInvitation, Interview, Notification, SourceRow, WorkflowCommand, now
from storage.portal import acknowledge, assigned_interviews, create_course, decide, finish_without_answer, schedule_due
from storage.workflow import SqlWorkflowStore
from test_course_architecture import program, records
from workers.course_worker import process_next


def user(role, suffix=""):
    return CurrentUser(id=role + suffix, email=f"{role}{suffix}@example.org", display_name=role, role=role)


class Model:
    def __init__(self, gap):
        self.gap, self.relevance = gap, True
        self.claim_status = "verified"

    async def generate(self, output, prompt, payload, context, flash=False):
        name = output.__name__
        if name == "AgentReport":
            value = {"findings": []}
        elif name == "AppliedActivityModel":
            count = 4 if self.gap and not any(s.source_type == "expert" for s in context["sources"].values()) else 5
            value = {"tasks": [{"id": f"task{i}", "title": "Запрос", "stack": ["SQL"], "artifacts": ["Отчёт"],
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
        elif name == "RelevanceDecision":
            value = {"requires_update": self.relevance, "explanation": "Изменение практики", "source_ids": ["HH"]}
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
            graph = build_course_graph(self.settings, self.model, self.evidence, self.store, saver)
            self.assertTrue(await process_next(self.engine, self.factory, graph))
        async with self.factory() as db:
            return await db.get(Course, "course")

    async def test_direct_course_review_export_and_audit(self):
        notes = ["Books: использован сохранённый снимок."]
        self.evidence.search.return_value = EvidenceResults(records().values(), notes)
        course = await self.process()
        self.assertEqual(course.status, "waiting_for_review")
        args = (self.factory, self.teacher, course.id, course.draft_version_id)
        with self.assertRaises(PermissionError):
            await decide(self.factory, user("teacher", "other"), *args[2:], "approve")
        hours = {"subject": dict(lectures=9, practice=9, independent=18, labs=0, hours_per_credit=36)}
        await decide(*args, "approve", workloads=hours)
        await decide(*args, "approve", workloads=hours)
        course = await self.process()
        self.assertEqual(course.status, "approved")
        active = course.active_version_id
        async with self.factory() as db:
            version = await db.get(CourseVersion, active)
            archive = export_course(course.course_meta, version, {k: v.model_dump(mode="json") for k, v in records().items()})
        with ZipFile(BytesIO(archive)) as package:
            rpd = json.loads(package.read("rpd/01/rpd.json"))
            self.assertEqual(rpd["workload"]["totalCredits"], 1)
            self.assertEqual(rpd["workload"]["totalHours"], 36)
            self.assertTrue(rpd["assessmentMaterials"])
            self.assertEqual(rpd["sourceNotes"], notes)
            self.assertEqual(json.loads(package.read("course.json"))["source_notices"], notes)
            self.assertIn("color:", package.read("rpd/01/rpd.html").decode())
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


class DirectoryTests(unittest.TestCase):
    def test_actual_text_layout_and_stable_lookup(self):
        items = list(accounts_from_text("Заголовок\n\nЛогин: Test@example.org\nПароль: xyz:123\n", "expert"))
        self.assertEqual(items[0].user.email, "test@example.org")
        self.assertTrue(ExistingPasswordVerifier().verify("xyz:123", items[0].password_hash))
        self.assertNotIn("xyz:123", repr(items[0]))
