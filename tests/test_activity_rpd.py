"""Раздел 7.14: курс по дисциплине программы получает задачи из модели деятельности без LLM-генерации модели."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from sqlalchemy.ext.asyncio import async_sessionmaker

from auth.settings import Settings
from orchestration.checkpointer import open_checkpointer
from orchestration.graph import build_course_graph
from storage.database import make_engine
from storage.models import Base, Course, CourseVersion, SourceRow
from storage.portal import create_course
from storage.workflow import SqlWorkflowStore
from test_course_architecture import records
from test_model_calls import batch_program
from test_portals import Model, user
from workers.course_worker import process_next


SNAPSHOT = {
    "professions": [{"id": "p1", "name": "Backend-разработчик", "version": 3, "as_of": "2026-11-21"}],
    "is_demo": True, "program_id": "prog", "program": "Программная инженерия", "discipline_id": "d1",
    "discipline": "Базы данных",
    "tasks": [{"id": f"task-{index}", "title": title, "profession_id": "p1", "profession": "Backend-разработчик",
               "model_version": 3, "as_of": "2026-11-21", "function_id": "f1", "function": "Работа с данными",
               "status": "confirmed", "status_label": "Подтверждено", "trend": "stable", "priority": 4,
               "coverage": "full", "coverage_reason": "совпали понятия", "tools": ["SQL"], "knowledge": [],
               "reason": ["Почему задача включена", "Подтвердили 36 из 39 экспертов."],
               "quotes": [{"text": title + " с SQL.", "source_title": "Вакансия", "date": "2026-10-01"}]}
              for index, title in enumerate(["Проектирует схемы баз данных под нагрузку", "Оптимизирует SQL-запросы"])],
}


class ProfessionModelCourseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        path = Path(self.tmp.name)
        self.engine = make_engine("sqlite+aiosqlite:///" + (path / "courses.db").as_posix())
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        self.factory = async_sessionmaker(self.engine, expire_on_commit=False)
        self.checkpoint = str(path / "checkpoints.db")
        self.settings = Settings(_env_file=None, session_signing_key="s" * 40)
        self.evidence = MagicMock()

        async def search(*args, **kwargs):
            return list(records().values())

        self.evidence.search = search
        await create_course(self.factory, user("teacher"), "course",
                            {"title": "Базы данных", "institution": "Вуз", "year": "2026", "expand_subjects": False,
                             "activity_model": SNAPSHOT}, ["Базы данных"])

    async def asyncTearDown(self):
        await self.engine.dispose()
        self.tmp.cleanup()

    async def test_course_uses_verified_tasks_and_skips_interview(self):
        model = Model(gap=False)
        original = model.generate
        requested = []

        async def generate(output, prompt, payload, context, flash=False, **options):
            requested.append(output.__name__)
            if output.__name__ in ("AppliedActivityModel", "SubjectPlan"):
                raise AssertionError(f"{output.__name__} не должен генерироваться: задачи берутся из модели профессии")
            if output.__name__ == "CourseProgramModel":
                value = batch_program(payload["batch_subjects"] if "batch_subjects" in payload else ["Базы данных"])
                for subject in value["subjects"]:
                    subject.update(origin="manual", locked=True, status="locked")
                    for topic in subject["topics"]:
                        topic["activity_task_ids"] = ["task-0"]
                return output.model_validate(value, context=context)
            return await original(output, prompt, payload, context, flash, **options)

        model.generate = generate
        async with open_checkpointer(self.checkpoint) as saver:
            graph = build_course_graph(self.settings, model, self.evidence, SqlWorkflowStore(self.factory), saver)
            self.assertTrue(await process_next(self.engine, self.factory, graph))
        async with self.factory() as db:
            course = await db.get(Course, "course")
            version = await db.get(CourseVersion, course.draft_version_id)
            sources = [row.record for row in (await db.execute(SourceRow.__table__.select())).all()]
        self.assertEqual(course.status, "waiting_for_review")
        self.assertNotIn("InterviewTurn", requested)
        self.assertEqual([task["id"] for task in version.content["activity"]["tasks"]], ["task-0", "task-1"])
        self.assertTrue(any(source["source_type"] == "ActivityModel" for source in sources))


if __name__ == "__main__":
    unittest.main()
