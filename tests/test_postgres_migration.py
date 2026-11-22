import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypedDict
import unittest
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
import psycopg
from psycopg import sql
from sqlalchemy import URL
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker

from orchestration.checkpointer import open_checkpointer
from scripts.migrate_sqlite_to_postgres import migrate
from storage.database import make_engine, postgres_dsn, run_async, upgrade_schema
from storage.models import Course, ProfessionBoundary


POSTGRES_URL = os.getenv("RPD_TEST_POSTGRES_URL")


class InterviewState(TypedDict):
    answer: str


def wait_for_answer(_state: InterviewState) -> InterviewState:
    return {"answer": interrupt("Уточните рабочую задачу")}


@unittest.skipUnless(POSTGRES_URL, "RPD_TEST_POSTGRES_URL is required")
class PostgresMigrationTests(unittest.TestCase):
    def test_saved_course_and_paused_interview_resume_in_postgres(self):
        schema = f"test_migration_{uuid4().hex}"
        base_url = postgres_dsn(POSTGRES_URL)
        parsed = make_url(base_url)
        options = f"{parsed.query.get('options', '')} -csearch_path={schema}".strip()
        target = parsed.update_query_dict({"options": options}).render_as_string(hide_password=False)
        with psycopg.connect(base_url, autocommit=True) as connection, TemporaryDirectory() as folder:
            connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
            try:
                app_path = Path(folder) / "app.sqlite"
                checkpoint_path = Path(folder) / "graph.sqlite"
                source_url = URL.create("sqlite+aiosqlite", database=str(app_path)).render_as_string(hide_password=False)
                upgrade_schema(source_url)
                config = {"configurable": {"thread_id": str(uuid4())}}
                graph = StateGraph(InterviewState)
                graph.add_node("interview", wait_for_answer)
                graph.add_edge(START, "interview")
                graph.add_edge("interview", END)
                pause = run_async(self.prepare(source_url, checkpoint_path, graph, config))

                counts = migrate(app_path, checkpoint_path, target)
                self.assertEqual(counts["courses"], 1)
                self.assertEqual(counts["profession_boundaries"], 1)
                self.assertGreater(counts["checkpoint_writes"], 0)
                with self.assertRaisesRegex(ValueError, "уже содержит данные"):
                    migrate(app_path, checkpoint_path, target)
                run_async(self.resume(target, graph, config, pause))
            finally:
                connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))

    async def prepare(self, url, checkpoints, graph, config):
        engine = make_engine(url)
        course_id = config["configurable"]["thread_id"]
        try:
            async with async_sessionmaker(engine).begin() as session:
                session.add(Course(id=course_id, thread_id=course_id, owner_user_id="owner", course_meta={"title": "Курс"},
                                   status="waiting_for_expert"))
                await session.flush()
                session.add(ProfessionBoundary(id=42, course_id=course_id, primary_profession="Инженер",
                                               adjacent_profession="Аналитик", skill_jaccard_overlap=0.5))
        finally:
            await engine.dispose()
        async with open_checkpointer(str(checkpoints)) as saver:
            result = await graph.compile(checkpointer=saver).ainvoke({"answer": ""}, config)
        return result["__interrupt__"][0]

    async def resume(self, url, graph, config, pause):
        engine = make_engine(url)
        course_id = config["configurable"]["thread_id"]
        try:
            async with async_sessionmaker(engine).begin() as session:
                saved = await session.get(Course, course_id)
                self.assertEqual(saved.course_meta, {"title": "Курс"})
                self.assertEqual(saved.status, "waiting_for_expert")
                boundary = ProfessionBoundary(course_id=course_id, primary_profession="Инженер",
                                              adjacent_profession="Разработчик", skill_jaccard_overlap=0.3)
                session.add(boundary)
                await session.flush()
                self.assertEqual(boundary.id, 43)
        finally:
            await engine.dispose()
        async with open_checkpointer(url) as saver:
            compiled = graph.compile(checkpointer=saver)
            state = await compiled.aget_state(config)
            self.assertEqual(state.interrupts, (pause,))
            result = await compiled.ainvoke(Command(resume={pause.id: "Проверка данных"}), config)
            self.assertEqual(result["answer"], "Проверка данных")
