"""Выполнение очереди команд с восстановлением из checkpoint."""
import argparse
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker

from auth.settings import Settings
from orchestration.checkpointer import open_checkpointer
from orchestration.clients import YandexEvidenceClient, YandexModelClient, source_record
from orchestration.graph import build_course_graph
from orchestration.rabota_business import BusinessEvidenceClient, RabotaBusiness
from orchestration.resume import resume_graph
from run_rpd import yandex_settings
from storage.database import make_engine, write_transaction
from storage.models import Course, CourseVersion, WorkflowCommand
from storage.workflow import SqlWorkflowStore


@asynccontextmanager
async def worker_lock(engine):
    if engine.dialect.name == "postgresql":
        async with engine.begin() as connection:
            acquired = await connection.scalar(text("SELECT pg_try_advisory_xact_lock(625149382)"))
            yield acquired
        return
    path = Path(engine.url.database).resolve().with_suffix(".worker.lock")
    with path.open("a+b") as handle:
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, PermissionError):
            yield False
        else:
            yield True


async def execute(graph, factory, cmd):
    if cmd.kind in {"resume", "review"}:
        key = cmd.payload.get("interrupt_id") or cmd.payload["version_id"]
        return await resume_graph(graph, cmd.course_id, key, cmd.payload)
    config = {"configurable": {"thread_id": cmd.course_id}, "recursion_limit": 100}
    snapshot = await graph.aget_state(config)
    if snapshot.values.get("run_id") == cmd.id:
        if snapshot.next and not any(task.interrupts for task in snapshot.tasks):
            return await graph.ainvoke(None, config)
        return snapshot.values
    if cmd.kind not in {"create", "relevance"}:
        raise ValueError("Неизвестная команда.")
    async with factory() as db:
        course = await db.get(Course, cmd.course_id)
        version = await db.get(CourseVersion, cmd.payload["version_id"]) if cmd.kind == "relevance" else None
        if cmd.kind == "relevance" and (version is None or version.id != course.active_version_id):
            raise ValueError("Версия для аудита изменилась.")
        state = {"request_id": course.id, "owner_user_id": course.owner_user_id, "run_id": cmd.id,
                 "mode": cmd.kind, "course_meta": course.course_meta,
                 "manual_subjects": course.course_meta.get("manual_subjects", []),
                 "baseline": version.content["program"] if version else None}
    return await graph.ainvoke(state, config)


async def process_next(engine, factory, graph):
    async with worker_lock(engine) as acquired:
        if not acquired:
            return False
        async with factory() as db:
            cmd = await db.scalar(select(WorkflowCommand).where(WorkflowCommand.status == "pending")
                                  .order_by(WorkflowCommand.created_at, WorkflowCommand.id).limit(1))
        if cmd is None:
            return False
        status = "done"
        try:
            await execute(graph, factory, cmd)
        except Exception as error:
            status = "failed"
            logging.error("Команда %s: %s", cmd.id, type(error).__name__)
        async with write_transaction(factory) as db:
            row = await db.get(WorkflowCommand, cmd.id)
            row.status = status
            if status == "failed":
                course = await db.get(Course, cmd.course_id)
                course.status = "failed"
        return True


async def run(once):
    settings = Settings()
    config = yandex_settings()
    engine = make_engine(settings.app_database_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with open_checkpointer(settings.checkpoint_url) as saver:
            evidence = YandexEvidenceClient(config)
            if rabota := RabotaBusiness.from_environment():
                evidence = BusinessEvidenceClient(evidence, rabota, source_record)
            graph = build_course_graph(settings, YandexModelClient(config), evidence,
                                       SqlWorkflowStore(factory), saver)
            while True:
                worked = await process_next(engine, factory, graph)
                if once:
                    return
                if not worked:
                    await asyncio.sleep(2)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    asyncio.run(run(parser.parse_args().once))
