"""Выполнение очереди команд с восстановлением из checkpoint."""
import argparse
import asyncio
import logging
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from sqlalchemy import and_, case, exists, or_, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.orm import aliased

from auth.settings import Settings
from domain.official_rpd import course_audit
from domain.workflow_queue import MAX_ATTEMPTS, QUEUED_COURSE_STATUS
from orchestration.checkpointer import open_checkpointer
from orchestration.clients import YandexEvidenceClient, YandexModelClient, source_record
from orchestration.graph import build_course_graph
from orchestration.rabota_business import BusinessEvidenceClient, RabotaBusiness
from orchestration.resume import resume_graph
from run_rpd import yandex_settings
from storage.database import DatabaseSetupError, ensure_schema, make_engine, write_transaction
from storage.models import Course, CourseVersion, WorkflowCommand, now
from storage.workflow import SqlWorkflowStore
from workers.retry_policy import classify_failure


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


async def audit_official_rpd(factory, evidence, cmd):
    version_id = cmd.payload["version_id"]
    async with factory() as db:
        course = await db.get(Course, cmd.course_id)
        if course is None or version_id not in {course.draft_version_id, course.active_version_id}:
            return
        version = await db.get(CourseVersion, version_id)
        subjects = version.content["program"]["subjects"]
        meta, owner = course.course_meta, course.owner_user_id

    slots = asyncio.Semaphore(4)

    async def search(subject):
        # Keep each search scoped to the exact discipline and course.
        query = json.dumps({**meta, "discipline": subject["title"]}, ensure_ascii=False, sort_keys=True)
        async with slots:
            return await evidence.search("official_rpd", query, owner_scope=owner)

    results = await asyncio.gather(*(search(subject) for subject in subjects))
    sources = {item.source_id: item for result in results for item in result if item.source_type == "RPD"}
    audit = course_audit(meta, subjects, sources.values(), None)
    audit["lookups"] = {subject["title"]: getattr(result, "rpd_lookup", None)
                        for subject, result in zip(subjects, results)}
    audit["notices"] = list(dict.fromkeys(note for result in results for note in getattr(result, "notices", [])))
    await SqlWorkflowStore(factory).save_sources(cmd.course_id, list(sources.values()))
    async with write_transaction(factory) as db:
        course = await db.scalar(select(Course).where(Course.id == cmd.course_id).with_for_update())
        if course and version_id in {course.draft_version_id, course.active_version_id}:
            version = await db.get(CourseVersion, version_id)
            version.content = {**version.content, "official_audit": audit}
            row = await db.get(WorkflowCommand, cmd.id)
            row.status, row.next_attempt_at, row.last_error_code = "done", None, None


async def execute(graph, factory, cmd, evidence=None):
    if cmd.kind == "official_rpd":
        return await audit_official_rpd(factory, evidence, cmd)
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


async def claim_command(factory):
    older = aliased(WorkflowCommand)
    # Keep commands in order per course, while other courses can run during a backoff.
    precedes = or_(older.created_at < WorkflowCommand.created_at,
                   and_(older.created_at == WorkflowCommand.created_at, older.id < WorkflowCommand.id))
    eligible = (
        select(WorkflowCommand)
        .where(or_(WorkflowCommand.status == "running",
                   and_(WorkflowCommand.status == "pending",
                        or_(WorkflowCommand.next_attempt_at.is_(None), WorkflowCommand.next_attempt_at <= now()),
                        ~exists().where(older.course_id == WorkflowCommand.course_id,
                                        older.status.in_(("pending", "running")), precedes))))
        .order_by(case((WorkflowCommand.status == "running", 0), else_=1),
                  WorkflowCommand.created_at, WorkflowCommand.id)
        .limit(1)
    )
    async with write_transaction(factory) as db:
        cmd = await db.scalar(eligible)
        if cmd is not None and cmd.status == "pending":
            cmd.attempt_count += 1
            cmd.status, cmd.next_attempt_at = "running", None
        # A leftover running command is a continuation of an interrupted attempt.
        return cmd


async def finish_command(factory, cmd, result, failure):
    async with write_transaction(factory) as db:
        row = await db.get(WorkflowCommand, cmd.id)
        if row.status != "running":
            return
        if failure is None:
            row.status, row.next_attempt_at, row.last_error_code = "done", None, None
            course_status = result.get("status") if isinstance(result, dict) else None
            if course_status and cmd.kind != "official_rpd":
                course = await db.get(Course, cmd.course_id)
                if course.status == QUEUED_COURSE_STATUS.get(cmd.kind):
                    course.status = course_status
            return

        row.last_error_code = failure.code
        if failure.retryable and row.attempt_count < MAX_ATTEMPTS:
            row.status = "pending"
            row.next_attempt_at = now() + failure.delay(row.attempt_count)
        else:
            row.status, row.next_attempt_at = "failed", None
            if cmd.kind != "official_rpd":
                course = await db.get(Course, cmd.course_id)
                course.status = "failed"


async def process_next(engine, factory, graph, evidence=None):
    async with worker_lock(engine) as acquired:
        if not acquired:
            return False
        cmd = await claim_command(factory)
        if cmd is None:
            return False
        result, failure = None, None
        try:
            result = await execute(graph, factory, cmd, evidence)
        except Exception as error:
            failure = classify_failure(error)
            logging.warning("Команда %s, попытка %s: %s (%s)", cmd.id, cmd.attempt_count,
                            failure.code, type(error).__name__)
        await finish_command(factory, cmd, result, failure)
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
                try:
                    worked = await process_next(engine, factory, graph, evidence)
                except SQLAlchemyError as error:
                    if once or not classify_failure(error).retryable:
                        raise
                    logging.warning("Очередь временно недоступна (%s).", type(error).__name__)
                    worked = False
                if once:
                    return
                if not worked:
                    await asyncio.sleep(2)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    arguments = parser.parse_args()
    try:
        ensure_schema(Settings().app_database_url)
    except DatabaseSetupError as error:
        raise SystemExit(str(error))
    asyncio.run(run(arguments.once))
