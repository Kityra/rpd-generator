from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from auth.user_repository import CurrentUser
from domain.workload import WorkloadParams, proposal_from_loads, propose_workload, workload
from storage.database import write_transaction
from storage.models import Course, CourseVersion, ExpertInvitation, Interview, Notification, WorkflowCommand


def require_teacher(actor: CurrentUser) -> None:
    if actor.role not in {"teacher", "admin"}:
        raise PermissionError("Требуется роль преподавателя.")


async def create_course(factory: async_sessionmaker[AsyncSession], actor: CurrentUser, course_id: str,
                        meta: dict, subjects: list[str]) -> None:
    require_teacher(actor)
    if not meta.get("title", "").strip():
        raise ValueError("Введите название курса.")
    async with write_transaction(factory) as db:
        existing = await db.get(Course, course_id)
        if existing:
            if existing.owner_user_id != actor.id or existing.course_meta != dict(meta, manual_subjects=subjects):
                raise ValueError("Заявка уже сохранена с другими параметрами.")
            return
        course = Course(id=course_id, thread_id=course_id, owner_user_id=actor.id,
                        course_meta=dict(meta, manual_subjects=subjects), status="queued")
        db.add(course)
        await db.flush()
        db.add(WorkflowCommand(id=f"create:{course_id}", course_id=course_id, kind="create", payload={}))


async def save_market(factory: async_sessionmaker[AsyncSession], actor: CurrentUser, course_id: str, *,
                      dataset: dict | None = ..., overrides: dict[str, list[str]] | None = ...) -> None:
    """Сохранить загруженные рыночные данные и/или ручные правки соответствия навыков дисциплинам."""
    require_teacher(actor)
    async with write_transaction(factory) as db:
        course = await db.scalar(select(Course).where(Course.id == course_id, Course.owner_user_id == actor.id).with_for_update())
        if course is None:
            raise PermissionError("Курс недоступен.")
        market = dict((course.course_meta or {}).get("market") or {})
        if dataset is not ...:
            if dataset is None:
                market.pop("dataset", None)
            else:
                market["dataset"] = dataset
        if overrides is not ...:
            market["overrides"] = {key: list(value) for key, value in (overrides or {}).items()}
        course.course_meta = {**course.course_meta, "market": market}


async def decide(factory: async_sessionmaker[AsyncSession], actor: CurrentUser, course_id: str, version_id: str,
                 action: str, content: str = "", attachments: tuple | list = (), workloads: dict | None = None) -> None:
    require_teacher(actor)
    if action not in {"approve", "reject", "revise", "refresh_materials"} or (action == "revise" and not (content.strip() or attachments)):
        raise ValueError("Укажите решение или правку.")
    async with write_transaction(factory) as db:
        course = await db.scalar(select(Course).where(Course.id == course_id, Course.owner_user_id == actor.id).with_for_update())
        if course is None:
            raise PermissionError("Курс недоступен.")
        proposal = None
        if workloads or action in {"approve", "revise", "refresh_materials"}:
            version = await db.get(CourseVersion, version_id)
            if not version or version.course_id != course_id:
                raise ValueError("Нагрузка не соответствует версии курса.")
            subjects = version.content["program"]["subjects"]
            proposal = version.content.get("workload_proposal")
            if proposal is None and course.active_version_id:
                active = await db.get(CourseVersion, course.active_version_id)
                if active:
                    proposal = proposal_from_loads(active.content.get("workloads", {}))
            if action == "approve" and workloads is None:
                workloads = propose_workload(subjects, proposal, WorkloadParams.from_meta(course.course_meta))["loads"]
            if set(workloads or {}) - {subject["id"] for subject in subjects}:
                raise ValueError("Нагрузка не соответствует версии курса.")
            for hours in (workloads or {}).values():
                workload(hours)
        payload = {"action": action, "actor_user_id": actor.id, "version_id": version_id,
                   "content": content.strip() or ("См. приложенные материалы." if attachments else ""),
                   "attachments": list(attachments), "workloads": workloads or {}}
        if proposal and action in {"revise", "refresh_materials"}:
            payload["workload_proposal"] = proposal
        key = f"review:{version_id}"
        existing = await db.get(WorkflowCommand, key)
        if existing and existing.payload == payload:
            return
        if existing or course.draft_version_id != version_id or course.status not in {"waiting_for_review", "pending_update_review"}:
            raise ValueError("Версия уже изменилась. Обновите страницу.")
        if action == "reject" and not course.active_version_id:
            raise ValueError("Отклонить можно обновление утверждённого курса.")
        db.add(WorkflowCommand(id=key, course_id=course_id, kind="review", payload=payload))
        course.status = "review_queued"


async def finish_without_answer(factory: async_sessionmaker[AsyncSession], actor: CurrentUser, course_id: str,
                                interview_id: str) -> None:
    require_teacher(actor)
    async with write_transaction(factory) as db:
        interview = await db.scalar(select(Interview).where(Interview.id == interview_id,
                                    Interview.course_id == course_id).with_for_update())
        course = await db.scalar(select(Course).where(Course.id == course_id,
                                   Course.owner_user_id == actor.id).with_for_update())
        if course is None or interview is None:
            raise PermissionError("Интервью недоступно.")
        key = f"interview:{interview.id}:{interview.interrupt_id}"
        existing = await db.get(WorkflowCommand, key)
        if existing and existing.payload.get("action") == "finish_without_answer":
            return
        if existing or interview.status != "invited" or course.status != "waiting_for_expert":
            raise ValueError("Курс уже продолжил обработку. Обновите страницу.")
        await db.execute(update(ExpertInvitation).where(ExpertInvitation.interview_id == interview_id,
                         ExpertInvitation.status == "active").values(status="revoked"))
        interview.status = "closed"
        course.status = "queued"
        db.add(WorkflowCommand(id=key, course_id=course.id, kind="resume",
                               payload={"interrupt_id": interview.interrupt_id, "actor_user_id": actor.id,
                                        "action": "finish_without_answer"}))


async def acknowledge(factory: async_sessionmaker[AsyncSession], actor: CurrentUser, course_id: str) -> None:
    require_teacher(actor)
    async with write_transaction(factory) as db:
        await db.execute(update(Notification).where(Notification.user_id == actor.id,
                         Notification.course_id == course_id, Notification.read_at.is_(None)).values(read_at=datetime.now(timezone.utc)))


async def schedule_due(factory: async_sessionmaker[AsyncSession], at: datetime | None = None) -> int:
    at = at or datetime.now(timezone.utc)
    async with write_transaction(factory) as db:
        due = (await db.scalars(select(Course).where(Course.status == "approved",
               Course.active_version_id.is_not(None), Course.next_audit_at <= at).with_for_update(skip_locked=True))).all()
        for course in due:
            db.add(WorkflowCommand(id=f"audit:{uuid4()}", course_id=course.id, kind="relevance",
                                   payload={"version_id": course.active_version_id}))
            course.status = "queued_audit"
        return len(due)


async def retry_command(factory: async_sessionmaker[AsyncSession], actor: CurrentUser, course_id: str) -> None:
    require_teacher(actor)
    async with write_transaction(factory) as db:
        course = await db.scalar(select(Course).where(Course.id == course_id, Course.owner_user_id == actor.id).with_for_update())
        if course is None or course.status != "failed":
            raise ValueError("Заявка недоступна для повторного запуска.")
        command = await db.scalar(select(WorkflowCommand).where(WorkflowCommand.course_id == course.id,
                                  WorkflowCommand.status == "failed").order_by(WorkflowCommand.created_at.desc()))
        if command is None:
            raise ValueError("Лимит уточнений исчерпан. Создайте заявку с уточнённой темой.")
        command.status = "pending"
        expert_reply = command.kind == "resume" and command.payload.get("action") != "finish_without_answer"
        course.status = "answer_queued" if expert_reply else "queued"


REBUILD_STATUSES = frozenset({"approved", "waiting_for_review", "pending_update_review", "failed"})


async def rebuild_course(factory: async_sessionmaker[AsyncSession], actor: CurrentUser, course_id: str) -> str:
    """Поставить в очередь полную пересборку курса моделью. Утверждённая версия остаётся действующей,
    новая придёт черновиком на проверку."""
    require_teacher(actor)
    async with write_transaction(factory) as db:
        course = await db.scalar(select(Course).where(Course.id == course_id, Course.owner_user_id == actor.id).with_for_update())
        if course is None:
            raise PermissionError("Курс недоступен.")
        if course.status not in REBUILD_STATUSES:
            raise ValueError("Курс сейчас обрабатывается. Дождитесь окончания текущего шага.")
        command_id = f"create:{course_id}:{uuid4()}"
        db.add(WorkflowCommand(id=command_id, course_id=course_id, kind="create", payload={"rebuild": True}))
        course.status = "queued"
    return command_id


def assigned_interviews(actor: CurrentUser, *, token_hash: str | None = None):
    if actor.role != "expert":
        raise PermissionError("Требуется роль эксперта.")
    stmt = select(ExpertInvitation, Interview, Course).join(Interview, ExpertInvitation.interview_id == Interview.id).join(
        Course, Interview.course_id == Course.id).where(ExpertInvitation.expert_user_id == actor.id,
        ExpertInvitation.status.in_(["used", "active"]))
    if token_hash is not None:
        stmt = stmt.where(ExpertInvitation.token_hash == token_hash)
    return stmt.order_by(Interview.created_at.desc())
