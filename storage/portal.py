from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from auth.user_repository import CurrentUser
from domain.export import workload
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


async def decide(factory: async_sessionmaker[AsyncSession], actor: CurrentUser, course_id: str, version_id: str,
                 action: str, content: str = "", attachments: tuple | list = (), workloads: dict | None = None) -> None:
    require_teacher(actor)
    if action not in {"approve", "reject", "revise"} or (action == "revise" and not (content.strip() or attachments)):
        raise ValueError("Укажите решение или правку.")
    async with write_transaction(factory) as db:
        course = await db.scalar(select(Course).where(Course.id == course_id, Course.owner_user_id == actor.id).with_for_update())
        if course is None:
            raise PermissionError("Курс недоступен.")
        if workloads:
            version = await db.get(CourseVersion, version_id)
            if not version or version.course_id != course_id or set(workloads) - {s["id"] for s in version.content["program"]["subjects"]}:
                raise ValueError("Нагрузка не соответствует версии курса.")
            for hours in workloads.values():
                workload(hours)
        payload = {"action": action, "actor_user_id": actor.id, "version_id": version_id,
                   "content": content.strip() or ("См. приложенные материалы." if attachments else ""),
                   "attachments": list(attachments), "workloads": workloads or {}}
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
        command.status, course.status = "pending", "queued"


def assigned_interviews(actor: CurrentUser, *, token_hash: str | None = None):
    if actor.role != "expert":
        raise PermissionError("Требуется роль эксперта.")
    stmt = select(ExpertInvitation, Interview, Course).join(Interview, ExpertInvitation.interview_id == Interview.id).join(
        Course, Interview.course_id == Course.id).where(ExpertInvitation.expert_user_id == actor.id,
        ExpertInvitation.status.in_(["used", "active"]))
    if token_hash is not None:
        stmt = stmt.where(ExpertInvitation.token_hash == token_hash)
    return stmt.order_by(Interview.created_at.desc())
