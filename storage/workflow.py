import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from uuid import NAMESPACE_URL, uuid5

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from auth.session import aware
from auth.user_repository import CurrentUser, UserRepository
from domain.schemas import CourseProgramModel, SourceRecord
from storage.database import write_transaction
from storage.models import Course, CourseVersion, ExpertInvitation, Interview, InterviewMessage, Notification, SourceRow, WorkflowCommand


class SqlWorkflowStore:
    def __init__(self, factory: async_sessionmaker[AsyncSession]) -> None:
        self.factory = factory

    async def set_status(self, course_id: str, status: str, owner: str) -> None:
        async with write_transaction(self.factory) as db:
            row = await db.scalar(select(Course).where(Course.id == course_id).with_for_update())
            if row is None or row.owner_user_id != owner or row.thread_id != course_id:
                raise PermissionError("Заявка недоступна или thread_id не совпадает.")
            if row.status == "cancelled":
                raise ValueError("Заявка отменена.")
            row.status = status

    async def save_sources(self, course_id: str, sources: list[SourceRecord]) -> None:
        async with write_transaction(self.factory) as db:
            for source in sources:
                key = (course_id, source.source_id)
                if await db.get(SourceRow, key) is None:
                    db.add(SourceRow(course_id=course_id, source_id=source.source_id,
                                     record=source.model_dump(mode="json")))

    async def prepare_interview(self, course_id: str, interrupt_id: str, report: dict) -> str:
        interview_id = str(uuid5(NAMESPACE_URL, f"{course_id}/{interrupt_id}"))
        async with write_transaction(self.factory) as db:
            if await db.get(Interview, interview_id) is None:
                db.add(Interview(id=interview_id, course_id=course_id, interrupt_id=interrupt_id, gap_report=report))
                previous_id = report.get("previous_interview_id")
                if previous_id:
                    previous = await db.scalar(select(ExpertInvitation).join(Interview).where(
                        Interview.course_id == course_id, Interview.id == previous_id, ExpertInvitation.status == "used"))
                    if previous:
                        await db.flush()
                        db.add(ExpertInvitation(interview_id=interview_id, expert_user_id=previous.expert_user_id,
                            token_hash=hashlib.sha256(secrets.token_bytes(32)).hexdigest(), created_by=previous.created_by,
                            expires_at=datetime.now(timezone.utc) + timedelta(days=7)))
        return interview_id

    async def record_verification(self, interview_id: str, report: dict) -> None:
        async with write_transaction(self.factory) as db:
            row = await db.get(Interview, interview_id)
            row.gap_report = {**row.gap_report, "verification": report}

    async def invite(self, interview_id: str, expert_id: str, teacher: CurrentUser,
                     users: UserRepository) -> str:
        expert = await users.get_by_id(expert_id)
        if teacher.role not in {"teacher", "admin"} or expert is None or expert.role != "expert":
            raise PermissionError("Требуются преподаватель и существующий эксперт.")
        token = secrets.token_urlsafe(32)
        async with write_transaction(self.factory) as db:
            interview = await db.scalar(select(Interview).where(Interview.id == interview_id).with_for_update())
            course = await db.get(Course, interview.course_id) if interview else None
            if course is None or course.owner_user_id != teacher.id or interview.status != "invited":
                raise PermissionError("Интервью недоступно.")
            active = (await db.scalars(select(ExpertInvitation).where(
                ExpertInvitation.interview_id == interview_id, ExpertInvitation.status == "active"))).all()
            for previous in active:
                previous.status = "revoked"
            db.add(ExpertInvitation(interview_id=interview_id, expert_user_id=expert.id,
                                    token_hash=hashlib.sha256(token.encode()).hexdigest(),
                                    expires_at=datetime.now(timezone.utc) + timedelta(days=7), created_by=teacher.id))
        return token

    async def submit_interview(self, token: str, expert: CurrentUser, interrupt_id: str,
                               submission_id: str, content: str, attachments: list[dict], *, invitation_id: str | None = None) -> None:
        if expert.role != "expert" or not content.strip() or len(content) > 16000:
            raise PermissionError("Требуется ответ авторизованного эксперта.")
        at = datetime.now(timezone.utc)
        async with write_transaction(self.factory) as db:
            locator = (ExpertInvitation.id == invitation_id if invitation_id else
                       ExpertInvitation.token_hash == hashlib.sha256(token.encode()).hexdigest())
            invitation = await db.scalar(select(ExpertInvitation).where(locator))
            if invitation is None or invitation.expert_user_id != expert.id:
                raise PermissionError("Приглашение недоступно.")
            interview = await db.scalar(select(Interview).where(Interview.id == invitation.interview_id).with_for_update())
            invitation = await db.scalar(select(ExpertInvitation).where(locator).with_for_update()
                                         .execution_options(populate_existing=True))
            command_id = f"interview:{interview.id}:{interrupt_id}"
            existing = await db.get(WorkflowCommand, command_id)
            if existing:
                expected = existing.payload
                if expected.get("action") == "finish_without_answer":
                    raise ValueError("Преподаватель завершил интервью и запросил черновик.")
                if expected["submission_id"] != submission_id or expected["content"] != content or expected["attachments"] != attachments:
                    raise ValueError("Повторная отправка отличается от сохранённого ответа.")
                return
            if invitation.status != "active" or invitation.used_at or aware(invitation.expires_at) <= at:
                raise PermissionError("Приглашение истекло или использовано.")
            if interview.interrupt_id != interrupt_id or interview.status != "invited":
                raise ValueError("Прерывание устарело.")
            payload = {"submission_id": submission_id, "interrupt_id": interrupt_id, "content": content,
                       "attachments": attachments, "expert_user_id": expert.id}
            db.add(InterviewMessage(interview_id=interview.id, submission_id=submission_id,
                                    role="expert", content=content, attachments=attachments))
            db.add(WorkflowCommand(id=command_id, course_id=interview.course_id, kind="resume", payload=payload))
            interview.status, interview.submitted_at = "submitted", at
            invitation.status, invitation.used_at = "used", at
            course = await db.get(Course, interview.course_id)
            course.status = "answer_queued"

    async def save_draft(self, course_id: str, version_id: str, program: CourseProgramModel,
                         scheduled: bool, activity: dict, evidence_gaps: dict) -> None:
        async with write_transaction(self.factory) as db:
            course = await db.scalar(select(Course).where(Course.id == course_id).with_for_update())
            if course.draft_version_id and course.draft_version_id != version_id:
                previous = await db.get(CourseVersion, course.draft_version_id)
                previous.status = "archived"
            if await db.get(CourseVersion, version_id) is None:
                revision = (await db.scalar(select(func.max(CourseVersion.revision)).where(
                    CourseVersion.course_id == course_id)) or 0) + 1
                db.add(CourseVersion(id=version_id, course_id=course_id, revision=revision,
                                     content={"program": program.model_dump(mode="json"), "activity": activity,
                                              "evidence_gaps": evidence_gaps}))
                await db.flush()
            course.draft_version_id = version_id
            course.status = "pending_update_review" if scheduled else "waiting_for_review"
            if scheduled:
                existing = await db.scalar(select(Notification).where(
                    Notification.course_id == course_id, Notification.version_id == version_id,
                    Notification.event == "relevance_update"))
                if existing is None:
                    db.add(Notification(user_id=course.owner_user_id, course_id=course_id, version_id=version_id,
                                        event="relevance_update", content="Курс обновлён по результатам 9-месячной проверки актуальности"))

    async def finish_review(self, course_id: str, version_id: str, action: str, actor: str, workloads=None) -> None:
        async with write_transaction(self.factory) as db:
            course = await db.scalar(select(Course).where(Course.id == course_id).with_for_update())
            if course is None or course.owner_user_id != actor:
                raise PermissionError("Курс недоступен.")
            version = await db.get(CourseVersion, version_id)
            if version is None or version.course_id != course_id:
                raise ValueError("Версия недоступна.")
            if action == "approve" and course.active_version_id == version_id:
                return
            if action == "reject" and version.status == "archived" and course.draft_version_id is None:
                return
            if course.draft_version_id != version_id:
                raise ValueError("Черновик изменился. Обновите страницу.")
            if action == "approve":
                version.content = {**version.content, "workloads": workloads or {}}
                if course.active_version_id:
                    previous = await db.get(CourseVersion, course.active_version_id)
                    previous.status = "archived"
                version.status = "approved"
                course.active_version_id = version_id
                course.last_approved_at = datetime.now(timezone.utc)
            elif action == "reject" and course.active_version_id:
                version.status = "archived"
            else:
                raise ValueError("Недопустимое решение ревью.")
            course.draft_version_id, course.status = None, "approved"
            course.next_audit_at = datetime.now(timezone.utc) + timedelta(days=270)

    async def defer_audit(self, course_id: str) -> None:
        async with write_transaction(self.factory) as db:
            course = await db.get(Course, course_id)
            course.status = "approved"
            course.next_audit_at = datetime.now(timezone.utc) + timedelta(days=270)
