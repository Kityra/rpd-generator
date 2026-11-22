import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from uuid import NAMESPACE_URL, uuid5

from sqlalchemy import func, select
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from auth.session import aware
from auth.user_repository import CurrentUser, UserRepository
from domain.schemas import CourseProgramModel, SourceRecord
from orchestration.profession_profile import ExpertDecision
from orchestration.profession_boundaries import role_key, with_boundary_warnings
from storage.database import write_transaction
from storage.models import (Course, CourseVersion, ExpertInvitation, Interview, InterviewMessage, ModelCall, Notification,
                            SourceRow, WorkflowCommand)
from storage.profession_boundaries import course_boundaries, public_boundary, sync_boundary_candidates


async def model_call_failures(factory: async_sessionmaker[AsyncSession], course_id: str,
                              limit: int = 5) -> list[tuple[str, str, int]]:
    """Главные причины сбоев вызовов модели по курсу: (этап, итог, число)."""
    try:
        async with factory() as db:
            rows = (await db.execute(
                select(ModelCall.stage, ModelCall.outcome, func.count())
                .where(ModelCall.course_id == course_id, ModelCall.outcome != "ok")
                .group_by(ModelCall.stage, ModelCall.outcome)
                .order_by(func.count().desc()).limit(limit))).all()
    except DBAPIError:  # журнал ещё не создан миграцией
        return []
    return [(stage, outcome, count) for stage, outcome, count in rows]


class SqlWorkflowStore:
    def __init__(self, factory: async_sessionmaker[AsyncSession]) -> None:
        self.factory = factory

    async def sync_boundaries(self, course_id: str, profile: dict) -> list[dict]:
        return await sync_boundary_candidates(self.factory, course_id, profile)

    async def get_boundaries(self, course_id: str, primary: str | None) -> list[dict]:
        async with self.factory() as db:
            return [public_boundary(row) for row in await course_boundaries(db, course_id)
                    if not primary or role_key(row.primary_profession) == role_key(primary)]

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

    async def lifespan_market_history(self, course_id: str) -> list[dict]:
        sample = CourseVersion.content["lifespan_market_sample"]
        async with self.factory() as db:
            return [item for item in await db.scalars(select(sample).where(CourseVersion.course_id == course_id)
                    .order_by(CourseVersion.revision.desc()).limit(100)) if isinstance(item, dict)]

    async def lifespan_sources(self, course_id: str, source_ids: list[str]) -> dict[str, SourceRecord]:
        async with self.factory() as db:
            rows = await db.scalars(select(SourceRow).where(SourceRow.course_id == course_id,
                                                           SourceRow.source_id.in_(source_ids)))
            return {row.source_id: SourceRecord.model_validate(row.record) for row in rows}

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
            raise PermissionError("Требуются методист и существующий эксперт.")
        token = secrets.token_urlsafe(32)
        async with write_transaction(self.factory) as db:
            interview = await db.scalar(select(Interview).where(Interview.id == interview_id).with_for_update())
            course = await db.get(Course, interview.course_id) if interview else None
            if (course is None or course.owner_user_id != teacher.id or course.status == "cancelled" or
                    interview.status != "invited"):
                raise PermissionError("Интервью недоступно.")
            if interview.gap_report.get("methodologist_handoff"):
                raise ValueError("Оставшиеся поля переданы методисту.")
            active = (await db.scalars(select(ExpertInvitation).where(
                ExpertInvitation.interview_id == interview_id, ExpertInvitation.status == "active"))).all()
            for previous in active:
                previous.status = "revoked"
            db.add(ExpertInvitation(interview_id=interview_id, expert_user_id=expert.id,
                                    token_hash=hashlib.sha256(token.encode()).hexdigest(),
                                    expires_at=datetime.now(timezone.utc) + timedelta(days=7), created_by=teacher.id))
        return token

    async def submit_interview(self, token: str, expert: CurrentUser, interrupt_id: str,
                               submission_id: str, content: str, attachments: list[dict], *, invitation_id: str | None = None,
                               profession_decisions: list[dict] | None = None, continue_interview: bool = False) -> None:
        if expert.role != "expert" or len(content) > 16000:
            raise PermissionError("Требуется ответ авторизованного эксперта.")
        at = datetime.now(timezone.utc)
        decisions = [ExpertDecision.model_validate(item).model_dump(mode="json") for item in profession_decisions or []]
        if len(decisions) > 40 or len({entry["claim_id"] for entry in decisions}) != len(decisions):
            raise ValueError("Для каждого утверждения допустимо одно решение.")
        async with write_transaction(self.factory) as db:
            locator = (ExpertInvitation.id == invitation_id if invitation_id else
                       ExpertInvitation.token_hash == hashlib.sha256(token.encode()).hexdigest())
            invitation = await db.scalar(select(ExpertInvitation).where(locator))
            if invitation is None or invitation.expert_user_id != expert.id:
                raise PermissionError("Приглашение недоступно.")
            interview = await db.scalar(select(Interview).where(Interview.id == invitation.interview_id).with_for_update())
            course = await db.get(Course, interview.course_id)
            if course is None or course.status == "cancelled":
                raise PermissionError("Интервью недоступно.")
            invitation = await db.scalar(select(ExpertInvitation).where(locator).with_for_update()
                                         .execution_options(populate_existing=True))
            command_id = f"interview:{interview.id}:{interrupt_id}"
            existing = await db.get(WorkflowCommand, command_id)
            if existing:
                expected = existing.payload
                if expected.get("action") == "finish_without_answer":
                    raise ValueError("Методист завершил интервью и запросил черновик.")
                if (expected["submission_id"] != submission_id or expected["content"] != content
                        or expected["attachments"] != attachments or expected.get("profession_decisions", []) != decisions
                        or expected.get("continue_interview", False) != continue_interview):
                    raise ValueError("Повторная отправка отличается от сохранённого ответа.")
                return
            if invitation.status != "active" or invitation.used_at or aware(invitation.expires_at) <= at:
                raise PermissionError("Приглашение истекло или использовано.")
            if interview.interrupt_id != interrupt_id or interview.status != "invited":
                raise ValueError("Прерывание устарело.")
            if not content.strip() and not interview.gap_report.get("profession_profile"):
                raise PermissionError("Требуется ответ авторизованного эксперта.")
            allowed = set(interview.gap_report.get("review_candidate_ids", []))
            if any(entry["action"] != "add" and entry["claim_id"] not in allowed for entry in decisions):
                raise ValueError("Утверждение отсутствует в текущем интервью.")
            report = interview.gap_report
            can_continue = report.get("interview_can_continue",
                                      report.get("interview_turn", 1) < report.get("interview_max_turns", 1))
            if continue_interview and not can_continue:
                raise ValueError("Достигнут лимит реплик по методике интервью.")
            payload = {"submission_id": submission_id, "interrupt_id": interrupt_id, "content": content,
                       "attachments": attachments, "expert_user_id": expert.id,
                       "profession_decisions": decisions, "continue_interview": continue_interview}
            candidates = {item["claim_id"]: item["statement"] for item in
                          interview.gap_report.get("profession_profile", {}).get("claims", [])}
            transcript = [content]
            labels = {"confirm": "Подтверждаю", "reject": "Отклоняю"}
            for entry in decisions:
                if entry["action"] in labels:
                    transcript.append(f"{labels[entry['action']]}: {candidates[entry['claim_id']]}")
                else:
                    transcript.append(entry["statement"])
            db.add(InterviewMessage(interview_id=interview.id, submission_id=submission_id,
                                    role="expert", content="\n".join(transcript), attachments=attachments))
            db.add(WorkflowCommand(id=command_id, course_id=interview.course_id, kind="resume", payload=payload))
            interview.status, interview.submitted_at = "submitted", at
            invitation.status, invitation.used_at = "used", at
            course.status = "answer_queued"

    async def save_draft(self, course_id: str, version_id: str, program: CourseProgramModel,
                         scheduled: bool, activity: dict, *, evidence_gaps: dict, source_notices: list[str],
                         vacancy_source_ids: list[str] | None = None,
                         generation: dict | None = None, composition: dict | None = None,
                         profession_profile: dict | None = None, profession_market: dict | None = None,
                         skill_mappings: dict | None = None,
                         lifespan_assessment: dict | None = None,
                         lifespan_market_sample: dict | None = None,
                         lifespan_note: str | None = None,
                         boundary_violation_threshold: float = 0.5) -> None:
        async with write_transaction(self.factory) as db:
            course = await db.scalar(select(Course).where(Course.id == course_id).with_for_update())
            if course.draft_version_id and course.draft_version_id != version_id:
                previous = await db.get(CourseVersion, course.draft_version_id)
                previous.status = "archived"
            if await db.get(CourseVersion, version_id) is None:
                revision = (await db.scalar(select(func.max(CourseVersion.revision)).where(
                    CourseVersion.course_id == course_id)) or 0) + 1
                content = {"program": program.model_dump(mode="json"), "activity": activity, "evidence_gaps": evidence_gaps,
                           "lifespan_assessment": lifespan_assessment}
                if lifespan_market_sample:
                    content["lifespan_market_sample"] = lifespan_market_sample
                if lifespan_note:
                    content["lifespan_note"] = lifespan_note
                primary = (profession_profile or {}).get("profession_name")
                boundaries = [public_boundary(row) for row in await course_boundaries(db, course_id)
                              if not primary or role_key(row.primary_profession) == role_key(primary)]
                if boundaries:
                    content["profession_boundaries"] = boundaries
                    content["program"] = with_boundary_warnings(content["program"], boundaries,
                                                               threshold=boundary_violation_threshold)
                if source_notices:
                    content["source_notices"] = source_notices
                if vacancy_source_ids:
                    content["vacancy_source_ids"] = vacancy_source_ids
                if skill_mappings:
                    content["skill_mappings"] = skill_mappings
                if generation:
                    content["generation"] = generation
                if composition:
                    content["composition_check"] = composition
                if profession_profile:
                    content["profession_profile"] = profession_profile
                    if profession_market:
                        content["profession_market"] = profession_market
                db.add(CourseVersion(id=version_id, course_id=course_id, revision=revision, content=content))
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

    async def finish_review(self, course_id: str, version_id: str, action: str, actor: str) -> None:
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

    async def record_model_call(self, entry: dict) -> None:
        async with write_transaction(self.factory) as db:
            db.add(ModelCall(**entry))

    async def defer_audit(self, course_id: str) -> None:
        async with write_transaction(self.factory) as db:
            course = await db.get(Course, course_id)
            course.status = "approved"
            course.next_audit_at = datetime.now(timezone.utc) + timedelta(days=270)
