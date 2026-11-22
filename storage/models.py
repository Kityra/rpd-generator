from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, false, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, synonym


def now() -> datetime:
    return datetime.now(timezone.utc)


def identifier() -> str:
    return str(uuid4())


class Base(DeclarativeBase):
    """Metadata содержит только новые таблицы приложения."""


JsonType = JSON().with_variant(JSONB(), "postgresql")


class Course(Base):
    __tablename__ = "courses"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=identifier)
    owner_user_id: Mapped[str] = mapped_column(String(256), index=True)
    thread_id: Mapped[str] = mapped_column(String(36), unique=True)
    course_meta: Mapped[dict] = mapped_column(JsonType)
    status: Mapped[str] = mapped_column(String(40), default="draft", index=True)
    active_version_id: Mapped[str | None] = mapped_column(String(36))
    draft_version_id: Mapped[str | None] = mapped_column(String(36))
    last_approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    next_audit_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, onupdate=now)


class CourseVersion(Base):
    __tablename__ = "course_versions"
    __table_args__ = (UniqueConstraint("course_id", "revision"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=identifier)
    course_id: Mapped[str] = mapped_column(ForeignKey("courses.id"), index=True)
    revision: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(30), default="draft")
    content: Mapped[dict] = mapped_column(JsonType)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ProfessionBoundary(Base):
    __tablename__ = "profession_boundaries"
    __table_args__ = (UniqueConstraint("course_id", "primary_profession", "adjacent_profession"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    course_id: Mapped[str] = mapped_column(ForeignKey("courses.id"), index=True)
    primary_profession: Mapped[str] = mapped_column(String(300), index=True)
    adjacent_profession: Mapped[str] = mapped_column(String(300), index=True)
    skill_jaccard_overlap: Mapped[float] = mapped_column(Float)
    overlap_tasks_json: Mapped[list] = mapped_column(JsonType, default=list, server_default="[]")
    exclusive_to_primary_json: Mapped[list] = mapped_column(JsonType, default=list, server_default="[]")
    exclusive_to_adjacent_json: Mapped[list] = mapped_column(JsonType, default=list, server_default="[]")
    shared_core_json: Mapped[list] = mapped_column(JsonType, default=list, server_default="[]")
    handoff_artifacts_json: Mapped[list] = mapped_column(JsonType, default=list, server_default="[]")
    boundary_conflicts_json: Mapped[list] = mapped_column(JsonType, default=list, server_default="[]")
    rejected_noise_turns_json: Mapped[list] = mapped_column(JsonType, default=list, server_default="[]")
    primary_claims_json: Mapped[list] = mapped_column(JsonType, default=list, server_default="[]")
    invite_token_hash: Mapped[str | None] = mapped_column(String(64), unique=True)
    invite_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    source_interview_id: Mapped[str | None] = mapped_column(String(36))
    turn_count: Mapped[int] = mapped_column(Integer, default=0, server_default=text("0"))
    interview_completed: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())
    verified_by_methodologist: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, server_default=func.now())


class Notification(Base):
    __tablename__ = "notifications"
    __table_args__ = (UniqueConstraint("course_id", "version_id", "event"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=identifier)
    user_id: Mapped[str] = mapped_column(String(256), index=True)
    course_id: Mapped[str] = mapped_column(ForeignKey("courses.id"))
    version_id: Mapped[str] = mapped_column(ForeignKey("course_versions.id"))
    event: Mapped[str] = mapped_column(String(50))
    content: Mapped[str] = mapped_column(Text)
    read_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class SessionRow(Base):
    __tablename__ = "sessions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(256), index=True)
    session_token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    ip: Mapped[str] = mapped_column(String(64))
    user_agent: Mapped[str] = mapped_column(Text)


class LoginAttempt(Base):
    __tablename__ = "login_attempts"
    email: Mapped[str] = mapped_column(String(320), primary_key=True)
    ip: Mapped[str] = mapped_column(String(64), primary_key=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AuthEvent(Base):
    __tablename__ = "auth_events"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=identifier)
    user_id: Mapped[str | None] = mapped_column(String(256))
    event: Mapped[str] = mapped_column(String(40))
    ip: Mapped[str] = mapped_column(String(64))
    user_agent: Mapped[str] = mapped_column(Text)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Interview(Base):
    __tablename__ = "interviews"
    __table_args__ = (UniqueConstraint("course_id", "interrupt_id"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    course_id: Mapped[str] = mapped_column(ForeignKey("courses.id"), index=True)
    interrupt_id: Mapped[str] = mapped_column(String(36))
    status: Mapped[str] = mapped_column(String(30), default="invited")
    gap_report: Mapped[dict] = mapped_column(JsonType)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ExpertInvitation(Base):
    __tablename__ = "expert_invitations"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=identifier)
    interview_id: Mapped[str] = mapped_column(ForeignKey("interviews.id"), index=True)
    expert_user_id: Mapped[str] = mapped_column(String(256), index=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    status: Mapped[str] = mapped_column(String(20), default="active")
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_by: Mapped[str] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class InterviewMessage(Base):
    __tablename__ = "interview_messages"
    __table_args__ = (UniqueConstraint("interview_id", "submission_id"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=identifier)
    interview_id: Mapped[str] = mapped_column(ForeignKey("interviews.id"))
    submission_id: Mapped[str] = mapped_column(String(100))
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    attachments: Mapped[list] = mapped_column(JsonType, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class SourceRow(Base):
    __tablename__ = "sources"
    course_id: Mapped[str] = mapped_column(ForeignKey("courses.id"), primary_key=True)
    source_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    record: Mapped[dict] = mapped_column(JsonType)


class ModelCall(Base):
    """Журнал вызовов модели. Секреты и ключи в текст ошибки не попадают."""
    __tablename__ = "model_calls"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=identifier)
    course_id: Mapped[str | None] = mapped_column(String(36), index=True)
    run_id: Mapped[str | None] = mapped_column(String(100))
    stage: Mapped[str] = mapped_column(String(60))
    stage_name = synonym("stage")
    subject: Mapped[str | None] = mapped_column(String(300))
    subject_id: Mapped[str | None] = mapped_column(Text)
    output: Mapped[str] = mapped_column(String(80))
    model: Mapped[str] = mapped_column(String(300))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    duration_ms: Mapped[int] = mapped_column(Integer)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    attempts: Mapped[int] = mapped_column(Integer, default=1)
    outcome: Mapped[str] = mapped_column(String(30), index=True)
    error: Mapped[str | None] = mapped_column(Text)
    parameters: Mapped[dict | None] = mapped_column(JsonType)


class WorkflowCommand(Base):
    __tablename__ = "workflow_commands"
    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    course_id: Mapped[str] = mapped_column(ForeignKey("courses.id"), index=True)
    kind: Mapped[str] = mapped_column(String(30))
    payload: Mapped[dict] = mapped_column(JsonType)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
