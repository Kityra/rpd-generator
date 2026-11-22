"""Tables of the profession activity model and of the educational programme (migration 0004).

IDs are UUID strings, as elsewhere in the project. The module is imported at the end of storage/models.py,
so Base.metadata contains these tables.
"""

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from storage.models import Base, JsonType, identifier, now


class Profession(Base):
    __tablename__ = "professions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=identifier)
    name: Mapped[str] = mapped_column(String(300))
    region: Mapped[str] = mapped_column(String(200), default="")
    horizon_years: Mapped[int] = mapped_column(Integer, default=3)
    standard_codes: Mapped[list] = mapped_column(JsonType, default=list)
    status: Mapped[str] = mapped_column(String(30), default="draft")
    # Demo data is labelled on every screen and export.
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)
    # {"delphi_round": 1|2, "market_totals": {"now": int, "prev": int, "source": str}}
    meta: Mapped[dict] = mapped_column(JsonType, default=dict)
    created_by: Mapped[str] = mapped_column(String(256), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ActivitySource(Base):
    __tablename__ = "activity_sources"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=identifier)
    profession_id: Mapped[str] = mapped_column(ForeignKey("professions.id"), index=True)
    # standard | vacancy | rpd | science | employer_doc | forecast | expert
    type: Mapped[str] = mapped_column(String(20), index=True)
    title: Mapped[str] = mapped_column(String(500))
    url: Mapped[str] = mapped_column(Text, default="")
    # Employer or publisher: 30 vacancies of one employer are one market vote.
    origin_key: Mapped[str] = mapped_column(String(300), index=True)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    collected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    snapshot_id: Mapped[str | None] = mapped_column(String(64))
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    content: Mapped[str] = mapped_column(Text)
    # Professional standard code and approval year (type=standard).
    standard_code: Mapped[str | None] = mapped_column(String(40))
    standard_year: Mapped[int | None] = mapped_column(Integer)
    # {"extraction": "done" | "failed", "extraction_error": str, "region": str, "added_by": str}
    meta: Mapped[dict] = mapped_column(JsonType, default=dict)


class ActivityClaim(Base):
    __tablename__ = "activity_claims"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=identifier)
    source_id: Mapped[str] = mapped_column(ForeignKey("activity_sources.id"), index=True)
    kind: Mapped[str] = mapped_column(String(20))  # task | knowledge | skill | tool
    text: Mapped[str] = mapped_column(Text)
    verb: Mapped[str | None] = mapped_column(String(200))
    object: Mapped[str | None] = mapped_column(Text)
    tool: Mapped[str | None] = mapped_column(String(300))
    result: Mapped[str | None] = mapped_column(Text)
    quote: Mapped[str] = mapped_column(Text, default="")
    quote_ok: Mapped[bool] = mapped_column(Boolean, default=False)
    foreign_role: Mapped[bool] = mapped_column(Boolean, default=False)
    # Why the claim does not take part in calculations ("Discarded at extraction" tab).
    rejection_reason: Mapped[str | None] = mapped_column(Text)
    # Labour function of the standard the claim comes from: {"code": "D/03.6", "name": "..."}.
    function_hint: Mapped[dict | None] = mapped_column(JsonType)
    extractor: Mapped[str] = mapped_column(String(200))  # rule:<kind> | llm:<model>:<prompt_version> | manual
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ActivityItem(Base):
    __tablename__ = "activity_items"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=identifier)
    profession_id: Mapped[str] = mapped_column(ForeignKey("professions.id"), index=True)
    level: Mapped[str] = mapped_column(String(20))  # function | task | knowledge | skill | tool
    # Task → its function; knowledge, skill or tool from a standard → the labour function it belongs to.
    parent_id: Mapped[str | None] = mapped_column(String(36), index=True)
    name: Mapped[str] = mapped_column(Text)
    canonical_key: Mapped[str] = mapped_column(String(500), index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    # Control card: foreign — a task of another profession, duplicate — a paraphrase of duplicate_of.
    is_control: Mapped[bool] = mapped_column(Boolean, default=False)
    control_kind: Mapped[str | None] = mapped_column(String(20))
    duplicate_of: Mapped[str | None] = mapped_column(String(36))
    knowledge_area_id: Mapped[str | None] = mapped_column(String(36))
    standard_code: Mapped[str | None] = mapped_column(String(40))
    # sources | expert_proposal | control | manual
    origin: Mapped[str] = mapped_column(String(30), default="sources")
    # Delphi round in which the item first reached experts (expert proposals go to round 2).
    first_round: Mapped[int] = mapped_column(Integer, default=1)
    # A merged item is switched off, never deleted: merges are reversible.
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    merged_into: Mapped[str | None] = mapped_column(String(36))
    # Earlier wordings: [{"version": 1, "name": "...", "changed_by": "...", "at": "..."}]
    history: Mapped[list] = mapped_column(JsonType, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ClaimLink(Base):
    __tablename__ = "claim_links"
    claim_id: Mapped[str] = mapped_column(ForeignKey("activity_claims.id"), primary_key=True)
    item_id: Mapped[str] = mapped_column(ForeignKey("activity_items.id"), primary_key=True, index=True)
    method: Mapped[str] = mapped_column(String(20))  # exact | synonym | embedding | manual
    similarity: Mapped[float | None] = mapped_column(Float)
    confirmed_by: Mapped[str | None] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ItemLink(Base):
    """Manual link "task → knowledge, skill or tool" set by a methodist (automatic links are computed)."""
    __tablename__ = "item_links"
    task_id: Mapped[str] = mapped_column(ForeignKey("activity_items.id"), primary_key=True)
    item_id: Mapped[str] = mapped_column(ForeignKey("activity_items.id"), primary_key=True)
    reason: Mapped[str] = mapped_column(Text, default="")
    created_by: Mapped[str] = mapped_column(String(256), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class KnowledgeArea(Base):
    __tablename__ = "knowledge_areas"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=identifier)
    name: Mapped[str] = mapped_column(String(300), unique=True)
    kind: Mapped[str] = mapped_column(String(20))  # fundamental | applied
    synonyms: Mapped[list] = mapped_column(JsonType, default=list)


class ActivityExpert(Base):
    __tablename__ = "experts"
    __table_args__ = (UniqueConstraint("user_id", "profession_id"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=identifier)
    user_id: Mapped[str] = mapped_column(String(256), index=True)
    profession_id: Mapped[str] = mapped_column(ForeignKey("professions.id"), index=True)
    years: Mapped[int | None] = mapped_column(Integer)
    role_kind: Mapped[str] = mapped_column(String(40), default="")  # practitioner | lead | hr | teacher
    company_size: Mapped[str] = mapped_column(String(40), default="")  # small | medium | large
    company_kind: Mapped[str] = mapped_column(String(60), default="")
    region: Mapped[str] = mapped_column(String(200), default="")
    excluded: Mapped[bool] = mapped_column(Boolean, default=False)
    exclusion_reason: Mapped[str | None] = mapped_column(Text)
    # "What will this role become in 3–5 years?" and the demand forecast (more | same | less).
    role_forecast_text: Mapped[str | None] = mapped_column(Text)
    role_demand: Mapped[str | None] = mapped_column(String(10))
    # "What is missing from the list?" — goes through the extractor as an expert source.
    missing_text: Mapped[str | None] = mapped_column(Text)
    # Cards per Delphi round: {"1": [item ids], "2": [item ids]}; rounds the expert has finished.
    cards: Mapped[dict] = mapped_column(JsonType, default=dict)
    submitted_rounds: Mapped[list] = mapped_column(JsonType, default=list)
    invited_by: Mapped[str] = mapped_column(String(256), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ExpertVote(Base):
    __tablename__ = "expert_votes"
    __table_args__ = (UniqueConstraint("expert_id", "item_id", "item_version", "round"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=identifier)
    expert_id: Mapped[str] = mapped_column(ForeignKey("experts.id"), index=True)
    item_id: Mapped[str] = mapped_column(ForeignKey("activity_items.id"), index=True)
    item_version: Mapped[int] = mapped_column(Integer, default=1)
    round: Mapped[int] = mapped_column(Integer, default=1)
    verdict: Mapped[str] = mapped_column(String(20))  # regular | sometimes | other_role | outdated | dont_know
    importance: Mapped[int | None] = mapped_column(Integer)
    forecast: Mapped[str | None] = mapped_column(String(20))  # more | same | less | automated
    wording_ok: Mapped[bool | None] = mapped_column(Boolean)
    wording_suggestion: Mapped[str | None] = mapped_column(Text)
    comment: Mapped[str | None] = mapped_column(Text)
    elapsed_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ModelVersionRow(Base):
    __tablename__ = "model_versions"
    __table_args__ = (UniqueConstraint("profession_id", "number"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=identifier)
    profession_id: Mapped[str] = mapped_column(ForeignKey("professions.id"), index=True)
    number: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    config_hash: Mapped[str] = mapped_column(String(64))
    # Hash of the inputs: same inputs and config give the same content.
    input_hash: Mapped[str] = mapped_column(String(64))
    counts: Mapped[dict] = mapped_column(JsonType, default=dict)
    diff_from_previous: Mapped[dict] = mapped_column(JsonType, default=dict)
    content: Mapped[dict] = mapped_column(JsonType)


class ItemMetrics(Base):
    __tablename__ = "item_metrics"
    item_id: Mapped[str] = mapped_column(ForeignKey("activity_items.id"), primary_key=True)
    model_version: Mapped[str] = mapped_column(ForeignKey("model_versions.id"), primary_key=True, index=True)
    metrics: Mapped[dict] = mapped_column(JsonType)
    status: Mapped[str] = mapped_column(String(30))
    trend: Mapped[str] = mapped_column(String(20))
    lag_flag: Mapped[str | None] = mapped_column(String(300))
    reason_text: Mapped[str] = mapped_column(Text)


class ActivityMergeLog(Base):
    """Log of merges, moves, rewordings and queue decisions: every action is recorded and can be undone."""
    __tablename__ = "activity_merge_log"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=identifier)
    profession_id: Mapped[str] = mapped_column(ForeignKey("professions.id"), index=True)
    action: Mapped[str] = mapped_column(String(30))  # merge | unmerge | reparent | reword | dismiss | function
    payload: Mapped[dict] = mapped_column(JsonType)
    actor: Mapped[str] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ExtractionCache(Base):
    """LLM extractor responses keyed by hash(source text + prompt version + model)."""
    __tablename__ = "activity_extraction_cache"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    prompt_version: Mapped[str] = mapped_column(String(40))
    model: Mapped[str] = mapped_column(String(300))
    response: Mapped[dict] = mapped_column(JsonType)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Program(Base):
    __tablename__ = "programs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=identifier)
    head_user_id: Mapped[str] = mapped_column(String(256), index=True)
    name: Mapped[str] = mapped_column(String(300))
    level: Mapped[str] = mapped_column(String(40), default="bachelor")
    institution: Mapped[str] = mapped_column(String(300), default="")
    academic_year: Mapped[str] = mapped_column(String(20), default="")
    total_zet: Mapped[int] = mapped_column(Integer, default=240)
    hours_per_zet: Mapped[int] = mapped_column(Integer, default=36)
    target_professions: Mapped[list] = mapped_column(JsonType, default=list)  # [{profession_id, weight}]
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ProgramDiscipline(Base):
    __tablename__ = "program_disciplines"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=identifier)
    program_id: Mapped[str] = mapped_column(ForeignKey("programs.id"), index=True)
    name: Mapped[str] = mapped_column(String(300))
    zet: Mapped[float] = mapped_column(Float, default=0)
    hours: Mapped[int] = mapped_column(Integer, default=0)
    semester: Mapped[int | None] = mapped_column(Integer)
    part: Mapped[str] = mapped_column(String(20), default="mandatory")  # mandatory_fgos | mandatory | elective
    # Methodologist assigned by the programme head to prepare the discipline's RPD.
    teacher_user_id: Mapped[str | None] = mapped_column(String(256), index=True)
    teacher_name: Mapped[str] = mapped_column(String(300), default="")
    course_id: Mapped[str | None] = mapped_column(String(36))
    content: Mapped[dict] = mapped_column(JsonType, default=dict)  # {"topics": [], "outcomes": [], "assignments": []}
    position: Mapped[int] = mapped_column(Integer, default=0)


class DisciplineLink(Base):
    """Manual coverage correction; automatic coverage is recomputed from the discipline content."""
    __tablename__ = "discipline_links"
    discipline_id: Mapped[str] = mapped_column(ForeignKey("program_disciplines.id"), primary_key=True)
    item_id: Mapped[str] = mapped_column(ForeignKey("activity_items.id"), primary_key=True, index=True)
    coverage: Mapped[str] = mapped_column(String(10))  # full | partial | none (none removes an automatic link)
    method: Mapped[str] = mapped_column(String(10), default="manual")  # auto | manual
    reason: Mapped[str] = mapped_column(Text, default="")
    created_by: Mapped[str] = mapped_column(String(256), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Recommendation(Base):
    __tablename__ = "recommendations"
    __table_args__ = (UniqueConstraint("program_id", "key"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=identifier)
    program_id: Mapped[str] = mapped_column(ForeignKey("programs.id"), index=True)
    # Stable key of the recommendation (rule + target + affected items): a decision survives recalculation.
    key: Mapped[str] = mapped_column(String(64))
    model_version: Mapped[str | None] = mapped_column(Text)
    rule: Mapped[str] = mapped_column(String(5))  # R1 … R7
    target: Mapped[str] = mapped_column(Text)
    action_text: Mapped[str] = mapped_column(Text)
    evidence: Mapped[dict] = mapped_column(JsonType, default=dict)
    confidence: Mapped[str] = mapped_column(String(10))  # high | medium | low
    current: Mapped[bool] = mapped_column(Boolean, default=True)
    decision: Mapped[str | None] = mapped_column(String(20))  # accepted | rejected | postponed
    decision_comment: Mapped[str | None] = mapped_column(Text)
    decided_by: Mapped[str | None] = mapped_column(String(256))
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


ACTIVITY_TABLES = ("professions", "activity_sources", "activity_claims", "activity_items", "claim_links", "item_links",
                   "knowledge_areas", "experts", "expert_votes", "model_versions", "item_metrics",
                   "activity_merge_log", "activity_extraction_cache",
                   "programs", "program_disciplines", "discipline_links", "recommendations")
