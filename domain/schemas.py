import json
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


SourceType = Literal["HH", "Priklad", "Crossref", "arXiv", "Books", "RPD", "Interview", "expert", "teacher", "documentation", "RabotaRu"]
ChangeStatus = Literal["unchanged", "added", "modified", "fundamental_protected"]
GapType = Literal["missing_tasks", "missing_stack", "missing_quality_criteria", "missing_artifacts", "conflicting_evidence"]


class SourceRecord(Model):
    source_id: str
    source_type: SourceType
    url: str
    title: str
    retrieved_at: datetime
    raw_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    content: str = Field(min_length=1)

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        from urllib.parse import urlsplit, urlunsplit
        parsed = urlsplit(value)
        if parsed.scheme not in {"https", "http"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Нужен абсолютный HTTP(S) URL без учётных данных.")
        return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path or "/", parsed.query, ""))


class Grounded(Model):
    source_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def check_sources(self, info: ValidationInfo) -> "Grounded":
        sources = (info.context or {}).get("sources")
        if sources is None:
            raise ValueError("Для проверки требуется серверный реестр sources.")
        missing = set(self.source_ids) - set(sources)
        if missing:
            raise ValueError(f"Несуществующие источники: {sorted(missing)}")
        if any(sources[key].source_type == "Interview" for key in self.source_ids):
            raise ValueError("Методика интервью не подтверждает факты профессиональной деятельности.")
        return self


class LearningReference(Grounded):
    source_id: str
    locator: str = Field(min_length=5)
    bibliographic_excerpt: str | None = None

    @model_validator(mode="after")
    def check_learning_reference(self, info: ValidationInfo) -> "LearningReference":
        source = info.context["sources"].get(self.source_id)
        if source is None or self.source_ids != [self.source_id]:
            raise ValueError("Некорректная ссылка на учебный материал.")
        if source.source_type not in {"Books", "Crossref", "arXiv", "documentation"}:
            raise ValueError("Нужна книга, статья или документация.")
        if " ".join(self.locator.split()).casefold() not in " ".join(source.content.split()).casefold():
            raise ValueError("Указанный фрагмент отсутствует в источнике.")
        if source.source_type == "Books":
            bibliography = self.bibliographic_excerpt
            if not bibliography or len(bibliography.strip()) < 10:
                raise ValueError("Для книги требуется точная библиографическая выдержка.")
            if " ".join(bibliography.split()).casefold() not in " ".join(source.content.split()).casefold():
                raise ValueError("Библиографическая выдержка отсутствует в источнике Books.")
        return self


class ActivityAspect(Grounded):
    id: str
    aspect_type: Literal["technological", "process", "architectural", "communication", "quality_debugging"]
    description: str
    task_ids: list[str]


class ActivityTask(Grounded):
    id: str
    title: str
    stack: list[str]
    artifacts: list[str]
    quality_criteria: list[str]
    typical_errors: list[str]


class ActivityProcess(Grounded):
    id: str
    title: str
    task_ids: list[str]


class AppliedActivityModel(Grounded):
    tasks: list[ActivityTask]
    processes: list[ActivityProcess]
    aspects: list[ActivityAspect]

    @model_validator(mode="before")
    @classmethod
    def collect_activity_sources(cls, data):
        if isinstance(data, dict) and data.get("source_ids") == []:
            children = [*data.get("tasks", []), *data.get("processes", []), *data.get("aspects", [])]
            references = [key for item in children if isinstance(item, dict) for key in item.get("source_ids", [])]
            return {**data, "source_ids": list(dict.fromkeys(references))}
        return data

    @model_validator(mode="after")
    def check_relations(self) -> "AppliedActivityModel":
        ids = [task.id for task in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("Повторяются идентификаторы задач.")
        for item in [*self.processes, *self.aspects]:
            if not set(item.task_ids) <= set(ids):
                raise ValueError("Процесс или аспект ссылается на неизвестную задачу.")
        return self

    def require_complete(self) -> None:
        if len(self.tasks) < 5 or len(self.processes) < 3:
            raise ValueError("Модель деятельности неполна.")
        if any(not task.stack or not task.artifacts or len(task.quality_criteria) < 2 for task in self.tasks):
            raise ValueError("Недостаточно инструментов, артефактов или критериев качества.")


class ExpertClaimVerification(Grounded):
    claim: str
    verification_status: Literal["verified", "needs_clarification", "rejected_hallucination"]
    explanation: str
    clarification_question: str | None

    @model_validator(mode="after")
    def check_verification(self, info: ValidationInfo) -> "ExpertClaimVerification":
        if self.verification_status == "needs_clarification" and not self.clarification_question:
            raise ValueError("Нужен уточняющий вопрос.")
        if self.verification_status == "verified":
            records = [info.context["sources"][key] for key in self.source_ids]
            if not any(source.source_type in {"HH", "Priklad", "Books", "Crossref", "arXiv", "documentation"} for source in records):
                raise ValueError("Подтверждение требует независимого источника.")
        return self


class VerificationReport(Model):
    claims: list[ExpertClaimVerification]


class ChangeJustification(Grounded):
    reason_type: Literal["market_demand", "scientific_trend", "expert_practice", "fundamental_core", "teacher_review", "scheduled_relevance_update"]
    evidence_summary: str = Field(min_length=1)


class ChangedBlock(Grounded):
    id: str
    title: str
    change_status: ChangeStatus
    previous_value: str | None
    is_fundamental: bool
    locked: bool
    justification: ChangeJustification
    learning_references: list[LearningReference] = Field(min_length=1)

    @model_validator(mode="after")
    def check_change(self) -> "ChangedBlock":
        if self.is_fundamental and not self.locked:
            raise ValueError("Фундаментальное ядро должно быть защищено.")
        if self.change_status == "modified" and not self.previous_value:
            raise ValueError("Для изменения требуется previous_value.")
        if self.change_status == "added" and self.previous_value is not None:
            raise ValueError("У нового блока отсутствует previous_value.")
        return self


class LearningAssignment(ChangedBlock):
    instructions: str
    expected_artifacts: list[str] = Field(min_length=1)
    quality_criteria: list[str] = Field(min_length=2)


class Topic(ChangedBlock):
    activity_task_ids: list[str] = Field(min_length=1)
    assignments: list[LearningAssignment]


class CourseSubject(ChangedBlock):
    origin: Literal["manual", "auto"]
    status: Literal["locked", "suggested", "accepted", "rejected"]
    conflict_with: str | None
    topics: list[Topic] = Field(min_length=1)


class MarketMetric(Grounded):
    name: Literal["active_vacancies", "demand_change", "junior_share", "middle_share", "median_salary"]
    value: float
    unit: str
    as_of: datetime
    region: str
    population_description: str

    @model_validator(mode="after")
    def require_measurement(self, info: ValidationInfo) -> "MarketMetric":
        measured = (info.context or {}).get("market_measurements", {})
        expected = measured.get(self.name)
        if expected is None or self.model_dump(mode="json") != expected:
            raise ValueError("Рыночная метрика должна совпадать с измерением API/выборки.")
        return self


class BusinessCourseAudit(Grounded):
    market_fit_score: float = Field(ge=0, le=1)
    metrics: list[MarketMetric]
    unavailable_metrics: list[str]
    evidence_summary: str
    suggested_changes: list[ChangeJustification]


def all_blocks(subjects: list[CourseSubject]) -> dict[str, ChangedBlock]:
    blocks: dict[str, ChangedBlock] = {}
    for subject in subjects:
        for block in [subject, *subject.topics, *(item for topic in subject.topics for item in topic.assignments)]:
            if block.id in blocks:
                raise ValueError("Повторяется ID блока курса.")
            blocks[block.id] = block
    return blocks


def block_value(block: ChangedBlock) -> str:
    fields = {"title", "instructions", "expected_artifacts", "quality_criteria", "activity_task_ids"}
    value = block.model_dump(mode="json", include=fields)
    if set(value) == {"title"}:
        return block.title
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


class CourseProgramModel(Grounded):
    title: str
    subjects: list[CourseSubject] = Field(min_length=1)
    business_audit: BusinessCourseAudit | None

    @model_validator(mode="after")
    def protect_baseline(self, info: ValidationInfo) -> "CourseProgramModel":
        context = info.context or {}
        current = all_blocks(self.subjects)
        baseline = context.get("baseline")
        if baseline:
            for key, old in all_blocks(baseline.subjects).items():
                new = current.get(key)
                if (old.is_fundamental or old.locked) and new is None:
                    raise ValueError(f"Удалён защищённый блок {key}.")
                if old.is_fundamental and (
                    not new.is_fundamental or not new.locked or new.title != old.title
                ):
                    raise ValueError(f"Изменено фундаментальное ядро {key}.")
                if new is not None and block_value(old) != block_value(new):
                    if new.change_status != "modified" or new.previous_value != block_value(old):
                        raise ValueError("Изменение блока должно содержать точный previous_value.")
            old_ids = set(all_blocks(baseline.subjects))
            if any(block.change_status != "added" for key, block in current.items() if key not in old_ids):
                raise ValueError("Новые блоки должны иметь change_status=added.")
        for name in context.get("manual_subjects", []):
            matching = [subject for subject in self.subjects if subject.title.casefold() == name.casefold()]
            if len(matching) != 1 or not matching[0].locked or matching[0].origin != "manual":
                raise ValueError(f"Не сохранён ручной предмет {name}.")
        for name in context.get("fundamental_subjects", []):
            matching = [subject for subject in self.subjects if subject.title.casefold() == name.casefold()]
            if len(matching) != 1 or not matching[0].is_fundamental or not matching[0].locked:
                raise ValueError(f"Не сохранён фундаментальный предмет {name}.")
        activity = context.get("activity")
        if activity:
            activity.require_complete()
            task_ids = {task.id for task in activity.tasks}
            for subject in self.subjects:
                for topic in subject.topics:
                    if not set(topic.activity_task_ids) <= task_ids:
                        raise ValueError("Тема ссылается на неизвестную задачу деятельности.")
        return self


class GapReport(Model):
    gap_types: list[GapType]
    questions: list[str]
    business_coverage: float = Field(ge=0, le=1)
    science_coverage: float = Field(ge=0, le=1)


class RelevanceDecision(Grounded):
    requires_update: bool
    explanation: str


class InterviewTurn(Model):
    prompt: str = Field(min_length=10, max_length=1200)

    @field_validator("prompt")
    @classmethod
    def require_expert_question(cls, value: str) -> str:
        if not value.rstrip().endswith("?"):
            raise ValueError("Ответ должен завершаться конкретным вопросом эксперту.")
        if any(marker in value.casefold() for marker in ("gap_report", "source_ids", "json", "схеме ниже")):
            raise ValueError("Вместо вопроса возвращена инструкция по генерации.")
        return value


class SubjectSuggestion(Grounded):
    name: str
    is_fundamental: bool
    conflict_with: str | None


class SubjectPlan(Model):
    subjects: list[SubjectSuggestion]
