import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Annotated, Any, Protocol, TypedDict
from uuid import NAMESPACE_URL, uuid5

from langgraph.graph import END, START, StateGraph
from langgraph.errors import GraphBubbleUp
from langgraph.types import interrupt
from openai import APIResponseValidationError, APIStatusError
from pydantic import ValidationError

from auth.settings import Settings
from domain.schemas import (AppliedActivityModel, BusinessCourseAudit, CourseProgramModel, GapReport,
                            Grounded, INDEPENDENT_EXPERT_SOURCE_TYPES, InterviewTurn, Model, RelevanceDecision, SourceRecord, SubjectPlan,
                            TopicAssignments, VerificationReport, all_blocks, block_value)
from domain.workload import WorkloadEdit, WorkloadParams, apply_workload_edit, proposal_from_loads, propose_workload
from orchestration.call_log import (CALL_SCOPE, GENERATION_NOTES, CallScope, RecordingModelClient, classify_outcome,
                                    failure_note)
from orchestration.clients import EvidenceClient, ModelClient, NoActivityEvidence, source_record
from orchestration import prompts
from orchestration.draft import make_outline
from orchestration.interview_methods import load_interview_methods
from orchestration.rabota_business import SOURCE, SOURCE_GUIDANCE
from orchestration.relevance_filter import gap_query, search_query
from orchestration.semantic_check import composition_report, limit_fundamental, weighted_coverage
from orchestration.resilience import ExternalUnavailable
from orchestration.subject_rationale import explain_subjects
from orchestration.vacancy_search import VACANCY_SOURCE_TYPES
from activity.rpd_bridge import activity_from_snapshot
from run_rpd import YandexModeratedResponse


AUTO_DRAFT_SUBJECT_LIMIT = 12
PROFESSION_MODEL = "profession_model"
# Одна дисциплина за вызов: короткий ответ реже обрывается и сбой затрагивает одну дисциплину.
DRAFT_SUBJECT_BATCH_SIZE = 1
# С этого числа дисциплин бизнес-аудит получает сжатую программу (названия и темы).
AUDIT_COMPACT_SUBJECTS = 4
RETRYABLE_OUTCOMES = frozenset({"timeout", "circuit_open", "unavailable"})
GAP_QUERY_LABELS = {"missing_tasks": "рабочие задачи и процессы", "missing_stack": "инструменты",
                    "missing_artifacts": "результаты работы", "missing_quality_criteria": "критерии качества",
                    "conflicting_evidence": "проверка утверждений по источникам"}


def is_truncated(error: Exception) -> bool:
    """Ответ модели оборван по длине: статус max_output_tokens или незакрытый JSON."""
    if classify_outcome(error) == "truncated":
        return True
    return isinstance(error, ValidationError) and any(item["type"] == "json_invalid" for item in error.errors())
BUSINESS_SOURCE_TYPES = frozenset({"Priklad", "HH", SOURCE}) | VACANCY_SOURCE_TYPES
WORKLOAD_REQUEST = re.compile(
    r"\b(?:час(?:ы|ов|а|ам|ами)?|з\s*\.\s*е\.?|зе|зач[её]тн\w*\s+единиц\w*|учебн\w*\s+нагрузк\w*)\b", re.I)


VOLATILE_SOURCE_FIELDS = frozenset({"retrieved_at", "relevance_score"})


def merge_records(previous: list[dict], additions: list[dict]) -> list[dict]:
    merged = {item["source_id"]: item for item in previous}
    for item in additions:
        if item["source_id"] in merged:
            # Один фрагмент может прийти по разным запросам с разной оценкой близости.
            old = {key: value for key, value in merged[item["source_id"]].items() if key not in VOLATILE_SOURCE_FIELDS}
            new = {key: value for key, value in item.items() if key not in VOLATILE_SOURCE_FIELDS}
            if old != new:
                raise ValueError("Конфликт источников с одинаковым ID.")
        else:
            merged[item["source_id"]] = item
    return list(merged.values())


class GraphState(TypedDict, total=False):
    request_id: str
    owner_user_id: str
    course_meta: dict
    run_id: str
    mode: str
    manual_subjects: list[str]
    known_tasks: list[str]
    auto_subjects: dict
    sources: Annotated[list[dict], merge_records]
    business_report: dict
    science_report: dict
    books_report: dict
    university_report: dict
    interview_source_notices: list[str]
    factcheck_source_notices: list[str]
    draft_source_notices: list[str]
    literature_source_notices: list[str]
    composition_notices: list[str]
    applied_activity: dict
    # "profession_model" — задачи взяты из проверенной модели деятельности профессии (activity/rpd_bridge.py).
    activity_origin: str
    gap_report: dict
    interview_session_id: str
    interrupt_id: str
    closed_interview_id: str
    interview_answers: list[dict]
    interview_round: int
    verification: dict
    course_program: dict
    baseline: dict | None
    revision_baseline: dict | None
    review_feedback: list[dict]
    review_iteration: int
    refresh_only: bool
    workload_only: bool
    workload_proposal: dict | None
    workload_notice: str
    status: str
    version_id: str
    consumed_review_id: str
    needs_update: bool


class Finding(Grounded):
    summary: str


class AgentReport(Model):
    findings: list[Finding]


def run_source_notices(state: GraphState) -> list[str]:
    notices = [note for role in ("business", "science", "books", "university")
               for note in state.get(f"{role}_report", {}).get("source_notices", [])]
    return list(dict.fromkeys([*notices, *state.get("interview_source_notices", []),
                               *state.get("composition_notices", []),
                               *state.get("factcheck_source_notices", []), *state.get("draft_source_notices", []),
                               *state.get("literature_source_notices", [])]))


class WorkflowStore(Protocol):
    async def set_status(self, course_id: str, status: str, owner: str) -> None:
        """Сохранить статус с проверкой владельца."""

    async def save_sources(self, course_id: str, sources: list[SourceRecord]) -> None:
        """Сохранить неизменяемые источники."""

    async def prepare_interview(self, course_id: str, interrupt_id: str, report: dict) -> str:
        """Идемпотентно создать интервью."""

    async def save_draft(self, course_id: str, version_id: str, program: CourseProgramModel,
                         scheduled: bool, activity: dict, *, evidence_gaps: dict, source_notices: list[str],
                         vacancy_source_ids: list[str], workload_proposal: dict, generation: dict | None = None,
                         composition: dict | None = None) -> None:
        """Сохранить версию и уведомление."""

    async def finish_review(self, course_id: str, version_id: str, action: str, actor: str, workloads=None) -> None:
        """Утвердить либо отклонить конкретную версию."""

    async def defer_audit(self, course_id: str) -> None:
        """Сохранить дату следующего аудита."""

    async def record_verification(self, interview_id: str, report: dict) -> None:
        """Сохранить результаты проверки ответов эксперта."""


def registry(state: GraphState) -> dict[str, SourceRecord]:
    return {item["source_id"]: SourceRecord.model_validate(item) for item in state.get("sources", [])}


def planned_subjects(state: GraphState) -> list[dict]:
    suggestions = state.get("auto_subjects", {}).get("subjects", [])
    if not state.get("gap_report", {}).get("gap_types"):
        return suggestions
    # Первый черновик оставляет место для решения преподавателя о расширении программы.
    unique = {}
    for subject in suggestions:
        unique.setdefault(" ".join(subject["name"].casefold().split()), subject)
    ordered = sorted(unique.values(), key=lambda subject: not subject["is_fundamental"])
    manual = {" ".join(name.casefold().split()) for name in state.get("manual_subjects", [])}
    known = [subject for subject in ordered if " ".join(subject["name"].casefold().split()) in manual]
    suggested = [subject for subject in ordered if " ".join(subject["name"].casefold().split()) not in manual]
    return [*known, *suggested[:AUTO_DRAFT_SUBJECT_LIMIT]]


def context(state: GraphState) -> dict:
    result: dict = {"sources": registry(state), "manual_subjects": state.get("manual_subjects", []),
                    # Задачи из модели профессии проверены экспертами, но без артефактов и критериев качества.
                    "allow_incomplete_activity": bool(state.get("gap_report", {}).get("gap_types"))
                    or state.get("activity_origin") == PROFESSION_MODEL,
                    "market_measurements": {}, "fundamental_subjects": [
                        subject["name"] for subject in planned_subjects(state)
                        if subject["is_fundamental"]]}
    if state.get("applied_activity"):
        result["activity"] = AppliedActivityModel.model_validate(state["applied_activity"], context=result)
    baseline = state.get("revision_baseline") or state.get("baseline")
    if baseline:
        result["baseline"] = CourseProgramModel.model_validate(baseline, context={
            **result, "manual_subjects": [], "fundamental_subjects": [], "activity": None,
            "allow_incomplete_activity": True})
    return result


def payload(state: GraphState) -> dict:
    excluded = {"interview_answers", "verification", "owner_user_id", "request_id", "run_id",
                "interview_session_id", "interrupt_id", "version_id", "consumed_review_id", "closed_interview_id"}
    excluded.update({"workload_proposal", "workload_only", "workload_notice"})
    result = {key: value for key, value in state.items() if key not in excluded}
    if "review_feedback" in result:
        result["review_feedback"] = [{"content": item.get("content", ""), "attachments": item.get("attachments", [])}
                                     for item in result["review_feedback"]]
    if "gap_report" in result:
        # В генерацию попадает описание пробелов без вопросов, цитирующих неподтверждённый ответ.
        report = result["gap_report"]
        result["gap_report"] = {key: report[key] for key in ("gap_types", "business_coverage", "science_coverage") if key in report}
        if report.get("gap_types"):
            result["auto_subjects"] = {"subjects": planned_subjects(state)}
            result["sources"] = [{**source, "content": source["content"][:1800]} for source in state.get("sources", [])]
    baseline = context(state).get("baseline")
    if baseline:
        result["previous_values"] = {key: block_value(block) for key, block in all_blocks(baseline.subjects).items()}
    return result


def mark_generated(program: dict, generated_by: str, note: str | None = None, *, keep_existing: bool = False) -> dict:
    """Записать происхождение каждой дисциплины: собрана моделью или шаблоном."""
    subjects = []
    for subject in program["subjects"]:
        if keep_existing and subject.get("generated_by"):
            subjects.append(subject)
        else:
            subjects.append({**subject, "generated_by": generated_by, "generation_note": note})
    return {**program, "subjects": subjects}


SUBJECT_SOURCE_LIMIT = 8
SUBJECT_TASK_LIMIT = 6
SUBJECT_SOURCE_CHARS = 1500


def outline_hint(title: str, name: str, suggestions: list[dict], ctx: dict) -> dict | None:
    """Каркас из сохранённых материалов — ориентир для модели, а не готовый результат."""
    try:
        outline = make_outline(title, [name], suggestions, ctx)
    except (ValueError, ValidationError):
        return None
    subject = outline.subjects[0]
    return {"topics": [{"title": topic.title, "activity_task_ids": topic.activity_task_ids,
                        "assignment": topic.assignments[0].title if topic.assignments else None}
                       for topic in subject.topics],
            "source_ids": subject.source_ids}


def subject_request(state: GraphState, names: list[str], suggestions: list[dict], ctx: dict) -> tuple[dict, list[str]]:
    """Запрос черновика одной дисциплины: только связанные задачи и источники, укладывается в контекст модели."""
    from orchestration.semantic_check import concepts
    sources: dict[str, SourceRecord] = ctx["sources"]
    keys = {" ".join(name.casefold().split()) for name in names}
    plan = [item for item in suggestions if " ".join(item["name"].casefold().split()) in keys]
    terms = concepts(" ".join(names))
    tasks = state.get("applied_activity", {}).get("tasks", [])
    ranked = sorted(tasks, key=lambda task: -len(terms & concepts(" ".join([task["title"], *task.get("stack", [])]))))
    chosen = [task for task in ranked if terms & concepts(" ".join([task["title"], *task.get("stack", [])]))]
    chosen = (chosen or ranked[:3])[:SUBJECT_TASK_LIMIT]
    hint = outline_hint(state["course_meta"]["title"], names[0], suggestions, ctx) if len(names) == 1 else None
    cited = [key for item in plan for key in item.get("source_ids", [])]
    cited += [key for task in chosen for key in task.get("source_ids", [])]
    cited += hint["source_ids"] if hint else []
    if ctx.get("manual_subject_source_id"):
        cited.append(ctx["manual_subject_source_id"])
    others = sorted((source for key, source in sources.items() if source.source_type != "Interview"),
                    key=lambda source: -len(terms & concepts(f"{source.title} {source.content[:2000]}")))
    selected = list(dict.fromkeys([key for key in cited if key in sources and sources[key].source_type != "Interview"]
                                  + [source.source_id for source in others]))[:SUBJECT_SOURCE_LIMIT]
    request = {
        "course": {key: state["course_meta"].get(key) for key in ("title", "profession") if state["course_meta"].get(key)},
        "batch_subjects": names, "subject_plan": plan,
        "applied_activity": {"tasks": [{key: task.get(key) for key in ("id", "title", "stack", "artifacts",
                                                                        "quality_criteria", "source_ids")}
                                       for task in chosen]},
        "gap_report": {"gap_types": state.get("gap_report", {}).get("gap_types", [])},
        "sources": [{**sources[key].model_dump(mode="json", exclude={"raw_hash", "relevance_score"}),
                     "content": sources[key].content[:SUBJECT_SOURCE_CHARS]} for key in selected],
    }
    if hint:
        request["outline_hint"] = hint
    if state.get("review_feedback"):
        request["review_feedback"] = [{"content": item.get("content", "")} for item in state["review_feedback"]]
    return request, selected


def with_importance(program: dict, suggestions: list[dict]) -> dict:
    """Перенести ранг важности из плана дисциплин (единственный вклад модели в расчёт часов)."""
    ranks = {" ".join(item["name"].casefold().split()): item for item in suggestions}
    subjects = []
    for subject in program["subjects"]:
        match = ranks.get(" ".join(subject["title"].casefold().split()))
        if match and not subject.get("importance"):
            subject = {**subject, "importance": match.get("importance", "standard"),
                       "importance_reason": match.get("importance_reason")}
        subjects.append(subject)
    return {**program, "subjects": subjects}


def build_course_graph(settings: Settings, llm: ModelClient, evidence: EvidenceClient,
                       store: WorkflowStore, checkpointer: Any, *,
                       university_literature_enricher: Callable[[dict, dict[str, SourceRecord]], Awaitable[tuple]] | None = None,
                       sleep: Callable[[float], Awaitable[None]] = asyncio.sleep):
    if not isinstance(llm, RecordingModelClient):
        llm = RecordingModelClient(llm, getattr(store, "record_model_call", None))

    async def start(state: GraphState) -> dict:
        if not state.get("run_id"):
            raise ValueError("Для запуска требуется устойчивый run_id.")
        await store.set_status(state["request_id"], "agents_running", state["owner_user_id"])
        return {"status": "agents_running", "interview_round": 0, "review_iteration": 0,
                "interview_answers": [], "review_feedback": [], "revision_baseline": None,
                "refresh_only": False, "workload_only": False, "workload_notice": "",
                "interview_source_notices": [], "factcheck_source_notices": [], "draft_source_notices": [],
                "literature_source_notices": [], "composition_notices": [],
                "applied_activity": {}, "activity_origin": "", "gap_report": {}, "verification": {}, "auto_subjects": {},
                "course_program": {}, "consumed_review_id": "", "closed_interview_id": "", "version_id": ""}

    def agent(role: str):
        async def run(state: GraphState) -> dict:
            # Запрос по смыслу: название курса, профессия и известные рабочие задачи, а не JSON заявки.
            query = search_query({**state["course_meta"], "manual_subjects": state.get("manual_subjects", [])},
                                 state.get("known_tasks", []))
            sources = await evidence.search(role, query, owner_scope=state["owner_user_id"])
            notices = getattr(sources, "notices", [])
            if not sources:
                return {f"{role}_report": {"findings": [], "source_notices": notices}, "sources": []}
            source_map = {item.source_id: item for item in sources}
            instructions = prompts.AGENT.format(role=role)
            if role == "business" and any(item.source_type == SOURCE for item in sources):
                instructions += "\n" + SOURCE_GUIDANCE
            try:
                report = await llm.generate(AgentReport, instructions,
                                            {"course": state["course_meta"],
                                             "sources": [item.model_dump(mode="json") for item in sources],
                                             "source_notices": notices}, {"sources": source_map}, flash=True)
            except YandexModeratedResponse:
                logging.getLogger(__name__).warning("Yandex ограничила анализ источников агента %s.", role)
                report = AgentReport(findings=[])
                sources = []
                notices = [*notices, f"{role}: Yandex AI Studio ограничила анализ этой подборки источников."]
            except (ExternalUnavailable, ValidationError):
                report = AgentReport(findings=[])
                notices = [*notices, f"{role}: модель не сформировала сводку. Сохранены исходные материалы."]
            report_data = {**report.model_dump(mode="json"), "source_notices": notices}
            if role == "business":
                report_data["vacancy_source_ids"] = [item.source_id for item in sources
                                                     if item.source_type in VACANCY_SOURCE_TYPES]
            return {f"{role}_report": report_data,
                    "sources": [item.model_dump(mode="json") for item in sources]}
        return run

    async def relevance(state: GraphState) -> dict:
        if state.get("mode") != "relevance":
            return {"needs_update": True}
        decision = await llm.generate(RelevanceDecision, prompts.RELEVANCE, payload(state), context(state))
        if not decision.requires_update:
            await store.defer_audit(state["request_id"])
        return {"needs_update": decision.requires_update,
                "status": "agents_running" if decision.requires_update else "approved"}

    async def formalize(state: GraphState) -> dict:
        snapshot = state["course_meta"].get("activity_model") or {}
        if snapshot.get("tasks"):
            # Задачи из модели деятельности профессии: без вызова LLM, каждая задача — со своим источником.
            activity, records = activity_from_snapshot(snapshot, settings.public_origin)
            known = {**registry(state), **{record.source_id: record for record in records}}
            model = AppliedActivityModel.model_validate(activity, context={"sources": known})
            return {"applied_activity": model.model_dump(mode="json"), "activity_origin": PROFESSION_MODEL,
                    "sources": [record.model_dump(mode="json") for record in records]}
        sources = {k: s for k, s in registry(state).items()
                   if s.source_type in BUSINESS_SOURCE_TYPES | {"Crossref", "arXiv", "expert", "documentation"}}
        if not sources:
            return {"applied_activity": {}}
        activity_input = {"course": state["course_meta"], "business": state["business_report"],
                          "science": state["science_report"], "sources": [s.model_dump(mode="json") for s in sources.values()]}
        try:
            instructions = prompts.ACTIVITY
            if any(source.source_type == SOURCE for source in sources.values()):
                instructions += "\n" + SOURCE_GUIDANCE
            model = await llm.generate(AppliedActivityModel, instructions, activity_input, {"sources": sources})
        except (NoActivityEvidence, YandexModeratedResponse):
            return {"applied_activity": {}}
        return {"applied_activity": model.model_dump(mode="json")}

    async def university_audit(state: GraphState) -> dict:
        if not state["applied_activity"]:
            return {"auto_subjects": {"subjects": []}}
        if state.get("activity_origin") == PROFESSION_MODEL:
            # Состав дисциплин решает руководитель программы: курс — это одна закреплённая дисциплина.
            ids = state["applied_activity"]["source_ids"]
            return {"auto_subjects": {"subjects": [
                {"name": name, "is_fundamental": False, "conflict_with": None, "importance": "standard",
                 "importance_reason": None, "source_ids": ids} for name in state.get("manual_subjects", [])]}}
        result = await llm.generate(SubjectPlan, prompts.UNIVERSITY, payload(state), context(state))
        # Флаг ядра: только с источником RPD/Books и в пределах доли из config/analysis.yaml.
        subjects, notes = limit_fundamental(result.model_dump(mode="json")["subjects"], registry(state))
        return {"auto_subjects": {"subjects": subjects}, "composition_notices": notes}

    async def gap_analysis(state: GraphState) -> dict:
        await store.set_status(state["request_id"], "gap_analysis", state["owner_user_id"])
        if not state["applied_activity"]:
            report = GapReport(gap_types=["missing_tasks"], business_coverage=0, science_coverage=0,
                               questions=["Какую рабочую задачу вы решали недавно, что сделали и как проверили результат?"])
            return {"gap_report": report.model_dump(mode="json"), "status": "gap_analysis"}
        activity = AppliedActivityModel.model_validate(state["applied_activity"], context=context(state))
        sources = registry(state)
        if state.get("activity_origin") == PROFESSION_MODEL:
            # Задачи уже прошли экспертную валидацию модели профессии: интервью по курсу не нужно.
            report = GapReport(gap_types=[], questions=[], business_coverage=1.0,
                               science_coverage=weighted_coverage(activity.tasks, sources, {"Crossref", "arXiv"}))
            return {"gap_report": report.model_dump(mode="json"), "status": "gap_analysis"}
        count = len(activity.tasks)
        # Покрытие с учётом качества: тип источника, свежесть и число независимых источников на задачу.
        business = weighted_coverage(activity.tasks, sources, BUSINESS_SOURCE_TYPES)
        science = weighted_coverage(activity.tasks, sources, {"Crossref", "arXiv"})
        gaps, questions = [], []
        if count < 5 or len(activity.processes) < 3:
            gaps.append("missing_tasks")
            questions.append("Какие задачи вы решаете регулярно и в какие рабочие процессы они входят?")
        for field, kind, question in (
            ("stack", "missing_stack", "Какие инструменты применяются к каждой задаче?"),
            ("artifacts", "missing_artifacts", "Какие проверяемые результаты даёт каждая задача?"),
            ("quality_criteria", "missing_quality_criteria", "По каким двум критериям вы оцениваете результат каждой задачи?"),
        ):
            minimum = 2 if field == "quality_criteria" else 1
            if any(len(getattr(task, field)) < minimum for task in activity.tasks):
                gaps.append(kind)
                questions.append(question)
        if business < settings.business_coverage_threshold or science < settings.science_coverage_threshold:
            gaps.append("conflicting_evidence")
            questions.append("Какие источники подтверждают применение задач на практике и их научную основу?")
        if state.get("verification"):
            for claim in state["verification"]["claims"]:
                if claim["verification_status"] == "needs_clarification":
                    gaps.append("conflicting_evidence")
                    questions.append(claim["clarification_question"])
        report = GapReport(gap_types=list(dict.fromkeys(gaps)), questions=questions,
                           business_coverage=business, science_coverage=science)
        return {"gap_report": report.model_dump(mode="json"), "status": "gap_analysis"}

    async def prepare_interview(state: GraphState) -> dict:
        round_number = state.get("interview_round", 0)
        if round_number:
            return {"status": "designing"}
        identifier = str(uuid5(NAMESPACE_URL, f"{state['request_id']}/{state['run_id']}/interview/{round_number}"))
        methods = await evidence.search("interview", gap_query(state["gap_report"], GAP_QUERY_LABELS),
                                        owner_scope=state["owner_user_id"])
        local_skill = load_interview_methods(settings.interview_skill_directory, state["gap_report"]["gap_types"])
        skill = (local_skill + "\n\n" + "\n\n".join(item.content for item in methods))[:18000]
        notices = [*state.get("interview_source_notices", []), *getattr(methods, "notices", [])]
        try:
            turn = await llm.generate(InterviewTurn, prompts.INTERVIEW,
                                      {"course": state["course_meta"]["title"], "gaps": state["gap_report"]["questions"],
                                       "tasks": [t["title"] for t in state["applied_activity"].get("tasks", [])],
                                       "interview_methods": skill, "history": [a["content"] for a in state.get("interview_answers", [])]},
                                      context(state), flash=True)
            question_origin = "interviewer"
        except (ValidationError, YandexModeratedResponse, ExternalUnavailable) as error:
            for question in reversed(state["gap_report"]["questions"]):
                try:
                    turn = InterviewTurn(prompt=question)
                except ValidationError:
                    continue
                break
            else:
                raise
            question_origin = "gap_report"
            notices.append("Интервью: использован резервный вопрос по выявленному пробелу.")
            logging.getLogger(__name__).warning("Интервью курса %s: использован резервный вопрос (%s).",
                                                state["request_id"], type(error).__name__)
        report = {**state["gap_report"], "prompt": turn.prompt,
                  "question_origin": question_origin,
                  "previous_interview_id": state.get("interview_session_id") if round_number else None,
                  "method_source_ids": [item.source_id for item in methods]}
        interview_id = await store.prepare_interview(state["request_id"], identifier, report)
        await store.save_sources(state["request_id"], [*registry(state).values(), *methods])
        await store.set_status(state["request_id"], "waiting_for_expert", state["owner_user_id"])
        return {"interrupt_id": identifier, "interview_session_id": interview_id,
                "gap_report": report, "status": "waiting_for_expert", "interview_source_notices": list(dict.fromkeys(notices))}

    async def interview(state: GraphState) -> dict:
        answer = interrupt({"interrupt_id": state["interrupt_id"], "interview_session_id": state["interview_session_id"],
                            "prompt": state["gap_report"]["prompt"], "kind": "expert"})
        if not isinstance(answer, dict) or answer.get("interrupt_id") != state["interrupt_id"]:
            raise ValueError("Ответ не соответствует текущему интервью.")
        if answer.get("action") == "finish_without_answer":
            if answer.get("actor_user_id") != state["owner_user_id"]:
                raise PermissionError("Завершение интервью доступно владельцу курса.")
            return {"closed_interview_id": state["interrupt_id"]}
        if not answer.get("content"):
            raise ValueError("Ответ эксперта пуст.")
        return {"interview_answers": [*state.get("interview_answers", []), answer],
                "interview_round": state.get("interview_round", 0) + 1}

    async def factcheck(state: GraphState) -> dict:
        answer = state["interview_answers"][-1]
        topic = " ".join(state["course_meta"]["title"].split())[:300]
        fresh = await evidence.search("factcheck", f"{topic}: рабочие задачи, инструменты, результаты и критерии качества",
                                      owner_scope=state["owner_user_id"])
        notices = getattr(fresh, "notices", [])
        combined = merge_records(state.get("sources", []), [item.model_dump(mode="json") for item in fresh])
        independent = {item["source_id"]: SourceRecord.model_validate(item) for item in combined
                       if item["source_type"] in INDEPENDENT_EXPERT_SOURCE_TYPES}
        try:
            result = await llm.generate(VerificationReport, prompts.FACTCHECK,
                                        {"answer": {"content": answer["content"], "attachments": answer.get("attachments", [])},
                                         "sources": [item.model_dump(mode="json") for item in independent.values()]},
                                        {"sources": independent}, flash=True)
            report = result.model_dump(mode="json")
        except (ValidationError, YandexModeratedResponse, ExternalUnavailable):
            report = {"claims": [], "assessment": "unavailable"}
        verified = [claim["claim"] for claim in report["claims"] if claim["verification_status"] == "verified"]
        records = [source_record("expert", f"{settings.public_origin}/teacher?course={state['request_id']}&interview={state['interview_session_id']}",
                                  "Подтверждённое утверждение эксперта", claim) for claim in verified]
        await store.save_sources(state["request_id"], [*fresh, *records])
        await store.record_verification(state["interview_session_id"], report)
        return {"verification": report,
                "factcheck_source_notices": list(dict.fromkeys([*state.get("factcheck_source_notices", []), *notices])),
                "sources": [item.model_dump(mode="json") for item in [*fresh, *records]]}

    async def design(state: GraphState) -> dict:
        await store.set_status(state["request_id"], "designing", state["owner_user_id"])
        suggestions = planned_subjects(state)
        manual = {" ".join(name.casefold().split()): name for name in state.get("manual_subjects", [])}
        automatic = {" ".join(item["name"].casefold().split()): item["name"] for item in suggestions}
        names = {**automatic, **manual}
        ctx = context(state)
        if ctx["allow_incomplete_activity"] and not ctx.get("baseline") and len(names) > DRAFT_SUBJECT_BATCH_SIZE:
            return await design_batched(state, list(names.items()), suggestions, manual)
        notes: list[str] = []
        GENERATION_NOTES.set(notes)
        program = await llm.generate(CourseProgramModel, prompts.DESIGN, payload(state), ctx)
        marked = mark_generated(program.model_dump(mode="json"), "model", "; ".join(notes) or None)
        return {"course_program": with_importance(marked, suggestions), "status": "designing"}

    async def design_batched(state: GraphState, names: list[tuple[str, str]], suggestions: list[dict],
                             manual: dict[str, str]) -> dict:
        semaphore = asyncio.Semaphore(settings.draft_concurrency)
        delays = settings.retry_delays()
        manual_source = None
        if manual:
            url = f"{settings.public_origin}/teacher?course={state['request_id']}"
            manual_source = source_record("teacher", url, "Список дисциплин преподавателя", "\n".join(manual.values()))
            state = {**state, "sources": merge_records(state.get("sources", []), [manual_source.model_dump(mode="json")])}

        async def generate_split(request: dict, ctx: dict, notes: list[str]) -> CourseProgramModel:
            """Сначала дисциплина с темами без заданий, затем задания каждой темы отдельным вызовом."""
            skeleton = await llm.generate(CourseProgramModel, prompts.DESIGN + prompts.DESIGN_OUTLINE_ONLY,
                                          {**request, "assignments_separately": True}, ctx)
            data = skeleton.model_dump(mode="json")
            for subject in data["subjects"]:
                for topic in subject["topics"]:
                    if topic["assignments"]:
                        continue
                    try:
                        part = await llm.generate(TopicAssignments, prompts.DESIGN_ASSIGNMENTS,
                                                  {**request, "subject": subject["title"], "topic": topic["title"]}, ctx)
                    except (ValidationError, RuntimeError) as error:
                        notes.append(f"Задания темы «{topic['title']}» не собраны — {failure_note(error)}")
                        continue
                    topic["assignments"] = [{**item.model_dump(mode="json"), "id": f"{topic['id']}-a{index}"}
                                            for index, item in enumerate(part.assignments, 1)]
            return CourseProgramModel.model_validate(data, context=ctx)

        async def generate_batch(offset: int) -> tuple[CourseProgramModel, str | None, list[str]]:
            batch = dict(names[offset:offset + DRAFT_SUBJECT_BATCH_SIZE])
            CALL_SCOPE.set(replace(CALL_SCOPE.get(), subject=", ".join(batch.values())[:300]))
            notes: list[str] = []
            GENERATION_NOTES.set(notes)
            manual_subjects = [name for key, name in batch.items() if key in manual]
            batch_state = {**state, "manual_subjects": manual_subjects,
                           "course_meta": {**state["course_meta"], "manual_subjects": manual_subjects},
                           "auto_subjects": {"subjects": [item for item in suggestions
                                             if " ".join(item["name"].casefold().split()) in batch]}}
            ctx = {**context(batch_state), "required_subject_names": set(batch)}
            if manual_source:
                ctx["manual_subject_source_id"] = manual_source.source_id
            # Короткий запрос по одной дисциплине: её задачи и источники, а шаблон — только ориентир.
            request, ctx["allowed_source_ids"] = subject_request(batch_state, list(batch.values()), suggestions, ctx)
            split = False
            error: Exception | None = None
            attempt = 0
            while True:
                try:
                    async with semaphore:
                        program = (await generate_split(request, ctx, notes) if split else
                                   await llm.generate(CourseProgramModel, prompts.DESIGN, request, ctx))
                    return program, None, notes
                except (ValidationError, RuntimeError) as caught:
                    error = caught
                if is_truncated(error) and not split:
                    split = True  # большой ответ: темы и задания отдельными вызовами
                    continue
                if classify_outcome(error) == "input_too_large" and len(request["sources"]) > 2:
                    # Вход всё ещё больше контекста модели: вдвое меньше источников и повтор без ожидания.
                    request = {**request, "sources": request["sources"][:len(request["sources"]) // 2]}
                    ctx["allowed_source_ids"] = [item["source_id"] for item in request["sources"]]
                    continue
                if classify_outcome(error) not in RETRYABLE_OUTCOMES or attempt >= len(delays):
                    break
                logging.getLogger(__name__).warning("Курс %s: повтор дисциплины %s через %s с (%s).", state["request_id"],
                                                    list(batch.values()), delays[attempt], classify_outcome(error))
                await sleep(delays[attempt])
                attempt += 1
            note = failure_note(error)
            if attempt:
                note += f" Повторов — {attempt}."
            logging.getLogger(__name__).warning("Курс %s: дисциплины %s собраны шаблоном — %s",
                                                state["request_id"], list(batch.values()), note)
            return make_outline(state["course_meta"]["title"], list(batch.values()), suggestions, ctx), note, notes

        async with asyncio.TaskGroup() as tasks:
            batches = [tasks.create_task(generate_batch(offset))
                       for offset in range(0, len(names), DRAFT_SUBJECT_BATCH_SIZE)]
        completed = [batch.result() for batch in batches]
        programs = [program for program, _, _ in completed]
        subjects = []
        for number, (program, failure, notes) in enumerate(completed, 1):
            note = "; ".join([*([failure] if failure else []), *notes]) or None
            for raw in program.subjects:
                subject = {**raw.model_dump(mode="json"), "generated_by": "outline" if failure else "model",
                           "generation_note": note}
                for block in [subject, *subject["topics"], *(a for topic in subject["topics"] for a in topic["assignments"])]:
                    block["id"] = f"b{number}-{block['id']}"
                subjects.append(subject)
        combined = {"title": state["course_meta"]["title"], "subjects": subjects, "business_audit": None,
                    "source_ids": list(dict.fromkeys(key for program in programs for key in program.source_ids))}
        program = CourseProgramModel.model_validate(with_importance(combined, suggestions), context=context(state))
        outlined = sum(len(part.subjects) for part, failure, _ in completed if failure)
        notices = [f"{outlined} из {len(subjects)} дисциплин собраны шаблоном из сохранённых материалов: "
                   "модель не сформировала их. Проверьте темы и учебные источники."] if outlined else []
        return {"course_program": program.model_dump(mode="json"), "status": "designing",
                "draft_source_notices": notices, "sources": [manual_source.model_dump(mode="json")] if manual_source else []}

    async def business_audit(state: GraphState) -> dict:
        audit_input = payload(state)
        batched = state["gap_report"]["gap_types"] and len(state["course_program"]["subjects"]) > AUDIT_COMPACT_SUBJECTS
        # Аудиту нужны названия дисциплин и тем: полная программа не помещается в контекст модели.
        program = state["course_program"]
        audit_input["course_program"] = {"title": program["title"], "subjects": [
            {"title": subject["title"], "topics": [topic["title"] for topic in subject["topics"]]}
            for subject in program["subjects"]]}
        try:
            audit = await llm.generate(BusinessCourseAudit, prompts.BUSINESS_AUDIT, audit_input, context(state), flash=True)
        except (ValidationError, RuntimeError):
            notices = [*state.get("draft_source_notices", []), "Рыночная оценка пока недоступна."]
            return {"draft_source_notices": list(dict.fromkeys(notices))}
        updated = {**state["course_program"], "business_audit": audit.model_dump(mode="json")}
        if audit.suggested_changes and not batched:
            try:
                program = await llm.generate(CourseProgramModel, prompts.DESIGN,
                                             {**payload(state), "business_enrichment": audit.model_dump(mode="json")},
                                             context(state))
                audit = await llm.generate(BusinessCourseAudit, prompts.BUSINESS_AUDIT,
                                           {**payload(state), "course_program": program.model_dump(mode="json")},
                                           context(state), flash=True)
                regenerated = with_importance(mark_generated(program.model_dump(mode="json"), "model"),
                                              planned_subjects(state))
                updated = {**regenerated, "business_audit": audit.model_dump(mode="json")}
            except (ValidationError, RuntimeError):
                notices = [*state.get("draft_source_notices", []), "Рыночные предложения сохранены в отчёте для проверки."]
                return {"course_program": updated, "draft_source_notices": list(dict.fromkeys(notices))}
        checked = CourseProgramModel.model_validate(updated, context=context(state))
        return {"course_program": checked.model_dump(mode="json")}

    async def subject_rationale(state: GraphState) -> dict:
        program, records = await explain_subjects(state["course_program"], registry(state), evidence, llm,
                                                  state["owner_user_id"])
        additions = [record.model_dump(mode="json") for record in records]
        combined = merge_records(state.get("sources", []), additions)
        checked = CourseProgramModel.model_validate(program, context=context({**state, "sources": combined}))
        return {"course_program": checked.model_dump(mode="json"), "sources": additions}

    async def university_literature(state: GraphState) -> dict:
        if not settings.literature_parser_enabled or university_literature_enricher is None:
            return {}
        await store.set_status(state["request_id"], "collecting_literature", state["owner_user_id"])
        try:
            current = registry(state)
            program, records, results = await university_literature_enricher(state["course_program"], current)
            additions = [record.model_dump(mode="json") for record in records]
            combined = merge_records(state.get("sources", []), additions)
            checked = CourseProgramModel.model_validate(program, context=context({**state, "sources": combined}))
        except Exception as error:
            logging.getLogger(__name__).warning("University: подбор литературы недоступен (%s)", type(error).__name__)
            return {"literature_source_notices": ["Подбор учебной литературы временно недоступен."]}
        notices = []
        missing = sum(not item.sources for item in results)
        if missing:
            notices.append(f"Для {missing} из {len(results)} тем поиск не добавил проверенный учебник или статью.")
        return {"course_program": checked.model_dump(mode="json"), "sources": additions,
                "literature_source_notices": notices}

    async def persist_review(state: GraphState) -> dict:
        identifier = str(uuid5(NAMESPACE_URL, f"{state['request_id']}/{state['run_id']}/version/{state.get('review_iteration', 0)}"))
        await store.save_sources(state["request_id"], list(registry(state).values()))
        program = CourseProgramModel.model_validate(state["course_program"], context=context(state))
        scheduled = state.get("mode") == "relevance"
        gaps = state["gap_report"]
        evidence_gaps = {key: gaps[key] for key in ("gap_types", "business_coverage", "science_coverage") if key in gaps}
        previous = state.get("workload_proposal")
        proposal = propose_workload(state["course_program"]["subjects"], previous,
                                    WorkloadParams.from_meta(state["course_meta"]))
        notices = run_source_notices(state)
        if notice := state.get("workload_notice"):
            notices.append(notice)
        composition = composition_report(state["course_program"], state["course_meta"], state.get("applied_activity"))
        await store.save_draft(state["request_id"], identifier, program, scheduled, state["applied_activity"],
                               evidence_gaps=evidence_gaps, source_notices=notices,
                               vacancy_source_ids=state["business_report"].get("vacancy_source_ids", []),
                               workload_proposal=proposal, generation=llm.describe(), composition=composition)
        return {"version_id": identifier, "workload_proposal": proposal,
                "status": "pending_update_review" if scheduled else "waiting_for_review"}

    async def review(state: GraphState) -> dict:
        decision = interrupt({"interrupt_id": state["version_id"], "prompt": "Утвердите курс или укажите правки.", "kind": "teacher"})
        if not isinstance(decision, dict) or decision.get("actor_user_id") != state["owner_user_id"]:
            raise PermissionError("Ревью доступно владельцу курса.")
        if decision.get("version_id") != state["version_id"]:
            raise ValueError("Решение относится к устаревшей версии.")
        action = decision.get("action")
        if action in {"approve", "reject"}:
            await store.finish_review(state["request_id"], state["version_id"], action, decision["actor_user_id"], decision.get("workloads"))
            return {"status": "approved", "consumed_review_id": state["version_id"]}
        if action == "refresh_materials":
            await store.set_status(state["request_id"], "collecting_literature", state["owner_user_id"])
            return {"consumed_review_id": state["version_id"], "refresh_only": True, "workload_only": False,
                    "review_iteration": state.get("review_iteration", 0) + 1,
                    "revision_baseline": state["course_program"], "status": "collecting_literature",
                    "workload_proposal": decision.get("workload_proposal") or state.get("workload_proposal")}
        if action != "revise" or not decision.get("content"):
            raise ValueError("Укажите правку или решение ревью.")
        subjects = state["course_program"]["subjects"]
        proposal = propose_workload(subjects, proposal_from_loads(decision.get("workloads") or {})
                                    or decision.get("workload_proposal") or state.get("workload_proposal"),
                                    WorkloadParams.from_meta(state["course_meta"]))
        workload_only = False
        notice = ""
        if WORKLOAD_REQUEST.search(decision["content"]):
            edit = None
            try:
                edit = await llm.generate(WorkloadEdit, prompts.WORKLOAD_EDIT, {
                    "feedback": decision["content"], "total_hours": proposal["total_hours"],
                    "hours_per_credit": proposal["hours_per_credit"],
                    "subjects": [{"subject_id": subject["id"], "title": subject["title"],
                                  "hours": proposal["loads"][subject["id"]]} for subject in subjects],
                    "attachments": [{"name": f["name"], "text": f["text"][:6000]} for f in decision.get("attachments", [])],
                }, {}, flash=True)
                workload_only = edit.workload_only
                if workload_only or any((edit.subjects, edit.distribute_evenly, edit.total_hours,
                                         edit.total_credits, edit.hours_per_credit)):
                    proposal = apply_workload_edit(subjects, proposal, edit)
            except ValueError as error:
                notice = str(error) if not isinstance(error, ValidationError) else "Уточните числовые параметры нагрузки."
                workload_only = edit is None or edit.workload_only
            except (RuntimeError, APIStatusError, APIResponseValidationError):
                notice = "Не удалось обработать уточнение нагрузки. Сохранена предыдущая разбивка часов."
                workload_only = True
        update = {"consumed_review_id": state["version_id"], "review_iteration": state.get("review_iteration", 0) + 1,
                  "revision_baseline": state["course_program"], "status": "designing", "refresh_only": False,
                  "workload_only": workload_only, "workload_proposal": proposal, "workload_notice": notice}
        if workload_only:
            return update
        url = f"{settings.public_origin}/teacher?course={state['request_id']}&version={state['version_id']}"
        records = [source_record("teacher", url, "Правка преподавателя", decision["content"])]
        records.extend(source_record("teacher", url, file["name"], file["text"]) for file in decision.get("attachments", []))
        return {**update, "review_feedback": [*state.get("review_feedback", []), decision],
                "sources": [r.model_dump(mode="json") for r in records]}

    def guarded(name, node):
        async def execute(state: GraphState) -> dict:
            CALL_SCOPE.set(CallScope(course_id=state.get("request_id"), run_id=state.get("run_id"), stage=name))
            try:
                return await node(state)
            except GraphBubbleUp:
                raise
            except Exception as error:
                logging.getLogger(__name__).error("Этап %s курса %s: %s", name, state["request_id"], type(error).__name__)
                await store.set_status(state["request_id"], "failed", state["owner_user_id"])
                raise
        return execute

    graph = StateGraph(GraphState)
    graph.add_node("start", start)
    roles = ("business", "science", "books", "university")
    for role in roles:
        graph.add_node(role, guarded(role, agent(role)))
        graph.add_edge("start", role)
    nodes = {"relevance": relevance, "formalize": formalize, "university_audit": university_audit,
             "gap_analysis": gap_analysis, "prepare_interview": prepare_interview, "interview": interview,
             "factcheck": factcheck, "design": design, "business_audit": business_audit,
             "subject_rationale": subject_rationale, "persist_review": persist_review, "review": review}
    for name, node in nodes.items():
        graph.add_node(name, guarded(name, node))
    # Сохраняем ключ узла для возобновления сохранённых запусков.
    graph.add_node("literature", guarded("literature", university_literature), metadata={"agent": "University"})
    graph.add_edge(START, "start")
    graph.add_edge(list(roles), "relevance")
    graph.add_conditional_edges("relevance", lambda state: "formalize" if state["needs_update"] else END)
    graph.add_edge("formalize", "university_audit")
    graph.add_edge("university_audit", "gap_analysis")
    graph.add_conditional_edges("gap_analysis", lambda state: "prepare_interview"
                                if state["gap_report"]["gap_types"] and not state.get("interview_round") else "design")
    graph.add_conditional_edges("prepare_interview", lambda state: "design" if state.get("interview_round") else "interview")
    graph.add_conditional_edges("interview", lambda state: "design"
                                if state.get("closed_interview_id") == state["interrupt_id"] else "factcheck")
    graph.add_edge("factcheck", "formalize")
    graph.add_edge("design", "business_audit")
    graph.add_edge("business_audit", "subject_rationale")
    graph.add_edge("subject_rationale", "literature")
    graph.add_edge("literature", "persist_review")
    graph.add_edge("persist_review", "review")
    def after_review(state):
        if state["status"] == "approved":
            return END
        if state.get("workload_only"):
            return "persist_review"
        return "subject_rationale" if state.get("refresh_only") else "design"

    graph.add_conditional_edges("review", after_review)
    return graph.compile(checkpointer=checkpointer)
