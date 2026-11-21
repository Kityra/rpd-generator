import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Annotated, Any, Protocol, TypedDict
from uuid import NAMESPACE_URL, uuid5

from langgraph.graph import END, START, StateGraph
from langgraph.errors import GraphBubbleUp
from langgraph.types import interrupt
from pydantic import ValidationError

from auth.settings import Settings
from domain.schemas import (AppliedActivityModel, BusinessCourseAudit, CourseProgramModel, GapReport,
                            Grounded, INDEPENDENT_EXPERT_SOURCE_TYPES, InterviewTurn, Model, RelevanceDecision, SourceRecord, SubjectPlan,
                            TopicAssignments, VerificationReport, all_blocks, block_value)
from domain.skill_mapper import skill_mapper
from orchestration.call_log import (CALL_SCOPE, GENERATION_NOTES, CallScope, RecordingModelClient, classify_outcome,
                                    failure_note)
from orchestration.clients import EvidenceClient, ModelClient, NoActivityEvidence, source_record
from orchestration import prompts
from orchestration.draft import make_outline
from orchestration.gap_coverage import GapParameters, profile_coverage
from orchestration.interview_methods import load_interview_methods
from orchestration.interview_skills import FALLBACK, load_interview_skills, select_interview_method
from orchestration.interviewer_turn import (CATEGORY_SLOT, SLOT_CATEGORY, SUBMISSION_PLACEHOLDERS, InterviewerRequest,
                                            classify_user_turn, extract_interview_claims, interview_result,
                                            missing_profile_fields, scoped_decisions, verified_tool_links)
from orchestration.market_analysis import course_market, market_vacancy_ids
from orchestration.profession_boundaries import negative_scope, with_boundary_warnings
from orchestration.profession_pipeline import (ACTIVITY_SOURCE_TYPES, activity_for_design, apply_expert_turn,
                                               initial_profile, manual_profile, market_evidence, with_missing_slots)
from orchestration.profession_profile import ProfessionProfile
from orchestration.profession_verifier import normalized, verify_profession_profile
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
RETRYABLE_OUTCOMES = frozenset({"timeout", "circuit_open", "network_error", "unavailable"})
GAP_QUERY_LABELS = {"missing_tasks": "рабочие задачи и процессы", "missing_stack": "инструменты",
                    "missing_artifacts": "результаты работы", "missing_quality_criteria": "критерии качества",
                    "conflicting_evidence": "проверка утверждений по источникам"}


def is_truncated(error: Exception) -> bool:
    """Ответ модели оборван по длине: статус max_output_tokens или незакрытый JSON."""
    if classify_outcome(error) == "truncated":
        return True
    return isinstance(error, ValidationError) and any(item["type"] == "json_invalid" for item in error.errors())
BUSINESS_SOURCE_TYPES = frozenset({"Priklad", "HH", SOURCE}) | VACANCY_SOURCE_TYPES



VOLATILE_SOURCE_FIELDS = frozenset({"retrieved_at", "relevance_score"})
SOURCE_METADATA_FIELDS = ("parent_document_id", "publication_year")


def merge_records(previous: list[dict], additions: list[dict]) -> list[dict]:
    merged = {item["source_id"]: item for item in previous}
    for item in additions:
        if item["source_id"] in merged:
            # Один фрагмент может прийти по разным запросам с разной оценкой близости.
            original = merged[item["source_id"]]
            excluded = VOLATILE_SOURCE_FIELDS.union(SOURCE_METADATA_FIELDS)
            old = {key: value for key, value in original.items() if key not in excluded}
            new = {key: value for key, value in item.items() if key not in excluded}
            if old != new:
                raise ValueError("Конфликт источников с одинаковым ID.")
            merged[item["source_id"]] = {**original, **{key: item[key] for key in SOURCE_METADATA_FIELDS
                                                         if original.get(key) is None and item.get(key) is not None}}
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
    profession_profile: dict
    profession_boundaries: list[dict]
    profession_rag: dict[str, str]
    profession_market: dict
    profession_market_sample: int
    skill_mappings: dict
    profession_turns: dict
    raw_activity: dict
    acknowledged_profession_claims: list[str]
    interview_max_turns: int
    interview_continue: bool
    interview_redirect: str | None
    interview_handoff: bool
    course_program: dict
    baseline: dict | None
    revision_baseline: dict | None
    review_feedback: list[dict]
    review_iteration: int
    refresh_only: bool
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
                         vacancy_source_ids: list[str], generation: dict | None = None,
                         composition: dict | None = None, profession_profile: dict | None = None,
                         profession_market: dict | None = None, skill_mappings: dict | None = None) -> None:
        """Сохранить версию и уведомление."""

    async def finish_review(self, course_id: str, version_id: str, action: str, actor: str) -> None:
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
    # Первый черновик оставляет место для решения методиста о расширении программы.
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
                    or (bool(state.get("profession_profile")) and not state.get("applied_activity"))
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
    excluded.update({"profession_profile", "profession_rag", "profession_market", "profession_market_sample", "profession_boundaries", "skill_mappings",
                     "profession_turns", "raw_activity", "acknowledged_profession_claims",
                     "interview_max_turns", "interview_continue", "interview_redirect", "interview_handoff"})
    result = {key: value for key, value in state.items() if key not in excluded}
    if "course_meta" in result:
        result["course_meta"] = {key: value for key, value in result["course_meta"].items() if key != "workload"}
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
    if state.get("profession_profile"):
        result["course_meta"] = {k: v for k, v in result["course_meta"].items() if k not in {"activity_model", "market"}}
        for role in ("business", "science", "university", "books"):
            result.pop(f"{role}_report", None)
        result["profession_basis"] = profession_basis(state)
        result["sources"] = synthesis_sources(state)
    if scope := negative_scope(state.get("profession_boundaries", [])):
        result["adjacent_profession_exclusions"] = scope
    baseline = context(state).get("baseline")
    if baseline:
        result["previous_values"] = {key: block_value(block) for key, block in all_blocks(baseline.subjects).items()}
    return result


def profession_basis(state: GraphState) -> list[dict]:
    acknowledged = set(state.get("acknowledged_profession_claims", []))
    return [{key: claim[key] for key in ("category", "statement", "verification_status")}
            for claim in state.get("profession_profile", {}).get("claims", [])
            if claim["verification_status"] == "verified" or
            (claim["verification_status"] == "needs_review" and claim["claim_id"] in acknowledged)]


def synthesis_sources(state: GraphState) -> list[dict]:
    sources = state.get("sources", [])
    if not state.get("profession_profile"):
        return sources
    excerpts: dict[str, list[str]] = {}
    for claim in state["profession_profile"]["claims"]:
        if claim["verification_status"] != "verified":
            continue
        for ref in claim["evidence"]:
            if ref["source_type"] == "rag_chunk" and ref["judge_verdict"] == "supports":
                excerpts.setdefault(ref["source_id"], []).append(ref["quote_or_metric"])
    selected = []
    activity_ids = set(state.get("applied_activity", {}).get("source_ids", []))
    for source in sources:
        source_id = source["source_id"]
        if source["source_type"] in {"Books", "RPD", "Literature", "Crossref", "arXiv", "documentation", "teacher"}:
            selected.append(source)
        elif fragments := excerpts.get(source_id):
            selected.append({**source, "content": "\n".join(dict.fromkeys(fragments))})
        elif source["source_type"] == "expert" and source_id in activity_ids:
            selected.append(source)
    return selected


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
    sources = ({row["source_id"]: SourceRecord.model_validate(row) for row in synthesis_sources(state)}
               if state.get("profession_profile") else ctx["sources"])
    keys = {" ".join(name.casefold().split()) for name in names}
    plan = [item for item in suggestions if " ".join(item["name"].casefold().split()) in keys]
    terms = concepts(" ".join(names))
    tasks = state.get("applied_activity", {}).get("tasks", [])
    ranked = sorted(tasks, key=lambda task: -len(terms & concepts(" ".join([task["title"], *task.get("stack", [])]))))
    chosen = [task for task in ranked if terms & concepts(" ".join([task["title"], *task.get("stack", [])]))]
    chosen = (chosen or ranked[:3])[:SUBJECT_TASK_LIMIT]
    judge_unavailable = state.get("profession_profile", {}).get("judge_unavailable", False)
    scope = negative_scope(state.get("profession_boundaries", []))
    hint = outline_hint(state["course_meta"]["title"], names[0], suggestions, ctx) if len(names) == 1 and not judge_unavailable and not scope else None
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
    if scope:
        request["adjacent_profession_exclusions"] = scope
    if state.get("profession_profile"):
        request["profession_basis"] = [claim for claim in profession_basis(state)
                                       if terms & concepts(claim["statement"])]
    if state.get("review_feedback"):
        request["review_feedback"] = [{"content": item.get("content", "")} for item in state["review_feedback"]]
    return request, selected


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
                "course_meta": {key: value for key, value in state["course_meta"].items() if key != "workload"},
                "interview_answers": [], "review_feedback": [], "revision_baseline": None,
                "refresh_only": False,
                "interview_source_notices": [], "factcheck_source_notices": [], "draft_source_notices": [],
                "literature_source_notices": [], "composition_notices": [],
                "applied_activity": {}, "activity_origin": "", "gap_report": {}, "verification": {}, "auto_subjects": {},
                "profession_profile": {}, "profession_boundaries": [], "profession_turns": {}, "acknowledged_profession_claims": [],
                "interview_max_turns": 0, "interview_continue": False, "interview_redirect": None,
                "interview_handoff": False,
                "course_program": {}, "consumed_review_id": "", "closed_interview_id": "", "version_id": ""}

    def agent(role: str):
        async def run(state: GraphState) -> dict:
            # Запрос по смыслу: название курса, профессия и известные рабочие задачи, а не JSON заявки.
            query = search_query({**state["course_meta"], "manual_subjects": state.get("manual_subjects", [])},
                                 state.get("known_tasks", []))
            sources = await evidence.search(role, query, owner_scope=state["owner_user_id"])
            notices = getattr(sources, "notices", [])
            if not sources:
                return {f"{role}_report": {"findings": [], "source_notices": notices, "retrieved_source_ids": []},
                        "sources": []}
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
            report_data = {**report.model_dump(mode="json"), "source_notices": notices,
                           "retrieved_source_ids": [item.source_id for item in sources]}
            if role == "business":
                cards = [item.source_id for item in sources if item.source_type in VACANCY_SOURCE_TYPES]
                report_data["vacancy_source_ids"] = market_vacancy_ids(
                    {item.source_id: {"source_type": item.source_type, "url": item.url} for item in sources}, cards)
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
        records = []
        raw = {}
        all_sources = registry(state)
        reports = [state[f"{role}_report"] for role in ("business", "science", "university", "books")]
        current_ids = {key for report in reports for key in report.get("retrieved_source_ids", [])}
        if any("retrieved_source_ids" not in report for report in reports):
            current_ids.update(all_sources)  # Чекпойнты, записанные до сохранения ID текущего сбора.
        if snapshot.get("tasks"):
            raw, records = activity_from_snapshot(snapshot, settings.public_origin)
            all_sources.update({record.source_id: record for record in records})
            current_ids.update(record.source_id for record in records)
        sources = {key: source for key, source in all_sources.items()
                   if key in current_ids and source.source_type in ACTIVITY_SOURCE_TYPES}
        if sources and not raw:
            activity_input = {"course": state["course_meta"], "business": state["business_report"],
                              "science": state["science_report"], "sources": [s.model_dump(mode="json") for s in sources.values()]}
            try:
                instructions = prompts.ACTIVITY
                if any(source.source_type == SOURCE for source in sources.values()):
                    instructions += "\n" + SOURCE_GUIDANCE
                model = await llm.generate(AppliedActivityModel, instructions, activity_input, {"sources": sources})
                raw = model.model_dump(mode="json")
            except (NoActivityEvidence, RuntimeError, ValidationError):
                logging.getLogger(__name__).warning("Не сформированы кандидаты профиля курса %s.", state["request_id"])
        rag = {key: source.content for key, source in sources.items() if source.source_type != "ActivityModel"}
        for task, record in zip(snapshot.get("tasks", []), records):
            if excerpts := "\n".join(quote["text"] for quote in task.get("quotes", []) if quote["text"].strip()):
                rag[record.source_id] = excerpts
        sample, market = market_evidence(state["course_meta"], sources, state["business_report"].get("vacancy_source_ids", []))
        meta = state["course_meta"]
        profile = initial_profile(meta.get("profession") or meta["title"], raw, rag, sample, market)
        profile = await verify_profile(profile, rag, market, {})
        update = project_profile(state, profile, raw, [], all_sources)
        update["profession_boundaries"] = await sync_boundaries(state, update["profession_profile"])
        return {**update, "raw_activity": raw, "profession_rag": rag, "profession_market": market,
                "profession_market_sample": sample, "activity_origin": PROFESSION_MODEL if snapshot.get("tasks") else "",
                "sources": [*[record.model_dump(mode="json") for record in records], *update["sources"]]}

    def project_profile(state, profile, raw, acknowledged, sources):
        url = f"{settings.public_origin}/teacher?course={state['request_id']}"
        profile = with_missing_slots(profile, state["course_meta"])
        activity, records = activity_for_design(profile, raw, acknowledged, url, sources)
        return {"profession_profile": profile.model_dump(mode="json"), "applied_activity": activity,
                "acknowledged_profession_claims": acknowledged,
                "sources": [record.model_dump(mode="json") for record in records]}

    async def verify_profile(profile, rag, market, turns, *, reuse=False):
        return await verify_profession_profile(profile, rag, market, turns, llm, reuse_recorded_verdicts=reuse)

    async def sync_boundaries(state, profile):
        if getattr(type(store), "sync_boundaries", None):
            return await store.sync_boundaries(state["request_id"], profile)
        return state.get("profession_boundaries", [])

    async def current_boundaries(state):
        if getattr(type(store), "get_boundaries", None):
            profile = state.get("profession_profile", {})
            return await store.get_boundaries(state["request_id"], profile.get("profession_name"))
        return state.get("profession_boundaries", [])

    async def university_audit(state: GraphState) -> dict:
        if not state["applied_activity"] and not synthesis_sources(state):
            return {"auto_subjects": {"subjects": []}}
        if state.get("activity_origin") == PROFESSION_MODEL and state.get("applied_activity"):
            # Состав дисциплин решает руководитель программы: курс — это одна закреплённая дисциплина.
            ids = state["applied_activity"]["source_ids"]
            return {"auto_subjects": {"subjects": [
                {"name": name, "is_fundamental": False, "conflict_with": None,
                 "source_ids": ids} for name in state.get("manual_subjects", [])]}}
        result = await llm.generate(SubjectPlan, prompts.UNIVERSITY, payload(state), context(state))
        # Флаг ядра: только с источником RPD/Books и в пределах доли из config/analysis.yaml.
        subjects, notes = limit_fundamental(result.model_dump(mode="json")["subjects"], registry(state))
        return {"auto_subjects": {"subjects": subjects}, "composition_notices": notes}

    async def gap_analysis(state: GraphState) -> dict:
        await store.set_status(state["request_id"], "gap_analysis", state["owner_user_id"])
        snapshot = state.get("profession_profile")
        sources = registry(state)
        expert_origins = {f"turn:{idx}": answer["expert_user_id"]
                          for idx, answer in enumerate(state.get("interview_answers", []), 1)
                          if answer.get("expert_user_id")}
        coverage = (profile_coverage(ProfessionProfile.model_validate(snapshot), sources,
                                     state.get("profession_market", {}), GapParameters.from_settings(settings),
                                     BUSINESS_SOURCE_TYPES, expert_origins=expert_origins) if snapshot else None)
        if not state["applied_activity"]:
            report = GapReport(gap_types=["missing_tasks"], **(coverage or {"business_coverage": 0, "science_coverage": 0}),
                               questions=["Какую рабочую задачу вы решали недавно, что сделали и как проверили результат?"])
            return {"gap_report": report.model_dump(mode="json"), "status": "gap_analysis"}
        activity = AppliedActivityModel.model_validate(state["applied_activity"], context=context(state))
        if state.get("activity_origin") == PROFESSION_MODEL and not any(
            claim["verification_status"] == "needs_review" and claim["category"] == "work_task"
            for claim in state.get("profession_profile", {}).get("claims", [])
        ):
            # Задачи уже прошли экспертную валидацию модели профессии: интервью по курсу не нужно.
            report = GapReport(gap_types=[], questions=[], **(coverage or {
                "business_coverage": 1.0, "science_coverage": weighted_coverage(activity.tasks, sources, {"Crossref", "arXiv"})
            }))
            return {"gap_report": report.model_dump(mode="json"), "status": "gap_analysis"}
        count = len(activity.tasks)
        if coverage is None:
            coverage = {"business_coverage": weighted_coverage(activity.tasks, sources, BUSINESS_SOURCE_TYPES),
                        "science_coverage": weighted_coverage(activity.tasks, sources, {"Crossref", "arXiv"})}
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
        evidence_missing = (any(row["category"] == "work_task" and row["status"] == "critical_gap"
                                for row in coverage["formula_breakdown"]) if snapshot else
                            coverage["business_coverage"] < settings.business_coverage_threshold
                            or coverage["science_coverage"] < settings.science_coverage_threshold)
        if evidence_missing:
            gaps.append("conflicting_evidence")
            questions.append("Какие источники подтверждают применение задач на практике и их научную основу?")
        if state.get("verification") and not state.get("profession_profile"):
            for claim in state["verification"]["claims"]:
                if claim["verification_status"] == "needs_clarification":
                    gaps.append("conflicting_evidence")
                    questions.append(claim["clarification_question"])
        if any(claim["verification_status"] == "needs_review"
               for claim in state.get("profession_profile", {}).get("claims", [])):
            gaps.append("conflicting_evidence")
            questions.append("Какие предложенные рабочие сведения подтверждаются вашей практикой?")
        report = GapReport(gap_types=list(dict.fromkeys(gaps)), questions=questions,
                           **coverage)
        return {"gap_report": report.model_dump(mode="json"), "status": "gap_analysis"}

    async def prepare_interview(state: GraphState) -> dict:
        round_number = state.get("interview_round", 0)
        saved = state.get("profession_profile")
        accepted = round_number - len((saved or {}).get("rejected_noise_turns", []))
        handoff = state.get("interview_handoff", False)
        if round_number and not handoff and (not saved or accepted >= state.get("interview_max_turns", 0)):
            return {"status": "designing"}
        identifier = str(uuid5(NAMESPACE_URL, f"{state['request_id']}/{state['run_id']}/interview/{round_number}"))
        redirect = state.get("interview_redirect")
        methods = [] if redirect or handoff else await evidence.search(
            "interview", gap_query(state["gap_report"], GAP_QUERY_LABELS), owner_scope=state["owner_user_id"])
        profile = ProfessionProfile.model_validate(saved) if saved else None
        limit = 1
        slots = []
        tool_competence = None
        if profile:
            slots = (state["gap_report"].get("requested_slots", []) if redirect or handoff else
                     missing_profile_fields(profile, state["gap_report"]["gap_types"], state["applied_activity"]))
            if redirect:
                tool_competence = state["gap_report"].get("tool_competence")
            elif not handoff:
                sources = registry(state)
                ids = state.get("business_report", {}).get("vacancy_source_ids", [])
                cards = {key: sources[key].model_dump(mode="json") for key in ids if key in sources}
                market = course_market(state["course_meta"], {}, cards, ids, None, profile=saved)
                configured = skill_mapper().competences_with_tools
                tool_competence = next((row.competence_name for row in market.competences if row.critical
                    and not row.tools and not row.expert_tools and row.competence_name in configured), None)
                if tool_competence and "tool_stack" not in slots:
                    slots = [*slots, "tool_stack"]
            if not slots and not handoff:
                return {"status": "designing"}
            pending = [claim for claim in profile.claims if claim.verification_status == "needs_review"
                       and CATEGORY_SLOT[claim.category] in slots]
            method = profile.interview_method_used or select_interview_method(
                sample_size=state["profession_market_sample"], rag_count=len(state["profession_rag"]),
                has_unverified=any(ref.source_type in {"rag_chunk", "market_cluster"} for claim in pending
                                   for ref in claim.evidence),
                missing_lab_tasks=not any(task["artifacts"] and len(task["quality_criteria"]) >= 2
                                          for task in state["applied_activity"].get("tasks", [])),
                missing_slots=profile.missing_slots)
            local = load_interview_skills(settings.interview_skill_directory).get(method, FALLBACK)
            limit = state.get("interview_max_turns") or local.max_turns
            profile = profile.model_copy(update={"interview_method_used": local.id, "unresolved_slots": slots})
            local_skill = local.instructions
        else:
            local_skill = load_interview_methods(settings.interview_skill_directory, state["gap_report"]["gap_types"])
        skill = (local_skill + "\n\n" + "\n\n".join(item.content for item in methods))[:18000]
        notices = [*state.get("interview_source_notices", []), *getattr(methods, "notices", [])]
        request = {"course": state["course_meta"]["title"], "gaps": state["gap_report"]["questions"],
                   "interview_methods": skill}
        if profile:
            request.update(requested_slots=slots, tool_competence=tool_competence, evidence=[
                {"source_id": ref.source_id, "quote_or_metric": ref.quote_or_metric}
                for claim in profile.claims if claim.verification_status == "verified" for ref in claim.evidence
                if ref.judge_verdict == "supports"][:12])
        else:
            request["tasks"] = [t["title"] for t in state["applied_activity"].get("tasks", [])]
        if redirect or handoff:
            question = ("Лимит уточнений исчерпан. Оставшиеся поля переданы методисту." if handoff else redirect)
            question_origin = "methodologist_handoff" if handoff else "interview_clarification"
        else:
            try:
                turn = await llm.generate(InterviewTurn, prompts.INTERVIEW, request, {}, flash=True)
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
            question = turn.prompt
        if tool_competence:
            question += (f"\n\nДля компетенции «{tool_competence}» назовите инструмент, который вы применяете, "
                         "и опишите одну рабочую операцию с ним. Если специальный инструмент не нужен, так и укажите.")
        report = {**state["gap_report"], "prompt": question,
                  "question_origin": question_origin,
                  "previous_interview_id": state.get("interview_session_id") if round_number and not handoff else None,
                  "methodologist_handoff": handoff,
                  "method_source_ids": [item.source_id for item in methods]}
        if profile:
            report.update(profession_profile=profile.model_dump(mode="json"), interview_max_turns=limit,
                          profession_market=state["profession_market"],
                          requested_slots=slots, tool_competence=tool_competence,
                          interview_can_continue=not handoff and accepted + 1 < limit,
                          interview_turn=round_number + 1, review_candidate_ids=[claim.claim_id for claim in pending[:20]],
                          requires_manual_input=not (synthesis_sources(state) or state.get("manual_subjects") or
                                                     state.get("applied_activity")))
        interview_id = await store.prepare_interview(state["request_id"], identifier, report)
        await store.save_sources(state["request_id"], [*registry(state).values(), *methods])
        await store.set_status(state["request_id"], "waiting_for_expert", state["owner_user_id"])
        return {"interrupt_id": identifier, "interview_session_id": interview_id, "interview_max_turns": limit,
                **({"profession_profile": profile.model_dump(mode="json")} if profile else {}),
                "gap_report": report, "status": "waiting_for_expert", "interview_redirect": None,
                "interview_source_notices": list(dict.fromkeys(notices))}

    async def interview(state: GraphState) -> dict:
        answer = interrupt({"interrupt_id": state["interrupt_id"], "interview_session_id": state["interview_session_id"],
                            "prompt": state["gap_report"]["prompt"], "kind": "expert"})
        if not isinstance(answer, dict) or answer.get("interrupt_id") != state["interrupt_id"]:
            raise ValueError("Ответ не соответствует текущему интервью.")
        if answer.get("action") == "finish_without_answer":
            if answer.get("actor_user_id") != state["owner_user_id"]:
                raise PermissionError("Завершение интервью доступно владельцу курса.")
            update = {"closed_interview_id": state["interrupt_id"], "interview_continue": False}
            if not state.get("profession_profile"):
                return update
            profile = ProfessionProfile.model_validate(state["profession_profile"])
            acknowledged = [key for key in answer.get("acknowledged_claim_ids", [])
                            if any(c.claim_id == key and c.verification_status == "needs_review" for c in profile.claims)]
            manual = answer.get("manual_tasks", [])
            sources = registry(state)
            if manual or state.get("manual_subjects"):
                url = f"{settings.public_origin}/teacher?course={state['request_id']}"
                record = source_record("teacher", url + "&input=methodologist", "Ручные исходные данные методиста",
                                       "\n".join([*manual, *state.get("manual_subjects", [])]))
                sources[record.source_id] = record
                profile = manual_profile(profile, manual, record)
                manual_texts = {normalized(text) for text in manual}
                acknowledged = list(dict.fromkeys([*acknowledged, *(c.claim_id for c in profile.claims
                    if c.verification_status == "needs_review" and normalized(c.statement) in manual_texts)]))
                update["sources"] = [record.model_dump(mode="json")]
            if acknowledged:
                projection = project_profile(state, profile, state["raw_activity"], acknowledged, sources)
                projection["sources"] = [*update.get("sources", []), *projection["sources"]]
                return {**update, **projection}
            return update
        if state.get("interview_handoff"):
            raise PermissionError("Оставшиеся поля переданы методисту.")
        if not answer.get("content") and not state.get("profession_profile"):
            raise ValueError("Ответ эксперта пуст.")
        return {"interview_answers": [*state.get("interview_answers", []), answer],
                "interview_round": state.get("interview_round", 0) + 1,
                "interview_continue": bool(answer.get("continue_interview"))}

    async def factcheck(state: GraphState) -> dict:
        answer = state["interview_answers"][-1]
        saved_profile = state.get("profession_profile")
        if saved_profile:
            profile = ProfessionProfile.model_validate(saved_profile)
            gap = state["gap_report"]
            slots = gap.get("requested_slots")
            if slots is None:  # Сохранённое приглашение до появления списка полей.
                slots = missing_profile_fields(profile, gap["gap_types"], state["applied_activity"])
            request = InterviewerRequest(requested_slots=slots, question=gap["prompt"], tool_competence=gap.get("tool_competence"))
            decisions, form_claims = scoped_decisions(profile, request, answer.get("profession_decisions", []))
            reply = answer.get("content", "")
            if reply in SUBMISSION_PLACEHOLDERS:
                reply = "\n".join(item["text"][:2000] for item in answer.get("attachments", [])[:2])
            notes_quality = await classify_user_turn(request, reply, llm)
            if notes_quality != "valid_answer":
                reply = ""
            typed = "\n".join(item["statement"] for item in decisions if item["statement"])
            if typed and await classify_user_turn(request, typed, llm) != "valid_answer":
                decisions, form_claims = scoped_decisions(profile, request,
                    [item for item in decisions if item["action"] in {"confirm", "reject"}])
            classification = "valid_answer" if reply or decisions else "nonsense_or_offtopic"
            if classification == "nonsense_or_offtopic":
                result = interview_result(request, classification, {}, reply)
                noise_turns = [*profile.rejected_noise_turns, state["interview_round"]]
                retry = bool(slots and len(noise_turns) <= settings.max_invalid_redirects)
                handoff = not retry and not (synthesis_sources(state) or state.get("manual_subjects") or
                                             state.get("applied_activity"))
                updated = profile.model_copy(update={"rejected_noise_turns": noise_turns,
                                                     "unresolved_slots": result.unresolved_slots})
                await store.record_verification(state["interview_session_id"],
                    {"claims": [], "interviewer_result": result.model_dump(mode="json"), "redirected": retry})
                return {"profession_profile": updated.model_dump(mode="json"), "interview_continue": False,
                        "interview_redirect": result.clarification if retry else None, "interview_handoff": handoff}
            extracted = await extract_interview_claims(request, reply, llm)
        topic = " ".join(state["course_meta"]["title"].split())[:300]
        fresh = await evidence.search("factcheck", f"{topic}: рабочие задачи, инструменты, результаты и критерии качества",
                                      owner_scope=state["owner_user_id"])
        notices = getattr(fresh, "notices", [])
        combined = merge_records(state.get("sources", []), [item.model_dump(mode="json") for item in fresh])
        independent = {item["source_id"]: SourceRecord.model_validate(item) for item in combined
                       if item["source_type"] in INDEPENDENT_EXPERT_SOURCE_TYPES}
        reply_for_check = reply if saved_profile else answer.get("content", "")
        attachments_for_check = answer.get("attachments", []) if reply_for_check else []
        try:
            result = await llm.generate(VerificationReport, prompts.FACTCHECK,
                                        {"answer": {"content": reply_for_check, "attachments": attachments_for_check},
                                         "sources": [item.model_dump(mode="json") for item in independent.values()]},
                                        {"sources": independent}, flash=True)
            report = result.model_dump(mode="json")
        except (ValidationError, YandexModeratedResponse, ExternalUnavailable):
            report = {"claims": [], "assessment": "unavailable"}
        if saved_profile:
            turn_id = f"turn:{state['interview_round']}"
            contradicted = {normalized(item["claim"]) for item in report["claims"]
                            if item["verification_status"] == "rejected_hallucination"}
            rejected = {item["claim_id"] for item in decisions if item["action"] == "reject"}
            contradicted.update(normalized(c.statement) for c in profile.claims
                                if c.verification_status == "rejected" or c.claim_id in rejected)
            extracted = extracted.model_copy(update={"items": [item for item in extracted.items
                                                               if normalized(item.statement) not in contradicted]})
            result = interview_result(request, classification, extracted, reply, form_claims=form_claims)
            existing = {(CATEGORY_SLOT[item.category], normalized(item.statement)): item for item in profile.claims}
            form_keys = {(item.slot_name, normalized(item.statement)) for item in form_claims}
            decided = {item["claim_id"] for item in decisions}
            verbatim = set()
            for index, item in enumerate(result.extracted_claims, 1):
                key = item.slot_name, normalized(item.statement)
                old = existing.get(key)
                if key in form_keys or (old and (old.verification_status != "needs_review" or old.claim_id in decided)):
                    continue
                if old:
                    same = old.statement == item.statement
                    decisions.append({"claim_id": old.claim_id, "action": "confirm" if same else "edit",
                                      "statement": None if same else item.statement})
                    if same:
                        verbatim.add(old.claim_id)
                else:
                    identifier = f"extracted-{index}"
                    while identifier in decided:
                        identifier += "-fact"
                    decisions.append({"claim_id": identifier, "action": "add", "statement": item.statement,
                                      "category": SLOT_CATEGORY[item.slot_name]})
                    decided.add(identifier)
            profile, turn = apply_expert_turn(profile, decisions, turn_id, verbatim_confirmations=verbatim)
            profile = profile.model_copy(update={"unresolved_slots": result.unresolved_slots})
            turns = {**state.get("profession_turns", {}), turn_id: turn}
            profile = await verify_profile(profile, state["profession_rag"], state["profession_market"], turns, reuse=True)
            blocked = {normalized(item["claim"]) for item in report["claims"] if item["verification_status"] != "verified"}
            links = verified_tool_links(request, extracted, reply, profile, turn_id, state["interview_session_id"], blocked)
            if links:
                profile = profile.model_copy(update={"expert_tool_links": list(dict.fromkeys([*profile.expert_tool_links, *links]))})
                report["expert_tool_links"] = [link.model_dump(mode="json") for link in links]
            update = project_profile(state, profile, state["raw_activity"], state["acknowledged_profession_claims"],
                                     registry(state))
            update["profession_boundaries"] = await sync_boundaries(state, update["profession_profile"])
            report["interviewer_result"] = result.model_dump(mode="json")
            await store.record_verification(state["interview_session_id"], report)
            return {**update, "profession_turns": turns, "verification": report, "factcheck_source_notices": notices,
                    "sources": [*[item.model_dump(mode="json") for item in fresh], *update["sources"]]}
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
        boundaries = await current_boundaries(state)
        state = {**state, "profession_boundaries": boundaries}
        suggestions = planned_subjects(state)
        manual = {" ".join(name.casefold().split()): name for name in state.get("manual_subjects", [])}
        automatic = {" ".join(item["name"].casefold().split()): item["name"] for item in suggestions}
        names = {**automatic, **manual}
        ctx = context(state)
        if ctx["allow_incomplete_activity"] and not ctx.get("baseline") and len(names) > DRAFT_SUBJECT_BATCH_SIZE:
            return {**await design_batched(state, list(names.items()), suggestions, manual),
                    "profession_boundaries": boundaries}
        notes: list[str] = []
        GENERATION_NOTES.set(notes)
        program = await llm.generate(CourseProgramModel, prompts.DESIGN, payload(state), ctx)
        marked = mark_generated(program.model_dump(mode="json"), "model", "; ".join(notes) or None)
        return {"course_program": marked, "status": "designing",
                "profession_boundaries": boundaries}

    async def design_batched(state: GraphState, names: list[tuple[str, str]], suggestions: list[dict],
                             manual: dict[str, str]) -> dict:
        semaphore = asyncio.Semaphore(settings.draft_concurrency)
        delays = settings.retry_delays()
        manual_source = None
        if manual:
            url = f"{settings.public_origin}/teacher?course={state['request_id']}"
            manual_source = source_record("teacher", url, "Список дисциплин методиста", "\n".join(manual.values()))
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
        program = CourseProgramModel.model_validate(combined, context=context(state))
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
                regenerated = mark_generated(program.model_dump(mode="json"), "model")
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
        sources = registry(state)
        await store.save_sources(state["request_id"], list(sources.values()))
        boundaries = await current_boundaries(state)
        threshold = getattr(settings, "boundary_violation_threshold", 0.5)
        checked = with_boundary_warnings(state["course_program"], boundaries, threshold=threshold)
        program = CourseProgramModel.model_validate(checked, context=context(state))
        scheduled = state.get("mode") == "relevance"
        gaps = state["gap_report"]
        evidence_gaps = {key: gaps[key] for key in ("gap_types", "business_coverage", "science_coverage",
                                                   "formula_breakdown", "coverage_parameters") if key in gaps}
        composition = composition_report(state["course_program"], state["course_meta"], state.get("applied_activity"))
        vacancy_ids = state["business_report"].get("vacancy_source_ids", [])
        meta = state["course_meta"]
        market = course_market(meta, checked, {key: sources[key].model_dump(mode="json") for key in vacancy_ids if key in sources},
                               vacancy_ids, state["applied_activity"])
        token = CALL_SCOPE.set(replace(CALL_SCOPE.get(), stage="skill_mapping"))
        try:
            mappings = await skill_mapper().normalize_unknown(
                [row.skill for row in market.skills], meta.get("profession") or meta["title"], llm,
                previous=state.get("skill_mappings"))
        finally:
            CALL_SCOPE.reset(token)
        boundary_options = {"boundary_violation_threshold": threshold} if getattr(type(store), "get_boundaries", None) else {}
        await store.save_draft(state["request_id"], identifier, program, scheduled, state["applied_activity"],
                               evidence_gaps=evidence_gaps, source_notices=run_source_notices(state),
                               vacancy_source_ids=vacancy_ids, skill_mappings=mappings,
                               generation=llm.describe(), composition=composition,
                               profession_profile=state.get("profession_profile"), profession_market=state.get("profession_market"),
                               **boundary_options)
        return {"version_id": identifier, "skill_mappings": mappings,
                "course_program": checked, "profession_boundaries": boundaries,
                "status": "pending_update_review" if scheduled else "waiting_for_review"}

    async def review(state: GraphState) -> dict:
        decision = interrupt({"interrupt_id": state["version_id"], "prompt": "Утвердите курс или укажите правки.", "kind": "teacher"})
        if not isinstance(decision, dict) or decision.get("actor_user_id") != state["owner_user_id"]:
            raise PermissionError("Ревью доступно владельцу курса.")
        if decision.get("version_id") != state["version_id"]:
            raise ValueError("Решение относится к устаревшей версии.")
        action = decision.get("action")
        if action in {"approve", "reject"}:
            await store.finish_review(state["request_id"], state["version_id"], action, decision["actor_user_id"])
            return {"status": "approved", "consumed_review_id": state["version_id"]}
        if action == "refresh_materials":
            await store.set_status(state["request_id"], "collecting_literature", state["owner_user_id"])
            return {"consumed_review_id": state["version_id"], "refresh_only": True,
                    "review_iteration": state.get("review_iteration", 0) + 1,
                    "revision_baseline": state["course_program"], "status": "collecting_literature"}
        if action != "revise" or not decision.get("content"):
            raise ValueError("Укажите правку или решение ревью.")
        url = f"{settings.public_origin}/teacher?course={state['request_id']}&version={state['version_id']}"
        records = [source_record("teacher", url, "Правка методиста", decision["content"])]
        records.extend(source_record("teacher", url, file["name"], file["text"]) for file in decision.get("attachments", []))
        feedback = {"content": decision["content"], "attachments": decision.get("attachments", [])}
        return {"consumed_review_id": state["version_id"], "review_iteration": state.get("review_iteration", 0) + 1,
                "revision_baseline": state["course_program"], "status": "designing", "refresh_only": False,
                "review_feedback": [*state.get("review_feedback", []), feedback],
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
                                if state["gap_report"]["gap_types"] and not state.get("closed_interview_id") and
                                not (state.get("profession_profile", {}).get("judge_unavailable") and
                                     (synthesis_sources(state) or state.get("manual_subjects"))) and
                                (not state.get("interview_round") or (state.get("interview_continue") and
                                 state["interview_round"] - len(state.get("profession_profile", {}).get("rejected_noise_turns", []))
                                 < state.get("interview_max_turns", 0))) else "design")
    graph.add_conditional_edges("prepare_interview", lambda state: "interview"
                                if state["status"] == "waiting_for_expert" else "design")
    graph.add_conditional_edges("interview", lambda state: (
        "university_audit" if state.get("acknowledged_profession_claims") else "design")
        if state.get("closed_interview_id") == state["interrupt_id"] else "factcheck")
    def after_factcheck(state):
        if not state.get("profession_profile"):
            return "formalize"
        if state.get("interview_redirect") or state.get("interview_handoff"):
            return "prepare_interview"
        if state["interview_round"] in state["profession_profile"].get("rejected_noise_turns", []):
            return "design"
        return "university_audit"

    graph.add_conditional_edges("factcheck", after_factcheck)
    graph.add_edge("design", "business_audit")
    graph.add_edge("business_audit", "subject_rationale")
    graph.add_edge("subject_rationale", "literature")
    graph.add_edge("literature", "persist_review")
    graph.add_edge("persist_review", "review")
    def after_review(state):
        if state["status"] == "approved":
            return END
        return "subject_rationale" if state.get("refresh_only") else "design"

    graph.add_conditional_edges("review", after_review)
    return graph.compile(checkpointer=checkpointer)
