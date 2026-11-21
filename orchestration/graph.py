import asyncio
import json
from typing import Annotated, Any, Protocol, TypedDict
from uuid import NAMESPACE_URL, uuid5

from langgraph.graph import END, START, StateGraph
from langgraph.errors import GraphBubbleUp
from langgraph.types import interrupt
from pydantic import ValidationError

from auth.settings import Settings
from domain.schemas import (AppliedActivityModel, BusinessCourseAudit, CourseProgramModel, GapReport,
                            Grounded, INDEPENDENT_EXPERT_SOURCE_TYPES, InterviewTurn, Model, RelevanceDecision, SourceRecord, SubjectPlan,
                            VerificationReport, all_blocks, block_value)
from orchestration.clients import EvidenceClient, ModelClient, NoActivityEvidence, source_record
from orchestration import prompts
from orchestration.draft import make_outline
from orchestration.interview_methods import load_interview_methods
from orchestration.rabota_business import SOURCE, SOURCE_GUIDANCE
from orchestration.resilience import ExternalUnavailable
from run_rpd import YandexModeratedResponse


AUTO_DRAFT_SUBJECT_LIMIT = 12
DRAFT_SUBJECT_BATCH_SIZE = 4


def merge_records(previous: list[dict], additions: list[dict]) -> list[dict]:
    merged = {item["source_id"]: item for item in previous}
    for item in additions:
        if item["source_id"] in merged:
            old = {key: value for key, value in merged[item["source_id"]].items() if key != "retrieved_at"}
            new = {key: value for key, value in item.items() if key != "retrieved_at"}
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
    auto_subjects: dict
    sources: Annotated[list[dict], merge_records]
    business_report: dict
    science_report: dict
    books_report: dict
    university_report: dict
    interview_source_notices: list[str]
    factcheck_source_notices: list[str]
    draft_source_notices: list[str]
    applied_activity: dict
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
                               *state.get("factcheck_source_notices", []), *state.get("draft_source_notices", [])]))


class WorkflowStore(Protocol):
    async def set_status(self, course_id: str, status: str, owner: str) -> None:
        """Сохранить статус с проверкой владельца."""

    async def save_sources(self, course_id: str, sources: list[SourceRecord]) -> None:
        """Сохранить неизменяемые источники."""

    async def prepare_interview(self, course_id: str, interrupt_id: str, report: dict) -> str:
        """Идемпотентно создать интервью."""

    async def save_draft(self, course_id: str, version_id: str, program: CourseProgramModel,
                         scheduled: bool, activity: dict, *, evidence_gaps: dict, source_notices: list[str]) -> None:
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
                    "allow_incomplete_activity": bool(state.get("gap_report", {}).get("gap_types")),
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


def build_course_graph(settings: Settings, llm: ModelClient, evidence: EvidenceClient,
                       store: WorkflowStore, checkpointer: Any):
    async def start(state: GraphState) -> dict:
        if not state.get("run_id"):
            raise ValueError("Для запуска требуется устойчивый run_id.")
        await store.set_status(state["request_id"], "agents_running", state["owner_user_id"])
        return {"status": "agents_running", "interview_round": 0, "review_iteration": 0,
                "interview_answers": [], "review_feedback": [], "revision_baseline": None,
                "interview_source_notices": [], "factcheck_source_notices": [], "draft_source_notices": [],
                "applied_activity": {}, "gap_report": {}, "verification": {}, "auto_subjects": {},
                "course_program": {}, "consumed_review_id": "", "closed_interview_id": "", "version_id": ""}

    def agent(role: str):
        async def run(state: GraphState) -> dict:
            query = json.dumps(state["course_meta"], ensure_ascii=False, sort_keys=True)
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
            except ExternalUnavailable:
                report = AgentReport(findings=[])
                notices = [*notices, f"{role}: модель не сформировала сводку. Сохранены исходные материалы."]
            return {f"{role}_report": {**report.model_dump(mode="json"), "source_notices": notices},
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
        sources = {k: s for k, s in registry(state).items()
                   if s.source_type in {"Priklad", "HH", "Crossref", "arXiv", "expert", "documentation", SOURCE}}
        if not sources:
            return {"applied_activity": {}}
        activity_input = {"course": state["course_meta"], "business": state["business_report"],
                          "science": state["science_report"], "sources": [s.model_dump(mode="json") for s in sources.values()]}
        try:
            instructions = prompts.ACTIVITY
            if any(source.source_type == SOURCE for source in sources.values()):
                instructions += "\n" + SOURCE_GUIDANCE
            model = await llm.generate(AppliedActivityModel, instructions, activity_input, {"sources": sources})
        except NoActivityEvidence:
            return {"applied_activity": {}}
        return {"applied_activity": model.model_dump(mode="json")}

    async def university_audit(state: GraphState) -> dict:
        if not state["applied_activity"]:
            return {"auto_subjects": {"subjects": []}}
        result = await llm.generate(SubjectPlan, prompts.UNIVERSITY, payload(state), context(state))
        return {"auto_subjects": result.model_dump(mode="json")}

    async def gap_analysis(state: GraphState) -> dict:
        await store.set_status(state["request_id"], "gap_analysis", state["owner_user_id"])
        if not state["applied_activity"]:
            report = GapReport(gap_types=["missing_tasks"], business_coverage=0, science_coverage=0,
                               questions=["Какую рабочую задачу вы решали недавно, что сделали и как проверили результат?"])
            return {"gap_report": report.model_dump(mode="json"), "status": "gap_analysis"}
        activity = AppliedActivityModel.model_validate(state["applied_activity"], context=context(state))
        sources = registry(state)
        count = len(activity.tasks)
        def coverage(types: set[str]) -> float:
            return sum(any(sources[key].source_type in types for key in task.source_ids)
                       for task in activity.tasks) / max(count, 1)
        business = coverage({"HH", "Priklad", SOURCE})
        science = coverage({"Crossref", "arXiv"})
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
        methods = await evidence.search("interview", json.dumps(state["gap_report"], ensure_ascii=False),
                                        owner_scope=state["owner_user_id"])
        local_skill = load_interview_methods(settings.interview_skill_directory, state["gap_report"]["gap_types"])
        skill = (local_skill + "\n\n" + "\n\n".join(item.content for item in methods))[:18000]
        try:
            turn = await llm.generate(InterviewTurn, prompts.INTERVIEW,
                                      {"course": state["course_meta"]["title"], "gaps": state["gap_report"]["questions"],
                                       "tasks": [t["title"] for t in state["applied_activity"].get("tasks", [])],
                                       "interview_methods": skill, "history": [a["content"] for a in state.get("interview_answers", [])]},
                                      context(state), flash=True)
            question_origin = "interviewer"
        except ValidationError:
            # После двух невалидных ответов задаём вопрос по реально обнаруженному пробелу.
            for question in reversed(state["gap_report"]["questions"]):
                try:
                    turn = InterviewTurn(prompt=question)
                except ValidationError:
                    continue
                break
            else:
                raise
            question_origin = "gap_report"
        report = {**state["gap_report"], "prompt": turn.prompt,
                  "question_origin": question_origin,
                  "previous_interview_id": state.get("interview_session_id") if round_number else None,
                  "method_source_ids": [item.source_id for item in methods]}
        notices = list(dict.fromkeys([*state.get("interview_source_notices", []), *getattr(methods, "notices", [])]))
        interview_id = await store.prepare_interview(state["request_id"], identifier, report)
        await store.save_sources(state["request_id"], [*registry(state).values(), *methods])
        await store.set_status(state["request_id"], "waiting_for_expert", state["owner_user_id"])
        return {"interrupt_id": identifier, "interview_session_id": interview_id,
                "gap_report": report, "status": "waiting_for_expert", "interview_source_notices": notices}

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
        program = await llm.generate(CourseProgramModel, prompts.DESIGN, payload(state), ctx)
        return {"course_program": program.model_dump(mode="json"), "status": "designing"}

    async def design_batched(state: GraphState, names: list[tuple[str, str]], suggestions: list[dict],
                             manual: dict[str, str]) -> dict:
        semaphore = asyncio.Semaphore(3)
        manual_source = None
        if manual:
            url = f"{settings.public_origin}/teacher?course={state['request_id']}"
            manual_source = source_record("teacher", url, "Список дисциплин преподавателя", "\n".join(manual.values()))
            state = {**state, "sources": merge_records(state.get("sources", []), [manual_source.model_dump(mode="json")])}

        async def generate_batch(offset: int) -> tuple[CourseProgramModel, bool]:
            batch = dict(names[offset:offset + DRAFT_SUBJECT_BATCH_SIZE])
            manual_subjects = [name for key, name in batch.items() if key in manual]
            batch_state = {**state, "manual_subjects": manual_subjects,
                           "course_meta": {**state["course_meta"], "manual_subjects": manual_subjects},
                           "auto_subjects": {"subjects": [item for item in suggestions
                                             if " ".join(item["name"].casefold().split()) in batch]}}
            ctx = {**context(batch_state), "required_subject_names": set(batch)}
            if manual_source:
                ctx["manual_subject_source_id"] = manual_source.source_id
            request = {**payload(batch_state), "batch_subjects": list(batch.values())}
            try:
                async with semaphore:
                    program = await llm.generate(CourseProgramModel, prompts.DESIGN, request, ctx)
                return program, False
            except (ValidationError, RuntimeError):
                return make_outline(state["course_meta"]["title"], list(batch.values()), suggestions, ctx), True

        async with asyncio.TaskGroup() as tasks:
            batches = [tasks.create_task(generate_batch(offset))
                       for offset in range(0, len(names), DRAFT_SUBJECT_BATCH_SIZE)]
        completed = [batch.result() for batch in batches]
        programs = [program for program, _ in completed]
        subjects = []
        for number, program in enumerate(programs, 1):
            for raw in program.subjects:
                subject = raw.model_dump(mode="json")
                for block in [subject, *subject["topics"], *(a for topic in subject["topics"] for a in topic["assignments"])]:
                    block["id"] = f"b{number}-{block['id']}"
                subjects.append(subject)
        combined = {"title": state["course_meta"]["title"], "subjects": subjects, "business_audit": None,
                    "source_ids": list(dict.fromkeys(key for program in programs for key in program.source_ids))}
        program = CourseProgramModel.model_validate(combined, context=context(state))
        notices = ["Часть дисциплин собрана из сохранённых материалов. Проверьте темы и учебные источники."] if any(
            used_outline for _, used_outline in completed) else []
        return {"course_program": program.model_dump(mode="json"), "status": "designing",
                "draft_source_notices": notices, "sources": [manual_source.model_dump(mode="json")] if manual_source else []}

    async def business_audit(state: GraphState) -> dict:
        audit_input = payload(state)
        batched = state["gap_report"]["gap_types"] and len(state["course_program"]["subjects"]) > DRAFT_SUBJECT_BATCH_SIZE
        if batched:
            program = state["course_program"]
            audit_input["course_program"] = {"title": program["title"], "subjects": [
                {"title": subject["title"], "topics": [topic["title"] for topic in subject["topics"]]}
                for subject in program["subjects"]]}
        try:
            audit = await llm.generate(BusinessCourseAudit, prompts.BUSINESS_AUDIT, audit_input, context(state), flash=True)
        except (ValidationError, RuntimeError):
            if not batched:
                raise
            notices = [*state.get("draft_source_notices", []), "Рыночная оценка пока недоступна."]
            return {"draft_source_notices": list(dict.fromkeys(notices))}
        updated = {**state["course_program"], "business_audit": audit.model_dump(mode="json")}
        if audit.suggested_changes and not batched:
            program = await llm.generate(CourseProgramModel, prompts.DESIGN,
                                        {**payload(state), "business_enrichment": audit.model_dump(mode="json")}, context(state))
            audit = await llm.generate(BusinessCourseAudit, prompts.BUSINESS_AUDIT,
                                       {**payload(state), "course_program": program.model_dump(mode="json")},
                                       context(state), flash=True)
            updated = {**program.model_dump(mode="json"), "business_audit": audit.model_dump(mode="json")}
        checked = CourseProgramModel.model_validate(updated, context=context(state))
        return {"course_program": checked.model_dump(mode="json")}

    async def persist_review(state: GraphState) -> dict:
        identifier = str(uuid5(NAMESPACE_URL, f"{state['request_id']}/{state['run_id']}/version/{state.get('review_iteration', 0)}"))
        await store.save_sources(state["request_id"], list(registry(state).values()))
        program = CourseProgramModel.model_validate(state["course_program"], context=context(state))
        scheduled = state.get("mode") == "relevance"
        gaps = state["gap_report"]
        evidence_gaps = {key: gaps[key] for key in ("gap_types", "business_coverage", "science_coverage") if key in gaps}
        await store.save_draft(state["request_id"], identifier, program, scheduled, state["applied_activity"],
                               evidence_gaps=evidence_gaps, source_notices=run_source_notices(state))
        return {"version_id": identifier, "status": "pending_update_review" if scheduled else "waiting_for_review"}

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
        if action != "revise" or not decision.get("content"):
            raise ValueError("Укажите правку или решение ревью.")
        url = f"{settings.public_origin}/teacher?course={state['request_id']}&version={state['version_id']}"
        records = [source_record("teacher", url, "Правка преподавателя", decision["content"])]
        records.extend(source_record("teacher", url, file["name"], file["text"]) for file in decision.get("attachments", []))
        return {"consumed_review_id": state["version_id"], "review_feedback": [*state.get("review_feedback", []), decision],
                "review_iteration": state.get("review_iteration", 0) + 1,
                "revision_baseline": state["course_program"], "sources": [r.model_dump(mode="json") for r in records],
                "status": "designing"}

    graph = StateGraph(GraphState)
    graph.add_node("start", start)
    roles = ("business", "science", "books", "university")
    for role in roles:
        graph.add_node(role, agent(role))
        graph.add_edge("start", role)
    nodes = {"relevance": relevance, "formalize": formalize, "university_audit": university_audit,
             "gap_analysis": gap_analysis, "prepare_interview": prepare_interview, "interview": interview,
             "factcheck": factcheck, "design": design, "business_audit": business_audit,
             "persist_review": persist_review, "review": review}
    def guarded(node):
        async def execute(state: GraphState) -> dict:
            try:
                return await node(state)
            except GraphBubbleUp:
                raise
            except Exception:
                await store.set_status(state["request_id"], "failed", state["owner_user_id"])
                raise
        return execute

    for name, node in nodes.items():
        graph.add_node(name, guarded(node))
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
    graph.add_edge("business_audit", "persist_review")
    graph.add_edge("persist_review", "review")
    graph.add_conditional_edges("review", lambda state: END if state["status"] == "approved" else "design")
    return graph.compile(checkpointer=checkpointer)
