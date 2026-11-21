"""Build the evidence packet for selecting subjects in the University stage."""

import re

from domain.schemas import SourceRecord


COURSE_FIELDS = ("title", "institution", "year", "program", "expand_subjects")
ROLES = ("university", "books", "science", "business")
SOURCE_LIMITS = {"RPD": 4, "Books": 4, "Crossref": 2, "arXiv": 1,
                 "Priklad": 2, "HH": 2, "RabotaRu": 2, "documentation": 1, "expert": 2}
EXCERPT_SIZE = 640


def _terms(text: str) -> set[str]:
    words = re.findall(r"[^\W_][\w+#-]*", text)
    return {word.casefold() for word in words if len(word) >= 4 or (len(word) >= 2 and word.isupper())}


def _excerpt(text: str, terms: set[str]) -> str:
    compact = " ".join(text.split())
    if len(compact) <= EXCERPT_SIZE:
        return compact
    # Look beyond the start of a chunk while keeping any returned excerpt contiguous.
    windows = (compact[offset:offset + EXCERPT_SIZE]
               for offset in range(0, len(compact), EXCERPT_SIZE // 2))
    best = max(windows, key=lambda excerpt: len(terms & _terms(excerpt)))
    return best


def build_university_request(state: dict, registry: dict[str, SourceRecord]) -> tuple[dict, dict[str, SourceRecord]]:
    course = {field: state["course_meta"][field] for field in COURSE_FIELDS if field in state["course_meta"]}
    manual = state.get("manual_subjects", [])
    activity = state["applied_activity"]
    tasks = activity.get("tasks", [])[:20]
    reports = {role: state.get(f"{role}_report", {}) for role in ROLES}

    # Prefer the chunks used by University, the specialists' findings and the current
    # activity model. Retain a quota per source type so one collection cannot crowd out the others.
    current_rpd = reports["university"].get("rpd_source_ids", [])
    findings = [item for report in reports.values() for item in report.get("findings", [])[:4]]
    priority = dict.fromkeys([
        *current_rpd, *(key for item in findings for key in item["source_ids"]),
        *(key for task in tasks for key in task["source_ids"]), *reversed(registry),
    ])
    counts: dict[str, int] = {}
    selected = {}
    for key in priority:
        source = registry.get(key)
        if source is None:
            continue
        kind = source.source_type
        count = counts.get(kind, 0)
        if count < SOURCE_LIMITS.get(kind, 0):
            selected[key] = source
            counts[kind] = count + 1

    terms = _terms(" ".join([course.get("title", ""), course.get("program", ""), *manual,
                             *(task["title"] for task in tasks)]))
    sources = [{"source_id": key, "source_type": source.source_type, "title": source.title[:240],
                "retrieved_at": source.retrieved_at.isoformat(), "content": _excerpt(source.content, terms)}
               for key, source in selected.items()]

    def visible_ids(item: dict) -> list[str]:
        return [key for key in item.get("source_ids", []) if key in selected]

    task_ids = {task["id"] for task in tasks}
    activity_summary = {
        "tasks": [{"id": task["id"], "title": task["title"][:240],
                   "stack": [item[:100] for item in task.get("stack", [])[:4]],
                   "artifacts": [item[:120] for item in task.get("artifacts", [])[:3]],
                   "source_ids": visible_ids(task)} for task in tasks],
        "processes": [{"title": process["title"][:240],
                       "task_ids": [key for key in process["task_ids"] if key in task_ids],
                       "source_ids": visible_ids(process)} for process in activity.get("processes", [])[:8]],
        "aspects": [{"aspect_type": aspect["aspect_type"], "description": aspect["description"][:260],
                     "task_ids": [key for key in aspect["task_ids"] if key in task_ids],
                     "source_ids": visible_ids(aspect)} for aspect in activity.get("aspects", [])[:8]],
    }
    request = {"course_meta": course, "manual_subjects": manual, "applied_activity": activity_summary,
               "sources": sources, "rpd_lookup": reports["university"].get("rpd_lookup")}
    for role, report in reports.items():
        request[f"{role}_report"] = [
            {"summary": item["summary"][:350], "source_ids": ids}
            for item in report.get("findings", [])[:4] if (ids := visible_ids(item))
        ]
    if baseline := state.get("baseline"):
        request["baseline_subjects"] = [
            {field: subject[field] for field in ("title", "is_fundamental", "locked")}
            for subject in baseline["subjects"]
        ]
    if subjects := state.get("auto_subjects", {}).get("subjects", []):
        request["previous_subjects"] = [
            {"name": subject["name"], "is_fundamental": subject["is_fundamental"],
             "source_ids": visible_ids(subject)} for subject in subjects[:20]
        ]
    return request, selected
