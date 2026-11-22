"""Черновые блоки курса по сохранённым источникам и модели деятельности."""
import re

from domain.schemas import CourseProgramModel, SourceRecord


def keywords(text: str) -> set[str]:
    return {word for word in re.findall(r"\w+", text.casefold()) if len(word) >= 3}


def make_outline(title: str, names: list[str], suggestions: list[dict], ctx: dict) -> CourseProgramModel:
    sources: dict[str, SourceRecord] = ctx["sources"]
    by_name = {" ".join(item["name"].casefold().split()): item for item in suggestions}
    manual = {" ".join(name.casefold().split()) for name in ctx["manual_subjects"]}
    activity = ctx.get("activity")
    subjects = []
    for number, name in enumerate(names, 1):
        key = " ".join(name.casefold().split())
        selection = by_name.get(key, {})
        terms = keywords(name)
        matched = [source for source in sources.values() if source.source_type != "Interview"
                   and terms & keywords(source.title)]
        weights = ctx.get("source_weights")
        if weights and weights.customized:
            matched.sort(key=weights.for_source, reverse=True)
        refs = [*selection.get("source_ids", []), *(source.source_id for source in matched[:2])]
        if key in manual and ctx.get("manual_subject_source_id"):
            refs.append(ctx["manual_subject_source_id"])
        if not refs:
            raise ValueError(f"Для дисциплины «{name}» нет подтверждающих материалов.")
        tasks = [task for task in activity.tasks if terms & keywords(" ".join([task.title, *task.stack]))] if activity else []
        task = max(tasks, key=lambda item: len(terms & keywords(item.title)) * (
            weights.for_references(item.source_ids, sources) if weights else 1), default=None)
        refs = list(dict.fromkeys([*refs, *(task.source_ids if task else [])]))
        if weights and weights.customized:
            refs.sort(key=lambda source_id: weights.for_source(sources[source_id]), reverse=True)
        learning = []
        for source_id in refs:
            source = sources[source_id]
            if source.source_type not in {"Crossref", "arXiv", "documentation"}:
                continue
            excerpt = " ".join(source.content.split())[:120].strip()
            if len(excerpt) >= 5:
                learning.append({"source_id": source_id, "source_ids": [source_id], "locator": excerpt,
                                 "bibliographic_excerpt": None})
                break
        grounded = {
            "source_ids": refs, "change_status": "added", "previous_value": None,
            "is_fundamental": bool(selection.get("is_fundamental")),
            "locked": key in manual or bool(selection.get("is_fundamental")),
            "justification": {"reason_type": "fundamental_core" if selection.get("is_fundamental") else "scientific_trend",
                              "evidence_summary": "Тема предложена для проверки методистом по приложенным материалам.",
                              "source_ids": refs},
            "learning_references": learning,
        }
        assignments = []
        if task and task.artifacts and len(task.quality_criteria) >= 2:
            assignments.append({**grounded, "id": f"assignment-{number}", "title": f"Разбор задачи «{task.title}»",
                                "instructions": "Разберите рабочую задачу по источникам и объясните, как проверяется результат.",
                                "expected_artifacts": task.artifacts, "quality_criteria": task.quality_criteria})
        subjects.append({**grounded, "id": f"subject-{number}", "title": name, "origin": "manual" if key in manual else "auto",
                         "status": "locked" if grounded["locked"] else "suggested", "conflict_with": selection.get("conflict_with"),
                         "topics": [{**grounded, "id": f"topic-{number}", "title": task.title if task else f"Основные понятия дисциплины «{name}»",
                                     "activity_task_ids": [task.id] if task else [], "assignments": assignments}]})
    return CourseProgramModel.model_validate({
        "title": title, "subjects": subjects, "business_audit": None,
        "source_ids": list(dict.fromkeys(source_id for subject in subjects for source_id in subject["source_ids"])),
    }, context=ctx)
