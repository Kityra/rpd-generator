"""Связь с генератором РПД (раздел 7.14).

Если дисциплина входит в программу и для её профессии есть модель деятельности, курс получает задачи из модели
(статусы 3, 4 и 6, покрытые дисциплиной), а Topic.activity_task_ids ссылается на них. Иначе остаётся прежняя
генерация модели деятельности LLM с пометкой «модель деятельности не проверена экспертами».
"""

from orchestration.evidence_cache import source_record

from activity.engine import ELIGIBLE, plural


UNVERIFIED_LABEL = "Модель деятельности не проверена экспертами: задачи курса предложены моделью по источникам курса."


def course_snapshot(models: dict[str, dict], links: dict[str, dict], *, program: dict, discipline: dict) -> dict:
    """Снимок задач для курса: подтверждённые и предварительные задачи целевых профессий, покрытые дисциплиной.

    models: {profession_id: содержимое последней версии модели}; links: покрытие дисциплины {"pid:task_id": link}.
    """
    tasks = []
    for key, link in sorted(links.items()):
        profession_id, task_id = key.split(":", 1)
        model = models.get(profession_id)
        task = (model or {}).get("tasks", {}).get(task_id)
        if task is None or task["status"] not in ELIGIBLE:
            continue
        functions = {function["id"]: function["name"] for function in model["functions"]}
        related = [model["knowledge"][item["id"]] for item in task["related"] if item["id"] in model["knowledge"]]
        tasks.append({
            "id": task_id, "title": task["name"], "profession_id": profession_id,
            "profession": model["profession"]["name"], "model_version": model.get("version"), "as_of": model["as_of"],
            "function_id": task["function_id"], "function": functions.get(task["function_id"], ""),
            "status": task["status"], "status_label": task["status_label"], "trend": task["trend"],
            "priority": task["priority"], "coverage": link["coverage"], "coverage_reason": link["reason"],
            "tools": sorted({item["name"] for item in related if item["level"] == "tool"}),
            "knowledge": sorted({item["name"] for item in related if item["level"] != "tool"}),
            "reason": [task["reason"]["title"], *task["reason"]["lines"]],
            "quotes": [{"text": quote["text"], "source_title": quote["source_title"], "date": quote["date"]}
                       for quote in task["quotes"][:3]],
        })
    tasks.sort(key=lambda task: (-task["priority"], task["title"]))
    used = sorted({task["profession_id"] for task in tasks})
    return {"professions": [{"id": pid, "name": models[pid]["profession"]["name"], "version": models[pid].get("version"),
                             "as_of": models[pid]["as_of"]} for pid in used],
            "is_demo": any(models[pid]["profession"].get("is_demo") for pid in used),
            "program_id": program["id"], "program": program["name"], "discipline_id": discipline["id"],
            "discipline": discipline["name"], "tasks": tasks}


def activity_from_snapshot(snapshot: dict, public_origin: str) -> tuple[dict, list]:
    """AppliedActivityModel и записи-источники из снимка модели (без вызова LLM)."""
    records, tasks, processes = [], [], {}
    for task in snapshot["tasks"]:
        lines = [f"Модель деятельности «{task['profession']}», версия {task['model_version']} на {task['as_of']}.",
                 f"Задача: {task['title']}. Функция: {task['function'] or 'не размечена'}. Статус: {task['status_label']}.",
                 *task["reason"], *(f"Цитата: «{quote['text']}» — {quote['source_title']} ({quote['date']})"
                                    for quote in task["quotes"])]
        url = f"{public_origin.rstrip('/')}/head?profession={task['profession_id']}&task={task['id']}"
        record = source_record("ActivityModel", url, f"Модель деятельности: {task['title']}"[:300], "\n".join(lines))
        records.append(record)
        tasks.append({"id": task["id"], "title": task["title"], "stack": task["tools"],
                      "artifacts": [], "quality_criteria": [], "typical_errors": [], "source_ids": [record.source_id]})
        process = processes.setdefault(task["function_id"], {"id": f"p-{task['function_id']}"[:60],
                                                             "title": task["function"] or "Задачи профессии",
                                                             "task_ids": [], "source_ids": []})
        process["task_ids"].append(task["id"])
        process["source_ids"].append(record.source_id)
    activity = {"tasks": tasks, "processes": list(processes.values()), "aspects": [],
                "source_ids": [record.source_id for record in records]}
    return activity, records


def origin_note(meta: dict) -> str:
    """Подпись в карточке курса: откуда взяты задачи деятельности."""
    snapshot = meta.get("activity_model") or {}
    if not snapshot.get("tasks"):
        return UNVERIFIED_LABEL
    professions = ", ".join(f"«{item['name']}» v{item['version']}" for item in snapshot["professions"])
    demo = " ДЕМО: синтетические данные." if snapshot.get("is_demo") else ""
    count = len(snapshot["tasks"])
    return (f"Задачи курса — из модели деятельности {professions} (программа «{snapshot['program']}»): "
            f"{count} {plural(count, 'проверенная задача', 'проверенные задачи', 'проверенных задач')}, "
            f"у каждой — источник и «почему здесь».{demo}")
