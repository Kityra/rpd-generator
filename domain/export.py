import json
from html import escape
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from jsonschema import Draft202012Validator
from sympy import Integer, Rational

HOURS = {"lectures": "Лекции", "practice": "Практика", "labs": "Лабораторные", "independent": "Самостоятельная работа"}
REASONS = {"market_demand": "Спрос работодателей", "scientific_trend": "Научные результаты",
           "expert_practice": "Практика эксперта", "fundamental_core": "Фундаментальная подготовка",
           "teacher_review": "Правка преподавателя", "scheduled_relevance_update": "Плановая проверка актуальности"}


def used_sources(value):
    if isinstance(value, list):
        for item in value:
            yield from used_sources(item)
    elif isinstance(value, dict):
        yield from value.get("source_ids", ())
        for child in value.values():
            yield from used_sources(child)


def workload(values):
    if any(type(values.get(key)) is not int or not 0 <= values[key] <= 10000 for key in HOURS):
        raise ValueError("Введите целые часы от 0 до 10 000.")
    rate = values.get("hours_per_credit")
    if type(rate) is not int or not 1 <= rate <= 1000:
        raise ValueError("Укажите число часов в зачётной единице.")
    total = sum(Integer(values[key]) for key in HOURS)
    if not total:
        raise ValueError("Суммарная нагрузка должна быть больше нуля.")
    return {"totalHours": int(total), "totalCredits": float(Rational(total, rate))}


def source_link(source):
    title, url = escape(source["title"]), source["url"]
    if url.startswith(("https://", "http://")) and source["source_type"] not in {"expert", "teacher"}:
        return f'<a href="{escape(url, quote=True)}">{title}</a>'
    return title


def block_html(block, sources, level=2):
    title = escape(block["title"])
    changed = block["change_status"] in {"added", "modified"}
    if changed:
        status = "Добавлено" if block["change_status"] == "added" else "Изменено"
        title = f'<span style="color:#b35400;font-weight:600">{title} · {status}</span>'
    parts = [f"<h{level}>{title}</h{level}>"]
    if block.get("is_fundamental"):
        parts.append("<p>Фундаментальное ядро</p>")
    if block.get("conflict_with"):
        parts.append("<p>Пересечение с дисциплиной «" + escape(block["conflict_with"]) + "»</p>")
    if block.get("previous_value"):
        parts.append("<p>Прежняя формулировка — " + escape(block["previous_value"]) + "</p>")
    why = block["justification"]
    parts.append("<p>" + REASONS[why["reason_type"]] + " — " + escape(why["evidence_summary"]) + " · " +
                 ", ".join(source_link(sources[k]) for k in why["source_ids"]) + "</p>")
    for ref in block["learning_references"]:
        parts.append("<p>" + source_link(sources[ref["source_id"]]) + " — " + escape(ref["locator"]) +
                     (" · " + escape(ref["bibliographic_excerpt"]) if ref.get("bibliographic_excerpt") else "") + "</p>")
    if "instructions" in block:
        tag = '<p style="color:#b35400">' if changed else "<p>"
        parts.extend(tag + escape(s) + "</p>" for s in (
            block["instructions"], "Результат — " + ", ".join(block["expected_artifacts"]),
            "Критерии — " + ", ".join(block["quality_criteria"])))
    for child in block.get("topics", []) + block.get("assignments", []):
        parts.append(block_html(child, sources, level + 1))
    return "\n".join(parts)


def export_course(meta, version, sources):
    if version.status != "approved":
        raise ValueError("Экспорт доступен после утверждения версии.")
    content = version.content
    program = content["program"]
    notices = content.get("source_notices", [])
    notices_html = ""
    if notices:
        items = "".join(f"<li>{escape(note)}</li>" for note in notices)
        notices_html = f"<h2>Примечания об источниках</h2><ul>{items}</ul>"
    validator = Draft202012Validator(json.loads((Path(__file__).resolve().parents[1] / "final_schema.json").read_text(encoding="utf-8")))
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        def write_json(path, data):
            archive.writestr(path, json.dumps(data, ensure_ascii=False, indent=2))
        write_json("course.json", content)
        write_json("sources.json", {k: sources[k] for k in sorted(set(used_sources(content)))})
        for index, subject in enumerate(program["subjects"], 1):
            tasks = [a for t in subject["topics"] for a in t["assignments"]]
            refs = [ref for b in [subject, *subject["topics"], *tasks] for ref in b["learning_references"]]
            reading = {r["source_id"]: {"title": r.get("bibliographic_excerpt") or sources[r["source_id"]]["title"],
                                       "url": sources[r["source_id"]]["url"]} for r in refs}
            document = {"documentType": "Рабочая программа дисциплины", "title": subject["title"],
                        "metadata": {"institution": meta["institution"], "academicYear": str(meta["year"]),
                                     "version": str(version.revision)}, "discipline": {"name": subject["title"]},
                        "assessmentMaterials": [{"title": a["title"], "instructions": a["instructions"],
                                                 "criteria": a["quality_criteria"]} for a in tasks],
                        "readingList": list(reading.values()), "subject": subject, "businessAudit": program["business_audit"]}
            load = content.get("workloads", {}).get(subject["id"])
            load_text = ""
            if notices:
                document["sourceNotes"] = notices
            if load:
                document["workload"] = workload(load)
                document["workloadInputs"] = load
                load_text = f"<p>Нагрузка — {document['workload']['totalHours']} ч., {document['workload']['totalCredits']:g} ЗЕ</p>"
            validator.validate(document)
            directory = f"rpd/{index:02}"
            write_json(f"{directory}/rpd.json", document)
            archive.writestr(f"{directory}/rpd.html", '<!doctype html><meta charset="utf-8"><title>' +
                             escape(subject["title"]) + "</title>" + load_text + block_html(subject, sources) + notices_html)
    return buffer.getvalue()
