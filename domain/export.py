import json
from html import escape
from io import BytesIO
from pathlib import Path
from urllib.parse import urlsplit
from zipfile import ZIP_DEFLATED, ZipFile

from jsonschema import Draft202012Validator
from domain.workload import HOURS, workload

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


def is_yandex_file(source: dict) -> bool:
    url = urlsplit(source["url"])
    return url.hostname == "ai.api.cloud.yandex.net" and url.path.startswith("/v1/files/")


def public_source_url(source: dict) -> str | None:
    url = source["url"]
    if is_yandex_file(source) or source["source_type"] in {"expert", "teacher"}:
        return None
    if url.startswith(("https://", "http://")):
        return url
    return None


def material_key(source: dict, source_id: str) -> tuple[str, str]:
    if is_yandex_file(source):
        return "file", " ".join(source["title"].casefold().split())
    if source["source_type"] == "Literature":
        for line in source["content"].splitlines():
            if line.startswith(("DOI: ", "ISBN: ")):
                identifier, _, value = line.partition(": ")
                return identifier, value.casefold().strip()
    if url := public_source_url(source):
        return "link", url
    return "record", source_id


def source_html(source: dict, *, number: int | None = None, fragments: list[str] | None = None) -> str:
    prefix = f"{number}. " if number is not None else ""
    title = prefix + escape(source["title"])
    if is_yandex_file(source):
        texts = list(dict.fromkeys(fragments if fragments is not None else [source["content"]]))
        parts = [f'<details><summary style="cursor:pointer;color:#2459ba">{title}</summary>']
        for index, text in enumerate(texts, 1):
            label = "Сохранённый фрагмент" if len(texts) == 1 else f"Сохранённый фрагмент {index}"
            parts.append(f"<p><strong>{label}</strong></p>")
            parts.append(f'<div style="white-space:pre-wrap;overflow-wrap:anywhere">{escape(text)}</div>')
        return "\n".join([*parts, "</details>"])
    if url := public_source_url(source):
        title = f'<a href="{escape(url, quote=True)}">{title}</a>'
    if source["source_type"] == "Literature":
        for line in source["content"].splitlines():
            if line.startswith("PDF: "):
                if pdf := public_source_url({**source, "url": line.removeprefix("PDF: ")}):
                    title += f' · <a href="{escape(pdf, quote=True)}">PDF</a>'
                break
    return f"<p>{title}</p>"


def rationale_report_html(program: dict, sources: dict) -> str:
    parts = []
    for subject in program["subjects"]:
        parts.append(f"<h2>{escape(subject['title'])}</h2>")
        rationale = subject.get("inclusion_rationale")
        if not rationale:
            parts.append("<p>В базе не подтверждено обоснование включения дисциплины в курс.</p>")
            continue
        parts.append(f"<p>{escape(rationale['explanation'])}</p>")
        for item in rationale["evidence"]:
            source = sources[item["source_id"]]
            title = escape(source["title"])
            if url := public_source_url(source):
                title = f'<a href="{escape(url, quote=True)}">{title}</a>'
            parts.append(f"<p>{title} · {escape(source['source_type'])}</p>")
            parts.append(f"<blockquote>{escape(item['excerpt'])}</blockquote>")
    return "\n".join(parts)


def course_report_html(content: dict, sources: dict, notices_html: str) -> str:
    program = content["program"]
    title = f"Отчёт по курсу «{escape(program['title'])}»"
    parts = ['<!doctype html><html lang="ru"><meta charset="utf-8">',
             f"<title>{title}</title><h1>{title}</h1>",
             rationale_report_html(program, sources)]
    audit = program.get("business_audit")
    vacancy_ids = content.get("vacancy_source_ids", [])
    if audit or vacancy_ids:
        parts.append("<h2>Оценка курса индустрией</h2>")
    if audit:
        parts.append(f"<p>Покрытие рабочих задач курсом (оценка модели) — {audit['market_fit_score']:.0%}</p>")
        parts.append(f"<p>{escape(audit['evidence_summary'])}</p>")
        for metric in audit["metrics"]:
            part = f"{metric['name']} — {metric['value']} {metric['unit']} · {metric['region']}"
            parts.append(f"<p>{escape(part)}</p>")
        if audit["unavailable_metrics"]:
            missing = escape(", ".join(audit["unavailable_metrics"]))
            parts.append(f"<p>Нет подтверждённых измерений — {missing}</p>")
    if vacancy_ids:
        parts.append("<h3>Вакансии из выборки</h3>")
        for key in vacancy_ids:
            parts.append(source_html(sources[key]))
    return "\n".join([*parts, notices_html, "</html>"])


def child_blocks(block):
    return [*block.get("topics", []), *block.get("assignments", [])]


def cited_sources(block, *, for_learning: bool):
    if for_learning:
        yield from (ref["source_id"] for ref in block["learning_references"])
    else:
        yield from block["justification"]["source_ids"]
    for child in child_blocks(block):
        yield from cited_sources(child, for_learning=for_learning)


def block_html(block, sources):
    materials = {}
    supporting = {}
    for source_id in dict.fromkeys(cited_sources(block, for_learning=True)):
        key = material_key(sources[source_id], source_id)
        materials.setdefault(key, []).append(source_id)
    for source_id in dict.fromkeys(cited_sources(block, for_learning=False)):
        key = material_key(sources[source_id], source_id)
        items = materials[key] if key in materials else supporting.setdefault(key, [])
        if source_id not in items:
            items.append(source_id)
    entries = [*materials.values(), *supporting.values()]
    numbers = {source_id: number for number, ids in enumerate(entries, 1) for source_id in ids}
    article = render_block(block, numbers, 2)
    catalog = [source_html(sources[ids[0]], number=number, fragments=[sources[key]["content"] for key in ids])
               for number, ids in enumerate(entries, 1)]
    parts = [article]
    if materials:
        parts.extend(["<h3>Материалы дисциплины</h3>", *catalog[:len(materials)]])
    if supporting:
        parts.extend(["<h3>Источники для обоснований</h3>", *catalog[len(materials):]])
    return "\n".join(parts)


def render_block(block, numbers, level):
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
    parts.append("<p>" + REASONS[why["reason_type"]] + " — " + escape(why["evidence_summary"]) + "</p>")
    annotations = {}
    for ref in block["learning_references"]:
        annotation = ref["locator"]
        if bibliography := ref.get("bibliographic_excerpt"):
            annotation += f" · {bibliography}"
        annotations[numbers[ref["source_id"]], annotation] = None
    annotated = {number for number, _ in annotations}
    cited = list(dict.fromkeys(numbers[key] for key in why["source_ids"] if numbers[key] not in annotated))
    if cited:
        label = "Источник" if len(cited) == 1 else "Источники"
        parts.append(f"<p>{label} {', '.join(map(str, cited))}</p>")
    parts.extend(f"<p>Источник {number} — {escape(annotation)}</p>" for number, annotation in annotations)
    if not block["learning_references"]:
        parts.append("<p><strong>Учебный источник требует уточнения.</strong></p>")
    if "instructions" in block:
        tag = '<p style="color:#b35400">' if changed else "<p>"
        parts.extend(tag + escape(s) + "</p>" for s in (
            block["instructions"], "Результат — " + ", ".join(block["expected_artifacts"]),
            "Критерии — " + ", ".join(block["quality_criteria"])))
    for child in child_blocks(block):
        parts.append(render_block(child, numbers, level + 1))
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
        source_ids = set(used_sources(content)) | set(content.get("vacancy_source_ids", ()))
        write_json("sources.json", {k: sources[k] for k in sorted(source_ids)})
        archive.writestr("course-report.html", course_report_html(content, sources, notices_html))
        for index, subject in enumerate(program["subjects"], 1):
            tasks = [a for t in subject["topics"] for a in t["assignments"]]
            refs = [ref for b in [subject, *subject["topics"], *tasks] for ref in b["learning_references"]]
            reading = {}
            for ref in refs:
                key = ref["source_id"]
                source = sources[key]
                item = {"title": ref.get("bibliographic_excerpt") or source["title"]}
                if url := public_source_url(source):
                    item["url"] = url
                reading.setdefault(material_key(source, key), item)
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
                load_text += "<ul>" + "".join(f"<li>{label} — {load.get(field, 0)} акад. ч.</li>"
                                               for field, label in HOURS.items()) + "</ul>"
            validator.validate(document)
            directory = f"rpd/{index:02}"
            write_json(f"{directory}/rpd.json", document)
            archive.writestr(f"{directory}/rpd.html", '<!doctype html><meta charset="utf-8"><title>' +
                             escape(subject["title"]) + "</title>" + load_text + block_html(subject, sources) + notices_html)
    return buffer.getvalue()
