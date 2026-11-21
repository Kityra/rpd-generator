import json
from html import escape
from io import BytesIO
from pathlib import Path
from urllib.parse import urlsplit
from zipfile import ZIP_DEFLATED, ZipFile

from jsonschema import Draft202012Validator
from domain.workload import HOURS, assessment_label, by_semester, curriculum, workload

REASONS = {"market_demand": "Спрос работодателей", "scientific_trend": "Научные результаты",
           "expert_practice": "Практика эксперта", "fundamental_core": "Фундаментальная подготовка",
           "teacher_review": "Правка преподавателя", "scheduled_relevance_update": "Плановая проверка актуальности"}


OUTLINE_LABEL = "Собрано шаблоном, нужна проверка"
PROFESSION_REVIEW_LABEL = "[Требует проверки преподавателя — нет прямого подтверждения в источниках]"
# Так помечал обоснование шаблон make_outline до появления поля generated_by.
LEGACY_OUTLINE_SUMMARY = "Тема предложена для проверки преподавателем по приложенным материалам."


def subject_origin(subject: dict) -> str | None:
    """«model», «outline» или None, если происхождение старой версии неизвестно."""
    if subject.get("generated_by"):
        return subject["generated_by"]
    if subject.get("justification", {}).get("evidence_summary") == LEGACY_OUTLINE_SUMMARY:
        return "outline"
    return None


def generation_summary(program: dict) -> dict:
    """Сколько дисциплин собрано моделью, сколько шаблоном, и причины перехода на шаблон."""
    counts = {"model": 0, "outline": 0, "unknown": 0}
    reasons: dict[str, int] = {}
    for subject in program.get("subjects", []):
        origin = subject_origin(subject) or "unknown"
        counts[origin] += 1
        if origin == "outline":
            reason = subject.get("generation_note") or "причина не сохранена (версия до журнала вызовов)"
            reasons[reason] = reasons.get(reason, 0) + 1
    return {**counts, "reasons": sorted(reasons.items(), key=lambda item: -item[1])}


def generation_summary_html(program: dict) -> str:
    summary = generation_summary(program)
    if not summary["outline"] and not summary["unknown"]:
        return ""
    parts = ["<h2>Происхождение дисциплин</h2>",
             f"<p>Собрано моделью — {summary['model']}, шаблоном — {summary['outline']}"
             + (f", не записано — {summary['unknown']}" if summary["unknown"] else "") + "</p>"]
    if summary["reasons"]:
        parts.append("<ul>" + "".join(f"<li>{escape(reason)} — {count}</li>" for reason, count in summary["reasons"]) + "</ul>")
    return "\n".join(parts)


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


CHECK_LABELS = {"weak": "цитаты подтверждают обоснование лишь косвенно",
                "unrelated": "цитаты не относятся к дисциплине", "missing": "подтверждение не найдено"}


def rationale_report_html(program: dict, sources: dict) -> str:
    parts = []
    for subject in program["subjects"]:
        parts.append(f"<h2>{escape(subject['title'])}</h2>")
        rationale = subject.get("inclusion_rationale")
        check = subject.get("rationale_check") or {}
        if not rationale:
            parts.append("<p>В базе не подтверждено обоснование включения дисциплины в курс.</p>")
            if check.get("status") in CHECK_LABELS:
                parts.append(f"<p><strong>Требует проверки</strong> — {escape(CHECK_LABELS[check['status']])}. "
                             f"{escape(check.get('reason') or '')}</p>")
            continue
        if check.get("status") == "supports" and check.get("judge") is False and "судья" in check.get("reason", ""):
            parts.append("<p>Обоснование проверено кодом; проверка смысла моделью-судьёй была недоступна.</p>")
        parts.append(f"<p>{escape(rationale['explanation'])}</p>")
        for item in rationale["evidence"]:
            source = sources[item["source_id"]]
            title = escape(source["title"])
            if url := public_source_url(source):
                title = f'<a href="{escape(url, quote=True)}">{title}</a>'
            parts.append(f"<p>{title} · {escape(source['source_type'])}</p>")
            parts.append(f"<blockquote>{escape(item['excerpt'])}</blockquote>")
    return "\n".join(parts)


def composition_html(check: dict | None) -> str:
    if not check or not (check.get("duplicates") or check.get("unlinked")):
        return ""
    parts = ["<h2>Состав курса — требует проверки</h2>"]
    for pair in check.get("duplicates", []):
        left, right = pair["subjects"]
        parts.append(f"<p>Возможный дубль: «{escape(left)}» и «{escape(right)}»</p>")
    if check.get("unlinked"):
        names = ", ".join(f"«{escape(name)}»" for name in check["unlinked"])
        parts.append(f"<p>Нет связи с профессией и рабочими задачами: {names}</p>")
    return "\n".join(parts)


LEVEL_LABELS = {"full": "покрыт темами", "partial": "частично (только в заданиях)", "none": "не покрыт"}


def market_html(analysis: dict | None) -> str:
    """Отчёт по навыкам рынка (расчёт кодом)."""
    if not analysis:
        return ""
    parts = ["<h2>Навыки рынка и покрытие курсом</h2>"]
    if analysis["status"] == "no_data":
        parts.extend(f"<p>{escape(item)}</p>" for item in analysis["warnings"])
        return "\n".join(parts)
    details = [analysis.get("source"), f"выборка — {analysis['sample_size']} вакансий" if analysis.get("sample_size") else None,
               f"период {analysis['period']}" if analysis.get("period") else None,
               f"регион {analysis['region']}" if analysis.get("region") else "регион не указан"]
    parts.append("<p>" + escape(" · ".join(item for item in details if item)) + "</p>")
    if analysis.get("market_fit") is not None:
        parts.append(f"<p>Покрытие критичных навыков курсом (расчёт) — {analysis['market_fit']:.0%}</p>")
    parts.extend(f"<p>{escape(item)}</p>" for item in analysis.get("warnings", []))
    parts.append("<table><tr><th>Навык</th><th>Доля вакансий</th><th>Критичный</th><th>Покрытие</th>"
                 "<th>Дисциплины</th><th>Приоритет</th></tr>")
    for row in analysis["skills"]:
        share = f"{row['share']:.0%} ({row['vacancies']})" if row.get("share") is not None else (
            f"вес {row['weight']:g}" if row.get("weight") is not None else "—")
        parts.append(f"<tr><td>{escape(row['skill'])}</td><td>{share}</td><td>{'да' if row['critical'] else 'нет'}</td>"
                     f"<td>{LEVEL_LABELS[row['level']]}{' (вручную)' if row.get('manual') else ''}</td>"
                     f"<td>{escape(', '.join(row['subjects']))}</td><td>{row['priority']:.2f}</td></tr>")
    parts.append("</table>")
    if analysis.get("uncovered"):
        parts.append("<p>Не покрыты курсом: " + escape(", ".join(analysis["uncovered"])) + "</p>")
    if analysis.get("weak_subjects"):
        parts.append("<p>Слабо связаны с рынком и задачами: " + escape(", ".join(analysis["weak_subjects"])) + "</p>")
    return "\n".join(parts)


def course_report_html(content: dict, sources: dict, notices_html: str, market: dict | None = None) -> str:
    program = content["program"]
    title = f"Отчёт по курсу «{escape(program['title'])}»"
    parts = ['<!doctype html><html lang="ru"><meta charset="utf-8">',
             f"<title>{title}</title><h1>{title}</h1>",
             generation_summary_html(program),
             profession_profile_html(content.get("profession_profile"), content.get("profession_market", {}), sources),
             composition_html(content.get("composition_check")),
             rationale_report_html(program, sources)]
    audit = program.get("business_audit")
    vacancy_ids = content.get("vacancy_source_ids", [])
    if audit or vacancy_ids:
        parts.append("<h2>Оценка курса индустрией</h2>")
    parts.append(market_html(market))
    if audit:
        parts.append(f"<p>Мнение модели: покрытие рабочих задач курсом — {audit['market_fit_score']:.0%} "
                     "(оценка модели, не расчёт)</p>")
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


def profession_counts(profile: dict) -> dict:
    counts = dict.fromkeys(("verified", "needs_review", "rejected"), 0)
    for claim in profile.get("claims", []):
        counts[claim["verification_status"]] += 1
    return counts


def profession_profile_html(profile: dict | None, market: dict, sources: dict) -> str:
    if not profile:
        return ""
    counts = profile.get("counts") or profession_counts(profile)
    parts = ["<h2>Профиль профессии</h2>", "<p>"
             f"Подтверждено: {counts['verified']} · Требует проверки: {counts['needs_review']} · "
             f"Отклонено экспертом: {counts['rejected']}</p>"]
    if profile["sample_adequacy"] == "empty_external_data":
        parts.append("<p>Внешних данных о рабочих задачах и измерений рынка нет.</p>")
    elif profile["sample_adequacy"] == "low_sample":
        parts.append("<p>Внешняя выборка ограничена.</p>")
    if profile["missing_slots"]:
        parts.append(f"<p>Требуется уточнить: {escape(', '.join(profile['missing_slots']))}.</p>")
    parts.append("<ul>")
    for claim in profile["claims"]:
        if claim["verification_status"] == "rejected":
            continue
        badges, fragments = [], []
        for ref in claim["evidence"]:
            if ref["expert_status"] in {"confirmed_by_expert", "edited_by_expert"}:
                number = ref["source_id"].removeprefix("turn:")
                badges.append(f"[Подтверждено экспертом, реплика #{number}]")
            if ref["judge_verdict"] == "supports" and ref["source_type"] == "market_cluster":
                provider = market.get(ref["source_id"], {}).get("provider", "выборка вакансий")
                metric = ref["quote_or_metric"].replace(" (", " вакансий (", 1)
                badges.append(f"[Рынок {provider}: {metric}, подтверждено]")
            if ref["judge_verdict"] == "supports" and ref["source_type"] == "rag_chunk":
                badges.append("[Подтверждено источником RAG]")
            title = sources.get(ref["source_id"], {}).get("title")
            fragments.append(f"<p>{escape(title + ' — ' if title else '')}{escape(ref['quote_or_metric'])}</p>")
        if claim["verification_status"] == "needs_review":
            badges = [PROFESSION_REVIEW_LABEL]
        labels = " ".join(escape(label) for label in dict.fromkeys(badges))
        details = (f"<details><summary>Основание</summary>{''.join(fragments)}</details>" if fragments else "")
        parts.append(f"<li>{escape(claim['statement'])} <strong>{labels}</strong>{details}</li>")
    parts.append("</ul>")
    return "\n".join(parts)


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
    if "topics" in block and subject_origin(block) == "outline":
        note = block.get("generation_note")
        parts.append(f'<p style="color:#b3261e"><strong>{OUTLINE_LABEL}.</strong>'
                     + (f" Причина — {escape(note)}" if note else "") + "</p>")
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


def export_course(meta, version, sources, market: dict | None = None):
    if version.status != "approved":
        raise ValueError("Экспорт доступен после утверждения версии.")
    content = version.content if market is None else {**version.content, "market_analysis": market}
    if profile := content.get("profession_profile"):
        content = {**content, "profession_profile": {**profile, "counts": profession_counts(profile),
                   "claims": [claim for claim in profile["claims"] if claim["verification_status"] != "rejected"]}}
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
        source_ids.update(ref["source_id"] for claim in content.get("profession_profile", {}).get("claims", [])
                          for ref in claim["evidence"] if ref["source_type"] in {"rag_chunk", "user_input"}
                          and ref["source_id"] in sources)
        write_json("sources.json", {k: sources[k] for k in sorted(source_ids)})
        archive.writestr("course-report.html", course_report_html(content, sources, notices_html, market))
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
            if origin := subject_origin(subject):
                document["generatedBy"] = origin
                if origin == "outline":
                    document["generationWarning"] = OUTLINE_LABEL
                    if subject.get("generation_note"):
                        document["generationNote"] = subject["generation_note"]
            load = content.get("workloads", {}).get(subject["id"])
            load_text = ""
            if notices:
                document["sourceNotes"] = notices
            if load:
                document["workload"] = {**workload(load), "bySemester": by_semester(load)}
                document["workloadInputs"] = load
                if plan := curriculum(subject, load):
                    document["curriculum"] = plan
                if form := assessment_label(load.get("assessment_form")):
                    document["assessment"] = {"interimForms": [form]}
                load_text = f"<p>Нагрузка — {document['workload']['totalHours']} ч., {document['workload']['totalCredits']:g} ЗЕ"
                semester = document["workload"]["bySemester"][0]
                load_text += f" · семестр {semester['semester']}"
                if form:
                    load_text += f" · {escape(form)}"
                load_text += "</p><ul>" + "".join(f"<li>{label} — {load.get(field, 0)} акад. ч.</li>"
                                                  for field, label in HOURS.items()) + "</ul>"
                if plan:
                    load_text += "<p>Часы по темам</p><ul>" + "".join(
                        f"<li>{escape(topic['title'])} — {sum(topic['activityHours'].values())} ч</li>"
                        for topic in plan[0]["topics"]) + "</ul>"
            validator.validate(document)
            directory = f"rpd/{index:02}"
            write_json(f"{directory}/rpd.json", document)
            archive.writestr(f"{directory}/rpd.html", '<!doctype html><meta charset="utf-8"><title>' +
                             escape(subject["title"]) + "</title>" + load_text + block_html(subject, sources) + notices_html)
    return buffer.getvalue()
