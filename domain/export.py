import json
from html import escape
from io import BytesIO
from pathlib import Path
from urllib.parse import urlsplit
from zipfile import ZIP_DEFLATED, ZipFile

from jsonschema import Draft202012Validator
from domain.schemas import MarketAnalysis
from domain.skill_mapper import TOOL_COLUMN_NOTE, competence_label, tool_labels
from orchestration.market_analysis import course_market, with_competences, with_expert_tools
from orchestration.profession_profile import ProfessionProfile

REASONS = {"market_demand": "Спрос работодателей", "scientific_trend": "Научные результаты",
           "expert_practice": "Практика эксперта", "fundamental_core": "Фундаментальная подготовка",
           "teacher_review": "Правка методиста", "scheduled_relevance_update": "Плановая проверка актуальности"}


OUTLINE_LABEL = "Собрано шаблоном, нужна проверка методиста"
PROFESSION_REVIEW_LABEL = "[Требует проверки методиста — нет прямого подтверждения в источниках]"
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
        problem = CHECK_LABELS.get(check.get("status"))
        if check.get("status") == "supports" and check.get("judge") is False:
            problem = "для обоснования нет вердикта модели-судьи"
        if problem:
            parts.append(f"<p><strong>{PROFESSION_REVIEW_LABEL}</strong> — {escape(problem)}. "
                         f"{escape(check.get('reason') or '')}</p>")
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
    report = with_competences(MarketAnalysis.model_validate(analysis), {})
    if report.market_fit is not None:
        parts.append(f"<p>Покрытие критичных компетенций курсом (расчёт) — {report.market_fit:.0%}</p>")
    parts.extend(f"<p>{escape(item)}</p>" for item in report.warnings)
    parts.append("<table><tr><th>Компетенция</th><th>Инструменты</th><th>Доля вакансий</th><th>Вакансий</th>"
                 "<th>Вес</th><th>Критичный</th><th>Покрытие</th><th>Дисциплина</th></tr>")
    for row in report.competences:
        share = f"{row.share:.0%}" if row.share is not None else "—"
        count = row.vacancies if row.vacancies is not None else "—"
        covered = row.level == "full"
        subjects = ", ".join(row.subjects) if covered else ""
        parts.append(f"<tr><td>{escape(row.competence_name)}</td><td>{escape(', '.join(tool_labels(row)) or '—')}</td>"
                     f"<td>{share}</td><td>{count}</td><td>{(row.weight or 0):g}</td><td>{'да' if row.critical else 'нет'}</td>"
                     f"<td>{'покрыт темами' if covered else 'не покрыт'}</td><td>{escape(subjects or '—')}</td></tr>")
    parts.append("</table>")
    if any(not tool_labels(row) or row.expert_tools for row in report.competences):
        parts.append(f"<p>{escape(TOOL_COLUMN_NOTE)}</p>")
    missing = [competence_label(row) for row in report.competences if row.level != "full"]
    if missing:
        parts.append("<p>Не покрыты курсом: " + escape(", ".join(missing)) + "</p>")
    if analysis.get("weak_subjects"):
        parts.append("<p>Слабо связаны с рынком и задачами: " + escape(", ".join(analysis["weak_subjects"])) + "</p>")
    return "\n".join(parts)


def course_report_html(content: dict, sources: dict, notices_html: str, market: dict | None = None) -> str:
    program = content["program"]
    title = f"Отчёт по курсу «{escape(program['title'])}»"
    parts = ['<!doctype html><html lang="ru"><meta charset="utf-8">',
             f"<title>{title}</title><h1>{title}</h1>",
             generation_summary_html(program),
             gap_coverage_html(content.get("evidence_gaps")),
             boundary_report_html(content.get("profession_boundaries", []), program),
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


def boundary_table_html(row: dict) -> str:
    headings = ("Ядро целевой профессии", "Общая база", "Зона смежной профессии (исключить)", "Конфликты")
    columns = ("exclusive_to_primary", "shared_core", "exclusive_to_adjacent", "boundary_conflicts")
    headings_html = "".join(f"<th>{escape(label)}</th>" for label in headings)
    values = ["<br>".join(escape(item) for item in row.get(key, [])) or "—" for key in columns]
    cells = "".join(f"<td>{value}</td>" for value in values)
    artifacts = ", ".join(row.get("handoff_artifacts", []))
    source = row.get("source_interview_id")
    return (f'<table><thead><tr>{headings_html}</tr></thead><tbody><tr>{cells}</tr></tbody></table>'
            + (f"<p>Артефакты передачи — {escape(artifacts)}.</p>" if artifacts else "")
            + (f"<p>Интервью — {escape(source)}.</p>" if source else ""))


def boundary_warning_html(subject: dict) -> str:
    return "\n".join(
        f"<p><strong>[Пересечение со смежной профессией «{escape(warning['adjacent_profession'])}» — "
        f"требует решения методиста]</strong> {escape(warning['matched_topic'])} · "
        f"{escape(warning['exclusive_item'])}</p>"
        + (f"<p>Интервью — {escape(warning['source_interview_id'])}.</p>" if warning.get("source_interview_id") else "")
        for warning in subject.get("boundary_warnings", [])
    )


def boundary_report_html(boundaries: list[dict], program: dict) -> str:
    verified = [row for row in boundaries if row["verified_by_methodologist"]]
    warnings = [(subject["title"], html) for subject in program["subjects"] if (html := boundary_warning_html(subject))]
    pending = [row for row in boundaries if not row["verified_by_methodologist"] and row.get("boundary_conflicts")]
    if not (verified or warnings or pending):
        return ""
    parts = ["<h2>Границы смежных профессий</h2>"]
    for row in verified:
        parts.append(f"<h3>{escape(row['primary_profession'])} и {escape(row['adjacent_profession'])} "
                     f"· {row['skill_jaccard_overlap']:.0%}</h3>{boundary_table_html(row)}")
    for row in pending:
        conflicts = ", ".join(row["boundary_conflicts"])
        parts.append(f"<p>Конфликты по роли «{escape(row['adjacent_profession'])}» — {escape(conflicts)}. "
                     "Требуется решение методиста.</p>")
    for title, html in warnings:
        parts.append(f"<h3>{escape(title)}</h3>{html}")
    return "\n".join(parts)


def gap_coverage_html(gaps: dict | None) -> str:
    params = (gaps or {}).get("coverage_parameters")
    if not params:
        return ""
    labels = {"covered": "Достаточно подтверждений", "partial_gap": "Частичный пробел",
              "critical_gap": "Критический пробел"}
    parts = ["<h2>Покрытие профиля источниками</h2>",
             "<p>Итог — лучшее произведение качества на свежесть × коэффициент независимости источников.</p>",
             f"<p>Коэффициент независимости — min(1, ln(1 + число источников) / "
             f"ln(1 + {params['target_independent_sources']})). Порог покрытия — {params['covered_threshold']:.2f}. "
             f"Год расчёта — {params['as_of_year']}. Слабая связь имеет вес {params['weak_weight']:.2f}. "
             f"За {params['freshness_half_life_years']:g} лет свежесть снижается с 1 до 0,5. "
             "Если год публикации неизвестен, свежесть равна 0,75.</p>"]
    rows = gaps.get("formula_breakdown", [])
    if not rows:
        return "\n".join([*parts, "<p>Нет утверждений с данными для расчёта.</p>"])
    parts.append("<details><summary>Расчёт по утверждениям</summary><ul>")
    for row in rows:
        parts.append(f"<li>{escape(row['statement'])} — "
                     f"{row['best_quality_freshness']:.3f} × {row['independence_factor']:.3f} = "
                     f"{row['final_score']:.3f}, независимых источников {row['independent_sources_count']}. "
                     f"{labels[row['status']]}.</li>")
    return "\n".join([*parts, "</ul></details>"])


def profession_counts(profile: dict) -> dict:
    counts = dict.fromkeys(("verified", "needs_review", "rejected"), 0)
    for claim in profile.get("claims", []):
        counts[claim["verification_status"]] += 1
    return counts


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
    if warnings := boundary_warning_html(block):
        parts.append(warnings)
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
    evidence_summary = why["evidence_summary"]
    if evidence_summary == LEGACY_OUTLINE_SUMMARY:
        evidence_summary = "Тема предложена для проверки методистом по приложенным материалам."
    parts.append("<p>" + REASONS[why["reason_type"]] + " — " + escape(evidence_summary) + "</p>")
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
    content = version.content
    # В старых версиях расчёты остаются в БД, но не попадают во вновь скачанные архивы.
    content = {key: value for key, value in content.items() if key not in {"workloads", "workload_proposal"}}
    if profile := content.get("profession_profile"):
        profile = ProfessionProfile.model_validate(profile).model_dump(mode="json")
        content = {**content, "profession_profile": {**profile, "counts": profession_counts(profile),
                   "claims": [claim for claim in profile["claims"] if claim["verification_status"] != "rejected"]}}
    program = content["program"]
    mappings = content.get("skill_mappings")
    if market is None:
        analysis = course_market(meta, program, sources, content.get("vacancy_source_ids", []), content.get("activity"),
                                 mappings=mappings, saved=content.get("market_analysis"), profile=profile)
    else:
        analysis = with_expert_tools(with_competences(MarketAnalysis.model_validate(market), meta, mappings), profile)
    market = analysis.model_dump(mode="json")
    content["market_analysis"] = market
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
                          for ref in claim["evidence"] if ref["source_type"] in {"rag_chunk", "methodologist_input"}
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
            if notices:
                document["sourceNotes"] = notices
            outcomes = discipline_outcomes(subject, analysis, meta, mappings)
            if outcomes:
                document["competenceLearningOutcomes"] = outcomes
            validator.validate(document)
            directory = f"rpd/{index:02}"
            write_json(f"{directory}/rpd.json", document)
            archive.writestr(f"{directory}/rpd.html", '<!doctype html><meta charset="utf-8"><title>' +
                             escape(subject["title"]) + "</title>" + block_html(subject, sources)
                             + discipline_outcomes_html(outcomes) + notices_html)
    return buffer.getvalue()


def discipline_outcomes(subject: dict, analysis: MarketAnalysis, meta: dict, mappings: dict | None) -> list[dict]:
    # Перегруппировка на уровне дисциплины исключает рыночные инструменты, которые в ней не изучаются.
    linked = [row for row in analysis.skills or analysis.competences if subject["title"] in row.subjects]
    result = with_competences(analysis.model_copy(update={"skills": linked, "competences": []}), meta, mappings)
    return [{"competence_name": row.competence_name, "learningOutcomes": {"skillsAndTools": row.tools}}
            for row in result.competences]


def discipline_outcomes_html(outcomes: list[dict]) -> str:
    if not outcomes:
        return ""
    rows = []
    for row in outcomes:
        tools = ", ".join(row["learningOutcomes"]["skillsAndTools"]) or "—"
        rows.append(f"<tr><td>{escape(row['competence_name'])}</td>"
                    f"<td><strong>Навыки и инструменты</strong><br>{escape(tools)}</td></tr>")
    return ("<h2>Планируемые результаты обучения</h2><table><thead><tr><th>Компетенция</th>"
            "<th>Результаты обучения</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")
