"""Выходные артефакты (раздел 9): «Модель деятельности профессии», «Рекомендации по программе», «Обоснование ОП».

Документ собирается из блоков, которые выводятся в HTML и в Markdown одинаково. Тексты — шаблоны, без LLM.
"""

import json
from html import escape

from activity.config import load_config
from activity.engine import ELIGIBLE, OTHER_FUNCTION, SOURCE_TYPE_LABELS, STATUS_LABELS, TREND_LABELS, percent
from activity.experts import COMPANY_SIZE_LABELS, ROLE_KIND_LABELS
from activity.program import PART_LABELS, RULE_LABELS, SIGNAL_LABELS, signal_lines


DEMO_NOTE = "ДЕМО: синтетические данные. Не является результатом реального опроса или анализа рынка."
DISCLAIMER = "Вакансии запаздывают, эксперты дают опережающий сигнал. Это повод для обсуждения, а не приговор."
DECISION_LABELS = {"accepted": "принято", "rejected": "отклонено", "postponed": "отложено", None: "не рассмотрено"}
CONFIDENCE_LABELS = {"high": "высокая", "medium": "средняя", "low": "низкая"}
METHOD_STEPS = [
    "Выберите целевые профессии и профстандарты: из приложения к ФГОС и (или) иные. Укажите регион и горизонт.",
    "Постройте модель деятельности. Наберите не меньше {n_min} экспертов на профессию из разных сегментов: размер компании, роль.",
    "Доведите задачи с высоким приоритетом до статуса «Подтверждено». Спорные отправьте во второй раунд.",
    "Сформулируйте ПК из функций (ОТФ/ТФ), а индикаторы достижения — из задач и знаний.",
    "Сопоставьте дисциплины, разберите рекомендации и зафиксируйте решения с комментариями.",
    "Закрепите за каждым методистом задачи и знания его дисциплины. Это вход для генерации РПД.",
    "Назначьте пересмотр раз в 6–12 месяцев: обновите вакансии и повторно опросите экспертов по спорным и растущим задачам.",
]
FGOS_NOTE = ("ФГОС ВО 3++ требует формировать профессиональные компетенции на основе профессиональных стандартов, анализа "
             "требований рынка труда и консультаций с работодателями. Номер пункта сверьте по своему стандарту. "
             "Этот отчёт — документальное подтверждение такого анализа.")


class Doc:
    """Блоки документа: ("h", level, text) | ("p", text) | ("note", text) | ("ul", items) | ("table", head, rows)."""

    def __init__(self, title: str, demo: bool):
        self.title, self.demo, self.blocks = title, demo, []

    def h(self, level: int, text: str):
        self.blocks.append(("h", level, text))

    def p(self, text: str):
        self.blocks.append(("p", text))

    def note(self, text: str):
        self.blocks.append(("note", text))

    def ul(self, items: list[str]):
        if items:
            self.blocks.append(("ul", items))

    def table(self, head: list[str], rows: list[list]):
        if rows:
            self.blocks.append(("table", head, rows))

    def markdown(self) -> str:
        lines = [f"# {self.title}", ""]
        if self.demo:
            lines += [f"> **{DEMO_NOTE}**", ""]
        for block in self.blocks:
            if block[0] == "h":
                lines += ["#" * (block[1] + 1) + " " + block[2], ""]
            elif block[0] == "p":
                lines += [block[1], ""]
            elif block[0] == "note":
                lines += [f"> {block[1]}", ""]
            elif block[0] == "ul":
                lines += [f"- {item}" for item in block[1]] + [""]
            else:
                head, rows = block[1], block[2]
                lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
                lines += ["| " + " | ".join(str(cell).replace("|", "/").replace("\n", " ") for cell in row) + " |" for row in rows]
                lines.append("")
        return "\n".join(lines)

    def html(self) -> str:
        parts = [f"<h1>{escape(self.title)}</h1>"]
        if self.demo:
            parts.append(f'<p class="demo">{escape(DEMO_NOTE)}</p>')
        for block in self.blocks:
            if block[0] == "h":
                parts.append(f"<h{block[1] + 1}>{escape(block[2])}</h{block[1] + 1}>")
            elif block[0] == "p":
                parts.append(f"<p>{escape(block[1])}</p>")
            elif block[0] == "note":
                parts.append(f'<p class="note">{escape(block[1])}</p>')
            elif block[0] == "ul":
                parts.append("<ul>" + "".join(f"<li>{escape(item)}</li>" for item in block[1]) + "</ul>")
            else:
                head = "".join(f"<th>{escape(cell)}</th>" for cell in block[1])
                rows = "".join("<tr>" + "".join(f"<td>{escape(str(cell))}</td>" for cell in row) + "</tr>" for row in block[2])
                parts.append(f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>")
        style = ("body{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;color:#1b263b;max-width:1100px;margin:2rem auto;"
                 "padding:0 1rem;line-height:1.5}h1{letter-spacing:-.02em}h2{margin-top:2rem;border-bottom:1px solid #e0e5ed;"
                 "padding-bottom:.3rem}table{border-collapse:collapse;width:100%;font-size:.9rem;margin:.6rem 0}"
                 "th,td{border:1px solid #e0e5ed;padding:.35rem .5rem;text-align:left;vertical-align:top}"
                 "th{background:#f1f4f9}.demo{background:#fff3dc;color:#744807;padding:.6rem .8rem;border-radius:10px;"
                 "font-weight:600}.note{color:#526176;border-left:3px solid #8eb0d9;padding-left:.7rem}")
        return (f"<!doctype html><html lang=\"ru\"><head><meta charset=\"utf-8\"><title>{escape(self.title)}</title>"
                f"<style>{style}</style></head><body>{''.join(parts)}</body></html>")


def experts_cell(task: dict) -> str:
    experts = task["metrics"]["experts"]
    if not experts["n"]:
        return "нет голосов"
    return f"{experts['k']}/{experts['n']} ({percent(experts['ci'][0])}–{percent(experts['ci'][1])})"


def market_cell(task: dict) -> str:
    market = task["metrics"]["market"]
    if not market["enough"]:
        return f"мало данных ({market['k_emp']} из {market['n_emp']})"
    return f"{market['k_emp']}/{market['n_emp']} ({percent(market['share'])})"


def standard_cell(task: dict) -> str:
    standard = task["metrics"]["standard"]
    if not standard["loaded"]:
        return "ПС не загружен"
    return ", ".join(standard["functions"]) if standard["in_standard"] else "нет"


def model_document(content: dict) -> Doc:
    profession = content["profession"]
    doc = Doc(f"Модель деятельности профессии «{profession['name']}»", bool(profession.get("is_demo")))
    passport = content["passport"]
    experts = content["experts"]
    doc.h(1, "Паспорт")
    doc.ul([f"Версия {content.get('version', '—')} на {content['as_of']}; хэш правил {content['config_hash'][:12]}, "
            f"хэш входных данных {content['input_hash'][:12]}.",
            f"Регион: {profession.get('region') or 'не указан'}; горизонт: {profession.get('horizon_years')} года; "
            f"профстандарты: {', '.join(profession.get('standard_codes') or []) or 'не указаны'}.",
            f"Окно рынка: {passport['window'][0]} — {passport['window'][1]}; работодателей сейчас {passport['employers_now']}, "
            f"в прошлом окне {passport['employers_prev']}.",
            f"Утверждений: {passport['claims']['total']}, прошли проверку цитаты {passport['claims']['verified']}, "
            f"отброшено {passport['claims']['rejected']}.",
            f"Эксперты: всего {experts['total']}, голосовали {experts['voted']}, надёжных {experts['reliable']}, "
            f"исключено {len(experts['excluded'])}."])
    doc.table(["Тип источника", "Источников", "Независимых (работодатели, издатели)", "Период"],
              [[SOURCE_TYPE_LABELS.get(kind, kind), item["count"], item["origins"], f"{item['from']} — {item['to']}"]
               for kind, item in passport["sources"].items()])
    segments = experts["segments"]
    doc.p("Сегменты надёжных экспертов: " + "; ".join(
        f"{ROLE_KIND_LABELS.get(key, key)} — {value}" for key, value in segments["role_kind"].items()) + "; " + "; ".join(
        f"{COMPANY_SIZE_LABELS.get(key, key)} — {value}" for key, value in segments["company_size"].items()))
    for item in passport["limitations"]:
        doc.note(item)
    tasks = content["tasks"]
    doc.h(1, "Функции и задачи")
    doc.note("Статусы и тренды рассчитаны кодом по формулам config/activity_model.yaml. Каждое утверждение — с цитатой источника.")
    for function in content["functions"]:
        members = [tasks[key] for key in function["tasks"]]
        if not members:
            continue
        code = f" ({function['code']})" if function.get("code") else ""
        doc.h(2, f"{function['name']}{code}")
        doc.table(["Задача", "Статус", "Тренд", "Приоритет", "Рынок k/n", "Эксперты k/n (95 % ДИ)", "ПС"],
                  [[task["name"], task["status_label"], task["trend_label"], task["priority"], market_cell(task),
                    experts_cell(task), standard_cell(task)] for task in members])
        for task in members:
            doc.h(3, task["name"])
            doc.p(task["reason"]["title"] + ".")
            doc.ul(task["reason"]["lines"])
            quotes = [f"«{quote['text']}» — {quote['source_title']} ({quote['date']})" for quote in task["quotes"][:5]]
            if len(task["quotes"]) > 5:
                quotes.append(f"…и ещё {len(task['quotes']) - 5} цитат(ы) в JSON-версии модели.")
            doc.ul(quotes)
            related = [content["knowledge"][link["id"]]["name"] + f" ({link['reason']})" for link in task["related"]]
            if related:
                doc.p("Знания, умения и инструменты: " + "; ".join(related[:12]))
            if task["areas"]:
                doc.p("Области знаний: " + ", ".join(task["areas"]))
    knowledge = sorted(content["knowledge"].values(), key=lambda item: (item["level"], item["name"]))
    doc.h(1, "Знания, умения и инструменты")
    labels = {"knowledge": "знание", "skill": "умение", "tool": "инструмент"}
    doc.table(["Пункт", "Вид", "Области", "Рынок k/n", "Задач"],
              [[item["name"], labels[item["level"]], ", ".join(item["areas"]) or "—",
                f"{item['market']['k_emp']}/{item['market']['n_emp']}" if item["market"]["enough"] else "мало данных",
                len(item["tasks"])] for item in knowledge])
    doc.h(1, "Карта областей знаний")
    doc.table(["Область", "Вид", "Вес в профессии", "Задач"],
              [[name, "фундаментальная" if area["kind"] == "fundamental" else "прикладная", percent(area["weight"]),
                len(area["tasks"])] for name, area in sorted(content["areas"].items(), key=lambda kv: -kv[1]["weight"])])
    for status, title in (("rejected", "Отклонено"), ("disputed", "Спорно"), ("hypothesis", "Гипотезы")):
        rows = [task for task in tasks.values() if task["status"] == status]
        doc.h(1, title)
        if not rows:
            doc.p("Нет.")
        for task in rows:
            doc.p(f"{task['name']}. " + " ".join(task["reason"]["lines"]))
    doc.h(1, "Отброшено при извлечении")
    doc.table(["Утверждение", "Цитата", "Причина", "Источник"],
              [[item["text"], item["quote"], item["reason"], item["source_title"]] for item in content["discarded_claims"]])
    if not content["discarded_claims"]:
        doc.p("Нет.")
    doc.h(1, "Исключённые эксперты")
    doc.p(f"Исключено {len(experts['excluded'])} из {experts['total']}. Голоса исключённых хранятся и не участвуют в расчёте.")
    doc.table(["Эксперт", "Сегмент", "Причины"],
              [[item["label"], item["segment"], "; ".join(item["reasons"])] for item in experts["excluded"]])
    doc.h(1, "Сигнал по профилю")
    signal = content["profile_signal"]
    doc.p(SIGNAL_LABELS[signal["label"]] + ".")
    doc.ul(signal_lines(signal))
    doc.note(DISCLAIMER)
    diff = content.get("diff") or {}
    doc.h(1, "Изменения с прошлой версии")
    if not diff:
        doc.p("Это первая версия модели.")
    else:
        names = {key: task["name"] for key, task in tasks.items()}
        doc.ul([f"Новых задач: {len(diff['added'])}; удалено: {len(diff['removed'])}.",
                *(f"«{names.get(change['id'], change['id'])}»: статус {STATUS_LABELS[change['from']].lower()} → "
                  f"{STATUS_LABELS[change['to']].lower()}" for change in diff["status"]),
                *(f"«{names.get(change['id'], change['id'])}»: тренд {TREND_LABELS[change['from']]} → "
                  f"{TREND_LABELS[change['to']]}" for change in diff["trend"])])
    return doc


def program_document(program: dict, analysis: dict, decisions: list[dict], models: dict[str, dict]) -> Doc:
    """program: {name, institution, academic_year, total_zet, is_demo, disciplines: [{id, name, zet, part, teacher_name}]}."""
    doc = Doc(f"Рекомендации по программе «{program['name']}»", bool(program.get("is_demo")))
    doc.p(" · ".join(filter(None, [program.get("institution"), program.get("academic_year"),
                                   f"{program['total_zet']} ЗЕТ"])))
    doc.p("Целевые профессии: " + "; ".join(f"«{model['profession']['name']}» (модель v{model.get('version')}, "
                                             f"на {model['as_of']})" for model in models.values()))
    recommendations = analysis["recommendations"]
    doc.h(1, "Главные выводы")
    doc.ul([f"{RULE_LABELS[rec['rule']]}: {rec['action_text']} Уверенность — {CONFIDENCE_LABELS[rec['confidence']]}."
            for rec in recommendations[:7]])
    balance = analysis["balance"]
    doc.h(1, "Баланс ЗЕТ")
    doc.ul([f"Сумма ЗЕТ дисциплин сейчас: {balance['disciplines_zet']:g} из {balance['program_zet']} по программе.",
            f"Освобождается по рекомендациям: {balance['frees']:g} ЗЕТ; нужно для новых дисциплин: {balance['needs']:g} ЗЕТ.",
            f"Если принять все рекомендации: {balance['after']:g} ЗЕТ."])
    doc.h(1, "Все рекомендации")
    for rec in recommendations:
        decision = next((item for item in decisions if item["key"] == rec["key"]), {})
        doc.h(2, f"{rec['rule']}. {rec['action_text']}")
        doc.ul(rec["lines"])
        doc.p(f"Уверенность: {CONFIDENCE_LABELS[rec['confidence']]}. Если не делать: {rec['consequence']} "
              f"Баланс: освобождается {rec['zet']['frees']:g} ЗЕТ, нужно {rec['zet']['needs']:g} ЗЕТ.")
        if rec["tasks"]:
            doc.table(["Задача", "Статус", "Тренд", "Приоритет", "Рынок", "Эксперты"],
                      [[task["name"], task["status_label"], TREND_LABELS[task["trend"]], task["priority"], task["market"],
                        task["experts"]] for task in rec["tasks"][:15]])
        comment = f" — {decision['comment']}" if decision.get("comment") else ""
        doc.p(f"Решение методиста: {DECISION_LABELS[decision.get('decision')]}{comment}.")
    doc.h(1, "Дисциплины программы")
    doc.table(["Дисциплина", "ЗЕТ", "Часть", "Методист", "Покрыто задач", "Доля тем без связи с задачами"],
              [[d["name"], f"{d['zet']:g}", PART_LABELS.get(d["part"], d["part"]), d.get("teacher_name") or "—",
                len(analysis["links"].get(d["id"], {})), percent(analysis["unlinked_topic_share"].get(d["id"], 0))]
               for d in program["disciplines"]])
    doc.h(1, "Сигнал по профилю")
    for model in models.values():
        signal = model["profile_signal"]
        doc.p(f"«{model['profession']['name']}»: {SIGNAL_LABELS[signal['label']].lower()}.")
        doc.ul(signal_lines(signal))
    doc.note(DISCLAIMER)
    doc.h(1, "Как разрабатывать программу")
    doc.ul([step.format(n_min=load_config()["experts"]["n_min"]) for step in METHOD_STEPS])
    doc.h(1, "Журнал решений методиста")
    logged = [item for item in decisions if item.get("decision")]
    doc.table(["Дата", "Правило", "Рекомендация", "Решение", "Комментарий"],
              [[item["decided_at"], item["rule"], item["action_text"], DECISION_LABELS[item["decision"]],
                item.get("comment") or ""] for item in logged])
    if not logged:
        doc.p("Решений пока нет.")
    return doc


def justification_document(program: dict, analysis: dict, decisions: list[dict], models: dict[str, dict]) -> Doc:
    """«Обоснование ОП»: выжимка модели и рекомендаций для аккредитационных материалов."""
    doc = Doc(f"Обоснование образовательной программы «{program['name']}»", bool(program.get("is_demo")))
    doc.note(FGOS_NOTE)
    for model in models.values():
        tasks = [task for task in model["tasks"].values() if task["status"] in ELIGIBLE]
        passport = model["passport"]
        doc.h(1, f"Профессия «{model['profession']['name']}»")
        doc.h(2, "Профессиональные стандарты")
        standard = [task for task in tasks if task["metrics"]["standard"]["in_standard"]]
        doc.p(f"Профстандарты: {', '.join(model['profession'].get('standard_codes') or []) or 'не указаны'}. "
              f"Задач из профстандарта в модели: {len(standard)} из {len(tasks)} подтверждённых и предварительных.")
        lagging = [task["name"] for task in tasks if any("отстаёт" in flag for flag in task["lag_flags"])]
        outdated = [task["name"] for task in model["tasks"].values() if any("Устарело" in flag for flag in task["lag_flags"])]
        if lagging:
            doc.p("Подтверждено рынком и экспертами, но нет в профстандарте: " + "; ".join(lagging) + ".")
        if outdated:
            doc.p("Есть в профстандарте, но отклонено экспертами и редко встречается на рынке: " + "; ".join(outdated) + ".")
        doc.h(2, "Анализ требований рынка труда")
        doc.p(f"Вакансии за {passport['window'][0]} — {passport['window'][1]}: {passport['employers_now']} работодателей "
              f"(доля считается по работодателям, а не по вакансиям).")
        top = sorted((task for task in tasks if task["metrics"]["market"]["enough"]),
                     key=lambda task: -task["metrics"]["market"]["share"])[:10]
        doc.table(["Задача", "Доля работодателей", "Тренд"],
                  [[task["name"], percent(task["metrics"]["market"]["share"]), task["trend_label"]] for task in top])
        doc.h(2, "Консультации с работодателями и экспертами")
        experts = model["experts"]
        doc.p(f"Опрошено экспертов: {experts['voted']}, надёжных {experts['reliable']}, исключено {len(experts['excluded'])} "
              f"по контрольным карточкам и времени ответов. Метод Дельфи, интервал Вильсона 95 %.")
        confirmed = [task for task in tasks if task["status"] in ("confirmed", "confirmed_early")]
        doc.table(["Задача", "Подтвердили экспертов", "Функция"],
                  [[task["name"], experts_cell(task),
                    next((f["name"] for f in model["functions"] if f["id"] == task["function_id"]), "—")]
                   for task in sorted(confirmed, key=lambda task: -task["priority"])[:20]])
        doc.h(2, "Функции деятельности — основа профессиональных компетенций")
        doc.ul([f"{function['name']}" + (f" ({function['code']})" if function.get("code") else "") +
                f": {sum(1 for key in function['tasks'] if model['tasks'][key]['status'] in ELIGIBLE)} задач"
                for function in model["functions"] if function["id"] != OTHER_FUNCTION])
    doc.h(1, "Решения по составу программы")
    accepted = [item for item in decisions if item.get("decision") == "accepted"]
    doc.ul([f"{item['action_text']}" + (f" Комментарий: {item['comment']}" if item.get("comment") else "") for item in accepted])
    if not accepted:
        doc.p("Принятых решений пока нет.")
    doc.p(f"Баланс ЗЕТ после принятия всех рекомендаций: {analysis['balance']['after']:g} из {analysis['balance']['program_zet']}.")
    return doc


def model_json(content: dict) -> str:
    return json.dumps(content, ensure_ascii=False, sort_keys=True, indent=1)


def program_json(program: dict, analysis: dict, decisions: list[dict]) -> str:
    return json.dumps({"program": program, "analysis": analysis, "decisions": decisions,
                       "demo": DEMO_NOTE if program.get("is_demo") else None},
                      ensure_ascii=False, sort_keys=True, indent=1, default=str)
