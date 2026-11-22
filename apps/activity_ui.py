"""Общие элементы интерфейса модели деятельности: чипы статусов, «Почему здесь», расчёт по клику, баннер ДЕМО."""

from html import escape

import streamlit as st

from activity.config import load_config
from activity.engine import SOURCE_TYPE_LABELS, STATUS_LABELS, TREND_LABELS, percent, plural
from activity.experts import COMPANY_SIZE_LABELS, FORECAST_LABELS, ROLE_KIND_LABELS, VERDICT_LABELS
from activity.report import DEMO_NOTE
from activity.stats import experts_needed


STATUS_ORDER = ("confirmed", "confirmed_early", "preliminary", "disputed", "rejected", "hypothesis")
UP, DOWN = frozenset({"new", "growing"}), frozenset({"declining", "disappearing", "automated"})
SEGMENT_LABELS = {**COMPANY_SIZE_LABELS, **ROLE_KIND_LABELS}


def demo_banner(is_demo: bool) -> None:
    if is_demo:
        st.html(f'<div class="am-banner">{escape(DEMO_NOTE)}</div>')


def status_chip(status: str) -> str:
    return f'<span class="am-chip am-chip--{status}">{escape(STATUS_LABELS[status])}</span>'


def trend_chip(trend: str) -> str:
    kind = " am-chip--up" if trend in UP else " am-chip--down" if trend in DOWN else ""
    arrow = "↑ " if trend in UP else "↓ " if trend in DOWN else ""
    return f'<span class="am-chip am-chip--trend{kind}">{arrow}{escape(TREND_LABELS[trend])}</span>'


def flag_chip(text: str) -> str:
    return f'<span class="am-chip am-chip--flag">{escape(text)}</span>'


def kpis(items: list[tuple[str, str]]) -> None:
    st.html('<div class="am-kpis">' + "".join(f'<div class="am-kpi"><b>{escape(str(value))}</b><span>{escape(label)}</span></div>'
                                              for value, label in items) + "</div>")


def status_bar(statuses: dict[str, int]) -> str:
    total = sum(count for key, count in statuses.items() if key != "control") or 1
    parts = "".join(f'<i class="s-{key}" style="width:{100 * statuses.get(key, 0) / total:.1f}%" title="{escape(STATUS_LABELS[key])}"></i>'
                    for key in STATUS_ORDER if statuses.get(key))
    return f'<div class="am-bar">{parts}</div>'


def task_label(task: dict) -> str:
    """Подпись задачи в дереве: формулировка, статус, тренд, приоритет (markdown expander)."""
    colors = {"confirmed": "green", "confirmed_early": "blue", "preliminary": "blue", "disputed": "orange",
              "rejected": "red", "hypothesis": "gray", "control": "gray"}
    arrow = " ↑" if task["trend"] in UP else " ↓" if task["trend"] in DOWN else ""
    name = task["name"].replace("[", "(").replace("]", ")")
    return (f"{name} · :{colors[task['status']]}[{STATUS_LABELS[task['status']]}] · "
            f"{TREND_LABELS[task['trend']]}{arrow} · приоритет {task['priority']}")


def quotes_html(quotes: list[dict], limit: int | None = None) -> str:
    """Цитаты с источниками; одинаковый текст из разных источников показывается один раз с их числом."""
    groups: dict[str, list[dict]] = {}
    for quote in quotes:
        groups.setdefault(" ".join(quote["text"].casefold().split()), []).append(quote)
    shown = list(groups.values())[:limit] if limit else list(groups.values())
    parts = []
    for group in shown:
        first = group[0]
        more = f" · и ещё {len(group) - 1} {plural(len(group) - 1, 'источник', 'источника', 'источников')}" if len(group) > 1 else ""
        parts.append(f'<div class="am-quote">«{escape(first["text"])}»<small>{escape(first["source_title"])} · '
                     f'{escape(SOURCE_TYPE_LABELS.get(first.get("source_type", ""), first.get("source_type", "")))} · '
                     f'{escape(first["date"])}{escape(more)}</small></div>')
    return "".join(parts)


def why_html(task: dict) -> str:
    reason = task["reason"]
    lines = "".join(f"<li>{escape(line)}</li>" for line in reason["lines"])
    return f'<div class="am-why"><h4>{escape(reason["title"])}</h4><ul>{lines}</ul></div>'


def market_calc(task: dict) -> None:
    market = task["metrics"]["market"]
    config = load_config()
    st.markdown("**emp_share = k_emp / n_emp** — доля работодателей, у которых хотя бы одна вакансия упоминает задачу. "
                "30 вакансий одного работодателя — один голос рынка.")
    st.write(f"Окно: {market['window'][0]} — {market['window'][1]}. k_emp = {market['k_emp']}, n_emp = {market['n_emp']} "
             f"(минимум {config['min_employers']}). Вакансий с упоминанием: {market['vacancies']}.")
    if market["enough"]:
        st.write(f"Доля {percent(market['share'])}, 95 % ДИ Вильсона {percent(market['ci'][0])}–{percent(market['ci'][1])}.")
    else:
        st.write("Работодателей меньше минимума — доля не считается («мало данных», а не 0).")
    if market["share_prev"] is not None:
        st.write(f"Прошлое окно {market['window_prev'][0]} — {market['window_prev'][1]}: {market['k_emp_prev']} из "
                 f"{market['n_emp_prev']} ({percent(market['share_prev'])}). Δ = {market['delta'] * 100:+.0f} п. п., "
                 f"{'значимо: интервалы не пересекаются' if market['delta_significant'] else 'в пределах шума: интервалы пересекаются'}.")
    if market["employers"]:
        st.caption("Работодатели с упоминанием: " + "; ".join(market["employers"]))


def experts_calc(task: dict) -> None:
    experts = task["metrics"]["experts"]
    config = load_config()["experts"]
    st.markdown("**p̂ = k / n**, где n — голоса надёжных экспертов без «не знаю», k — «регулярно» + «иногда». "
                f"Интервал Вильсона, z = {config['z']}.")
    st.dataframe([{"Ответ": VERDICT_LABELS[key], "Голосов": count} for key, count in experts["verdicts"].items()],
                 hide_index=True, width="stretch")
    if experts["n"]:
        st.write(f"k = {experts['k']}, n = {experts['n']}, p̂ = {experts['p']:.2f}, ДИ {experts['ci'][0]:.3f}–{experts['ci'][1]:.3f}. "
                 f"Подтверждено, если нижняя граница > {config['majority']}; отклонено, если верхняя < {config['majority']}.")
    if experts["n_needed"] and task["status"] == "disputed":
        st.write(f"n_needed = ⌈z² · p̂(1 − p̂) / (p̂ − 0.5)²⌉ = {experts['n_needed']} (предел {config['n_max']}).")
    if experts["excluded_votes"]:
        st.caption(f"Не учтены голоса исключённых экспертов: {experts['excluded_votes']}.")
    if experts["outdated_version_votes"]:
        st.caption(f"Голоса за прежнюю формулировку (другая версия пункта): {experts['outdated_version_votes']}.")
    forecast = experts["forecast"]
    if experts["n_forecast"]:
        st.write("Прогноз «через 3 года»: " + "; ".join(f"{FORECAST_LABELS[key]} — {value}" for key, value in forecast.items())
                 + f". forecast_net = {experts['forecast_net']:+.2f}, доля автоматизации {percent(experts['automated_share'])}.")


def support_calc(task: dict) -> None:
    support = task["metrics"]["support"]
    st.markdown("**support = min(1, Σ w_тип · свежесть)** — внутри типа берётся лучший источник, каждый тип один раз.")
    st.dataframe([{"Тип": SOURCE_TYPE_LABELS.get(part["type"], part["type"]), "Вес": part["weight"],
                   "Свежесть": part["freshness"], "Вклад": part["contribution"]} for part in support["parts"]],
                 hide_index=True, width="stretch")
    st.write(f"support = {support['value']:.2f}; документальных типов (кроме экспертов): {len(support['doc_types'])}.")


def priority_calc(task: dict) -> None:
    bonus = load_config()["trend"]["bonus"][task["trend"]]
    st.markdown("**priority = importance + trend_bonus** (не ниже 1).")
    st.write(f"importance = {task['importance']} ({task['importance_source']}); trend_bonus = {bonus:+d} "
             f"(тренд «{TREND_LABELS[task['trend']]}»); priority = {task['priority']}.")
    st.caption(f"Строка таблицы статусов 7.6: № {task['status_rule']}.")


def task_detail(task: dict, content: dict) -> None:
    """Карточка задачи: «Почему здесь», цифры с раскрытием расчёта, цитаты, знания, сегменты."""
    chips = status_chip(task["status"]) + trend_chip(task["trend"]) + "".join(flag_chip(flag) for flag in task["lag_flags"])
    if task["origin"] == "expert_proposal":
        chips += flag_chip("предложено экспертами")
    if task["metrics"]["experts"]["segments"]["flag"]:
        chips += flag_chip("расслоение по сегментам")
    st.html(f"<div>{chips}</div>" + why_html(task))
    columns = st.columns(4)
    market, experts = task["metrics"]["market"], task["metrics"]["experts"]
    with columns[0].popover(f"Рынок: {market['k_emp']}/{market['n_emp']}" if market["enough"] else "Рынок: мало данных",
                            width="stretch"):
        market_calc(task)
    with columns[1].popover(f"Эксперты: {experts['k']}/{experts['n']}" if experts["n"] else "Эксперты: нет голосов",
                            width="stretch"):
        experts_calc(task)
    with columns[2].popover(f"Источники: {task['metrics']['support']['value']:.2f}", width="stretch"):
        support_calc(task)
    with columns[3].popover(f"Приоритет: {task['priority']}", width="stretch"):
        priority_calc(task)
    segments = experts["segments"]["segments"]
    if any(segments.values()):
        with st.expander("Ответы по сегментам экспертов"):
            for field, values in segments.items():
                for value, row in values.items():
                    st.write(f"{SEGMENT_LABELS.get(value, value)}: {row['k']} из {row['n']} ({percent(row['p'])})")
    if task["quotes"]:
        st.html(quotes_html(task["quotes"], 3))
        distinct = len({" ".join(quote["text"].casefold().split()) for quote in task["quotes"]})
        if distinct > 3 or len(task["quotes"]) > 3:
            with st.expander(f"Все источники ({len(task['quotes'])})"):
                st.dataframe([{"Цитата": q["text"], "Источник": q["source_title"],
                               "Тип": SOURCE_TYPE_LABELS.get(q["source_type"], q["source_type"]), "Дата": q["date"],
                               "Извлечение": q["extractor"], "Связь": q["link"]} for q in task["quotes"]],
                             hide_index=True, width="stretch")
    related = [(content["knowledge"][link["id"]], link["reason"]) for link in task["related"]]
    if related:
        with st.expander(f"Знания, умения и инструменты ({len(related)})"):
            for item, reason in related:
                st.write(f"{item['name']} — {reason}")
    if task["areas"]:
        st.caption("Области знаний: " + ", ".join(task["areas"]))
    if experts["comments"]:
        with st.expander(f"Комментарии экспертов (обезличены, {len(experts['comments'])})"):
            for comment in experts["comments"]:
                st.write(f"«{comment}»")


def needed_hint() -> None:
    """Подсказка руководителю: почему 10 или 50 экспертов (по формуле n_needed)."""
    config = load_config()["experts"]
    rows = []
    for share in (0.9, 0.8, 0.7, 0.65, 0.6, 0.55):
        needed = experts_needed(share, config["z"], config["majority"])
        verdict = (f"хватает {config['n_min']}" if needed <= config["n_min"] else
                   f"≈ {needed} {plural(needed, 'эксперт', 'эксперта', 'экспертов')}" if needed <= config["n_max"]
                   else "спорно по существу — число экспертов не поможет")
        rows.append({"Согласие экспертов": percent(share), "Нужно голосов": needed, "Вывод": verdict})
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption("n_needed = ⌈z² · p̂(1 − p̂) / (p̂ − 0.5)²⌉. Если мнения делятся почти поровну, больше людей не поможет: "
               "значит, формулировка двусмысленна или роль расслаивается. Недобросовестных отсеивают контрольные карточки.")
