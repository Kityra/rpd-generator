"""Отбор фрагментов векторного поиска по смыслу: порог близости и простые правила для служебных фрагментов."""

import re
from collections.abc import Iterable

from domain.schemas import SourceRecord


DEFAULT_MIN_SCORE = 0.3
_BIBLIOGRAPHY = re.compile(r"(?:\b(?:М|СПб|Ростов-на-Дону|Новосибирск)\s*\.?\s*:|изд-во|издательство|ISBN|\bс\.\s*\d|"
                           r"\b\d+\s*с\.|URL:|ЭБС|электронно-библиотечн)", re.I)
_WORKLOAD = re.compile(r"(?:трудо[её]мкост|зач[её]тн\w*\s+единиц|\bз\.\s*е\b|\(час\)|акад\.?\s*ч|"
                       r"самостоятельн\w*\s+работ\w*\s*\(|контактн\w*\s+работ|виды\s+учебной\s+работы)", re.I)
# «(час) 302», «трудоемкость … 302»: число часов рядом с упоминанием трудоёмкости.
_HOURS_VALUE = re.compile(r"\(\s*час\.?\s*\)\s*\d|трудо[её]мкост\w*[^.\n]{0,80}?(?:\bчас\w*|\bз\.\s*е\b|\bзет\b)"
                          r"[^.\n]{0,30}?\d", re.I)
_CONTENTS = re.compile(r"(?:оглавление|содержание)\s*\n", re.I)
_DOTS = re.compile(r"\.{4,}\s*\d+\s*$", re.M)


def search_query(meta: dict, tasks: Iterable[str] = (), *, limit: int = 600) -> str:
    """Поисковый запрос из названия курса, профессии и (на повторных этапах) рабочих задач — не JSON."""
    parts = [" ".join(str(meta.get("title", "")).split())]
    if profession := " ".join(str(meta.get("profession", "")).split()):
        if profession.casefold() != parts[0].casefold():
            parts.append(f"профессия {profession}")
    manual = [" ".join(name.split()) for name in meta.get("manual_subjects", []) if name.strip()]
    if manual:
        parts.append("дисциплины: " + ", ".join(manual[:6]))
    task_list = [" ".join(task.split()) for task in tasks if task and task.strip()]
    if task_list:
        parts.append("рабочие задачи: " + "; ".join(dict.fromkeys(task_list[:8])))
    # Первая строка — название курса: по ней поиск вакансий выбирает тему.
    return "\n".join(part for part in parts if part)[:limit]


def service_reason(text: str) -> str | None:
    """Причина, по которой фрагмент — служебный (не учебное содержание), либо None."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "пустой фрагмент"
    bibliography = sum(bool(_BIBLIOGRAPHY.search(line)) for line in lines)
    if bibliography >= 3 and bibliography / len(lines) >= 0.4:
        return "список литературы"
    if _WORKLOAD.search(text):
        digits = sum(char.isdigit() for char in text)
        letters = sum(char.isalpha() for char in text)
        if (digits >= 12 and digits / max(letters, 1) > 0.08) or _HOURS_VALUE.search(text):
            return "таблица трудоёмкости"
    if _CONTENTS.search(text) or len(_DOTS.findall(text)) >= 3:
        return "оглавление"
    return None


def keep_fragment(source: SourceRecord, min_score: float) -> str | None:
    """None — фрагмент оставить; иначе причина, по которой он отброшен."""
    if reason := service_reason(source.content):
        return reason
    if source.relevance_score is not None and source.relevance_score < min_score:
        return f"близость {source.relevance_score:.2f} ниже порога {min_score:.2f}"
    return None


def filter_fragments(sources: Iterable[SourceRecord], min_score: float) -> tuple[list[SourceRecord], dict[str, int]]:
    kept, dropped = [], {}
    for source in sources:
        reason = keep_fragment(source, min_score)
        if reason is None:
            kept.append(source)
        else:
            label = reason.split(" ")[0] if reason.startswith("близость") else reason
            dropped[label] = dropped.get(label, 0) + 1
    return kept, dropped


def dropped_notice(index: str, dropped: dict[str, int]) -> str:
    labels = {"близость": "ниже порога близости"}
    parts = [f"{labels.get(reason, reason)} — {count}" for reason, count in dropped.items()]
    return f"Индекс {index}: отброшены нерелевантные фрагменты ({', '.join(parts)})."


def gap_query(gap_report: dict, labels: dict[str, str]) -> str:
    """Запрос к индексу методик интервью — по типам пробелов, а не по JSON отчёта."""
    kinds = [labels.get(kind, kind) for kind in gap_report.get("gap_types", [])]
    return "методики интервью эксперта" + (": " + ", ".join(kinds) if kinds else "")
