"""Проверки смысла кодом: понятия текста, покрытие названия дисциплины цитатой, дубли, связь с задачами,
качество подтверждения задач источниками. Настройки — config/analysis.yaml."""

import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

import yaml

from orchestration.relevance_filter import service_reason


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "analysis.yaml"
_TOKEN = re.compile(r"[a-zа-я0-9+#]+", re.I)
_SUFFIXES = sorted("""ностями ностях ностью остями ениями ениях ением ения ений ение ости ость ами ями ого его ому ему
ыми ими ых их ую юю ая яя ое ее ые ие ый ий ой ей ом ем ах ях ов ев ам ям ы и а я о е у ю ь й""".split(), key=len, reverse=True)
_YEAR = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")
_FUNCTION_WORDS = frozenset("""and the for with from into this that are was were has have its not but или для при как это
его она они что чем над под без про через также между после перед всех всего если тоже""".split())


@lru_cache(maxsize=4)
def load_analysis(path: str | None = None) -> dict:
    with open(path or CONFIG_PATH, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def stem(word: str) -> str:
    word = word.casefold().replace("ё", "е")
    if word.isascii():
        return word[:-1] if len(word) > 4 and word.endswith("s") else word
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def tokens(text: str) -> list[str]:
    """Основы всех слов текста (включая сокращения), по порядку."""
    return [stem(token) for token in _TOKEN.findall(text.casefold().replace("ё", "е"))]


@lru_cache(maxsize=4)
def _vocabulary(path: str | None = None) -> tuple[list[tuple[tuple[str, ...], str]], frozenset[str]]:
    config = load_analysis(path)
    phrases = []
    for group in config["synonyms"]:
        canonical = "c:" + group[0]
        for phrase in group:
            phrases.append((tuple(tokens(phrase)), canonical))
    phrases.sort(key=lambda item: len(item[0]), reverse=True)
    generic = frozenset(stem(word) for word in [*config["generic_terms"], *_FUNCTION_WORDS])
    return phrases, generic


def concepts(text: str) -> set[str]:
    """Понятия текста: синонимы сведены к каноническому имени, остальные слова — к основе."""
    phrases, generic = _vocabulary()
    words = tokens(text)
    found, index = set(), 0
    while index < len(words):
        for phrase, canonical in phrases:
            if phrase and tuple(words[index:index + len(phrase)]) == phrase:
                found.add(canonical)
                index += len(phrase)
                break
        else:
            token = words[index]
            if len(token) >= 3 and token not in generic and not token.isdigit():
                found.add(token)
            index += 1
    return found


def title_coverage(title: str, text: str) -> float:
    required = concepts(title)
    if not required:
        return 1.0
    return len(required & concepts(text)) / len(required)


def excerpt_supports_title(title: str, excerpt: str) -> bool:
    """Цитата относится к дисциплине: она не служебная и покрывает понятия названия."""
    if service_reason(excerpt):
        return False
    required = concepts(title)
    threshold = 1.0 if len(required) <= 3 else load_analysis()["rationale"]["min_title_coverage"]
    return title_coverage(title, excerpt) >= threshold - 1e-9


def explanation_matches(explanation: str, excerpts: list[str]) -> bool:
    shared = concepts(explanation) & set().union(*(concepts(text) for text in excerpts)) if excerpts else set()
    return len(shared) >= load_analysis()["rationale"]["min_explanation_overlap"]


def rationale_problem(title: str, explanation: str, excerpts: list[str]) -> str | None:
    """Проверка кодом: подтверждают ли цитаты объяснение и дисциплину. None — проверка пройдена."""
    if any(service_reason(text) for text in excerpts):
        return "цитата из служебного фрагмента (таблица трудоёмкости, список литературы)"
    if not any(excerpt_supports_title(title, text) for text in excerpts):
        return "цитаты не содержат понятий из названия дисциплины"
    if not explanation_matches(explanation, excerpts):
        return "объяснение не опирается на приведённые цитаты"
    return None


def similarity(left: str, right: str) -> float:
    a, b = concepts(left), concepts(right)
    return len(a & b) / len(a | b) if a and b else 0.0


def duplicate_subjects(subjects: list[dict]) -> list[dict]:
    """Дубли и почти-дубли дисциплин по названию."""
    threshold = load_analysis()["composition"]["duplicate_similarity"]
    pairs = []
    for index, left in enumerate(subjects):
        for right in subjects[index + 1:]:
            score = similarity(left["title"], right["title"])
            a, b = concepts(left["title"]), concepts(right["title"])
            contained = bool(a and b) and (a <= b or b <= a) and min(len(a), len(b)) >= 2
            if score >= threshold or contained:
                pairs.append({"subjects": [left["title"], right["title"]], "similarity": round(score, 2)})
    return pairs


def unlinked_subjects(subjects: list[dict], course_meta: dict, activity: dict | None) -> list[str]:
    """Дисциплины без связи с профессией и задачами деятельности."""
    tasks = (activity or {}).get("tasks", [])
    reference = concepts(" ".join([course_meta.get("title", ""), course_meta.get("profession", ""),
                                   *(task.get("title", "") for task in tasks),
                                   *(item for task in tasks for item in task.get("stack", []))]))
    result = []
    for subject in subjects:
        linked_topics = any(topic.get("activity_task_ids") for topic in subject.get("topics", []))
        text = " ".join([subject["title"], *(topic["title"] for topic in subject.get("topics", []))])
        if not linked_topics and not (concepts(text) & reference):
            result.append(subject["title"])
    return result


def composition_report(program: dict, course_meta: dict, activity: dict | None) -> dict:
    subjects = program.get("subjects", [])
    fundamental = [subject["title"] for subject in subjects if subject.get("is_fundamental")]
    return {"duplicates": duplicate_subjects(subjects),
            "unlinked": unlinked_subjects(subjects, course_meta, activity),
            "fundamental_share": round(len(fundamental) / len(subjects), 2) if subjects else 0.0}


def limit_fundamental(suggestions: list[dict], sources: dict) -> tuple[list[dict], list[str]]:
    """Флаг «фундаментальное ядро» — только с источником RPD или Books и не больше заданной доли дисциплин."""
    config = load_analysis()["composition"]
    allowed_types = set(config["fundamental_source_types"])
    notes = []
    result = []
    for item in suggestions:
        if item.get("is_fundamental") and not any(
            getattr(sources.get(key), "source_type", None) in allowed_types for key in item.get("source_ids", [])
        ):
            item = {**item, "is_fundamental": False}
            notes.append(f"«{item['name']}»: флаг фундаментального ядра снят — нет подтверждения в РПД или учебниках.")
        result.append(item)
    limit = max(1, int(len(result) * config["max_fundamental_share"])) if result else 0
    flagged = [index for index, item in enumerate(result) if item.get("is_fundamental")]
    if len(flagged) > limit:
        rank = {"core": 0, "major": 1, "standard": 2, "minor": 3}
        keep = set(sorted(flagged, key=lambda index: (rank.get(result[index].get("importance"), 2), index))[:limit])
        for index in flagged:
            if index not in keep:
                result[index] = {**result[index], "is_fundamental": False}
        notes.append(f"Фундаментальное ядро ограничено {limit} из {len(result)} дисциплин "
                     f"(предложено {len(flagged)}).")
    return result, notes


def source_year(source) -> int | None:
    match = _YEAR.search(source.content[:600]) or _YEAR.search(source.title)
    return int(match.group(1)) if match else None


def source_quality(source, now: datetime | None = None) -> float:
    config = load_analysis()["coverage"]
    weight = float(config["source_weights"].get(source.source_type, 0.5))
    year = source_year(source)
    if year is None:
        return weight
    age = (now or datetime.now(timezone.utc)).year - year
    for step in config["freshness"]:
        if age <= step["max_age_years"]:
            return weight * float(step["factor"])
    return weight * float(config["freshness"][-1]["factor"])


def independence_key(source) -> tuple[str, str]:
    """Фрагменты одного файла или страницы — один источник."""
    host = urlsplit(source.url).hostname or ""
    if host == "ai.api.cloud.yandex.net":
        return source.source_type, " ".join(source.title.casefold().split())
    return source.source_type, source.url


def task_support(source_ids: list[str], sources: dict, types, now: datetime | None = None) -> float:
    """Подтверждение задачи: сумма качества независимых источников нужного типа, не больше 1."""
    best: dict[tuple[str, str], float] = {}
    for key in source_ids:
        source = sources.get(key)
        if source is None or source.source_type not in types:
            continue
        group = independence_key(source)
        best[group] = max(best.get(group, 0.0), source_quality(source, now))
    required = float(load_analysis()["coverage"]["required_weight"])
    return min(1.0, sum(best.values()) / required)


def weighted_coverage(tasks, sources: dict, types, now: datetime | None = None) -> float:
    tasks = list(tasks)
    if not tasks:
        return 0.0
    return sum(task_support(task.source_ids, sources, types, now) for task in tasks) / len(tasks)
