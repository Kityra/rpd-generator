"""Анализ рынка и навыков кодом: частоты навыков, покрытие дисциплинами, пробелы и приоритеты.

Входные данные — загруженный методистом файл (JSON или CSV) или карточки вакансий из реестра курса.
Модель в расчёте не участвует; без данных отчёт честно сообщает «рыночные данные не загружены».
Настройки и словарь навыков — config/market_skills.yaml.
"""

import csv
import io
import json
import re
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

import yaml

from domain.schemas import ExpertToolLink, MarketAnalysis, MarketDataset, MarketSkill, MarketVacancy, SkillCoverage
from domain.skill_mapper import key, skill_mapper
from orchestration.semantic_check import tokens


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "market_skills.yaml"
VACANCY_TYPES = frozenset({"HH", "Rostrud", "EmployerCareer", "RabotaRu"})
_PUBLISHED = re.compile(r"^Опубликована:\s*(\d{4}-\d{2}-\d{2})", re.M)
_SKILL_LINES = re.compile(r"^(?:Навыки|Ключевые навыки|Технологии|Skills|Key skills):[ \t]*(.+)$", re.I | re.M)


@lru_cache(maxsize=4)
def load_market_config(path: str | None = None) -> dict:
    with open(path or CONFIG_PATH, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _contains(haystack: list[str], needle: tuple[str, ...]) -> bool:
    size = len(needle)
    return bool(size) and any(tuple(haystack[index:index + size]) == needle for index in range(len(haystack) - size + 1))


class SkillDictionary:
    """Канонические навыки и их названия; навыки из загруженных данных добавляются к словарю."""

    def __init__(self, groups: list[list[str]]):
        self.aliases: dict[str, list[tuple[str, ...]]] = {}
        for group in groups:
            self.add(group[0], group)

    def add(self, name: str, aliases) -> str:
        name = " ".join(name.split())
        variants = {tuple(tokens(alias)) for alias in [name, *aliases] if tokens(alias)}
        existing = self.find(name)
        canonical = existing or name
        self.aliases.setdefault(canonical, [])
        for variant in variants:
            if variant not in self.aliases[canonical]:
                self.aliases[canonical].append(variant)
        return canonical

    def find(self, name: str) -> str | None:
        key = tuple(tokens(name))
        for canonical, variants in self.aliases.items():
            if key in variants:
                return canonical
        return None

    def detect(self, text: str) -> set[str]:
        words = tokens(text)
        return {name for name, variants in self.aliases.items() if any(_contains(words, variant) for variant in variants)}


def dictionary(dataset: MarketDataset | None = None) -> SkillDictionary:
    result = SkillDictionary([*load_market_config()["skills"], *skill_mapper().tools])
    for skill in dataset.skills if dataset else ():
        result.add(skill.name, skill.aliases)
    return result


def _split_skills(value: str) -> list[str]:
    return [part.strip() for part in re.split(r"[;|\n]|,(?![^()]*\))", value or "") if part.strip()]


def parse_market_file(name: str, data: bytes) -> MarketDataset:
    """JSON по шаблону templates/market_data.json или CSV: навыки (skill, weight, vacancies) либо вакансии (title, skills)."""
    text = data.decode("utf-8-sig")
    if name.casefold().endswith(".json"):
        return MarketDataset.model_validate(json.loads(text))
    rows = list(csv.DictReader(io.StringIO(text), delimiter=";" if text.count(";") > text.count(",") else ","))
    if not rows:
        raise ValueError("CSV-файл пуст.")
    columns = {key.strip().casefold(): key for key in rows[0] if key}
    def column(*names):
        return next((columns[item] for item in names if item in columns), None)
    skill_col, title_col = column("skill", "навык"), column("title", "вакансия", "название")
    source = f"Файл {name}"
    if skill_col:
        weight_col, count_col = column("weight", "вес"), column("vacancies", "вакансий", "частота")
        alias_col, task_col = column("aliases", "синонимы"), column("tasks", "задачи")
        skills = []
        for row in rows:
            if not (row.get(skill_col) or "").strip():
                continue
            skills.append(MarketSkill(
                name=row[skill_col].strip(),
                weight=float(row[weight_col].replace(",", ".")) if weight_col and row.get(weight_col) else None,
                vacancies=int(row[count_col]) if count_col and (row.get(count_col) or "").strip().isdigit() else None,
                aliases=_split_skills(row.get(alias_col, "")) if alias_col else [],
                tasks=_split_skills(row.get(task_col, "")) if task_col else []))
        return MarketDataset(source=source, skills=skills)
    if title_col:
        id_col = column("id", "vacancy_id", "идентификатор")
        skills_col, text_col = column("skills", "навыки"), column("text", "requirements", "требования", "описание")
        vacancies = [MarketVacancy(id=(row.get(id_col) or "").strip() or None, title=row[title_col].strip(),
                                   skills=_split_skills(row.get(skills_col, "")) if skills_col else [],
                                   text=(row.get(text_col) or "") if text_col else "")
                     for row in rows if (row.get(title_col) or "").strip()]
        return MarketDataset(source=source, vacancies=vacancies)
    raise ValueError("В CSV нужен столбец skill (навыки) или title (вакансии). См. templates/.")


def market_vacancy_ids(registry: dict, vacancy_ids: list[str]) -> list[str]:
    """Добавить найденные индивидуальные карточки HH и Работа.ру к выборке курса."""
    cards = list(vacancy_ids)
    for source_id, record in registry.items():
        url = urlsplit(record["url"])
        if record["source_type"] == "RabotaRu" or (
            record["source_type"] == "HH" and (url.hostname or "").split(".")[-2:] == ["hh", "ru"]
            and re.fullmatch(r"/vacancy/\d+/?", url.path)
        ):
            cards.append(source_id)
    return list(dict.fromkeys(cards))


def dataset_from_registry(registry: dict, vacancy_ids: list[str]) -> MarketDataset | None:
    """Выборка из карточек вакансий, найденных для курса (Rostrud, карьерные сайты, Работа.ру, HH)."""
    cards = {}
    for source_id in vacancy_ids:
        if source_id in registry and registry[source_id]["source_type"] in VACANCY_TYPES:
            card = registry[source_id]
            url = urlsplit(card["url"])
            identity = (url.hostname or "").removeprefix("www.") + url.path.rstrip("/")
            cards.setdefault(identity or source_id, card)
    if not cards:
        return None
    dates = sorted(match.group(1) for card in cards.values() for match in [_PUBLISHED.search(card["content"])] if match)
    period = f"{dates[0]} — {dates[-1]}" if dates else None
    kinds = ", ".join(sorted({card["source_type"] for card in cards.values()}))
    return MarketDataset(source=f"Найденные вакансии ({kinds})", period=period,
                         collected_at=max((card.get("retrieved_at") or "")[:10] for card in cards.values()) or None,
                         vacancies=[MarketVacancy(id=identity, title=card["title"][:300],
                             skills=[skill for line in _SKILL_LINES.findall(card["content"]) for skill in _split_skills(line)],
                             text=card["content"][:20_000]) for identity, card in cards.items()])


def _subject_texts(subject: dict) -> tuple[str, list[tuple[str, str]], str]:
    topics = [(topic["title"], topic["title"]) for topic in subject.get("topics", [])]
    assignments = " ".join(" ".join([item.get("title", ""), item.get("instructions", ""),
                                     *item.get("expected_artifacts", []), *item.get("quality_criteria", [])])
                           for topic in subject.get("topics", []) for item in topic.get("assignments", []))
    return subject["title"], topics, assignments


def analyse_market(program: dict, dataset: MarketDataset | None, activity: dict | None = None,
                   overrides: dict[str, list[str]] | None = None) -> MarketAnalysis:
    """Детерминированный расчёт покрытия навыков рынка дисциплинами и темами курса."""
    if dataset is None:
        return MarketAnalysis(status="no_data", warnings=["Рыночные данные не загружены: загрузите файл с навыками или "
                                                          "вакансиями (шаблон в templates/) либо дождитесь поиска вакансий."])
    config = load_market_config()["thresholds"]
    skills_dictionary = dictionary(dataset)
    occurrences: dict[str, set[str]] = {}
    original_tags: dict[str, set[str]] = {}
    # Сначала зарегистрировать все явные теги, чтобы распознавать их и в тексте других карточек.
    for vacancy in dataset.vacancies:
        for name in vacancy.skills:
            original_tags.setdefault(skills_dictionary.add(name, []), set()).add(name)
    sample_ids = set()
    for index, vacancy in enumerate(dataset.vacancies):
        identity = f"id:{vacancy.id}" if vacancy.id else f"row:{index}"
        sample_ids.add(identity)
        found = {skills_dictionary.find(name) or name for name in vacancy.skills}
        found |= skills_dictionary.detect(f"{vacancy.title}\n{vacancy.text}")
        for name in found:
            occurrences.setdefault(name, set()).add(identity)
    sample = len(sample_ids) or None
    declared = {skills_dictionary.find(item.name) or item.name: item for item in dataset.skills}
    names = list(dict.fromkeys([*declared, *sorted(occurrences, key=lambda key: (-len(occurrences[key]), key))]))
    max_weight = max((item.weight for item in declared.values() if item.weight), default=None)
    warnings = []
    if sample is not None and sample < config["min_sample"]:
        warnings.append(f"В выборке {sample} вакансий — доли навыков ненадёжны (нужно не меньше {config['min_sample']}).")

    tasks = (activity or {}).get("tasks", [])
    overrides = overrides or {}
    subjects = program.get("subjects", [])
    texts = {}
    for subject in subjects:
        title, topics, assignments = _subject_texts(subject)
        texts[subject["title"]] = tokens(title), [(label, tokens(text)) for label, text in topics], tokens(assignments)
    task_texts = [(task["title"], tokens(" ".join([task["title"], *task.get("stack", [])]))) for task in tasks]
    credits = {"full": 1.0, "partial": config["partial_credit"], "none": 0.0}
    rows = []
    for name in names:
        item = declared.get(name)
        ids = sorted(occurrences.get(name, set())) if sample else None
        vacancies = len(ids) if ids is not None else (item.vacancies if item else None)
        share = (vacancies / sample) if sample and vacancies is not None else None
        weight = item.weight if item else None
        if share is not None:
            critical, base = share >= config["critical_share"], share
        elif weight is not None and max_weight:
            critical, base = weight / max_weight >= config["critical_weight"], weight / max_weight
        else:
            critical, base = False, 0.5
        variants = skills_dictionary.aliases.get(name, [tuple(tokens(name))])

        def mentions(words: list[str]) -> bool:
            return any(_contains(words, variant) for variant in variants)

        full_subjects, topics, partial_subjects = [], [], []
        for title, (subject_title, topic_list, assignments) in texts.items():
            hit_topics = [label for label, text in topic_list if mentions(text)]
            if mentions(subject_title) or hit_topics:
                full_subjects.append(title)
                topics.extend(hit_topics)
            elif mentions(assignments):
                partial_subjects.append(title)
        linked_tasks = [title for title, words in task_texts if mentions(words)]
        linked_tasks += [task for task in (item.tasks if item else []) if task not in linked_tasks]
        manual = name in overrides
        if manual:
            full_subjects = [title for title in overrides[name] if title in texts]
            partial_subjects = []
        level = "full" if full_subjects else ("partial" if partial_subjects else "none")
        rows.append(SkillCoverage(skill=name, vacancies=vacancies, share=round(share, 3) if share is not None else None,
                                  weight=weight, critical=critical, level=level,
                                  subjects=full_subjects or partial_subjects, topics=topics, tasks=linked_tasks,
                                  priority=round(base * (1 - credits[level]), 3), manual=manual, vacancy_ids=ids,
                                  raw_skills=sorted(original_tags.get(name, set())) or [item.name if item else name]))
    rows.sort(key=lambda row: (-row.priority, not row.critical, row.skill))
    covered_subjects = {title for row in rows if row.level != "none" for title in row.subjects}
    weak = [subject["title"] for subject in subjects if subject["title"] not in covered_subjects
            and not any(topic.get("activity_task_ids") for topic in subject.get("topics", []))]
    scored = [row for row in rows if row.critical] or rows
    fit = round(sum(credits[row.level] for row in scored) / len(scored), 3) if scored else None
    task_scores: dict[str, dict] = {}
    for row in rows:
        for task in row.tasks:
            entry = task_scores.setdefault(task, {"task": task, "priority": 0.0, "skills": []})
            entry["priority"] = round(entry["priority"] + row.priority, 3)
            entry["skills"].append(row.skill)
    return MarketAnalysis(status="ok", source=dataset.source, sample_size=sample, period=dataset.period,
                          region=dataset.region, skills=rows, uncovered=[row.skill for row in rows if row.level == "none"],
                          weak_subjects=weak, market_fit=fit,
                          task_priorities=sorted(task_scores.values(), key=lambda entry: -entry["priority"]),
                          warnings=warnings)


def with_competences(analysis: MarketAnalysis, meta: dict, mappings: dict | None = None) -> MarketAnalysis:
    if analysis.competences:
        return analysis
    config = load_market_config()["thresholds"]
    rows = skill_mapper().aggregate(analysis.skills, analysis.sample_size, config,
                                    profession=meta.get("profession") or meta.get("title", ""), learned=mappings)
    scored = [row for row in rows if row.critical] or rows
    credit = {"full": 1.0, "partial": config["partial_credit"], "none": 0.0}
    fit = round(sum(credit[row.level] for row in scored) / len(scored), 3) if scored else None
    warnings = list(analysis.warnings)
    if any(row.vacancy_ids is None and row.vacancies is not None and len(row.raw_skills) > 1 for row in rows):
        warnings.append("Для объединённых навыков без идентификаторов вакансий показана нижняя граница числа вакансий.")
    return analysis.model_copy(update={"competences": rows, "market_fit": fit, "warnings": list(dict.fromkeys(warnings))})


def course_market(meta: dict, program: dict, registry: dict, vacancy_ids: list[str], activity: dict | None,
                  *, mappings: dict | None = None, saved: dict | None = None, profile: dict | None = None) -> MarketAnalysis:
    """Рыночный отчёт курса: загруженный файл важнее найденных вакансий."""
    market = meta.get("market") or {}
    if market.get("dataset"):
        dataset = MarketDataset.model_validate(market["dataset"])
    else:
        # Старые версии могли содержать отдельные карточки HH без списка ID выборки.
        ids = vacancy_ids or ([] if saved else market_vacancy_ids(registry, []))
        dataset = dataset_from_registry(registry, ids)
    if dataset is None and saved:
        analysis = MarketAnalysis.model_validate(saved)
        if analysis.skills:
            analysis = analysis.model_copy(update={"competences": []})
    else:
        analysis = analyse_market(program, dataset, activity, market.get("overrides"))
    return with_expert_tools(with_competences(analysis, meta, mappings), profile)


def with_expert_tools(analysis: MarketAnalysis, profile: dict | None) -> MarketAnalysis:
    """Добавить сведения интервью к таблице, сохранив статистику вакансий."""
    mapper = skill_mapper()
    tools = {}
    for raw in (profile or {}).get("expert_tool_links", []):
        link = ExpertToolLink.model_validate(raw)
        known = mapper.rules.get(key(link.tool_name))
        name = (known.tool_name if known else None) or link.tool_name
        tools.setdefault(link.competence_name, {}).setdefault(key(name), name)
    if not tools:
        return analysis
    rows = []
    for row in analysis.competences:
        market_keys = {key(name) for name in row.tools}
        annotated = [name for tag, name in tools.get(row.competence_name, {}).items() if tag not in market_keys]
        rows.append(row.model_copy(update={"expert_tools": annotated}))
    return analysis.model_copy(update={"competences": rows})
