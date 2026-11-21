"""Анализ рынка и навыков кодом: частоты навыков, покрытие дисциплинами, пробелы и приоритеты.

Входные данные — загруженный преподавателем файл (JSON или CSV) или карточки вакансий из реестра курса.
Модель в расчёте не участвует; без данных отчёт честно сообщает «рыночные данные не загружены».
Настройки и словарь навыков — config/market_skills.yaml.
"""

import csv
import io
import json
import re
from functools import lru_cache
from pathlib import Path

import yaml

from domain.schemas import MarketAnalysis, MarketDataset, MarketSkill, MarketVacancy, SkillCoverage
from orchestration.semantic_check import tokens


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "market_skills.yaml"
VACANCY_TYPES = frozenset({"HH", "Rostrud", "EmployerCareer", "RabotaRu"})
_PUBLISHED = re.compile(r"^Опубликована:\s*(\d{4}-\d{2}-\d{2})", re.M)


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
    result = SkillDictionary(load_market_config()["skills"])
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
        skills_col, text_col = column("skills", "навыки"), column("text", "requirements", "требования", "описание")
        vacancies = [MarketVacancy(title=row[title_col].strip(),
                                   skills=_split_skills(row.get(skills_col, "")) if skills_col else [],
                                   text=(row.get(text_col) or "") if text_col else "")
                     for row in rows if (row.get(title_col) or "").strip()]
        return MarketDataset(source=source, vacancies=vacancies)
    raise ValueError("В CSV нужен столбец skill (навыки) или title (вакансии). См. templates/.")


def dataset_from_registry(registry: dict, vacancy_ids: list[str]) -> MarketDataset | None:
    """Выборка из карточек вакансий, найденных для курса (Rostrud, карьерные сайты, Работа.ру, HH)."""
    cards = [registry[key] for key in vacancy_ids if key in registry and registry[key]["source_type"] in VACANCY_TYPES]
    if not cards:
        return None
    dates = sorted(match.group(1) for card in cards for match in [_PUBLISHED.search(card["content"])] if match)
    period = f"{dates[0]} — {dates[-1]}" if dates else None
    kinds = ", ".join(sorted({card["source_type"] for card in cards}))
    return MarketDataset(source=f"Найденные вакансии ({kinds})", period=period,
                         collected_at=max((card.get("retrieved_at") or "")[:10] for card in cards) or None,
                         vacancies=[MarketVacancy(title=card["title"][:300], text=card["content"][:20_000]) for card in cards])


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
    # Частоты навыков по вакансиям выборки.
    counts: dict[str, int] = {}
    for vacancy in dataset.vacancies:
        found = {skills_dictionary.add(name, []) for name in vacancy.skills}
        found |= skills_dictionary.detect(f"{vacancy.title}\n{vacancy.text}")
        for name in found:
            counts[name] = counts.get(name, 0) + 1
    sample = len(dataset.vacancies) or None
    declared = {skills_dictionary.find(item.name) or item.name: item for item in dataset.skills}
    names = list(dict.fromkeys([*declared, *sorted(counts, key=lambda key: -counts[key])]))
    max_weight = max((item.weight for item in declared.values() if item.weight), default=None)
    warnings = []
    if sample is not None and sample < config["min_sample"]:
        warnings.append(f"В выборке {sample} вакансий — доли навыков ненадёжны (нужно не меньше {config['min_sample']}).")

    tasks = (activity or {}).get("tasks", [])
    overrides = overrides or {}
    subjects = program.get("subjects", [])
    texts = {subject["title"]: _subject_texts(subject) for subject in subjects}
    rows = []
    for name in names:
        item = declared.get(name)
        vacancies = counts.get(name) if sample else (item.vacancies if item else None)
        share = (vacancies / sample) if sample and vacancies is not None else None
        weight = item.weight if item else None
        if share is not None:
            critical, base = share >= config["critical_share"], share
        elif weight is not None and max_weight:
            critical, base = weight / max_weight >= config["critical_weight"], weight / max_weight
        else:
            critical, base = False, 0.5
        variants = skills_dictionary.aliases.get(name, [tuple(tokens(name))])

        def mentions(text: str) -> bool:
            words = tokens(text)
            return any(_contains(words, variant) for variant in variants)

        full_subjects, topics, partial_subjects = [], [], []
        for title, (subject_title, topic_list, assignments) in texts.items():
            hit_topics = [label for label, text in topic_list if mentions(text)]
            if mentions(subject_title) or hit_topics:
                full_subjects.append(title)
                topics.extend(hit_topics)
            elif mentions(assignments):
                partial_subjects.append(title)
        linked_tasks = [task["title"] for task in tasks if mentions(" ".join([task["title"], *task.get("stack", [])]))]
        linked_tasks += [task for task in (item.tasks if item else []) if task not in linked_tasks]
        manual = name in overrides
        if manual:
            full_subjects = [title for title in overrides[name] if title in texts]
            partial_subjects = []
        level = "full" if full_subjects else ("partial" if partial_subjects else "none")
        credit = {"full": 1.0, "partial": config["partial_credit"], "none": 0.0}[level]
        rows.append(SkillCoverage(skill=name, vacancies=vacancies, share=round(share, 3) if share is not None else None,
                                  weight=weight, critical=critical, level=level,
                                  subjects=full_subjects or partial_subjects, topics=topics, tasks=linked_tasks,
                                  priority=round(base * (1 - credit), 3), manual=manual))
    rows.sort(key=lambda row: (-row.priority, not row.critical, row.skill))
    covered_subjects = {title for row in rows if row.level != "none" for title in row.subjects}
    weak = [subject["title"] for subject in subjects if subject["title"] not in covered_subjects
            and not any(topic.get("activity_task_ids") for topic in subject.get("topics", []))]
    scored = [row for row in rows if row.critical] or rows
    credit = {"full": 1.0, "partial": config["partial_credit"], "none": 0.0}
    fit = round(sum(credit[row.level] for row in scored) / len(scored), 3) if scored else None
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


def course_market(meta: dict, program: dict, registry: dict, vacancy_ids: list[str], activity: dict | None) -> MarketAnalysis:
    """Рыночный отчёт курса: загруженный файл важнее найденных вакансий."""
    market = meta.get("market") or {}
    dataset = MarketDataset.model_validate(market["dataset"]) if market.get("dataset") else \
        dataset_from_registry(registry, vacancy_ids)
    return analyse_market(program, dataset, activity, market.get("overrides"))
