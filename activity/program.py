"""Сопоставление программы с моделями деятельности и правила рекомендаций R1–R5 и R7 (разделы 7.10–7.12).

Чистые функции: вход — дисциплины программы и сохранённые версии моделей, выход — покрытие, рекомендации
и баланс ЗЕТ. Пороги — config/activity_model.yaml, раздел recommendations.
"""

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field

from orchestration.semantic_check import concepts

from activity.config import load_areas, load_config
from activity.engine import CONFIDENCE, DECLINING, ELIGIBLE, OTHER_FUNCTION, TREND_LABELS, percent, plural
from activity.text import AreaDictionary, jaccard, object_concepts, terms


PART_LABELS = {"mandatory_fgos": "обязательная по ФГОС", "mandatory": "обязательная часть", "elective": "часть, формируемая участниками"}
RULE_LABELS = {"R1": "Создать дисциплину", "R2": "Обновить дисциплину", "R3": "Сократить или убрать",
               "R4": "Объединить", "R5": "Сохранить как фундамент",
               "R7": "Сигнал по профилю"}
BAD_STATUSES = frozenset({"rejected", "hypothesis"})
SIGNAL_LABELS = {"risk": "Профиль под риском", "transforming": "Профиль трансформируется", "stable": "Профиль устойчив"}


@dataclass(frozen=True)
class Discipline:
    id: str
    name: str
    zet: float
    hours: int
    part: str = "mandatory"
    semester: int | None = None
    topics: tuple[str, ...] = ()
    outcomes: tuple[str, ...] = ()
    assignments: tuple[str, ...] = ()


@dataclass
class ProgramInput:
    id: str
    name: str
    total_zet: float
    targets: list[tuple[str, float]]  # (profession_id, weight)
    disciplines: list[Discipline]
    models: dict[str, dict]  # profession_id → content of the latest model version
    overrides: dict[tuple[str, str], str] = field(default_factory=dict)  # (discipline_id, task_id) → full | partial | none


def _terms(task_name: str) -> set[str]:
    return object_concepts(task_name) or terms(task_name)


def _label(term: str) -> str:
    return term[2:] if term.startswith("c:") else term


def match(task_name: str, texts: tuple[str, ...], share: float) -> list[tuple[int, list[str]]]:
    """Тексты (темы, результаты), где есть не меньше share понятий задачи. Возвращает индексы и совпавшие слова."""
    wanted = _terms(task_name)
    hits = []
    if not wanted:
        return hits
    for index, text in enumerate(texts):
        common = wanted & terms(text)
        if len(common) / len(wanted) >= share - 1e-9:
            hits.append((index, sorted(_label(term) for term in common)))
    return hits


def coverage(discipline: Discipline, task: dict, config: dict, override: str | None = None) -> dict | None:
    """full — пункт есть в темах и в результатах или заданиях; partial — только в одном из них."""
    if override == "none":
        return None
    share = config["coverage"]["match_share"]
    topics = discipline.topics or (discipline.name,)
    topic_hits = match(task["name"], topics, share)
    outcome_hits = match(task["name"], discipline.outcomes + discipline.assignments, share)
    if override in ("full", "partial"):
        return {"coverage": override, "method": "manual", "topics": [i for i, _ in topic_hits],
                "reason": "задано вручную методистом"}
    if not topic_hits and not outcome_hits:
        return None
    words = sorted({word for _, common in topic_hits + outcome_hits for word in common})
    places = []
    if topic_hits:
        places.append("темы «" + "», «".join(topics[i] for i, _ in topic_hits[:2]) + "»")
    if outcome_hits:
        others = discipline.outcomes + discipline.assignments
        places.append("результаты/задания «" + "», «".join(others[i] for i, _ in outcome_hits[:2]) + "»")
    return {"coverage": "full" if topic_hits and outcome_hits else "partial", "method": "auto",
            "topics": [i for i, _ in topic_hits],
            "reason": f"совпали понятия {', '.join(words)}: " + "; ".join(places)}


def evidence_task(profession_id: str, task: dict) -> dict:
    market, experts = task["metrics"]["market"], task["metrics"]["experts"]
    return {"profession_id": profession_id, "id": task["id"], "name": task["name"], "status": task["status"],
            "status_label": task["status_label"], "trend": task["trend"], "priority": task["priority"],
            "market": f"{market['k_emp']} из {market['n_emp']} работодателей" if market["enough"] else "мало данных",
            "experts": f"{experts['k']} из {experts['n']}" if experts["n"] else "нет голосов"}


def confidence_of(statuses: list[str]) -> str:
    levels = [CONFIDENCE.get(status, "low") for status in statuses] or ["low"]
    return "low" if "low" in levels else "medium" if "medium" in levels else "high"


def stable_key(rule: str, target: str, items: list[str]) -> str:
    return hashlib.sha256(json.dumps([rule, target, sorted(items)], ensure_ascii=False).encode()).hexdigest()[:32]


def topic_shares(discipline: Discipline, covered: dict[str, dict]) -> tuple[dict[str, float], float]:
    """Доли содержания по задачам и доля тем без связи. Каждая тема имеет одинаковый вес."""
    topics = discipline.topics or (discipline.name,)
    per_topic = 1 / len(topics)
    members: dict[int, list[str]] = {index: [] for index in range(len(topics))}
    floating = []
    for key, link in covered.items():
        indexes = members.keys() & link["topics"]
        if not indexes:
            floating.append(key)
        for index in indexes:
            members[index].append(key)
    if floating:
        empty = [index for index, group in members.items() if not group] or list(members)
        for index in empty:
            members[index].extend(floating)
    shares = dict.fromkeys(covered, 0.0)
    nothing = 0.0
    for group in members.values():
        if group:
            for key in group:
                shares[key] += per_topic / len(group)
        else:
            nothing += per_topic
    return shares, nothing


def _covered_task_keys(links: dict[str, dict]) -> set[str]:
    return {key for discipline_links in links.values() for key in discipline_links}


def analyse(program: ProgramInput, all_models: dict[str, dict] | None = None, config: dict | None = None,
            areas: list[dict] | None = None) -> dict:
    """Покрытие дисциплин, рекомендации R1–R5 и R7 и баланс ЗЕТ."""
    config = config or load_config()
    rules = config["recommendations"]
    area_dictionary = AreaDictionary(areas or load_areas())
    all_models = all_models or program.models
    weights = {profession_id: weight for profession_id, weight in program.targets if profession_id in program.models}
    links: dict[str, dict[str, dict]] = {d.id: {} for d in program.disciplines}
    tasks: dict[str, tuple[str, dict]] = {}
    for profession_id in weights:
        for task in program.models[profession_id]["tasks"].values():
            if task["is_control"]:
                continue
            key = f"{profession_id}:{task['id']}"
            tasks[key] = (profession_id, task)
            for discipline in program.disciplines:
                link = coverage(discipline, task, config, program.overrides.get((discipline.id, task["id"])))
                if link:
                    links[discipline.id][key] = link
    covered_tasks = _covered_task_keys(links)
    topic_coverage = {d.id: topic_shares(d, links[d.id]) for d in program.disciplines}
    by_id = {d.id: d for d in program.disciplines}
    recommendations = []

    def weighted(key: str) -> float:
        profession_id, task = tasks[key]
        return task["priority"] * weights[profession_id]

    # R5. Фундамент: дисциплина покрывает фундаментальную область, нужную многим задачам или профессиям.
    protected: dict[str, list[dict]] = {}
    for discipline in program.disciplines:
        text = " ".join([discipline.name, *discipline.topics, *discipline.outcomes])
        for area in area_dictionary.detect(text):
            if area_dictionary.kind(area) != "fundamental":
                continue
            users = [(profession_id, program.models[profession_id]["tasks"][task_id])
                     for profession_id in weights for task_id in program.models[profession_id]["areas"].get(area, {}).get("tasks", [])]
            professions = sorted({profession_id for profession_id, _ in users})
            if len(users) >= rules["fundamental_min_tasks"] or len(professions) >= 2:
                protected.setdefault(discipline.id, []).append({"area": area, "tasks": users, "professions": professions})
    for discipline_id, found in protected.items():
        discipline = by_id[discipline_id]
        involved = [evidence_task(profession_id, task) for item in found for profession_id, task in item["tasks"]]
        names = ", ".join(f"«{item['area']}»" for item in found)
        lines = [f"Область {item['area']}: нужна {len(item['tasks'])} "
                 f"{plural(len(item['tasks']), 'подтверждённой задаче', 'подтверждённым задачам', 'подтверждённым задачам')}"
                 f" в {len(item['professions'])} {plural(len(item['professions']), 'профессии', 'профессиях', 'профессиях')}: "
                 + "; ".join(f"«{task['name']}»" for _, task in item["tasks"][:5]) for item in found]
        recommendations.append(recommendation(
            "R5", discipline.name, f"Сохранить «{discipline.name}» как фундамент: покрывает {names}.", lines,
            involved, confidence_of([task["status"] for task in involved]),
            "Без дисциплины задачи профессии останутся без фундаментальной основы.", {"frees": 0, "needs": 0},
            [task["id"] for task in involved]))

    # R3. Сократить или убрать: доля содержания, которое не поддержано моделью деятельности.
    reductions: list[str] = []
    removed: set[str] = set()
    for discipline in program.disciplines:
        if discipline.part == "mandatory_fgos" or discipline.id in protected:
            continue
        shares, unlinked = topic_coverage[discipline.id]
        bad_keys = [key for key in shares if tasks[key][1]["status"] in BAD_STATUSES
                    or tasks[key][1]["trend"] == "disappearing"]
        share = unlinked + sum(shares[key] for key in bad_keys)
        if share < rules["reduce_share"] - 1e-9:
            continue
        remove = share >= rules["remove_share"] - 1e-9
        frees = discipline.zet if remove else round(discipline.zet * share * 2) / 2
        involved = [evidence_task(*tasks[key]) for key in bad_keys]
        lines = [f"{percent(share)} содержания приходится на отклонённые, исчезающие, непроверенные задачи "
                 "или на темы без связи с задачами. Темы имеют одинаковый вес."]
        if unlinked:
            lines.append(f"Доля тем без связи с задачами профессии — {percent(unlinked)}.")
        for task in involved:
            lines.append(f"«{task['name']}» — {task['status_label'].lower()}, тренд {TREND_LABELS[task['trend']]}.")
        statuses = [tasks[key][1]["status"] for key in bad_keys]
        confidence = "high" if statuses and all(s == "rejected" for s in statuses) and not unlinked else "medium"
        action = (f"Убрать или переработать «{discipline.name}»" if remove else f"Сократить «{discipline.name}»")
        recommendations.append(recommendation(
            "R3", discipline.name, f"{action}: освобождается ≈ {frees:g} ЗЕТ.", lines, involved, confidence,
            "В программе останется содержание, не подтверждённое моделью деятельности.", {"frees": frees, "needs": 0},
            [discipline.id, *bad_keys], weight=share * discipline.zet))
        reductions.append(discipline.name)
        if remove:
            removed.add(discipline.id)

    # R1. Создать дисциплину: функция с подтверждёнными задачами без покрытия.
    for profession_id, weight in weights.items():
        model = program.models[profession_id]
        total = sum(task["priority"] for task in model["tasks"].values() if task["status"] in ELIGIBLE)
        if not total:
            continue
        for function in model["functions"]:
            if function["id"] == OTHER_FUNCTION:
                continue
            members = [model["tasks"][task_id] for task_id in function["tasks"]]
            uncovered = [task for task in members if task["status"] in ELIGIBLE
                         and f"{profession_id}:{task['id']}" not in covered_tasks]
            share = weight * sum(task["priority"] for task in uncovered) / total
            if not uncovered or share < rules["new_discipline_min_share"] - 1e-9:
                continue
            zet = min(rules["zet_max"], max(rules["zet_min"], round(share * rules["elective_zet_pool"])))
            knowledge = sorted({model["knowledge"][link["id"]]["name"] for task in uncovered for link in task["related"]
                                if model["knowledge"][link["id"]]["level"] != "tool"})
            tools = sorted({model["knowledge"][link["id"]]["name"] for task in uncovered for link in task["related"]
                            if model["knowledge"][link["id"]]["level"] == "tool"})
            involved = [evidence_task(profession_id, task) for task in uncovered]
            lines = [f"Функция «{function['name']}» профессии «{model['profession']['name']}»: {len(uncovered)} "
                     f"{plural(len(uncovered), 'задача', 'задачи', 'задач')} без покрытия, "
                     f"{percent(share)} приоритета профессии в программе (порог {percent(rules['new_discipline_min_share'])}).",
                     "Задачи: " + "; ".join(f"«{task['name']}»" for task in uncovered)]
            if knowledge:
                lines.append("Знания и умения: " + "; ".join(knowledge[:8]))
            if tools:
                lines.append("Инструменты: " + ", ".join(tools[:10]))
            lines.append(f"Объём ≈ {zet} ЗЕТ = округл.({share:.2f} × {rules['elective_zet_pool']}) в пределах "
                         f"[{rules['zet_min']}; {rules['zet_max']}].")
            if reductions:
                lines.append("Кандидаты на сокращение: " + ", ".join(f"«{name}»" for name in reductions))
            recommendations.append(recommendation(
                "R1", function["name"], f"Создать дисциплину или модуль «{function['name']}» (≈ {zet} ЗЕТ).", lines,
                involved, confidence_of([task["status"] for task in uncovered]),
                f"{len(uncovered)} {plural(len(uncovered), 'подтверждённая задача', 'подтверждённые задачи', 'подтверждённых задач')} "
                "профессии останутся непокрытыми.", {"frees": 0, "needs": zet},
                [f"{profession_id}:{task['id']}" for task in uncovered],
                weight=sum(task["priority"] * weight for task in uncovered)))

    # R2. Обновить: добавить растущие и новые задачи функции, заменить отклонённое, исчезающее, автоматизируемое.
    # Растущую задачу функции рекомендуем той дисциплине, которая покрывает больше всего задач этой функции.
    owners: dict[tuple[str, str], tuple[int, str]] = {}
    for discipline_id in sorted(links):
        counts: Counter[tuple[str, str]] = Counter()
        for key in links[discipline_id]:
            profession_id, task = tasks[key]
            if task["status"] in ELIGIBLE:
                counts[profession_id, task["function_id"]] += 1
        for function_key, count in counts.items():
            previous = owners.get(function_key)
            if previous is None or count > previous[0]:
                owners[function_key] = (count, discipline_id)
    additions: dict[str, list[str]] = {}
    for key, (profession_id, task) in tasks.items():
        if (key in covered_tasks or task["function_id"] == OTHER_FUNCTION
                or task["status"] not in ELIGIBLE or task["trend"] not in ("new", "growing")):
            continue
        owner = owners.get((profession_id, task["function_id"]))
        if owner is not None:
            additions.setdefault(owner[1], []).append(key)
    for discipline in program.disciplines:
        if discipline.id in removed:
            continue  # дисциплину предлагается убрать (R3): обновлять её незачем
        covered = links[discipline.id]
        add = additions.get(discipline.id, [])
        replace = [key for key in covered if tasks[key][1]["status"] == "rejected"
                   or tasks[key][1]["trend"] in ("disappearing", "automated")]
        if add:
            involved = [evidence_task(*tasks[key]) for key in add]
            lines = [f"«{task['name']}» — {task['status_label'].lower()}, тренд {TREND_LABELS[task['trend']]}" for task in involved]
            recommendations.append(recommendation(
                "R2", discipline.name, f"Обновить «{discipline.name}»: добавить темы — "
                + "; ".join(f"«{task['name']}»" for task in involved[:4]) + ".", lines, involved,
                confidence_of([task["status"] for task in involved]),
                "Растущие задачи профессии останутся вне программы.", {"frees": 0, "needs": 0}, [discipline.id, *add],
                weight=sum(weighted(key) for key in add)))
        if replace:
            involved = [evidence_task(*tasks[key]) for key in replace]
            lines = []
            for key, task in zip(replace, involved, strict=True):
                if tasks[key][1]["trend"] == "automated":
                    lines.append(f"«{task['name']}» автоматизируется: учить контролю результата инструмента, "
                                 "а не ручному выполнению.")
                else:
                    lines.append(f"«{task['name']}» — {task['status_label'].lower()}, тренд {TREND_LABELS[task['trend']]}: "
                                 "заменить тему.")
            recommendations.append(recommendation(
                "R2", discipline.name, f"Обновить «{discipline.name}»: заменить темы — "
                + "; ".join(f"«{task['name']}»" for task in involved[:4]) + ".", lines, involved,
                confidence_of([task["status"] if task["status"] != "rejected" else "confirmed" for task in involved]),
                "Студенты будут учиться тому, что практика отклоняет или передаёт инструментам.",
                {"frees": 0, "needs": 0}, [discipline.id, "replace", *replace],
                weight=sum(tasks[key][1]["priority"] for key in replace)))

    # R4. Объединить: сходство Жаккара по покрытым пунктам.
    ordered = sorted(program.disciplines, key=lambda d: d.name)
    for index, left in enumerate(ordered):
        for right in ordered[index + 1:]:
            if "mandatory_fgos" in (left.part, right.part):
                continue
            a, b = set(links[left.id]), set(links[right.id])
            similarity = jaccard(a, b)
            if similarity < rules["merge_similarity"] - 1e-9:
                continue
            shared = sorted(a & b)
            frees = round(min(left.zet, right.zet) * similarity * 2) / 2
            involved = [evidence_task(*tasks[key]) for key in shared]
            recommendations.append(recommendation(
                "R4", f"{left.name} + {right.name}", f"Объединить «{left.name}» и «{right.name}»: "
                f"общие пункты — {percent(similarity)}; освобождается ≈ {frees:g} ЗЕТ.",
                [f"Сходство Жаккара по покрытым пунктам: {len(shared)} общих из {len(a | b)} = {similarity:.2f} "
                 f"(порог {rules['merge_similarity']}).", "Общие задачи: " + "; ".join(f"«{t['name']}»" for t in involved[:6])],
                involved, confidence_of([task["status"] for task in involved]),
                "Одно и то же содержание будет преподаваться дважды.", {"frees": frees, "needs": 0},
                [left.id, right.id], weight=sum(weighted(key) for key in shared)))

    # R7. Сигнал по профилю и смежные роли.
    for profession_id, weight in weights.items():
        model = program.models[profession_id]
        signal = model["profile_signal"]
        if signal["label"] == "stable":
            continue
        lines = signal_lines(signal)
        lines.extend(adjacent(program, profession_id, all_models, config))
        recommendations.append(recommendation(
            "R7", model["profession"]["name"], f"{SIGNAL_LABELS[signal['label']]}: «{model['profession']['name']}».",
            lines + ["Вакансии запаздывают, эксперты дают опережающий сигнал. Это повод для обсуждения, а не приговор."],
            [evidence_task(profession_id, model["tasks"][task_id]) for task_id in signal["tasks"]["falling"]],
            "medium", "Программа может готовить роль, спрос на которую снижается.", {"frees": 0, "needs": 0},
            [profession_id, signal["label"]],
            weight=weight * sum(model["tasks"][task_id]["priority"] for task_id in signal["tasks"]["falling"])))

    recommendations.sort(key=lambda rec: (-rec["weight"], rec["rule"], rec["target"]))
    frees = sum(rec["zet"]["frees"] for rec in recommendations)
    needs = sum(rec["zet"]["needs"] for rec in recommendations)
    disciplines_zet = sum(d.zet for d in program.disciplines)
    matrix = coverage_matrix(program, weights, links)
    return {"recommendations": recommendations, "matrix": matrix,
            "links": links,
            "unlinked_topic_share": {key: unlinked for key, (_, unlinked) in topic_coverage.items()},
            "protected": sorted(protected),
            "balance": {"frees": frees, "needs": needs, "disciplines_zet": disciplines_zet,
                        "program_zet": program.total_zet, "after": disciplines_zet - frees + needs}}


def recommendation(rule: str, target: str, action: str, lines: list[str], tasks: list[dict], confidence: str,
                   consequence: str, zet: dict, items: list[str], weight: float | None = None) -> dict:
    return {"key": stable_key(rule, target, items), "rule": rule, "rule_label": RULE_LABELS[rule], "target": target,
            "action_text": action, "lines": lines, "tasks": tasks, "confidence": confidence,
            "consequence": consequence, "zet": zet,
            "weight": round(weight if weight is not None else sum(task["priority"] for task in tasks), 4)}


def signal_lines(signal: dict) -> list[str]:
    market, tasks, experts = signal["market"], signal["tasks"], signal["experts"]
    lines = []
    if market["value"] is None:
        lines.append(f"A. Рынок: {market['note']}.")
    else:
        note = " (нормировано на объём выборки)" if market["normalized"] else f" ({market['note']})"
        lines.append(f"A. Рынок: работодателей сейчас {market['employers_now']}, год назад {market['employers_prev']}, "
                     f"изменение {market['value'] * 100:+.0f} %{note}.")
    lines.append(f"B. Задачи: {percent(tasks['value'])} приоритета подтверждённых задач снижается, исчезает или "
                 "автоматизируется." if tasks["value"] is not None else "B. Задачи: нет подтверждённых задач.")
    if experts["value"] is None:
        lines.append(f"C. Эксперты: прогноз по роли дали {experts['answered']} — меньше минимума.")
    else:
        lines.append(f"C. Эксперты: «спрос вырастет» минус «снизится» = {experts['value']:+.2f} "
                     f"({experts['answered']} ответов).")
    if signal["successors"]:
        lines.append("Во что превратится роль: " + "; ".join(
            f"«{item['name']}» — {item['count']}" for item in signal["successors"]))
    return lines


def adjacent(program: ProgramInput, profession_id: str, all_models: dict[str, dict], config: dict) -> list[str]:
    """Сколько содержания программы сохраняется при переходе на смежную роль и что добавить (R1 относительно X)."""
    target = program.models[profession_id]
    signal = target["profile_signal"]
    lines = []
    for successor in signal["successors"]:
        name = successor["name"]
        model = find_profession(all_models, name, profession_id, config)
        if model is None:
            count = successor["count"]
            lines.append(f"Постройте модель деятельности для «{name}» (назвали {count} из {signal['successors_answered']} экспертов).")
            continue
        retained = retention(target, model, config)
        uncovered = []
        for function in model["functions"]:
            if function["id"] == OTHER_FUNCTION:
                continue
            missing = [model["tasks"][task_id]["name"] for task_id in function["tasks"]
                       if model["tasks"][task_id]["status"] in ELIGIBLE and not any(
                           coverage(d, model["tasks"][task_id], config, program.overrides.get((d.id, task_id)))
                           for d in program.disciplines)]
            if missing:
                uncovered.append(f"«{function['name']}» ({len(missing)})")
        growth = model["profile_signal"]["market"]["value"]
        growth_text = f", рост рынка {growth * 100:+.0f} %" if growth is not None else ""
        lines.append(f"При переходе на «{model['profession']['name']}» сохраняется ≈ {percent(retained)} содержания "
                     f"программы{growth_text}. Добавить: " + (", ".join(uncovered) if uncovered else "—") + ".")
    return lines


def find_profession(models: dict[str, dict], name: str, exclude: str, config: dict) -> dict | None:
    """Модель профессии в базе, совпадающая с названной экспертами ролью по понятиям названия."""
    best, score = None, 0.0
    for profession_id, model in sorted(models.items()):
        value = jaccard(concepts(model["profession"]["name"]), concepts(name))
        if profession_id != exclude and value > score:
            best, score = model, value
    return best if score >= config["adjacent"]["name_similarity"] - 1e-9 else None


def retention(target: dict, other: dict, config: dict) -> float | None:
    """overlap = Σ priority задач цели, у которых есть похожая задача у X или все области знаний которых есть у X."""
    threshold = config["adjacent"]["task_similarity"]
    other_tasks = [task for task in other["tasks"].values() if task["status"] in ELIGIBLE]
    other_areas = {area for area, data in other["areas"].items() if data["tasks"]}
    own = [task for task in target["tasks"].values() if task["status"] in ELIGIBLE]
    total = sum(task["priority"] for task in own)
    if not total:
        return None
    kept = 0
    for task in own:
        similar = any(jaccard(concepts(task["name"]), concepts(item["name"])) >= threshold for item in other_tasks)
        if similar or (task["areas"] and set(task["areas"]) <= other_areas):
            kept += task["priority"]
    return kept / total


def coverage_matrix(program: ProgramInput, weights: dict, links: dict) -> list[dict]:
    """Матрица «дисциплины × функции»: сколько подтверждённых задач функции покрыто дисциплиной."""
    rows = []
    discipline_links = {discipline.id: links[discipline.id] for discipline in program.disciplines}
    covered_tasks = _covered_task_keys(discipline_links)
    for profession_id in weights:
        model = program.models[profession_id]
        for function in model["functions"]:
            eligible = [task_id for task_id in function["tasks"] if model["tasks"][task_id]["status"] in ELIGIBLE]
            eligible_keys = [f"{profession_id}:{task_id}" for task_id in eligible]
            cells = {}
            for discipline_id, coverage_links in discipline_links.items():
                matches = [coverage_links[key] for key in eligible_keys if key in coverage_links]
                cells[discipline_id] = {"covered": len(matches),
                                        "full": sum(link["coverage"] == "full" for link in matches)}
            covered = covered_tasks.intersection(eligible_keys)
            rows.append({"profession_id": profession_id, "profession": model["profession"]["name"],
                         "function_id": function["id"], "function": function["name"], "eligible": len(eligible),
                         "covered": len(covered), "cells": cells,
                         "declining": sum(model["tasks"][t]["trend"] in DECLINING for t in eligible)})
    return rows
