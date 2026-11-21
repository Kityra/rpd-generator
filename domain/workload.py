"""Расчёт и проверка учебной нагрузки. Числа считает только код; параметры — config/workload_norms.yaml.

Модель может лишь предложить относительную важность дисциплины (ранг из фиксированного набора).
"""

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from functools import lru_cache
import math
from pathlib import Path

import yaml
from pydantic import Field

from domain.schemas import Model


HOURS = {"lectures": "Лекции", "practice": "Практические занятия", "labs": "Лабораторные",
         "independent": "Самостоятельная работа", "assessment": "Аттестация"}
CLASS_HOURS = ("lectures", "practice", "labs", "independent")
MAX_HOURS = 10_000
NORMS_PATH = Path(__file__).resolve().parents[1] / "config" / "workload_norms.yaml"
# Значения для курсов, созданных до появления параметров нагрузки: 60 з.е. по 36 ч.
DEFAULT_RATE = 36
DEFAULT_TOTAL = 60 * DEFAULT_RATE


@lru_cache(maxsize=4)
def load_norms(path: str | None = None) -> dict:
    with open(path or NORMS_PATH, encoding="utf-8") as handle:
        norms = yaml.safe_load(handle)
    for program, table in norms["shares"].items():
        for kind, shares in table.items():
            if set(shares) != set(CLASS_HOURS) or abs(sum(shares.values()) - 1) > 1e-6:
                raise ValueError(f"Доли нагрузки {program}/{kind} должны покрывать {CLASS_HOURS} и давать 1.")
    return norms


@dataclass(frozen=True)
class WorkloadParams:
    """Параметры нагрузки курса из course_meta (форма создания курса)."""
    program_type: str
    hours_per_credit: int
    total_hours: int
    semesters: int
    assessment_form: str
    explicit: bool = False  # True — параметры заданы преподавателем, иначе значения по умолчанию

    @classmethod
    def from_meta(cls, meta: dict | None, norms: dict | None = None) -> "WorkloadParams":
        norms = norms or load_norms()
        meta = meta or {}
        defaults = norms["defaults"]
        settings = meta.get("workload") or {}
        program = settings.get("program_type") or defaults["program_type"]
        if program not in norms["program_types"]:
            program = "other"
        rate = int(settings.get("hours_per_credit") or defaults["hours_per_credit"])
        if settings.get("total_hours"):
            total = int(settings["total_hours"])
        elif settings.get("total_credits"):
            total = round(float(settings["total_credits"]) * rate)
        elif settings:
            total = int(norms["program_types"][program]["total_credits"]) * rate
        else:
            total = DEFAULT_TOTAL if rate == DEFAULT_RATE else 60 * rate
        form = settings.get("assessment_form") or defaults["assessment_form"]
        if form not in norms["assessment_forms"]:
            form = defaults["assessment_form"]
        return cls(program, rate, total, max(1, int(settings.get("semesters") or defaults["semesters"])),
                   form, explicit=bool(settings))


def workload(values: dict) -> dict:
    # Archived versions may predate the separate assessment field.
    hours = [values.get(field, 0) if field == "assessment" else values.get(field) for field in HOURS]
    if any(type(value) is not int or not 0 <= value <= MAX_HOURS for value in hours):
        raise ValueError("Введите целые часы от 0 до 10 000.")
    rate = values.get("hours_per_credit")
    if type(rate) is not int or not 1 <= rate <= 1000:
        raise ValueError("Укажите число часов в зачётной единице.")
    total = sum(hours)
    if not total:
        raise ValueError("Суммарная нагрузка должна быть больше нуля.")
    return {"totalHours": total, "totalCredits": total / rate}


class SubjectLoadEdit(Model):
    subject_id: str
    total_hours: int | None = Field(ge=1, le=MAX_HOURS)
    credits: float | None = Field(gt=0, allow_inf_nan=False)
    lectures: int | None = Field(ge=0, le=MAX_HOURS)
    practice: int | None = Field(ge=0, le=MAX_HOURS)
    labs: int | None = Field(ge=0, le=MAX_HOURS)
    independent: int | None = Field(ge=0, le=MAX_HOURS)
    assessment: int | None = Field(ge=0, le=MAX_HOURS)


class WorkloadEdit(Model):
    workload_only: bool
    total_hours: int | None = Field(ge=1, le=MAX_HOURS)
    total_credits: float | None = Field(gt=0, allow_inf_nan=False)
    hours_per_credit: int | None = Field(ge=1, le=1000)
    distribute_evenly: bool
    subjects: list[SubjectLoadEdit]


def _split(total: int, weights: list, minimum: int = 0) -> list[int]:
    """Целочисленное деление по весам методом наибольших остатков."""
    if not weights:
        if total:
            raise ValueError("Распределённые часы не совпадают с общим объёмом.")
        return []
    remaining = total - minimum * len(weights)
    if remaining < 0:
        raise ValueError("Общего объёма часов недостаточно для всех дисциплин.")
    weights = [Fraction(value).limit_denominator(10_000) for value in weights]
    if not any(weights):
        weights = [Fraction(1)] * len(weights)
    quotas = [remaining * value / sum(weights) for value in weights]
    result = [math.floor(quota) for quota in quotas]
    remainder = remaining - sum(result)
    order = sorted(range(len(quotas)), key=lambda index: quotas[index] - result[index], reverse=True)
    for index in order[:remainder]:
        result[index] += 1
    return [value + minimum for value in result]


def _assignment_text(assignment: dict) -> str:
    return " ".join([assignment.get("title", ""), assignment.get("instructions", ""),
                     *assignment.get("expected_artifacts", [])]).casefold()


def assignment_kind(assignment: dict, norms: dict | None = None) -> str | None:
    """«laboratory», «practical» или None — по содержанию задания, а не по факту его наличия."""
    markers = (norms or load_norms())["markers"]
    text = _assignment_text(assignment)
    if any(marker in text for marker in markers["laboratory"]):
        return "laboratory"
    if any(marker in text for marker in markers["practical"]):
        return "practical"
    return None


def subject_character(subject: dict, norms: dict | None = None) -> str:
    kinds = {assignment_kind(item, norms) for topic in subject.get("topics", []) for item in topic.get("assignments", [])}
    if "laboratory" in kinds:
        return "laboratory"
    return "practical" if "practical" in kinds else "theoretical"


def subject_weight(subject: dict, norms: dict | None = None) -> float:
    norms = norms or load_norms()
    weights = norms["weights"]
    topics = subject.get("topics", [])
    practical = sum(assignment_kind(item, norms) is not None for topic in topics for item in topic.get("assignments", []))
    value = weights["base"] + weights["per_topic"] * len(topics) + weights["per_practical_assignment"] * practical
    if subject.get("is_fundamental"):
        value *= weights["fundamental"]
    return value * norms["importance"].get(subject.get("importance") or "standard", 1.0)


def assessment_hours(form: str, norms: dict | None = None) -> int:
    return int((norms or load_norms())["assessment_forms"][form]["hours"])


def assessment_label(form: str | None, norms: dict | None = None) -> str | None:
    forms = (norms or load_norms())["assessment_forms"]
    return forms[form]["label"] if form in forms else None


def topic_hours(subject: dict, row: dict) -> dict[str, int]:
    """Часы дисциплины без аттестации по темам: вес темы — 1 + число её заданий."""
    topics = subject.get("topics", [])
    if not topics:
        return {}
    available = sum(row.get(field, 0) for field in CLASS_HOURS)
    weights = [1 + len(topic.get("assignments", [])) for topic in topics]
    return dict(zip((topic["id"] for topic in topics), _split(available, weights), strict=True))


def _annotate(subject: dict, row: dict, rate: int) -> dict:
    total = sum(row.get(field, 0) for field in HOURS)
    result = {**row, "hours_per_credit": rate, "topics": topic_hours(subject, row)}
    if total % rate == 0:
        result["credits"] = total // rate
    else:
        result.pop("credits", None)
    return result


def _subject_row(subject: dict, size: int, params: WorkloadParams, norms: dict, *, semester: int,
                 form: str | None = None) -> dict:
    form = form or params.assessment_form
    assessment = min(assessment_hours(form, norms), size)
    table = norms["shares"].get(params.program_type) or norms["shares"]["other"]
    shares = table[subject_character(subject, norms)]
    classes = dict(zip(CLASS_HOURS, _split(size - assessment, [shares[key] for key in CLASS_HOURS]), strict=True))
    row = {**classes, "assessment": assessment, "semester": semester, "assessment_form": form}
    return _annotate(subject, row, params.hours_per_credit)


def _semesters(subjects: list[dict], sizes: dict[str, int], count: int) -> dict[str, int]:
    """Последовательное заполнение семестров: сначала фундаментальное ядро, затем остальные."""
    if count <= 1:
        return {subject["id"]: 1 for subject in subjects}
    ordered = sorted(subjects, key=lambda subject: not subject.get("is_fundamental"))
    target = sum(sizes.values()) / count
    result, semester, filled = {}, 1, 0
    for subject in ordered:
        if filled >= target * semester and semester < count:
            semester += 1
        result[subject["id"]] = semester
        filled += sizes[subject["id"]]
    return result


def _proposal(loads: dict, rate: int, origin: str, params: WorkloadParams | None = None) -> dict:
    result = {"loads": loads, "hours_per_credit": rate, "origin": origin,
              "total_hours": sum(workload(row)["totalHours"] for row in loads.values())}
    if params is not None:
        result["parameters"] = {"program_type": params.program_type, "semesters": params.semesters,
                                "assessment_form": params.assessment_form, "target_hours": params.total_hours,
                                "explicit": params.explicit}
    return result


def proposal_from_loads(loads: dict) -> dict | None:
    if not loads:
        return None
    rate = next(iter(loads.values()))["hours_per_credit"]
    rows = {key: {**row, "hours_per_credit": rate} for key, row in loads.items()}
    return _proposal(rows, rate, "teacher")


def propose_workload(subjects: list[dict], previous: dict | None = None, params: WorkloadParams | None = None,
                     norms: dict | None = None) -> dict:
    """Распределить объём курса между дисциплинами, видами работ, темами и семестрами."""
    norms = norms or load_norms()
    previous = previous or {}
    params = params or WorkloadParams.from_meta(None, norms)
    old = previous.get("loads", {})
    rate = previous.get("hours_per_credit", params.hours_per_credit)
    total = previous.get("total_hours", params.total_hours)
    keys = [subject["id"] for subject in subjects]
    if not subjects:
        return _proposal({}, rate, previous.get("origin", "default"), params)
    if set(keys) == set(old):
        return previous
    params = WorkloadParams(params.program_type, rate, total, params.semesters, params.assessment_form, params.explicit)
    total = max(total, len(keys))
    computed = {subject["id"]: subject_weight(subject, norms) for subject in subjects}
    kept = [key for key in keys if key in old]
    scale = (sum(workload(old[key])["totalHours"] for key in kept) / sum(computed[key] for key in kept)) if kept else 1
    weights = [workload(old[key])["totalHours"] if key in old else computed[key] * scale for key in keys]
    # Дисциплина получает целое число з.е., если объём курса кратен з.е. и его хватает на всех.
    unit = rate if total % rate == 0 and total // rate >= len(keys) else 1
    sizes = dict(zip(keys, (size * unit for size in _split(total // unit, weights, minimum=1)), strict=True))
    semesters = _semesters(subjects, sizes, params.semesters)
    rows = {}
    for subject in subjects:
        key = subject["id"]
        if key in old:
            row = {**old[key], "semester": old[key].get("semester", semesters[key]),
                   "assessment_form": old[key].get("assessment_form", params.assessment_form)}
            rows[key] = _annotate(subject, {**row, **dict(zip(HOURS, _split(sizes[key], [old[key].get(f, 0) for f in HOURS]),
                                                                strict=True))}, rate)
        else:
            rows[key] = _subject_row(subject, sizes[key], params, norms, semester=semesters[key])
    return _proposal(rows, rate, previous.get("origin", "default"), params)


def _requested_hours(hours: int | None, credits: float | None, rate: int) -> int | None:
    if credits is None:
        return hours
    converted = Decimal(str(credits)) * rate
    if converted != converted.to_integral_value() or not 1 <= converted <= MAX_HOURS:
        raise ValueError("Указанные зачётные единицы не переводятся в допустимое целое число часов.")
    if hours is not None and hours != int(converted):
        raise ValueError("Указанные часы и зачётные единицы расходятся.")
    return int(converted)


def apply_workload_edit(subjects: list[dict], current: dict, edit: WorkloadEdit, norms: dict | None = None) -> dict:
    norms = norms or load_norms()
    old = current["loads"]
    rate = edit.hours_per_credit or current["hours_per_credit"]
    requested_total = _requested_hours(edit.total_hours, edit.total_credits, rate)
    changes = {item.subject_id: item for item in edit.subjects}
    if len(changes) != len(edit.subjects) or set(changes) - set(old):
        raise ValueError("В уточнении указаны неизвестные или повторяющиеся дисциплины.")
    pinned = {}
    parts = {}
    for key, change in changes.items():
        size = _requested_hours(change.total_hours, change.credits, rate)
        fixed = {field: getattr(change, field) for field in HOURS if getattr(change, field) is not None}
        if size is None and len(fixed) == len(HOURS):
            size = sum(fixed.values())
        if size is not None:
            pinned[key] = size
        parts[key] = fixed
    total = requested_total
    if total is None:
        total = sum(pinned.values()) if len(pinned) == len(subjects) else current["total_hours"]
    if not (edit.subjects or edit.distribute_evenly or edit.total_hours is not None or
            edit.total_credits is not None or edit.hours_per_credit is not None):
        raise ValueError("Уточните, какие часы или доли нужно изменить.")
    floating = [subject["id"] for subject in subjects if subject["id"] not in pinned]
    minima = {key: max(sum(parts.get(key, {}).values()), 1) for key in floating}
    if total == current["total_hours"] and not edit.distribute_evenly and all(
        pinned.get(key, workload(old[key])["totalHours"]) == workload(old[key])["totalHours"] for key in old
    ) and all(minima[key] <= workload(old[key])["totalHours"] for key in floating):
        sizes = {key: workload(row)["totalHours"] for key, row in old.items()}
    else:
        remainder = total - sum(pinned.values()) - sum(minima.values())
        if remainder < 0:
            raise ValueError("Указанных часов больше, чем доступно в общем объёме.")
        weights = [1 if edit.distribute_evenly else workload(old[key])["totalHours"] for key in floating]
        sizes = {**pinned, **{key: minima[key] + value
                             for key, value in zip(floating, _split(remainder, weights), strict=True)}}
    rows = {}
    for subject in subjects:
        key = subject["id"]
        fixed = dict(parts.get(key, {}))
        form = old[key].get("assessment_form")
        # Часы на аттестацию следуют форме аттестации, пока преподаватель не указал их явно.
        if "assessment" not in fixed and form in norms["assessment_forms"]:
            fixed["assessment"] = min(assessment_hours(form, norms), max(0, sizes[key] - sum(fixed.values())))
        rest = [field for field in HOURS if field not in fixed]
        leftover = sizes[key] - sum(fixed.values())
        if leftover < 0:
            raise ValueError("По дисциплине сумма отдельных видов занятий превышает её общий объём.")
        weights = [old[key].get(field, 0) for field in rest]
        distributed = dict(zip(rest, _split(leftover, weights), strict=True))
        extra = {name: old[key][name] for name in ("semester", "assessment_form") if name in old[key]}
        rows[key] = _annotate(subject, {**extra, **fixed, **distributed}, rate)
    result = _proposal(rows, rate, "teacher")
    if "parameters" in current:
        result["parameters"] = current["parameters"]
    return result


def validate_workload(subjects: list[dict], proposal: dict, params: WorkloadParams | None = None,
                      norms: dict | None = None) -> list[str]:
    """Арифметические проверки нагрузки до утверждения. Возвращает список нарушений."""
    norms = norms or load_norms()
    issues = []
    rate = proposal["hours_per_credit"]
    titles = {subject["id"]: subject["title"] for subject in subjects}
    by_id = {subject["id"]: subject for subject in subjects}
    loads = proposal["loads"]
    for key, row in loads.items():
        title = titles.get(key, key)
        total = sum(row.get(field, 0) for field in HOURS)
        if "credits" in row and total != row["credits"] * rate:
            issues.append(f"«{title}»: сумма по видам занятий {total} ч не равна объёму {row['credits']} з.е. × {rate} ч.")
        if total % rate:
            issues.append(f"«{title}»: {total} ч не кратно зачётной единице ({rate} ч).")
        form = row.get("assessment_form")
        if form in norms["assessment_forms"] and row.get("assessment", 0) != assessment_hours(form, norms):
            issues.append(f"«{title}»: на аттестацию ({assessment_label(form, norms)}) отведено {row.get('assessment', 0)} ч, "
                          f"по настройкам нужно {assessment_hours(form, norms)} ч.")
        if not sum(row.get(field, 0) for field in CLASS_HOURS):
            issues.append(f"«{title}»: после аттестации на занятия не осталось часов — увеличьте объём дисциплины.")
        if row.get("topics") and key in by_id:
            classes = sum(row.get(field, 0) for field in CLASS_HOURS)
            if sum(row["topics"].values()) != classes:
                issues.append(f"«{title}»: часы по темам ({sum(row['topics'].values())}) не равны часам занятий ({classes}).")
    course_total = sum(sum(row.get(field, 0) for field in HOURS) for row in loads.values())
    if course_total != proposal["total_hours"]:
        issues.append(f"Сумма по дисциплинам {course_total} ч не равна объёму курса {proposal['total_hours']} ч.")
    if params is not None and params.explicit and course_total != params.total_hours:
        issues.append(f"Сумма по дисциплинам {course_total} ч не совпадает с объёмом курса из параметров "
                      f"({params.total_hours} ч = {params.total_hours / rate:g} з.е.).")
    if params is not None and params.explicit and params.total_hours % rate:
        issues.append(f"Объём курса {params.total_hours} ч не кратен зачётной единице ({rate} ч).")
    return issues


def by_semester(row: dict, norms: dict | None = None) -> list[dict]:
    """Запись bySemester для РПД: дисциплина целиком в одном семестре."""
    hours = sum(row.get(field, 0) for field in HOURS)
    entry = {"semester": int(row.get("semester") or 1), "hours": hours,
             "credits": hours / row["hours_per_credit"], "lectureHours": row.get("lectures", 0),
             "seminarHours": row.get("practice", 0), "laboratoryHours": row.get("labs", 0),
             "independentHours": row.get("independent", 0), "assessmentHours": row.get("assessment", 0)}
    if label := assessment_label(row.get("assessment_form"), norms):
        entry["assessmentForm"] = label
    return [entry]


def curriculum(subject: dict, row: dict) -> list[dict]:
    """Тематический план для РПД: часы темы по видам работ в пропорции дисциплины."""
    hours = row.get("topics") or topic_hours(subject, row)
    weights = [row.get(field, 0) for field in CLASS_HOURS]
    names = ("lectures", "seminars", "laboratories", "independentStudy")
    topics = []
    for number, topic in enumerate(subject.get("topics", []), 1):
        total = hours.get(topic["id"], 0)
        parts = _split(total, weights) if any(weights) else [0, 0, 0, total]
        topics.append({"number": number, "title": topic["title"],
                       "activityHours": dict(zip(names, parts, strict=True))})
    return [{"semester": int(row.get("semester") or 1), "topics": topics}] if topics else []


def reported_workload_issues(document_workload: dict, hours_per_credit: int) -> list[str]:
    """Проверка часов, которые написала модель в консольном конвейере: ставка и сумма по семестрам."""
    issues = []
    total_hours = document_workload.get("totalHours")
    total_credits = document_workload.get("totalCredits")
    if isinstance(total_hours, (int, float)) and isinstance(total_credits, (int, float)):
        if abs(total_credits * hours_per_credit - total_hours) > 1e-6:
            issues.append(f"Трудоёмкость {total_hours} ч не равна {total_credits:g} з.е. × {hours_per_credit} ч.")
    semesters = document_workload.get("bySemester") or []
    if semesters and isinstance(total_hours, (int, float)):
        hours = [item.get("hours") for item in semesters if isinstance(item, dict)]
        if all(isinstance(value, (int, float)) for value in hours) and abs(sum(hours) - total_hours) > 1e-6:
            issues.append(f"Сумма часов по семестрам ({sum(hours):g}) не равна общей трудоёмкости ({total_hours}).")
        for item in semesters:
            if not isinstance(item, dict) or not isinstance(item.get("hours"), (int, float)):
                continue
            kinds = [item.get(name) for name in ("lectureHours", "seminarHours", "laboratoryHours",
                                                  "consultationHours", "independentHours", "assessmentHours")]
            numbers = [value for value in kinds if isinstance(value, (int, float))]
            if numbers and abs(sum(numbers) - item["hours"]) > 1e-6:
                issues.append(f"Семестр {item.get('semester')}: сумма видов работ ({sum(numbers):g}) "
                              f"не равна часам семестра ({item['hours']}).")
    return issues
