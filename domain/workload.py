"""Calculate and validate proposed academic workloads independently of generated RPD text."""

from decimal import Decimal
from fractions import Fraction
import math

from pydantic import Field

from domain.schemas import Model


HOURS = {"lectures": "Лекции", "practice": "Практические занятия", "labs": "Лабораторные",
         "independent": "Самостоятельная работа", "assessment": "Аттестация"}
DEFAULT_RATE = 36
DEFAULT_TOTAL = 60 * DEFAULT_RATE
MAX_HOURS = 10_000


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


def _split(total: int, weights: list[int], minimum: int = 0) -> list[int]:
    if not weights:
        if total:
            raise ValueError("Распределённые часы не совпадают с общим объёмом.")
        return []
    remaining = total - minimum * len(weights)
    if remaining < 0:
        raise ValueError("Общего объёма часов недостаточно для всех дисциплин.")
    if not any(weights):
        weights = [1] * len(weights)
    quotas = [Fraction(remaining * value, sum(weights)) for value in weights]
    result = [math.floor(quota) for quota in quotas]
    remainder = remaining - sum(result)
    order = sorted(range(len(quotas)), key=lambda index: quotas[index] - result[index], reverse=True)
    for index in order[:remainder]:
        result[index] += 1
    return [value + minimum for value in result]


def _default_weights(subject: dict) -> list[int]:
    practical = any(topic.get("assignments") for topic in subject.get("topics", []))
    return [25, 20, 15, 35, 5] if practical else [30, 20, 0, 45, 5]


def _proposal(loads: dict, rate: int, origin: str) -> dict:
    return {"loads": loads, "hours_per_credit": rate, "origin": origin,
            "total_hours": sum(workload(row)["totalHours"] for row in loads.values())}


def proposal_from_loads(loads: dict) -> dict | None:
    if not loads:
        return None
    rate = next(iter(loads.values()))["hours_per_credit"]
    rows = {key: {**row, "hours_per_credit": rate} for key, row in loads.items()}
    return _proposal(rows, rate, "teacher")


def propose_workload(subjects: list[dict], previous: dict | None = None) -> dict:
    previous = previous or {}
    old = previous.get("loads", {})
    rate = previous.get("hours_per_credit", DEFAULT_RATE)
    total = previous.get("total_hours", DEFAULT_TOTAL)
    keys = [subject["id"] for subject in subjects]
    if not subjects:
        return _proposal({}, rate, previous.get("origin", "default"))
    if set(keys) == set(old):
        return previous
    if total < len(keys):
        total = len(keys)
    weights = [workload(old[key])["totalHours"] if key in old else
               max(1, len(subject.get("topics", []))) * rate for key, subject in zip(keys, subjects, strict=True)]
    unit = rate if total % rate == 0 and total // rate >= len(keys) else 1
    totals = _split(total // unit, weights, minimum=1)
    rows = {}
    for subject, size in zip(subjects, totals, strict=True):
        key = subject["id"]
        mix = [old[key].get(field, 0) for field in HOURS] if key in old else _default_weights(subject)
        rows[key] = {**dict(zip(HOURS, _split(size * unit, mix), strict=True)), "hours_per_credit": rate}
    return _proposal(rows, rate, previous.get("origin", "default"))


def _requested_hours(hours: int | None, credits: float | None, rate: int) -> int | None:
    if credits is None:
        return hours
    converted = Decimal(str(credits)) * rate
    if converted != converted.to_integral_value() or not 1 <= converted <= MAX_HOURS:
        raise ValueError("Указанные зачётные единицы не переводятся в допустимое целое число часов.")
    if hours is not None and hours != int(converted):
        raise ValueError("Указанные часы и зачётные единицы расходятся.")
    return int(converted)


def apply_workload_edit(subjects: list[dict], current: dict, edit: WorkloadEdit) -> dict:
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
        fixed = parts.get(key, {})
        rest = [field for field in HOURS if field not in fixed]
        leftover = sizes[key] - sum(fixed.values())
        if leftover < 0:
            raise ValueError("По дисциплине сумма отдельных видов занятий превышает её общий объём.")
        weights = [old[key].get(field, 0) for field in rest]
        distributed = dict(zip(rest, _split(leftover, weights), strict=True))
        rows[key] = {**fixed, **distributed, "hours_per_credit": rate}
    return _proposal(rows, rate, "teacher")
