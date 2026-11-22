"""Доверительный интервал Вильсона и оценка числа экспертов (раздел 7.4, 7.5)."""

import math


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95 % интервал Вильсона для доли k из n. None — данных нет."""
    if n <= 0:
        return None
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def experts_needed(p: float, z: float = 1.96, majority: float = 0.5) -> int | None:
    """Сколько голосов нужно, чтобы интервал отошёл от порога большинства. None — доля ровно на пороге."""
    gap = p - majority
    if abs(gap) < 1e-12:
        return None
    return math.ceil(z * z * p * (1 - p) / (gap * gap) - 1e-9)


def round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def median(values: list[float]) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    middle = len(ordered) // 2
    return float(ordered[middle]) if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def overlap(left: tuple[float, float] | None, right: tuple[float, float] | None) -> bool:
    """Пересекаются ли интервалы. Если одного нет, различие значимым не считается."""
    if left is None or right is None:
        return True
    return left[0] <= right[1] and right[0] <= left[1]
