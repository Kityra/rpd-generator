"""Экспертная валидация (раздел 7.5): анкета, контрольные карточки, надёжность экспертов, раунды Дельфи."""

import hashlib
import random
from collections import Counter
from dataclasses import dataclass

from activity.stats import median


YES = frozenset({"regular", "sometimes"})
NO = frozenset({"other_role", "outdated"})
VERDICT_LABELS = {"regular": "Да, регулярно", "sometimes": "Да, иногда", "other_role": "Нет, это другая роль",
                  "outdated": "Устарело, так уже не делают", "dont_know": "Не знаю"}
FORECAST_LABELS = {"more": "станет важнее", "same": "не изменится", "less": "станет менее важной",
                   "automated": "в основном будет автоматизирована"}
ROLE_KIND_LABELS = {"practitioner": "практик", "lead": "тимлид", "hr": "HR", "teacher": "преподаватель"}
COMPANY_SIZE_LABELS = {"small": "малая компания", "medium": "средняя компания", "large": "крупная компания"}
DEMAND_LABELS = {"more": "спрос вырастет", "same": "не изменится", "less": "спрос снизится"}
EXCLUSION_LABELS = {"foreign": "ответ «да» на «чужие» задачи", "duplicates": "противоречия в дублях",
                    "speed": "слишком быстрые ответы", "pattern": "одинаковые ответы подряд"}


@dataclass(frozen=True)
class Expert:
    id: str
    label: str
    years: int | None = None
    role_kind: str = ""
    company_size: str = ""
    company_kind: str = ""
    region: str = ""
    role_forecast_text: str | None = None
    role_demand: str | None = None


@dataclass(frozen=True)
class Vote:
    expert_id: str
    item_id: str
    verdict: str
    item_version: int = 1
    round: int = 1
    importance: int | None = None
    forecast: str | None = None
    wording_ok: bool | None = None
    wording_suggestion: str | None = None
    comment: str | None = None
    elapsed_ms: int | None = None


def latest_votes(votes: list[Vote]) -> dict[tuple[str, str], Vote]:
    """Последний ответ каждого эксперта по каждому пункту (раунд 2 заменяет раунд 1)."""
    result: dict[tuple[str, str], Vote] = {}
    for vote in sorted(votes, key=lambda item: item.round):
        result[(vote.expert_id, vote.item_id)] = vote
    return result


def side(verdict: str) -> str | None:
    return "yes" if verdict in YES else "no" if verdict in NO else None


def assess(experts: list[Expert], votes: list[Vote], controls: dict[str, dict], names: dict[str, str],
           config: dict) -> dict[str, dict]:
    """Надёжность каждого эксперта по четырём правилам. Голоса исключённых не удаляются, а не учитываются.

    controls: {item_id: {"kind": "foreign" | "duplicate", "duplicate_of": item_id | None}}.
    """
    rules = config["experts"]["exclude"]
    latest = latest_votes(votes)
    by_expert: dict[str, list[Vote]] = {}
    for vote in votes:
        by_expert.setdefault(vote.expert_id, []).append(vote)
    result = {}
    for expert in experts:
        own = {item: vote for (expert_id, item), vote in latest.items() if expert_id == expert.id}
        reasons, codes, checks = [], [], {}
        foreign = [vote for item, vote in own.items() if controls.get(item, {}).get("kind") == "foreign"
                   and vote.verdict != "dont_know"]
        errors = [vote for vote in foreign if vote.verdict in YES]
        checks["foreign"] = {"answered": len(foreign), "errors": len(errors)}
        if foreign and len(errors) / len(foreign) >= rules["foreign_errors_share"]:
            listed = "», «".join(sorted(names.get(vote.item_id, "") for vote in errors)[:3])
            reasons.append(f"ответил «да» на {len(errors)} из {len(foreign)} «чужих» задач: «{listed}»")
            codes.append("foreign")
        pairs = mismatched = 0
        for item, vote in own.items():
            control = controls.get(item, {})
            original = own.get(control.get("duplicate_of") or "")
            if control.get("kind") == "duplicate" and original is not None:
                left, right = side(vote.verdict), side(original.verdict)
                if left and right:
                    pairs += 1
                    mismatched += left != right
        checks["duplicates"] = {"pairs": pairs, "mismatched": mismatched}
        if pairs and mismatched / pairs >= rules["duplicate_mismatch_share"]:
            reasons.append(f"ответил по-разному на одну и ту же задачу в {mismatched} из {pairs} пар-дублей")
            codes.append("duplicates")
        first_round = [vote for vote in by_expert.get(expert.id, []) if vote.round == 1]
        timings = [vote.elapsed_ms / 1000 for vote in first_round if vote.elapsed_ms is not None]
        seconds = median(timings)
        checks["median_seconds"] = round(seconds, 2) if seconds is not None else None
        if seconds is not None and seconds < rules["median_seconds_per_card"]:
            reasons.append(f"медианное время на карточку {seconds:.1f} с — меньше {rules['median_seconds_per_card']} с")
            codes.append("speed")
        # «Одинаковый ответ» — совпадение всей карточки: вердикт, важность и прогноз.
        answers = Counter((vote.verdict, vote.importance, vote.forecast) for vote in first_round)
        cards = sum(answers.values())
        share = max(answers.values()) / cards if cards else 0.0
        checks["same_answers"] = {"cards": cards, "share": round(share, 4)}
        if cards >= rules["min_cards_for_pattern"] and share >= rules["same_answer_share"]:
            reasons.append(f"{share:.0%} одинаковых ответов на {cards} карточках")
            codes.append("pattern")
        result[expert.id] = {"excluded": bool(reasons), "reasons": reasons, "codes": codes, "checks": checks}
    return result


def _rng(*parts: str) -> random.Random:
    seed = int(hashlib.sha256("\0".join(parts).encode()).hexdigest()[:16], 16)
    return random.Random(seed)


def assign_cards(expert_id: str, tasks: list[str], disputed: list[str], controls: list[dict], load: Counter,
                 config: dict) -> list[str]:
    """Карточки анкеты одного эксперта: задачи + около 10 % контрольных, порядок перемешан.

    Если задач больше max_cards_per_expert, эксперт получает все спорные и задачи с наименьшим числом
    назначений (так на каждую задачу набирается не меньше n_min голосов). Выбор детерминирован по id эксперта.
    controls: [{"id", "kind", "duplicate_of"}].
    """
    rules = config["experts"]
    share, minimum = rules["control_share"], config["controls"]["min_foreign"]
    rng = _rng("cards", expert_id)
    task_budget = max(1, rules["max_cards_per_expert"] - max(minimum, round(rules["max_cards_per_expert"] * share)))
    chosen = list(dict.fromkeys(item for item in disputed if item in tasks))[:task_budget]
    rest = [item for item in tasks if item not in chosen]
    rng.shuffle(rest)
    rest.sort(key=lambda item: load[item])  # устойчивая сортировка: случайный порядок среди равных
    chosen += rest[:task_budget - len(chosen)]
    controls_count = max(minimum, round(len(chosen) * share / (1 - share)))
    duplicates = [control["id"] for control in controls
                  if control["kind"] == "duplicate" and control.get("duplicate_of") in chosen]
    foreign = [control["id"] for control in controls if control["kind"] == "foreign"]
    rng.shuffle(duplicates)
    rng.shuffle(foreign)
    duplicates = duplicates[:max(0, controls_count - minimum)]
    cards = chosen + duplicates + foreign[:max(minimum, controls_count - len(duplicates))]
    rng.shuffle(cards)
    return cards


def changed_share(votes: list[Vote]) -> float | None:
    """Доля ответов, изменившихся во втором раунде Дельфи (по вердикту)."""
    first = {(vote.expert_id, vote.item_id): vote.verdict for vote in votes if vote.round == 1}
    second = [(vote.expert_id, vote.item_id, vote.verdict) for vote in votes if vote.round == 2]
    compared = [(key, verdict) for *key, verdict in second if tuple(key) in first]
    if not compared:
        return None
    changed = sum(1 for key, verdict in compared if first[tuple(key)] != verdict)
    return changed / len(compared)
