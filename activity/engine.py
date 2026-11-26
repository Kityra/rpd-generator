"""Расчёт модели деятельности (разделы 7.4–7.9, 7.12): метрики, статусы, тренды, «Почему здесь».

Чистые функции без базы данных и без LLM: те же входные данные и тот же конфиг дают тот же результат.
"""

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta

from orchestration.semantic_check import concepts

from activity.config import config_hash, load_areas, load_config
from activity.experts import (COMPANY_SIZE_LABELS, DEMAND_LABELS, EXCLUSION_LABELS, ROLE_KIND_LABELS, VERDICT_LABELS,
                              YES, Expert, Vote, assess, latest_votes)
from activity.stats import experts_needed, median, overlap, round_half_up, wilson
from activity.text import AreaDictionary


SOURCE_TYPE_LABELS = {"standard": "профстандарт", "vacancy": "вакансии", "rpd": "РПД других вузов",
                      "science": "научные и технологические публикации", "employer_doc": "документы работодателей",
                      "forecast": "прогнозы и тренды", "expert": "ответы и интервью экспертов"}
STATUS_LABELS = {"control": "Контрольная карточка", "rejected": "Отклонено экспертами", "confirmed": "Подтверждено",
                 "confirmed_early": "Подтверждено экспертами · опережающий сигнал", "disputed": "Спорно",
                 "preliminary": "Предварительно", "hypothesis": "Гипотеза"}
TREND_LABELS = {"new": "новая", "growing": "растёт", "stable": "стабильна", "no_data": "нет данных",
                "ambiguous": "неоднозначно", "declining": "снижается", "automated": "автоматизируется",
                "disappearing": "исчезает"}
# Статусы, которые участвуют в рекомендациях (строки 3, 4 и 6 таблицы 7.6).
ELIGIBLE = frozenset({"confirmed", "confirmed_early", "preliminary"})
CONFIDENCE = {"confirmed": "high", "confirmed_early": "high", "preliminary": "medium", "disputed": "low"}
DECLINING = frozenset({"declining", "disappearing", "automated"})
OTHER_FUNCTION = "other"
OTHER_FUNCTION_NAME = "Прочее (требует разметки)"
MONTHS = ("янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек")


@dataclass(frozen=True)
class Source:
    id: str
    type: str
    title: str
    origin_key: str
    collected_at: date
    url: str = ""
    published_at: date | None = None
    standard_code: str | None = None
    standard_year: int | None = None

    @property
    def day(self) -> date:
        return self.published_at or self.collected_at


@dataclass(frozen=True)
class Claim:
    id: str
    source_id: str
    kind: str
    text: str
    quote: str
    quote_ok: bool
    extractor: str
    item_id: str | None = None
    link_method: str | None = None
    foreign_role: bool = False
    rejection_reason: str | None = None
    tool: str | None = None
    result: str | None = None
    function_hint: dict | None = None


@dataclass(frozen=True)
class Item:
    id: str
    level: str
    name: str
    canonical_key: str = ""
    parent_id: str | None = None
    version: int = 1
    is_control: bool = False
    control_kind: str | None = None
    duplicate_of: str | None = None
    standard_code: str | None = None
    origin: str = "sources"
    first_round: int = 1


@dataclass
class ModelData:
    profession: dict
    sources: list[Source]
    claims: list[Claim]
    items: list[Item]
    experts: list[Expert] = field(default_factory=list)
    votes: list[Vote] = field(default_factory=list)
    manual_links: list[tuple[str, str, str]] = field(default_factory=list)  # (task_id, item_id, reason)


def plural(number: int, one: str, few: str, many: str) -> str:
    tail = number % 100
    if 11 <= tail <= 14:
        return many
    return one if number % 10 == 1 else few if 2 <= number % 10 <= 4 else many


def percent(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.0f} %"


def month(day: date) -> str:
    return f"{MONTHS[day.month - 1]} {day.year}"


def _r(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def windows(as_of: date, config: dict) -> tuple[tuple[date, date], tuple[date, date]]:
    """Текущее окно window_months и предыдущее окно той же длины (прошлый снимок рынка)."""
    days = round(config["window_months"] * 30.4375)
    start = as_of - timedelta(days=days)
    return (start, as_of), (start - timedelta(days=days), start)


def inside(day: date, window: tuple[date, date]) -> bool:
    return window[0] < day <= window[1]


def freshness(day: date, as_of: date, steps: list[dict]) -> float:
    age = max(0.0, (as_of - day).days / 365.25)
    for step in steps:
        if age <= step["max_age_years"]:
            return float(step["factor"])
    return float(steps[-1]["factor"])


class Context:
    """Общие для всех пунктов величины: источники, окна рынка, надёжные эксперты, последние голоса."""

    def __init__(self, data: ModelData, as_of: date, config: dict, areas: list[dict]):
        self.data, self.as_of, self.config = data, as_of, config
        self.areas = AreaDictionary(areas)
        self.sources = {source.id: source for source in data.sources}
        self.now, self.prev = windows(as_of, config)
        vacancies = [source for source in data.sources if source.type == "vacancy"]
        self.vacancies_now = [source for source in vacancies if inside(source.day, self.now)]
        self.vacancies_prev = [source for source in vacancies if inside(source.day, self.prev)]
        self.employers_now = {source.origin_key for source in self.vacancies_now}
        self.employers_prev = {source.origin_key for source in self.vacancies_prev}
        self.has_standard = any(source.type == "standard" for source in data.sources)
        self.claims_by_item: dict[str, list[Claim]] = {}
        for claim in data.claims:
            if claim.item_id and claim.quote_ok and not claim.foreign_role and claim.source_id in self.sources:
                self.claims_by_item.setdefault(claim.item_id, []).append(claim)
        controls = {item.id: {"kind": item.control_kind, "duplicate_of": item.duplicate_of}
                    for item in data.items if item.is_control}
        names = {item.id: item.name for item in data.items}
        self.reliability = assess(data.experts, data.votes, controls, names, config)
        self.reliable = {expert.id for expert in data.experts if not self.reliability[expert.id]["excluded"]}
        self.experts = {expert.id: expert for expert in data.experts}
        versions = {item.id: item.version for item in data.items}
        current = [vote for vote in data.votes if vote.item_version == versions.get(vote.item_id)]
        self.latest = latest_votes(current)
        self.votes_by_item: dict[str, list[Vote]] = {}
        for (_, item_id), vote in sorted(self.latest.items()):
            self.votes_by_item.setdefault(item_id, []).append(vote)
        self.outdated_votes = Counter(vote.item_id for vote in data.votes
                                      if vote.item_id in versions and vote.item_version != versions[vote.item_id])


def market_metrics(ctx: Context, claims: list[Claim]) -> dict:
    """Доля работодателей (а не вакансий), у которых хотя бы одна вакансия упоминает пункт."""
    config = ctx.config
    z = config["experts"]["z"]
    linked = {claim.source_id for claim in claims}
    hits_now = sorted({s.origin_key for s in ctx.vacancies_now if s.id in linked})
    hits_prev = sorted({s.origin_key for s in ctx.vacancies_prev if s.id in linked})
    n_now, n_prev = len(ctx.employers_now), len(ctx.employers_prev)
    enough_now, enough_prev = n_now >= config["min_employers"], n_prev >= config["min_employers"]
    share = len(hits_now) / n_now if enough_now else None
    share_prev = len(hits_prev) / n_prev if enough_prev else None
    ci = wilson(len(hits_now), n_now, z) if enough_now else None
    ci_prev = wilson(len(hits_prev), n_prev, z) if enough_prev else None
    delta = share - share_prev if share is not None and share_prev is not None else None
    return {"k_emp": len(hits_now), "n_emp": n_now, "share": _r(share), "ci": [_r(v) for v in ci] if ci else None,
            "enough": enough_now, "employers": hits_now,
            "k_emp_prev": len(hits_prev), "n_emp_prev": n_prev, "share_prev": _r(share_prev),
            "ci_prev": [_r(v) for v in ci_prev] if ci_prev else None,
            "delta": _r(delta), "delta_significant": delta is not None and not overlap(ci, ci_prev),
            "window": [ctx.now[0].isoformat(), ctx.now[1].isoformat()],
            "window_prev": [ctx.prev[0].isoformat(), ctx.prev[1].isoformat()],
            "vacancies": len({claim.source_id for claim in claims if ctx.sources[claim.source_id].type == "vacancy"})}


def support_metrics(ctx: Context, claims: list[Claim]) -> dict:
    """support = min(1, Σ_тип w_тип · свежесть). Внутри типа — лучший источник, каждый тип один раз."""
    weights, steps = ctx.config["source_weights"], ctx.config["freshness"]
    best: dict[str, tuple[float, str]] = {}
    for claim in claims:
        source = ctx.sources[claim.source_id]
        fresh = freshness(source.day, ctx.as_of, steps)
        if source.type not in best or (fresh, source.id) > best[source.type]:
            best[source.type] = (fresh, source.id)
    parts = [{"type": kind, "weight": weights.get(kind, 0.0), "freshness": fresh, "source_id": source_id,
              "contribution": _r(weights.get(kind, 0.0) * fresh)} for kind, (fresh, source_id) in sorted(best.items())]
    value = min(1.0, sum(part["contribution"] for part in parts))
    doc_types = sorted(kind for kind in best if kind != "expert")
    return {"value": _r(value), "parts": parts, "doc_types": doc_types,
            "sources": sorted({claim.source_id for claim in claims})}


def standard_metrics(ctx: Context, claims: list[Claim]) -> dict:
    hits = [(ctx.sources[claim.source_id], claim) for claim in claims if ctx.sources[claim.source_id].type == "standard"]
    codes = sorted({(claim.function_hint or {}).get("code") for _, claim in hits if (claim.function_hint or {}).get("code")})
    standards = sorted({source.standard_code for source, _ in hits if source.standard_code})
    years = sorted({source.standard_year for source, _ in hits if source.standard_year})
    return {"loaded": ctx.has_standard, "in_standard": bool(hits), "functions": codes, "standards": standards,
            "year": years[-1] if years else None}


def segment_flag(votes: list[Vote], experts: dict[str, Expert], config: dict) -> dict:
    """Расслоение: доля согласия в сегментах экспертов различается больше segment_gap."""
    rules = config["experts"]
    labels = {"company_size": COMPANY_SIZE_LABELS, "role_kind": ROLE_KIND_LABELS}
    answered = [vote for vote in votes if vote.verdict != "dont_know"]
    result = {"checked": len(answered) >= 2 * rules["segment_min_n"], "segments": {}, "flag": None}
    if not result["checked"]:
        return result
    for name in rules["segment_fields"]:
        groups: dict[str, list[Vote]] = {}
        for vote in answered:
            value = getattr(experts[vote.expert_id], name, "") or ""
            if value:
                groups.setdefault(value, []).append(vote)
        shares = {value: (sum(v.verdict in YES for v in group), len(group)) for value, group in sorted(groups.items())
                  if len(group) >= rules["segment_min_n"]}
        result["segments"][name] = {value: {"k": k, "n": n, "p": _r(k / n)} for value, (k, n) in shares.items()}
        if len(shares) >= 2 and result["flag"] is None:
            high = max(shares, key=lambda key: (shares[key][0] / shares[key][1], key))
            low = min(shares, key=lambda key: (shares[key][0] / shares[key][1], key))
            gap = shares[high][0] / shares[high][1] - shares[low][0] / shares[low][1]
            if gap > rules["segment_gap"]:
                names = labels.get(name, {})
                result["flag"] = (f"{names.get(high, high)}: это делают {percent(shares[high][0] / shares[high][1])} "
                                  f"экспертов, {names.get(low, low)}: {percent(shares[low][0] / shares[low][1])}")
    return result


def expert_metrics(ctx: Context, item: Item) -> dict:
    config = ctx.config["experts"]
    votes = ctx.votes_by_item.get(item.id, [])
    reliable = [vote for vote in votes if vote.expert_id in ctx.reliable]
    verdicts = Counter(vote.verdict for vote in reliable)
    answered = [vote for vote in reliable if vote.verdict != "dont_know"]
    k = sum(vote.verdict in YES for vote in answered)
    n = len(answered)
    ci = wilson(k, n, config["z"]) if n else None
    importance = median([vote.importance for vote in answered if vote.verdict in YES and vote.importance])
    forecasts = Counter(vote.forecast for vote in answered if vote.forecast)
    n_forecast = sum(forecasts.values())
    net = (forecasts["more"] - forecasts["less"] - forecasts["automated"]) / n_forecast if n_forecast else None
    p = k / n if n else None
    needed = experts_needed(p, config["z"], config["majority"]) if p is not None else None
    return {
        "n": n, "k": k, "p": _r(p), "ci": [_r(v) for v in ci] if ci else None,
        "verdicts": {key: verdicts[key] for key in VERDICT_LABELS},
        "dont_know": verdicts["dont_know"],
        "excluded_votes": len(votes) - len(reliable), "outdated_version_votes": ctx.outdated_votes.get(item.id, 0),
        "importance": _r(importance, 2), "forecast": {key: forecasts[key] for key in ("more", "same", "less", "automated")},
        "n_forecast": n_forecast, "forecast_net": _r(net),
        "automated_share": _r(forecasts["automated"] / n_forecast) if n_forecast else None,
        "n_needed": needed, "segments": segment_flag(answered, ctx.experts, ctx.config),
        "comments": sorted({" ".join(vote.comment.split()) for vote in reliable if vote.comment and vote.comment.strip()}),
        "wording_suggestions": sorted({" ".join(vote.wording_suggestion.split()) for vote in reliable
                                       if vote.wording_ok is False and vote.wording_suggestion
                                       and vote.wording_suggestion.strip()}),
        "rounds": sorted({vote.round for vote in reliable}),
    }


def status_of(item: Item, metrics: dict, config: dict) -> tuple[str, int]:
    """Таблица решений 7.6: срабатывает первая подходящая строка."""
    rules = config["experts"]
    experts = metrics["experts"]
    if item.is_control:
        return "control", 1
    if experts["n"] >= rules["n_min"]:
        low, high = experts["ci"]
        if high < rules["majority"]:
            return "rejected", 2
        if low > rules["majority"]:
            return ("confirmed", 3) if metrics["doc_types"] >= 1 else ("confirmed_early", 4)
        return "disputed", 5
    if metrics["support"]["value"] >= 1 - 1e-9 and metrics["doc_types"] >= 2:
        return "preliminary", 6
    return "hypothesis", 7


def dispute(metrics: dict, config: dict) -> dict | None:
    """Сколько ещё экспертов нужно спорной задаче, или «спорно по существу»."""
    experts, rules = metrics["experts"], config["experts"]
    needed = experts["n_needed"]
    if needed is None or needed > rules["n_max"]:
        return {"essential": True, "n_needed": needed}
    return {"essential": False, "n_needed": needed, "more": max(1, needed - experts["n"])}


def trend_of(item: Item, status: str, metrics: dict, config: dict) -> tuple[str, str]:
    """Метка тренда 7.7 и её основание. Тренд не зависит от статуса, кроме «предложили и подтвердили эксперты»."""
    rules, experts, market = config["trend"], metrics["experts"], metrics["market"]
    forecast_ok = experts["n_forecast"] >= config["experts"]["n_min"]
    net = experts["forecast_net"] if forecast_ok else None
    automated = experts["automated_share"] if forecast_ok else None
    delta, significant = market["delta"], market["delta_significant"]
    up_market = delta is not None and significant and delta > 0
    down_market = delta is not None and significant and delta < 0
    up_experts = net is not None and net >= rules["forecast_threshold"]
    down_experts = net is not None and net <= -rules["forecast_threshold"]
    market_note = (f"{delta * 100:+.0f} п. п. за год, {'значимо' if significant else 'в пределах шума'}"
                   if delta is not None else None)
    expert_note = f"прогноз экспертов {net:+.2f}" if net is not None else None
    if automated is not None and automated >= rules["automated_min"]:
        return "automated", f"{percent(automated)} экспертов: «в основном будет автоматизирована»"
    if (up_market and down_experts) or (down_market and up_experts):
        return "ambiguous", f"рынок: {market_note}; {expert_note}"
    declining = down_market or down_experts
    basis = "; ".join(note for note in (market_note if down_market else None, expert_note if down_experts else None) if note)
    # «Исчезает» — только если задача была на рынке раньше: иначе снижение видно лишь по прогнозу экспертов.
    if (declining and market["share"] is not None and market["share"] < rules["disappearing_max_share"]
            and market["k_emp_prev"] > 0):
        return "disappearing", f"{basis}; сейчас у {percent(market['share'])} работодателей"
    if declining:
        return "declining", basis
    new_market = (not metrics["standard"]["in_standard"] and market["share_prev"] is not None
                  and market["share_prev"] < rules["new_prev_max_share"] and market["share"] is not None
                  and market["share"] >= rules["new_now_min_share"])
    if new_market:
        return "new", f"год назад — {percent(market['share_prev'])} работодателей, сейчас — {percent(market['share'])}"
    if item.origin == "expert_proposal" and status in ("confirmed", "confirmed_early"):
        return "new", "предложили и подтвердили эксперты"
    if up_market or up_experts:
        basis = "; ".join(note for note in (market_note if up_market else None, expert_note if up_experts else None) if note)
        return "growing", basis
    if delta is not None or forecast_ok:
        return "stable", "; ".join(note for note in (market_note, expert_note) if note)
    return "no_data", "нет второго снимка рынка и мало прогнозов экспертов"


def lag_flags(status: str, metrics: dict, config: dict) -> list[str]:
    standard, market = metrics["standard"], metrics["market"]
    flags = []
    if not standard["loaded"]:
        return flags
    if (standard["in_standard"] and market["share"] is not None
            and market["share"] < config["trend"]["disappearing_max_share"] and status == "rejected"):
        year = f" (утв. {standard['year']})" if standard["year"] else ""
        flags.append(f"Устарело в профстандарте{year}")
    if not standard["in_standard"] and market["k_emp"] > 0 and status == "confirmed":
        flags.append("Профстандарт отстаёт: на рынке и у экспертов есть, в профстандарте нет")
    return flags


def importance_of(metrics: dict, config: dict) -> tuple[int, str]:
    experts, market, rules = metrics["experts"], metrics["market"], config["importance"]
    if experts["importance"] is not None:
        return round_half_up(experts["importance"]), "медиана оценок экспертов"
    if market["share"] is not None:
        for step in rules["from_market"]:
            if market["share"] >= step["min_share"]:
                return step["importance"], "оценена по вакансиям"
        return rules["market_default"], "оценена по вакансиям"
    return rules["no_data_default"], "по умолчанию: нет оценок экспертов и мало вакансий"


def quotes_of(ctx: Context, claims: list[Claim], limit: int | None = None) -> list[dict]:
    order = {"standard": 0, "vacancy": 1, "employer_doc": 2, "expert": 3, "forecast": 4, "science": 5, "rpd": 6}
    ranked = sorted(claims, key=lambda c: (order.get(ctx.sources[c.source_id].type, 9), ctx.sources[c.source_id].title, c.id))
    result, seen = [], set()
    for claim in ranked:
        source = ctx.sources[claim.source_id]
        if (source.id, claim.quote) in seen:
            continue
        seen.add((source.id, claim.quote))
        result.append({"claim_id": claim.id, "text": claim.quote, "formulation": claim.text, "source_id": source.id,
                       "source_title": source.title, "source_type": source.type, "url": source.url,
                       "date": source.day.isoformat(), "extractor": claim.extractor, "link": claim.link_method})
    return result[:limit] if limit else result


def reason_of(ctx: Context, task: dict) -> dict:
    """«Почему здесь» по шаблону, без LLM (раздел 7.8)."""
    config = ctx.config
    status, metrics = task["status"], task["metrics"]
    market, experts, standard, support = metrics["market"], metrics["experts"], metrics["standard"], metrics["support"]
    titles = {"confirmed": "Почему задача включена", "confirmed_early": "Почему задача включена: опережающий сигнал",
              "preliminary": "Почему задача включена предварительно", "disputed": "Почему задача спорная",
              "rejected": "Почему задача не включена", "hypothesis": "Почему пока не учитывается",
              "control": "Контрольная карточка"}
    lines = []
    if status == "control":
        lines.append("Карточка-ловушка для проверки внимательности экспертов. В модель и рекомендации не входит.")
        return {"title": titles[status], "lines": lines, "quotes": []}
    if status == "hypothesis":
        count = len(support["sources"])
        lines.append(f"{count} {plural(count, 'источник', 'источника', 'источников')}, "
                     f"{len(support['doc_types'])} {plural(len(support['doc_types']), 'тип', 'типа', 'типов')} документов, "
                     f"голосов экспертов — {experts['n']} (нужно не меньше {config['experts']['n_min']}).")
    if market["n_emp"] == 0:
        lines.append("Рынок: вакансии за период не загружены — доля работодателей не считается.")
    elif not market["enough"]:
        lines.append(f"Рынок: мало данных — {market['n_emp']} {plural(market['n_emp'], 'работодатель', 'работодателя', 'работодателей')} "
                     f"при минимуме {config['min_employers']}, доля не считается. Упоминают задачу: {market['k_emp']}.")
    else:
        period = f"{month(ctx.now[0] + timedelta(days=1))} – {month(ctx.now[1])}"
        lines.append(f"Упоминается у {market['k_emp']} из {market['n_emp']} работодателей ({percent(market['share'])}, "
                     f"95 % ДИ {percent(market['ci'][0])}–{percent(market['ci'][1])}, вакансии за {period}).")
    if experts["n"] == 0:
        lines.append("Эксперты ещё не голосовали." if not experts["excluded_votes"]
                     else f"Надёжных голосов нет; голоса исключённых экспертов ({experts['excluded_votes']}) не учтены.")
    else:
        text = (f"Подтвердили {experts['k']} из {experts['n']} экспертов ({percent(experts['p'])}, "
                f"95 % ДИ {percent(experts['ci'][0])}–{percent(experts['ci'][1])}).")
        if experts["n"] < config["experts"]["n_min"]:
            text += f" Это меньше минимума {config['experts']['n_min']}: нужно ещё {config['experts']['n_min'] - experts['n']}."
        if experts["excluded_votes"]:
            text += (f" Ещё {experts['excluded_votes']} {plural(experts['excluded_votes'], 'голос', 'голоса', 'голосов')} "
                     "исключённых экспертов не учтены.")
        if experts["dont_know"]:
            text += f" Ответили «не знаю»: {experts['dont_know']}."
        lines.append(text)
    if status == "disputed":
        detail = task["dispute"]
        if detail["essential"]:
            needed = f"≈ {detail['n_needed']}" if detail["n_needed"] else "бесконечно много"
            lines.append(f"Спорно по существу: при доле {percent(experts['p'])} для вывода нужно {needed} голосов — больше "
                         f"{config['experts']['n_max']}. Число экспертов не поможет: переформулируйте задачу, разбейте её "
                         "или посмотрите сегменты.")
        else:
            lines.append(f"Нужно ещё ≈ {detail['more']} {plural(detail['more'], 'эксперт', 'эксперта', 'экспертов')} "
                         f"(оценка: всего ≈ {detail['n_needed']} голосов).")
    if not standard["loaded"]:
        lines.append("Профстандарт для профессии не загружен.")
    elif standard["in_standard"]:
        year = f" (утв. {standard['year']})" if standard["year"] else ""
        codes = f" {', '.join(standard['standards'])}" if standard["standards"] else ""
        lines.append(f"Есть в профстандарте{codes}, ТФ {', '.join(standard['functions']) or '—'}{year}.")
    else:
        lines.append("В загруженном профстандарте нет.")
    lines.append(f"Подкреплённость источниками: {support['value']:.2f} из 1 "
                 f"({', '.join(SOURCE_TYPE_LABELS[kind] for kind in support['doc_types']) or 'нет документальных источников'}).")
    trend_text = f"Тренд: {TREND_LABELS[task['trend']]}"
    lines.append(f"{trend_text} ({task['trend_basis']})." if task["trend_basis"] else f"{trend_text}.")
    lines.extend(f"{flag}." for flag in task["lag_flags"])
    if experts["segments"]["flag"]:
        lines.append(f"Расслоение: {experts['segments']['flag']}.")
    lines.append(f"Приоритет {task['priority']} = важность {task['importance']} ({task['importance_source']}) "
                 f"{config['trend']['bonus'][task['trend']]:+d} за тренд.")
    return {"title": titles[status], "lines": lines, "quotes": task["quotes"][:2]}


def related_items(ctx: Context, task: Item, knowledge: list[Item]) -> list[dict]:
    """Знания, умения и инструменты задачи: тот же ТФ профстандарта, упоминание в формулировках, ручная связь."""
    texts = [task.name, *(c.text for c in ctx.claims_by_item.get(task.id, [])),
             *(c.tool for c in ctx.claims_by_item.get(task.id, []) if c.tool)]
    words = set().union(*(concepts(text) for text in texts))
    manual = {(task_id, item_id): reason for task_id, item_id, reason in ctx.data.manual_links}
    result = []
    for item in knowledge:
        reason = None
        if (task.id, item.id) in manual:
            reason = f"вручную: {manual[(task.id, item.id)]}" if manual[(task.id, item.id)] else "вручную"
        elif item.parent_id and item.parent_id == task.parent_id and item.standard_code:
            reason = f"из того же ТФ профстандарта {item.standard_code}"
        elif (terms := concepts(item.name)) and terms <= words:
            reason = "упоминается в формулировках задачи"
        if reason:
            result.append({"id": item.id, "reason": reason})
    return result


def build(data: ModelData, as_of: date, config: dict | None = None, areas: list[dict] | None = None,
          previous: dict | None = None) -> dict:
    """Версия модели деятельности: JSON без времени сборки (детерминизм)."""
    config, areas = config or load_config(), areas or load_areas()
    ctx = Context(data, as_of, config, areas)
    functions = sorted((item for item in data.items if item.level == "function"),
                       key=lambda item: (item.standard_code or "~", item.name, item.id))
    function_ids = {item.id for item in functions}
    knowledge = sorted((item for item in data.items if item.level in ("knowledge", "skill", "tool")),
                       key=lambda item: (item.level, item.name, item.id))
    tasks_out: dict[str, dict] = {}
    for item in sorted((item for item in data.items if item.level == "task"), key=lambda item: (item.name, item.id)):
        claims = ctx.claims_by_item.get(item.id, [])
        support = support_metrics(ctx, claims)
        metrics = {"market": market_metrics(ctx, claims), "support": support, "doc_types": len(support["doc_types"]),
                   "standard": standard_metrics(ctx, claims), "experts": expert_metrics(ctx, item)}
        status, rule = status_of(item, metrics, config)
        trend, basis = trend_of(item, status, metrics, config) if status != "control" else ("no_data", "")
        importance, importance_source = importance_of(metrics, config)
        priority = max(1, importance + config["trend"]["bonus"][trend])
        related = related_items(ctx, item, knowledge)
        area_names = set(ctx.areas.detect(" ".join([item.name, *(c.text for c in claims), *(c.tool or "" for c in claims)])))
        task = {
            "id": item.id, "name": item.name, "version": item.version, "origin": item.origin,
            "function_id": item.parent_id if item.parent_id in function_ids else OTHER_FUNCTION,
            "is_control": item.is_control, "control_kind": item.control_kind, "duplicate_of": item.duplicate_of,
            "status": status, "status_rule": rule, "status_label": STATUS_LABELS[status],
            "confidence": CONFIDENCE.get(status), "trend": trend, "trend_label": TREND_LABELS[trend],
            "trend_basis": basis, "importance": importance, "importance_source": importance_source,
            "priority": priority, "metrics": metrics, "lag_flags": lag_flags(status, metrics, config),
            "dispute": dispute(metrics, config) if status == "disputed" else None,
            "quotes": quotes_of(ctx, claims), "related": related, "areas": sorted(area_names),
        }
        tasks_out[item.id] = task
    knowledge_out = {}
    for item in knowledge:
        claims = ctx.claims_by_item.get(item.id, [])
        knowledge_out[item.id] = {
            "id": item.id, "level": item.level, "name": item.name, "standard_code": item.standard_code,
            "function_id": item.parent_id if item.parent_id in function_ids else None,
            "areas": ctx.areas.detect(item.name), "market": market_metrics(ctx, claims),
            "quotes": quotes_of(ctx, claims, 3), "tasks": []}
    by_area: dict[str, list[dict]] = {}
    by_function: dict[str, list[dict]] = {}
    total_priority = 0
    for task in tasks_out.values():
        area_names = set(task["areas"])
        for link in task["related"]:
            related = knowledge_out[link["id"]]
            related["tasks"].append(task["id"])
            area_names.update(related["areas"])
        task["areas"] = sorted(area_names)
        task["reason"] = reason_of(ctx, task)
        eligible = task["status"] in ELIGIBLE
        if eligible:
            total_priority += task["priority"]
        for name in task["areas"]:
            members = by_area.setdefault(name, [])
            if eligible:
                members.append(task)
        if not task["is_control"]:
            by_function.setdefault(task["function_id"], []).append(task)
    area_weights = {}
    for name, using in sorted(by_area.items()):
        area_weights[name] = {"kind": ctx.areas.kind(name), "tasks": [task["id"] for task in using],
                              "weight": _r(sum(task["priority"] for task in using) / total_priority) if total_priority else 0.0}
    function_list = [{"id": item.id, "name": item.name, "code": item.standard_code, "origin": item.origin}
                     for item in functions]
    if OTHER_FUNCTION in by_function:
        function_list.append({"id": OTHER_FUNCTION, "name": OTHER_FUNCTION_NAME, "code": None, "origin": "system"})
    for function in function_list:
        members = by_function.get(function["id"], [])
        members.sort(key=lambda task: (-task["priority"], task["name"], task["id"]))
        function["tasks"] = [task["id"] for task in members]
        function["priority"] = sum(task["priority"] for task in members if task["status"] in ELIGIBLE)
    content = {
        "profession": {key: data.profession.get(key) for key in ("id", "name", "region", "horizon_years",
                                                                  "standard_codes", "is_demo")},
        "as_of": as_of.isoformat(), "config_hash": config_hash(config, areas),
        "functions": function_list, "tasks": tasks_out, "knowledge": knowledge_out, "areas": area_weights,
        "experts": experts_report(ctx), "passport": passport(ctx, tasks_out),
        "discarded_claims": discarded(ctx), "profile_signal": profile_signal(ctx, tasks_out),
        "totals": {"eligible_priority": total_priority},
    }
    content["input_hash"] = input_hash(data, as_of, content["config_hash"])
    content["diff"] = diff(previous, content) if previous else {}
    return content


def experts_report(ctx: Context) -> dict:
    """Агрегаты по экспертам без персональных данных: сегменты, исключённые с причинами."""
    experts = sorted(ctx.data.experts, key=lambda expert: expert.label)
    excluded = [{"label": expert.label, "reasons": ctx.reliability[expert.id]["reasons"],
                 "checks": ctx.reliability[expert.id]["checks"],
                 "segment": ", ".join(filter(None, [ROLE_KIND_LABELS.get(expert.role_kind, expert.role_kind),
                                                    COMPANY_SIZE_LABELS.get(expert.company_size, expert.company_size),
                                                    f"стаж {expert.years} лет" if expert.years is not None else None]))}
                for expert in experts if ctx.reliability[expert.id]["excluded"]]
    reasons = Counter(EXCLUSION_LABELS[code] for expert in experts for code in ctx.reliability[expert.id]["codes"])
    voted = {vote.expert_id for vote in ctx.data.votes}
    reliable = [expert for expert in experts if expert.id in ctx.reliable and expert.id in voted]
    segments = {field: dict(sorted(Counter(getattr(expert, field) or "не указано" for expert in reliable).items()))
                for field in ("role_kind", "company_size", "region")}
    return {"total": len(experts), "voted": len(voted), "reliable": len(reliable), "excluded": excluded,
            "excluded_reason_counts": dict(sorted(reasons.items())), "segments": segments}


def passport(ctx: Context, tasks: dict) -> dict:
    by_type = {}
    for source in ctx.data.sources:
        entry = by_type.setdefault(source.type, {"count": 0, "origins": set(), "from": source.day, "to": source.day})
        entry["count"] += 1
        entry["origins"].add(source.origin_key)
        entry["from"], entry["to"] = min(entry["from"], source.day), max(entry["to"], source.day)
    claims = [claim for claim in ctx.data.claims if claim.source_id in ctx.sources]
    statuses = Counter(task["status"] for task in tasks.values())
    limitations = []
    if len(ctx.employers_now) < ctx.config["min_employers"]:
        limitations.append(f"Мало вакансий: {len(ctx.employers_now)} работодателей за период при минимуме "
                           f"{ctx.config['min_employers']} — доли рынка не считаются.")
    if len(ctx.employers_prev) < ctx.config["min_employers"]:
        limitations.append("Нет достаточного прошлого снимка рынка — динамика вакансий не считается.")
    if not ctx.has_standard:
        limitations.append("Профстандарт не загружен: сравнение с ПС и флаги запаздывания не считаются.")
    if not ctx.data.votes:
        limitations.append("Эксперты ещё не голосовали: статусы — только «предварительно» или «гипотеза».")
    return {
        "sources": {kind: {"count": entry["count"], "origins": len(entry["origins"]), "from": entry["from"].isoformat(),
                           "to": entry["to"].isoformat()} for kind, entry in sorted(by_type.items())},
        "employers_now": len(ctx.employers_now), "employers_prev": len(ctx.employers_prev),
        "window": [ctx.now[0].isoformat(), ctx.now[1].isoformat()],
        "claims": {"total": len(claims), "verified": sum(c.quote_ok for c in claims),
                   "rejected": sum(not c.quote_ok for c in claims),
                   "linked": sum(1 for c in claims if c.item_id and c.quote_ok)},
        "statuses": dict(sorted(statuses.items())), "limitations": limitations,
    }


def discarded(ctx: Context) -> list[dict]:
    result = []
    for claim in sorted(ctx.data.claims, key=lambda claim: (claim.source_id, claim.id)):
        if claim.quote_ok or claim.source_id not in ctx.sources:
            continue
        source = ctx.sources[claim.source_id]
        result.append({"claim_id": claim.id, "kind": claim.kind, "text": claim.text, "quote": claim.quote,
                       "reason": claim.rejection_reason or "цитата не прошла проверку", "source_id": source.id,
                       "source_title": source.title, "extractor": claim.extractor})
    return result


def successor_roles(ctx: Context) -> tuple[list[dict], int]:
    """Ответы «во что превратится роль», объединённые по понятиям, топ-3."""
    groups: dict[str, list[str]] = {}
    answered = 0
    for expert in sorted(ctx.data.experts, key=lambda expert: expert.id):
        if expert.id not in ctx.reliable or not (expert.role_forecast_text or "").strip():
            continue
        answered += 1
        key = " ".join(sorted(concepts(expert.role_forecast_text))) or expert.role_forecast_text.casefold()
        groups.setdefault(key, []).append(" ".join(expert.role_forecast_text.split()))
    top = sorted(groups.values(), key=lambda names: (-len(names), Counter(names).most_common(1)[0][0]))[:3]
    return [{"name": Counter(names).most_common(1)[0][0], "count": len(names)} for names in top], answered


def profile_signal(ctx: Context, tasks: dict) -> dict:
    """Три компонента сигнала по профилю (7.12) показываются отдельно, без единого индекса."""
    config, rules = ctx.config, ctx.config["profile_signal"]
    now, prev = len(ctx.employers_now), len(ctx.employers_prev)
    totals = (ctx.data.profession.get("meta") or {}).get("market_totals") or {}
    market = {"employers_now": now, "employers_prev": prev, "value": None, "normalized": False, "note": None}
    if now < config["min_employers"] or prev < config["min_employers"]:
        market["note"] = "мало данных: нужно не меньше {} работодателей в каждом окне".format(config["min_employers"])
    elif totals.get("now") and totals.get("prev"):
        market.update(value=_r((now / totals["now"]) / (prev / totals["prev"]) - 1), normalized=True,
                      totals={"now": totals["now"], "prev": totals["prev"], "source": totals.get("source", "")})
    else:
        market.update(value=_r(now / prev - 1), note="без нормировки на общий объём выборки источника")
    eligible = [task for task in tasks.values() if task["status"] in ELIGIBLE]
    total = sum(task["priority"] for task in eligible)
    falling = [task for task in eligible if task["trend"] in DECLINING]
    task_share = _r(sum(task["priority"] for task in falling) / total) if total else None
    reliable = [expert for expert in ctx.data.experts if expert.id in ctx.reliable]
    demands = Counter(expert.role_demand for expert in reliable if expert.role_demand)
    answered = sum(demands.values())
    expert_value = _r((demands["more"] - demands["less"]) / answered) if answered >= config["experts"]["n_min"] else None
    successors, named = successor_roles(ctx)
    reliable_count = len([expert for expert in reliable if expert.id in {vote.expert_id for vote in ctx.data.votes}])
    for successor in successors:
        successor["share"] = _r(successor["count"] / reliable_count) if reliable_count else None
    risk_conditions = {"A": market["value"] is not None and market["value"] <= rules["a_risk"],
                       "B": task_share is not None and task_share >= rules["b_risk"],
                       "C": expert_value is not None and expert_value <= rules["c_risk"]}
    transforming = (task_share is not None and task_share >= rules["b_transform"]) or any(
        (successor["share"] or 0) >= rules["successor_min_share"] for successor in successors)
    at_risk = sum(risk_conditions.values()) >= 2
    label = "risk" if at_risk else "transforming" if transforming else "stable"
    return {"label": label, "transforming": transforming, "at_risk": at_risk, "risk_conditions": risk_conditions,
            "market": market,
            "tasks": {"value": task_share, "falling": [task["id"] for task in falling], "total_priority": total},
            "experts": {"value": expert_value, "answered": answered,
                        "demand": {key: demands[key] for key in DEMAND_LABELS}},
            "successors": successors, "successors_answered": named}


def input_hash(data: ModelData, as_of: date, rules_hash: str) -> str:
    payload = {key: sorted((json.dumps(asdict(item), ensure_ascii=False, sort_keys=True, default=str)
                            for item in getattr(data, key))) for key in ("sources", "claims", "items", "experts", "votes")}
    payload.update(profession=data.profession, manual_links=sorted(data.manual_links), as_of=as_of.isoformat(),
                   config=rules_hash)
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


def diff(previous: dict, current: dict) -> dict:
    old, new = previous.get("tasks", {}), current["tasks"]
    changes = {"added": sorted(set(new) - set(old)), "removed": sorted(set(old) - set(new)),
               "status": [], "trend": []}
    for key in sorted(set(old) & set(new)):
        if old[key]["status"] != new[key]["status"]:
            changes["status"].append({"id": key, "from": old[key]["status"], "to": new[key]["status"]})
        if old[key]["trend"] != new[key]["trend"]:
            changes["trend"].append({"id": key, "from": old[key]["trend"], "to": new[key]["trend"]})
    return changes


def dumps(content: dict) -> str:
    """Каноническая сериализация версии модели (для экспорта и проверки детерминизма)."""
    return json.dumps(content, ensure_ascii=False, sort_keys=True, indent=1)
