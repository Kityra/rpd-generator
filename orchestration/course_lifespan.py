"""Calculate a review horizon from dated lifecycle data and comparable vacancy samples."""

import asyncio
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, timezone
from urllib.parse import urlsplit

import httpx
from pydantic import Field, model_validator

from domain.schemas import CourseLifespanAssessment, Model, SourceRecord
from orchestration.evidence_cache import source_record
from orchestration.market_analysis import dictionary
from orchestration.vacancy_search import WINDOW_DAYS


MIN_MARKET_SAMPLE = 20
MIN_MARKET_DROP_POINTS = 25
MAX_COMPARISON_DAYS = 365
LIFESPAN_UNAVAILABLE_NOTE = "Оценка актуальности курса недоступна"
INSUFFICIENT_EVIDENCE = (f"{LIFESPAN_UNAVAILABLE_NOTE}: нет подтверждённого срока поддержки "
                         "для указанной версии инструмента или заметного изменения в сопоставимых выборках вакансий.")
SUPPORT_FEED_UNAVAILABLE = (f"{LIFESPAN_UNAVAILABLE_NOTE}: официальный источник дат поддержки временно недоступен.")
_PUBLISHED = re.compile(r"^Опубликована:[ \t]*(\d{4}-\d{2}-\d{2})", re.M)
_VERSION_BOUNDARY = r"(?![\w.]|\s*[+–-]\s*\d|\s*\+)"
_NODE = re.compile(r"(?<!\w)Node(?:\.js|js)\s+v?(\d+)(?:\.(?:\d+|x)){0,2}" + _VERSION_BOUNDARY, re.I)
_DOTNET = re.compile(r"(?<!\w)\.NET(?:\s+Core)?\s+v?(\d+)(?:\.(\d+)(?:\.\d+)?)?" + _VERSION_BOUNDARY, re.I)
NODE_SCHEDULE = "https://raw.githubusercontent.com/nodejs/Release/main/schedule.json"
DOTNET_RELEASES = "https://builds.dotnet.microsoft.com/dotnet/release-metadata/releases-index.json"


class MarketSample(Model):
    topic: str
    collected_on: date
    source_ids: list[str] = Field(min_length=MIN_MARKET_SAMPLE, max_length=50)
    vacancy_keys: list[str]
    skill_counts: dict[str, int]

    @model_validator(mode="after")
    def check_counts(self):
        size = len(self.source_ids)
        if len(set(self.source_ids)) != size or len(set(self.vacancy_keys)) != size or len(self.vacancy_keys) != size:
            raise ValueError("Карточки выборки должны быть уникальны.")
        if any(not 0 < count <= size for count in self.skill_counts.values()):
            raise ValueError("Число вакансий с навыком должно входить в размер выборки.")
        return self


@dataclass(frozen=True)
class SupportDeadline:
    technology: str
    subject: str
    end: date
    source: SourceRecord


@dataclass(frozen=True)
class MarketDrop:
    skill: str
    previous: MarketSample
    current: MarketSample


def _dated(value: object) -> date | None:
    if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        try:
            return date.fromisoformat(value)
        except ValueError:
            pass
    return None


def _titles(program: dict):
    for subject in program["subjects"]:
        yield subject["title"], [subject["title"], *(topic["title"] for topic in subject["topics"])]


def collect_market_sample(topic: str, records: list[SourceRecord]) -> MarketSample | None:
    """Use the dated Rostrud cohort from one complete Business collection."""
    cards = {}
    if not records:
        return None
    as_of = max(record.retrieved_at.astimezone(timezone.utc).date() for record in records)
    for record in sorted(records, key=lambda item: item.source_id):
        published = _PUBLISHED.search(record.content)
        published_on = _dated(published[1]) if published else None
        if (record.source_type != "Rostrud" or published_on is None or not 0 <= (as_of - published_on).days < WINDOW_DAYS
                or record.retrieved_at.astimezone(timezone.utc).date() != as_of):
            continue
        url = urlsplit(record.url)
        cards.setdefault(f"{url.hostname}{url.path.rstrip('/')}", record)
    if len(cards) < MIN_MARKET_SAMPLE:
        return None
    skills = dictionary()
    counts = Counter(skill for record in cards.values() for skill in skills.detect(f"{record.title}\n{record.content}"))
    return MarketSample(topic=" ".join(topic.casefold().split()), collected_on=as_of,
                        source_ids=[record.source_id for record in cards.values()], vacancy_keys=list(cards),
                        skill_counts=dict(sorted(counts.items())))


def compare_market_samples(program: dict, current: MarketSample, history: list[MarketSample], today: date) -> MarketDrop | None:
    if not 0 <= (today - current.collected_on).days < WINDOW_DAYS:
        return None
    previous = max((sample for sample in history if sample.topic == current.topic
                    and WINDOW_DAYS <= (current.collected_on - sample.collected_on).days <= MAX_COMPARISON_DAYS
                    and not set(sample.vacancy_keys).intersection(current.vacancy_keys)),
                   key=lambda item: item.collected_on, default=None)
    if previous is None:
        return None
    skills = dictionary()
    covered = {skill for _, titles in _titles(program) for title in titles for skill in skills.detect(title)}
    old_size, new_size = len(previous.source_ids), len(current.source_ids)
    drops = [(previous.skill_counts.get(skill, 0) * new_size - current.skill_counts.get(skill, 0) * old_size, skill)
             for skill in covered]
    delta, skill = min(drops, key=lambda item: (-item[0], item[1]), default=(0, ""))
    if 100 * delta < MIN_MARKET_DROP_POINTS * old_size * new_size:
        return None
    return MarketDrop(skill, previous, current)


async def fetch_support_deadlines(program: dict, *, transport: httpx.AsyncBaseTransport | None = None
                                  ) -> tuple[list[SupportDeadline], bool]:
    # Only explicit version references in a discipline or topic identify a release channel.
    targets: dict[str, dict[str, str]] = {NODE_SCHEDULE: {}, DOTNET_RELEASES: {}}
    for subject, titles in _titles(program):
        for title in titles:
            for match in _NODE.finditer(title):
                targets[NODE_SCHEDULE].setdefault(f"v{match[1]}", subject)
            for match in _DOTNET.finditer(title):
                targets[DOTNET_RELEASES].setdefault(f"{match[1]}.{match[2] or '0'}", subject)
    if not any(targets.values()):
        return [], False

    async def read(client: httpx.AsyncClient, url: str, versions: dict[str, str]) -> tuple[list[SupportDeadline], bool]:
        try:
            async with asyncio.timeout(6):
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
                if url == NODE_SCHEDULE:
                    dates = {key: entry.get("end") for key, entry in data.items() if isinstance(entry, dict)}
                    name, page = "Node.js", "https://github.com/nodejs/Release/blob/main/schedule.json"
                else:
                    dates = {entry["channel-version"]: entry.get("eol-date") for entry in data["releases-index"]}
                    name, page = ".NET", DOTNET_RELEASES
        except (httpx.HTTPError, TimeoutError, ValueError, KeyError, TypeError, AttributeError):
            return [], True
        deadlines = []
        for version, subject in versions.items():
            if end := _dated(dates.get(version)):
                technology = f"{name} {version.removeprefix('v')}"
                evidence = source_record("documentation", page, f"{technology} · официальный график поддержки",
                                         f"Версия {version}\nОкончание поддержки: {end.isoformat()}")
                deadlines.append(SupportDeadline(technology, subject, end, evidence))
        return deadlines, False

    async with httpx.AsyncClient(timeout=5, follow_redirects=False, transport=transport) as client:
        results = await asyncio.gather(*(read(client, url, versions) for url, versions in targets.items() if versions))
    return [item for items, _ in results for item in items], any(failed for _, failed in results)


def calculate_lifespan(today: date, deadlines: list[SupportDeadline], drop: MarketDrop | None,
                       registry: dict[str, SourceRecord]) -> CourseLifespanAssessment | None:
    if not deadlines and drop is None:
        return None
    ordered = sorted(deadlines, key=lambda item: (item.end, item.technology, item.subject))
    earliest = ordered[0] if ordered else None
    points, source_ids = [], []
    if earliest:
        sources = [item.source for item in ordered]
        registry = {**registry, **{record.source_id: record for record in sources}}
        source_ids.extend(record.source_id for record in sources)
        points.append(f"Официальное окончание поддержки {earliest.technology} — {earliest.end:%d.%m.%Y}.")
        points.append(f"Версия указана в дисциплине «{earliest.subject[:150]}».")
    if drop:
        old, new = drop.previous, drop.current
        source_ids.extend([*old.source_ids, *new.source_ids])
        points.append(f"{drop.skill} в выборках «Работы России»: {old.skill_counts.get(drop.skill, 0)}/{len(old.source_ids)} "
                      f"на {old.collected_on:%d.%m.%Y} и {new.skill_counts.get(drop.skill, 0)}/{len(new.source_ids)} "
                      f"на {new.collected_on:%d.%m.%Y}.")
        points.append(f"Доля снизилась минимум на {MIN_MARKET_DROP_POINTS} п. п. в двух непересекающихся выборках. "
                      "Правило предлагает проверить этот навык сейчас.")
        points.append("Вывод относится к этой выборке вакансий и к указанному навыку.")
    else:
        points.append("Горизонт рассчитан до ближайшего подтверждённого окончания поддержки версии в программе.")
    review_by = today if drop else max(today, earliest.end)
    years = min(15.0, round((review_by - today).days / 365.2425, 2))
    method = "combined" if earliest and drop else "support_end" if earliest else "market_shift"
    return CourseLifespanAssessment.model_validate(
        {"estimated_lifespan_years": years, "is_short_lived": years <= 5, "hr_digest_points": points,
         "source_ids": list(dict.fromkeys(source_ids)), "review_by": review_by, "calculated_on": today,
         "calculation_method": method}, context={"sources": registry})
