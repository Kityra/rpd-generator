"""Retrieve recent job postings from open, first-party sources for Business."""

import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from html.parser import HTMLParser
import json
import logging
import re
from urllib.parse import urlsplit

import httpx

from domain.schemas import EvidenceResults, SourceType
from orchestration.evidence_cache import source_record


LOG = logging.getLogger(__name__)
ROSTRUD_URL = "https://opendata.trudvsem.ru/api/v1/vacancies"
CAREERS_URL = "https://yandex.ru/jobs/vacancies"
WINDOW_DAYS = 45
VACANCY_SOURCE_TYPES = frozenset({"Rostrud", "EmployerCareer"})
_ROLES = (
    (("mlops", "млопс", "llmops"), ("MLOps", "LLMOps"), ("mlops", "llmops", "млопс")),
    (("ml", "llm", "ai инженер", "ии инженер", "data scientist", "data science", "машинное обучение",
      "машинного обучения", "mlинженер", "nlp"),
     ("ML", "LLM", "Data Scientist"), ("ml", "llm", "data scientist", "машинного обучения",
                                       "data science", "ai engineer", "ai инженер", "ии инженер", "nlp",
                                       "по искусственному интеллекту")),
    (("devops", "девопс", "sre"), ("DevOps", "SRE"), ("devops", "sre", "девопс")),
    (("qa", "тестировщик", "тестирование", "тестированию", "quality assurance"),
     ("QA", "тестировщик", "инженер по тестированию"),
     ("qa", "тестировщик", "тестированию", "quality assurance")),
    (("аналитик данных", "анализ данных", "data analyst"), ("аналитик данных", "Data Analyst"),
     ("аналитик данных", "data analyst")),
    (("data engineer", "инженер данных", "дата инженер"), ("Data Engineer", "инженер данных"),
     ("data engineer", "инженер данных", "инженер по данным")),
    (("frontend", "фронтенд", "фронт енд"), ("Frontend", "фронтенд"), ("frontend", "фронтенд", "фронт енд")),
    (("backend", "бэкенд", "бекенд"), ("Backend", "бэкенд"), ("backend", "бэкенд", "бекенд")),
    (("python",), ("Python",), ("python",)),
)


def _normal(text: str) -> str:
    return " ".join(re.findall(r"[\w+#]+", text.casefold().replace("ё", "е")))


def _matches(value: str, terms: tuple[str, ...]) -> bool:
    text = f" {_normal(value)} "
    return any(f" {term} " in text for term in terms)


def _search_terms(topic: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    normalized = _normal(topic)
    for aliases, queries, titles in _ROLES:
        if _matches(topic, aliases):
            return queries, titles
    return (topic,), (normalized,) if normalized else ()


def _matches_job(title: str, terms: tuple[str, ...]) -> bool:
    if terms[0] == "ml" and _matches(title, ("product manager", "продакт менеджер", "менеджер продукта")):
        return False
    return _matches(title, terms)


def _date(value: object) -> date | None:
    try:
        return date.fromisoformat(value[:10]) if isinstance(value, str) else None
    except ValueError:
        return None


class _PageData(HTMLParser):
    def __init__(self, *, listing: bool = False):
        super().__init__()
        self.listing = listing
        self.links: dict[str, str] = {}
        self.structured: list[str] = []
        self._path: str | None = None
        self._json = False
        self._parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if self.listing and tag == "a":
            href = attributes.get("href") or ""
            path = urlsplit(href).path if href.startswith("/jobs/vacancies/") else ""
            if re.fullmatch(r"/jobs/vacancies/[a-zA-Z0-9_-]+-\d+", path):
                self._path = path
                self._parts = []
        elif not self.listing and tag == "script" and attributes.get("type") == "application/ld+json":
            self._json = True
            self._parts = []

    def handle_data(self, data):
        if self._path or self._json:
            self._parts.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._path:
            self.links.setdefault(self._path, " ".join(self._parts))
            self._path = None
        elif tag == "script" and self._json:
            self.structured.append("".join(self._parts))
            self._json = False


class _PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data):
        self.parts.append(data)


def _plain(value: object, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    parser = _PlainText()
    parser.feed(value[:12000])
    parser.close()
    return " ".join(" ".join(parser.parts).split())[:limit]


@dataclass(frozen=True)
class Vacancy:
    kind: SourceType
    url: str
    title: str
    employer: str
    published: date
    details: str

    def content(self) -> str:
        return f"Работодатель: {self.employer}\nОпубликована: {self.published.isoformat()}\n{self.details}"


def _rostrud_card(raw: object, cutoff: date, today: date, terms: tuple[str, ...]) -> Vacancy | None:
    if not isinstance(raw, dict) or not isinstance(item := raw.get("vacancy"), dict):
        return None
    title = _plain(item.get("job-name"), 200)
    published = _date(item.get("creation-date"))
    url = item.get("vac_url")
    if not (title and published and cutoff <= published <= today and _matches_job(title, terms)):
        return None
    if not isinstance(url, str) or not re.fullmatch(r"https://trudvsem\.ru/vacancy/card/[\w-]+/[\w-]+/?", url):
        return None
    company = item.get("company")
    employer = _plain(company.get("name"), 180) if isinstance(company, dict) else ""
    if not employer:
        return None
    requirements = item.get("requirement")
    requirements = requirements if isinstance(requirements, dict) else {}
    region = item.get("region")
    region = region if isinstance(region, dict) else {}
    lines = []
    for label, value, limit in (
        ("Обязанности", item.get("duty"), 800),
        ("Требования", item.get("requirements") or requirements.get("qualification") or item.get("qualification"), 650),
        ("Опыт", requirements.get("experience"), 80),
        ("Регион", region.get("name"), 100),
    ):
        if value := _plain(str(value) if type(value) is int else value, limit):
            lines.append(f"{label}: {value}")
    if not any(line.startswith(("Обязанности:", "Требования:")) for line in lines):
        return None
    return Vacancy("Rostrud", url, title, employer, published, "\n".join(lines))


def _career_card(html: str, url: str, cutoff: date, today: date, terms: tuple[str, ...]) -> Vacancy | None:
    page = _PageData()
    page.feed(html)
    page.close()
    for raw in page.structured:
        try:
            card = json.loads(raw)
        except (ValueError, RecursionError):
            continue
        if not isinstance(card, dict) or card.get("@type") != "JobPosting":
            continue
        title = _plain(card.get("title"), 200)
        published = _date(card.get("datePosted"))
        expires = _date(card.get("validThrough"))
        details = _plain(card.get("description"), 1450)
        if (title and details and published and cutoff <= published <= today
                and (not expires or expires >= today) and _matches_job(title, terms)):
            return Vacancy("EmployerCareer", url, title, "Яндекс", published, details)
    return None


async def _download(client: httpx.AsyncClient, url: str, *, params=None, accept: str, limit: int) -> str:
    async with client.stream("GET", url, params=params, headers={"Accept": accept}) as response:
        response.raise_for_status()
        body = bytearray()
        async for chunk in response.aiter_bytes(chunk_size=65536):
            body.extend(chunk)
            if len(body) > limit:
                raise ValueError("Превышен размер ответа.")
        return body.decode(response.encoding or "utf-8", errors="replace")


async def _rostrud(client, queries, terms, cutoff, today) -> tuple[list[Vacancy], bool]:
    async def search(query):
        data = json.loads(await _download(
            client, ROSTRUD_URL, params={"text": query, "limit": 50, "modifiedFrom": f"{cutoff}T00:00:00Z"},
            accept="application/json", limit=4_000_000,
        ))
        if not isinstance(data, dict) or str(data.get("status")) != "200":
            raise ValueError("Некорректный ответ Роструда.")
        result = data.get("results")
        meta = data.get("meta")
        if result is None and isinstance(meta, dict) and meta.get("total") in (0, "0"):
            return []
        if not isinstance(result, dict) or not isinstance(rows := result.get("vacancies", []), list):
            raise ValueError("Некорректный список вакансий.")
        return [card for raw in rows if (card := _rostrud_card(raw, cutoff, today, terms))]

    return await _collect(*(search(query) for query in queries))


async def _careers(client, query, terms, cutoff, today) -> tuple[list[Vacancy], bool]:
    page = _PageData(listing=True)
    async with asyncio.timeout(22):
        page.feed(await _download(client, CAREERS_URL, params={"text": query}, accept="text/html", limit=1_500_000))
        page.close()
        paths = [path for path, title in page.links.items() if _matches_job(title, terms)][:6]

        async def detail(path):
            url = "https://yandex.ru" + path
            page = await _download(client, url, accept="text/html", limit=1_500_000)
            card = _career_card(page, url, cutoff, today, terms)
            return [card] if card else []

        return await _collect(*(detail(path) for path in paths))


async def _collect(*requests) -> tuple[list[Vacancy], bool]:
    async def checked(request):
        try:
            async with asyncio.timeout(20):
                return await request
        except (httpx.HTTPError, TimeoutError, ValueError, RecursionError) as error:
            LOG.info("Поиск вакансий: %s", type(error).__name__)
            return None

    results = await asyncio.gather(*(checked(request) for request in requests))
    return [card for result in results if result for card in result], any(result is None for result in results)


def _employer_name(name: str) -> str:
    return " ".join(part for part in _normal(name).split() if part not in {"ооо", "пао", "ао", "ип", "зао"})


class PublicVacancySearch:
    def __init__(self, *, employer_careers: bool = True, transport: httpx.AsyncBaseTransport | None = None):
        self.employer_careers = employer_careers
        self.transport = transport

    async def records(self, topic: str) -> EvidenceResults:
        if not topic or not _normal(topic):
            return EvidenceResults()
        now = datetime.now(timezone.utc)
        cutoff = now.date() - timedelta(days=WINDOW_DAYS - 1)
        queries, titles = _search_terms(topic)
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(18, connect=7), headers={"User-Agent": "RPD-Generator/1.0"},
            transport=self.transport, limits=httpx.Limits(max_connections=6), follow_redirects=False,
        ) as client:
            async def careers():
                try:
                    return await _careers(client, queries[0], titles, cutoff, now.date())
                except (httpx.HTTPError, TimeoutError, ValueError, RecursionError) as error:
                    LOG.info("Карьерный сайт Яндекса: %s", type(error).__name__)
                    return [], True

            providers = [("Работа в России", _rostrud(client, queries, titles, cutoff, now.date()))]
            if self.employer_careers:
                providers.append(("Карьера Яндекса", careers()))
            results = await asyncio.gather(*(request for _, request in providers))

        notices = [f"{label}: часть запросов не выполнена либо источник недоступен."
                   for (label, _), (_, incomplete) in zip(providers, results, strict=True) if incomplete]
        unique: dict[tuple[str, str], Vacancy] = {}
        seen_urls = set()
        for item in sorted((card for cards, _ in results for card in cards), key=lambda card: card.published, reverse=True):
            if item.url not in seen_urls:
                unique.setdefault((_employer_name(item.employer), _normal(item.title)), item)
                seen_urls.add(item.url)
        if unique:
            employer_count = len({name for name, _ in unique})
            notices.append(f"Выборка вакансий за {cutoff:%d.%m.%Y}–{now:%d.%m.%Y} · "
                           f"вакансий после удаления повторов: {len(unique)} · работодателей: {employer_count}.")
        else:
            notices.append(f"По выполненным запросам вакансии по теме за последние {WINDOW_DAYS} дней не найдены.")
        rostrud = [item for item in unique.values() if item.kind == "Rostrud"]
        careers = [item for item in unique.values() if item.kind == "EmployerCareer"]
        selected = [*rostrud[:6], *careers[:4]]
        return EvidenceResults(
            [source_record(card.kind, card.url, card.title, card.content(), retrieved_at=now) for card in selected],
            notices,
        )
