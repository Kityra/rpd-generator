"""Подбор и верификация академической литературы для тем РПД.

Для каждой темы модуль параллельно опрашивает белый список академических источников,
проверяет PDF реальными HTTP-запросами, валидирует библиографические признаки,
ранжирует кандидатов (Relevance & Trust Score) и удаляет дубли.

Источники:
  * OpenAlex API — статьи и книги открытого доступа (фильтр language:ru, is_oa:true);
  * Crossref API — статьи с DOI, в том числе Math-Net.Ru (префикс DOI 10.4213)
    и журналы РАН/ИСП РАН через их DOI-префиксы;
  * КиберЛенинка — открытые статьи ВАК/RSCI с проверкой PDF, УДК, журнала и года;
  * Semantic Scholar API — только при SCIENCE_SEMANTIC_SCHOLAR_ENABLED=1 (как в run_science.py);
  * библиотека МЦНМО (www.mccme.ru/free-books) — PDF-учебники по математике;
  * вузовские репозитории (mipt.ru, msu.ru, hse.ru, itmo.ru, spbu.ru, nsu.ru, *.edu.ru …)
    и карточки ЭБС (Юрайт, Лань, Znanium …) через Yandex Search API v2.

Модуль самостоятелен и зависит только от httpx и pypdf из requirements.txt.
Интеграция с курсом — функция enrich_course_program, CLI — `python -m literature_parser`.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import copy
import hashlib
import json
import logging
import math
import os
import random
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Literal, Sequence
from urllib.parse import parse_qsl, quote, urlencode, urljoin, urlsplit, urlunsplit
from xml.etree import ElementTree

import httpx

from literature_catalog import references_for_discipline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

OPENALEX_WORKS = "https://api.openalex.org/works"
CROSSREF_WORKS = "https://api.crossref.org/works"
SEMANTIC_SCHOLAR_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"
CYBERLENINKA = "https://cyberleninka.ru"
CYBERLENINKA_SEARCH = CYBERLENINKA + "/api/search"
MCCME_FREE_BOOKS = "https://www.mccme.ru/free-books/"
YANDEX_WEB_SEARCH = "https://searchapi.api.cloud.yandex.net/v2/web/search"

MIN_PDF_BYTES = 100 * 1024
PROBE_BYTES = MIN_PDF_BYTES + 1
FINGERPRINT_BYTES = 64 * 1024
MIN_SCORE = 0.6
DUPLICATE_SIMILARITY = 0.85
MODERN_SINCE = 2018
RETRY_STATUSES = {429, 500, 502, 503, 504}

SourceKind = Literal["textbook", "article"]

# DOI-префиксы рецензируемых российских изданий по математике и информатике.
TRUSTED_DOI_PREFIXES: dict[str, str] = {
    "10.4213": "Math-Net.Ru",          # журналы МИАН и Math-Net.Ru
    "10.15514": "Труды ИСП РАН",       # Институт системного программирования РАН
    "10.14357": "ФИЦ ИУ РАН",          # «Информатика и её применения» и др.
    "10.20537": "Компьютерные исследования и моделирование",
}
MATH_PREFIXES = ("10.4213",)
IT_PREFIXES = ("10.15514", "10.14357", "10.20537")

# Вузовские и академические домены (поиск PDF через Yandex Search).
EDU_DOMAINS = (
    "mipt.ru", "msu.ru", "hse.ru", "itmo.ru", "ifmo.ru", "spbu.ru", "nsu.ru", "mccme.ru",
    "mathnet.ru", "math-net.ru", "mi-ras.ru", "ispras.ru", "ccas.ru", "machinelearning.ru",
    "bmstu.ru", "spbstu.ru", "urfu.ru", "kpfu.ru", "tsu.ru", "unn.ru", "keldysh.ru", "edu.ru",
)
EBS_DOMAINS: dict[str, str] = {
    "urait.ru": "Юрайт", "e.lanbook.com": "Лань", "znanium.ru": "Znanium", "znanium.com": "Znanium",
    "ibooks.ru": "Ibooks", "book.ru": "BOOK.ru", "biblioclub.ru": "Университетская библиотека онлайн",
    "iprbookshop.ru": "IPR SMART", "studentlibrary.ru": "Консультант студента",
}
TRUSTED_ARTICLE_DOMAINS = ("cyberleninka.ru", "mathnet.ru", "doi.org", "elibrary.ru", "ispras.ru")
BLOCKED_DOMAINS = (
    "studfile.net", "studfiles.net", "studfiles.ru", "studopedia.ru", "studopedia.org", "studwood.net",
    "studbooks.net", "studme.org", "allbest.ru", "knowledge.allbest.ru", "otvet.mail.ru", "mail.ru",
    "twirpx.com", "twirpx.org", "docplayer.ru", "docplayer.com", "dokumen.pub", "pdfcoffee.com",
    "vdoc.pub", "libgen.rs", "libgen.is", "libgen.li", "z-lib.org", "bookfi.net", "scribd.com",
    "myshared.ru", "ppt-online.org", "infourok.ru", "nsportal.ru", "multiurok.ru", "5fan.ru",
    "bibliofond.ru", "referat.ru", "vk.com", "t.me", "ozlib.com", "litmir.me", "rulit.me",
    "obuchalka.org", "djvu.online", "en.ppt-online.org", "studentopedia.ru", "lektsii.org",
)
JUNK_EXTENSIONS = {
    ".csv", ".tsv", ".xls", ".xlsx", ".json", ".jsonl", ".xml", ".parquet", ".zip", ".rar", ".7z",
    ".gz", ".tar", ".txt", ".log", ".sql", ".py", ".ipynb", ".doc", ".docx", ".ppt", ".pptx",
    ".rtf", ".odt", ".pkl", ".npy", ".exe",
}

# Издательства: (каноническое имя, регулярное выражение). Общие слова требуют контекста «М.: …».
_CITY = r"(?:М\.|Москва|СПб\.|Санкт-Петербург|Новосибирск|Ижевск|Долгопрудный|Казань|Екатеринбург)\s*[:,]\s*"
PUBLISHERS: tuple[tuple[str, str], ...] = (
    ("Физматлит", r"физматлит|fizmatlit"),
    ("МЦНМО", r"мцнмо|mccme"),
    ("ДМК Пресс", r"дмк\s*пресс|dmk\s*press"),
    ("БХВ-Петербург", r"бхв[\s-]*петербург"),
    ("Юрайт", r"юрайт|urait"),
    ("Лань", r"(?:изд(?:ательство|-во)?\.?\s*[«\"]?|" + _CITY + r"[«\"]?)лань\b|e\.lanbook\.com"),
    ("Питер", r"(?:изд(?:ательство|-во)?\.?\s*[«\"]?|" + _CITY + r"[«\"]?|издательский\s+дом\s+[«\"]?)питер\b"),
    ("Вильямс", r"вильямс|williams\s+publishing"),
    ("Диалектика", r"(?:изд(?:ательство|-во)?\.?\s*[«\"]?|" + _CITY + r"[«\"]?|ооо\s+[«\"]?)диалектика"),
    ("Бином. Лаборатория знаний", r"бином\.?\s*лаборатория\s+знаний|лаборатория\s+знаний"),
    ("Наука", _CITY + r"[«\"]?наука\b|издательство\s+[«\"]?наука"),
    ("Высшая школа", _CITY + r"[«\"]?высшая\s+школа|издательство\s+[«\"]?высшая\s+школа"),
    ("Мир", _CITY + r"[«\"]?мир\b|издательство\s+[«\"]?мир[»\"]"),
    ("URSS", r"\burss\b|едиториал\s+урсс|ленанд"),
    ("Инфра-М", r"инфра[\s-]*м\b"),
    ("Солон-пресс", r"солон[\s-]*пресс"),
    ("Интуит", r"интуит|intuit\.ru"),
    ("Издательский дом ВШЭ", r"(?:изд(?:ательский\s+дом|ательство|-во)\.?\s*)(?:ниу\s+)?вшэ|издательский\s+дом\s+высшей\s+школы\s+экономики"),
    ("Издательство МГУ", r"изд(?:ательство|-во)\.?\s*(?:московского\s+университета|мгу)"),
    ("МФТИ", r"изд(?:ательство|-во)?\.?\s*мфти|московский\s+физико-технический\s+институт"),
    ("Издательство МГТУ им. Н. Э. Баумана", r"мгту\s+им\.?\s*н\.?\s*э\.?\s*баумана"),
    ("Университет ИТМО", r"университет\s+итмо|ниу\s+итмо"),
    ("Издательство СПбГУ", r"изд(?:ательство|-во)\.?\s*(?:спбгу|санкт-петербургского\s+(?:государственного\s+)?университета)"),
    ("НГУ", r"(?:изд(?:ательство|-во)|ипц)\.?\s*нгу|новосибирский\s+государственный\s+университет"),
    ("Регулярная и хаотическая динамика", r"регулярная\s+и\s+хаотическая\s+динамика"),
)
ACADEMIC_AFFILIATION = re.compile(
    r"университет|институт|академи|училищ|\bниу\b|\bнии\b|\bран\b|мфти|мгу|вшэ|итмо|спбгу|нгу|"
    r"universit|institut|academy|college|school\s+of|polytechnic|russian\s+academy",
    re.IGNORECASE,
)

# Области знаний, которые отсекают медицину, экономику, юриспруденцию и т. п.
FORBIDDEN_FIELDS = (
    "medicine", "nursing", "health", "dentistry", "veterinary", "pharmacology", "economics",
    "business", "law", "psychology", "arts and humanities", "agricultural", "immunology",
    "neuroscience", "sociology", "political", "history", "philosophy", "education",
    "медицин", "здоров", "экономик", "бизнес", "право", "юрид", "психолог", "образован",
    "сельск", "фармац", "социол", "истори", "философ", "политолог",
)
NEGATIVE_WORDS = (
    "медицина", "медицинский", "клинический", "пациент", "пациентов", "болезнь", "заболевание",
    "хирургия", "терапия", "фармакология", "стоматология", "кардиология", "онкология",
    "экономический", "экономика", "бухгалтерский", "налоговый", "маркетинг", "юридический",
    "правовой", "судебный", "уголовный", "законодательство", "психологический", "педагогический",
    "школьники", "преподавание", "clinical", "patients", "disease", "medical", "economic",
    "legal", "law", "therapy", "surgery",
)
POSITIVE_WORDS = (
    "алгоритм", "нейронный", "нейросеть", "вычислительный", "матрица", "уравнение", "геометрия",
    "вероятность", "информационный", "компьютерный", "программирование", "машинный", "модель",
    "оптимизация", "обучение", "агент", "функция", "пространство", "граф", "данные",
    "algorithm", "neural", "learning", "matrix", "equation", "geometry", "optimization", "agent",
)

# ---------------------------------------------------------------------------
# Нормализация текста, стемминг, сходство строк
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"[0-9a-zа-я]+", re.IGNORECASE)
_RU_SUFFIXES = tuple(sorted({
    "иями", "ями", "ами", "ением", "ием", "ией", "иях", "ость", "ости", "остей", "ения", "ение",
    "ений", "ениям", "ениях", "ого", "его", "ому", "ему", "ыми", "ими", "ых", "их", "ая", "яя",
    "ое", "ее", "ые", "ие", "ой", "ей", "ый", "ий", "ов", "ев", "ам", "ям", "ах", "ях", "ом",
    "ем", "ую", "юю", "а", "я", "ы", "и", "у", "ю", "е", "о", "ь",
}, key=len, reverse=True))
_EN_SUFFIXES = ("ations", "ation", "ings", "ing", "ies", "es", "ed", "s")
STEM_LENGTH = 7
STOPWORDS = {
    "и", "в", "во", "на", "с", "со", "по", "для", "от", "до", "из", "к", "о", "об", "при", "как",
    "или", "а", "но", "не", "что", "это", "его", "их", "the", "of", "and", "in", "for", "on",
    "with", "to", "a", "an", "by", "at", "from", "is", "are", "основе", "основы", "основа",
    "введение", "учебное", "учебник", "пособие", "курс", "лекции", "лекций", "часть", "тема",
    "раздел", "издание", "изд", "pdf", "book", "textbook", "introduction",
}


def normalise(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").casefold().replace("ё", "е")


def clean(text: Any) -> str:
    return " ".join(str(text or "").split())


def strip_tags(text: str) -> str:
    return clean(_unescape(re.sub(r"<[^>]+>", " ", text or "")))


def _unescape(text: str) -> str:
    import html as _html
    return _html.unescape(text)


def stem(word: str) -> str:
    token = normalise(word)
    if re.fullmatch(r"[a-z0-9]+", token):
        for suffix in _EN_SUFFIXES:
            if token.endswith(suffix) and len(token) - len(suffix) >= 4:
                token = token[: -len(suffix)]
                break
    else:
        for suffix in _RU_SUFFIXES:
            if token.endswith(suffix) and len(token) - len(suffix) >= 3:
                token = token[: -len(suffix)]
                break
    return token[:STEM_LENGTH]


def stems(text: str, *, drop_stopwords: bool = True) -> list[str]:
    tokens = _TOKEN.findall(normalise(text))
    return [stem(token) for token in tokens if not (drop_stopwords and token in STOPWORDS)]


def stem_set(text: str) -> frozenset[str]:
    return frozenset(stems(text))


def phrase_stems(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(stems(text)))


def has_cyrillic(text: str) -> bool:
    letters = [char for char in text or "" if char.isalpha()]
    if not letters:
        return False
    return sum("а" <= char.lower() <= "я" or char.lower() == "ё" for char in letters) / len(letters) > 0.4


_EDITION = re.compile(
    r"\(?\b\d+\s*-?\s*е\s+изд\w*\.?[^)]*\)?|\bиздание\s+\d+\w*|\b\d+(?:st|nd|rd|th)\s+ed(?:ition)?\.?",
    re.IGNORECASE,
)


def title_key(title: str) -> str:
    text = _EDITION.sub(" ", normalise(title))
    return " ".join(_TOKEN.findall(text))[:220]


def levenshtein_ratio(first: str, second: str) -> float:
    if first == second:
        return 1.0
    if not first or not second:
        return 0.0
    if len(first) < len(second):
        first, second = second, first
    previous = list(range(len(second) + 1))
    for i, left in enumerate(first, 1):
        current = [i]
        for j, right in enumerate(second, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (left != right)))
        previous = current
    return 1 - previous[-1] / max(len(first), len(second))


def _trigrams(text: str) -> Counter:
    padded = f"  {text} "
    return Counter(padded[i:i + 3] for i in range(len(padded) - 2))


def trigram_cosine(first: str, second: str) -> float:
    if not first or not second:
        return 0.0
    left, right = _trigrams(first), _trigrams(second)
    dot = sum(count * right[gram] for gram, count in left.items())
    norm = math.sqrt(sum(v * v for v in left.values())) * math.sqrt(sum(v * v for v in right.values()))
    return dot / norm if norm else 0.0


def similar_titles(first: str, second: str, threshold: float = DUPLICATE_SIMILARITY) -> bool:
    left, right = title_key(first), title_key(second)
    if not left or not right:
        return False
    if left == right:
        return True
    return levenshtein_ratio(left, right) > threshold or trigram_cosine(left, right) > threshold


def surname(author: str) -> str:
    """Фамилия из «Р. С. Саттон», «Саттон Р. С.», «Sutton, R.» или «Richard S. Sutton»."""
    text = clean(author)
    if "," in text:
        text = text.split(",")[0]
    tokens = re.findall(r"[^\W\d_]+(?:-[^\W\d_]+)*\.?", text)
    words = [token for token in tokens if not token.endswith(".") and len(token) > 1]
    if not words:
        return ""
    initials_after = len(tokens) > 1 and tokens[0] in words and all(token.endswith(".") for token in tokens[1:])
    return normalise(words[0] if initials_after else words[-1])


def normalise_doi(value: str) -> str:
    doi = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", clean(value), flags=re.IGNORECASE)
    doi = doi.rstrip(".,;)")
    return doi.lower() if re.fullmatch(r"10\.\d{4,9}/\S+", doi) else ""


def valid_isbn(raw: str) -> str:
    digits = re.sub(r"[^0-9XxХх]", "", raw or "").upper().replace("Х", "X")
    if len(digits) == 10 and re.fullmatch(r"\d{9}[\dX]", digits):
        total = sum((10 - i) * (10 if char == "X" else int(char)) for i, char in enumerate(digits))
        return digits if total % 11 == 0 else ""
    if len(digits) == 13 and digits.isdigit() and digits[:3] in {"978", "979"}:
        total = sum(int(char) * (1 if i % 2 == 0 else 3) for i, char in enumerate(digits))
        return digits if total % 10 == 0 else ""
    return ""


def format_isbn(isbn: str) -> str:
    if len(isbn) == 13:
        return f"{isbn[:3]}-{isbn[3]}-{isbn[4:9]}-{isbn[9:12]}-{isbn[12]}"
    return isbn

# ---------------------------------------------------------------------------
# URL: нормализация и анти-мусорные фильтры
# ---------------------------------------------------------------------------


def host_of(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def domain_in(host: str, domains: Iterable[str]) -> bool:
    return any(host == domain or host.endswith("." + domain) for domain in domains)


def url_key(url: str) -> str:
    parts = urlsplit(clean(url))
    if not parts.scheme or not parts.hostname:
        return ""
    host = host_of(url)
    query = urlencode(sorted((k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                             if not k.lower().startswith(("utm_", "fbclid", "yclid", "gclid"))))
    path = re.sub(r"/+$", "", parts.path) or "/"
    if host in {"doi.org", "dx.doi.org"}:
        return "doi:" + normalise_doi(path.lstrip("/"))
    return urlunsplit(("https", host, path, query, ""))


def url_extension(url: str) -> str:
    path = urlsplit(url).path.lower()
    match = re.search(r"(\.[a-z0-9]{1,8})$", path)
    return match.group(1) if match else ""


def blocked_reason(url: str) -> str:
    """Причина, по которой ссылку нельзя использовать как академический источник."""
    parts = urlsplit(clean(url))
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        return "некорректный URL"
    if parts.username or parts.password:
        return "URL с учётными данными"
    host = host_of(url)
    if domain_in(host, BLOCKED_DOMAINS):
        return f"домен {host} в чёрном списке"
    if url_extension(url) in JUNK_EXTENSIONS:
        return f"файл {url_extension(url)} не является академическим источником"
    return ""


def is_non_literature_file(name: str) -> bool:
    """Файл заведомо не учебная литература (таблицы, выгрузки, архивы, код)."""
    return Path(clean(name).lower()).suffix in JUNK_EXTENSIONS


def looks_like_filename(title: str) -> bool:
    """Название похоже на имя файла («Мл4.pdf», «lecture_5»), а не на библиографическое заглавие."""
    text = clean(title)
    return bool(re.search(r"\.[A-Za-z0-9]{2,5}$", text) or "_" in text or re.fullmatch(r"[\w-]+\d+", text))


def is_internal_file(name: str) -> bool:
    """Внутренний рабочий файл без библиографического описания, например «Мл4.pdf» или выгрузка HH."""
    text = clean(name)
    if not text:
        return True
    if is_non_literature_file(text):
        return True
    base = re.sub(r"\.[A-Za-z0-9]{1,6}$", "", Path(text).name)
    lowered = normalise(base)
    if re.search(r"(?:^|[_\-\s])(?:part|chunk|dump|export|copy|draft|tmp|temp|черновик|копия|выгрузка)"
                 r"(?:[_\-\s]*\d+)?(?:$|[_\-\s])", lowered):
        return True
    if re.search(r"(?:^|[_\-\s])(?:hh|jobix)(?:$|[_\-\s])", lowered):
        return True
    words = re.findall(r"[a-zа-я]{3,}", lowered)
    return len(words) < 2

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "да"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class ParserConfig:
    http_timeout: float = 15.0
    source_timeout: float = 45.0
    retries: int = 2
    max_backoff: float = 8.0
    per_host_concurrency: int = 3
    max_connections: int = 24
    topic_concurrency: int = 3
    contact_email: str = ""
    proxy: str = ""
    min_score: float = MIN_SCORE
    max_textbooks: int = 2
    max_articles: int = 3
    articles_to_verify: int = 10
    textbooks_to_verify: int = 8
    max_pdf_bytes: int = 30 * 1024 * 1024
    deep_pdf_check: bool = True
    english_fallback: bool = True
    use_llm: bool = True
    yandex_search_enabled: bool = False
    yandex_search_api_key: str = ""
    yandex_folder_id: str = ""
    yandex_max_requests: int = 4
    semantic_scholar_enabled: bool = False
    semantic_scholar_api_key: str = ""

    @classmethod
    def from_env(cls) -> "ParserConfig":
        key = (os.environ.get("YANDEX_SEARCH_API_KEY") or os.environ.get("YANDEX_API_KEY") or "").strip()
        folder = os.environ.get("YANDEX_FOLDER_ID", "").strip()
        return cls(
            http_timeout=_env_float("LITERATURE_HTTP_TIMEOUT", 15.0),
            source_timeout=_env_float("LITERATURE_SOURCE_TIMEOUT", 45.0),
            retries=max(0, _env_int("LITERATURE_RETRIES", 2)),
            topic_concurrency=max(1, _env_int("LITERATURE_TOPIC_CONCURRENCY", 3)),
            contact_email=os.environ.get("LITERATURE_CONTACT_EMAIL", "").strip(),
            proxy=os.environ.get("LITERATURE_PROXY", "").strip(),
            min_score=_env_float("LITERATURE_MIN_SCORE", MIN_SCORE),
            max_pdf_bytes=max(1, _env_int("LITERATURE_MAX_PDF_MB", 30)) * 1024 * 1024,
            deep_pdf_check=_env_bool("LITERATURE_DEEP_PDF_CHECK", True),
            english_fallback=_env_bool("LITERATURE_ENGLISH_FALLBACK", True),
            use_llm=_env_bool("LITERATURE_USE_LLM", True),
            yandex_search_enabled=_env_bool("LITERATURE_YANDEX_SEARCH", False) and bool(key and folder),
            yandex_search_api_key=key,
            yandex_folder_id=folder,
            yandex_max_requests=max(0, _env_int("LITERATURE_YANDEX_MAX_REQUESTS", 4)),
            semantic_scholar_enabled=os.environ.get("SCIENCE_SEMANTIC_SCHOLAR_ENABLED", "0").strip() == "1",
            semantic_scholar_api_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip(),
        )

# ---------------------------------------------------------------------------
# HTTP-слой: повторные запросы, лимиты на хост, кеш
# ---------------------------------------------------------------------------


class SourceUnavailable(RuntimeError):
    """Источник не ответил или вернул неожиданный формат."""


@dataclass(frozen=True)
class Fetched:
    status: int
    headers: httpx.Headers
    body: bytes
    final_url: str
    truncated: bool


def decode_body(body: bytes, content_type: str = "") -> str:
    match = re.search(r"charset=([\w-]+)", content_type or "", re.IGNORECASE)
    encodings = [match.group(1)] if match else []
    meta = re.search(rb"<meta[^>]+charset=[\"']?([\w-]+)", body[:4096], re.IGNORECASE)
    if meta:
        encodings.append(meta.group(1).decode("ascii", "ignore"))
    encodings += ["utf-8", "cp1251"]
    for encoding in encodings:
        try:
            return body.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    return body.decode("utf-8", "replace")


class Http:
    def __init__(self, config: ParserConfig, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.config = config
        options: dict[str, Any] = {
            "timeout": httpx.Timeout(config.http_timeout, connect=min(10.0, config.http_timeout)),
            "follow_redirects": True,
            "event_hooks": {"request": [self.check_request]},
            "headers": {"Accept-Language": "ru-RU,ru;q=0.9,en;q=0.7"},
            "limits": httpx.Limits(max_connections=config.max_connections, max_keepalive_connections=12),
        }
        if transport is not None:
            options["transport"] = transport
        elif config.proxy:
            options["proxy"] = config.proxy
        self.client = httpx.AsyncClient(**options)
        self._hosts: dict[str, asyncio.Semaphore] = {}
        self._cache: dict[str, Any] = {}

    async def check_request(self, request: httpx.Request) -> None:
        url = request.url
        provider_hosts = {urlsplit(endpoint).hostname for endpoint in (
            OPENALEX_WORKS, CROSSREF_WORKS, SEMANTIC_SCHOLAR_SEARCH, YANDEX_WEB_SEARCH,
        )}
        academic_domains = (*EDU_DOMAINS, *EBS_DOMAINS, *TRUSTED_ARTICLE_DOMAINS, "arxiv.org")
        host = url.host
        if (url.scheme != "https" or url.port not in {None, 443} or url.userinfo
                or not (host in provider_hosts or domain_in(host, academic_domains))):
            raise SourceUnavailable("Адрес вне списка академических источников.")
        if host != urlsplit(YANDEX_WEB_SEARCH).hostname:
            request.headers.pop("Authorization", None)
        if host != urlsplit(SEMANTIC_SCHOLAR_SEARCH).hostname:
            request.headers.pop("x-api-key", None)

    async def aclose(self) -> None:
        await self.client.aclose()

    def _semaphore(self, url: str) -> asyncio.Semaphore:
        host = host_of(url)
        if host not in self._hosts:
            self._hosts[host] = asyncio.Semaphore(self.config.per_host_concurrency)
        return self._hosts[host]

    def _headers(self, api: bool, extra: dict[str, str] | None) -> dict[str, str]:
        contact = f"; mailto:{self.config.contact_email}" if api and self.config.contact_email else ""
        agent = f"rpd-generator-literature/1.0 (+https://github.com/Kityra/rpd-generator{contact})"
        return {"User-Agent": agent, **(extra or {})}

    def _delay(self, attempt: int, retry_after: str | None = None) -> float:
        if retry_after and retry_after.strip().isdigit():
            return min(float(retry_after.strip()), self.config.max_backoff)
        return min(0.6 * (2 ** attempt) + random.uniform(0, 0.4), self.config.max_backoff)

    async def request(self, method: str, url: str, *, api: bool = False, retries: int | None = None,
                      headers: dict[str, str] | None = None, **kwargs: Any) -> httpx.Response:
        attempts = (self.config.retries if retries is None else retries) + 1
        for attempt in range(attempts):
            try:
                async with self._semaphore(url):
                    response = await self.client.request(method, url, headers=self._headers(api, headers), **kwargs)
            except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError):
                if attempt + 1 >= attempts:
                    raise
                await asyncio.sleep(self._delay(attempt))
                continue
            if response.status_code in RETRY_STATUSES and attempt + 1 < attempts:
                await asyncio.sleep(self._delay(attempt, response.headers.get("Retry-After")))
                continue
            return response
        raise SourceUnavailable(f"{url}: исчерпаны попытки")

    def _cache_key(self, *parts: Any) -> str:
        return hashlib.sha256(json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()

    async def get_json(self, url: str, *, params: dict[str, Any] | None = None, api: bool = True,
                       headers: dict[str, str] | None = None) -> Any:
        key = self._cache_key("GET", url, params)
        if key not in self._cache:
            response = await self.request("GET", url, params=params, api=api, headers=headers)
            response.raise_for_status()
            self._cache[key] = response.json()
        return self._cache[key]

    async def post_json(self, url: str, payload: Any, *, api: bool = False,
                        headers: dict[str, str] | None = None) -> Any:
        key = self._cache_key("POST", url, payload)
        if key not in self._cache:
            response = await self.request("POST", url, json=payload, api=api, headers=headers)
            response.raise_for_status()
            self._cache[key] = response.json()
        return self._cache[key]

    async def get_text(self, url: str, *, api: bool = False, headers: dict[str, str] | None = None) -> str:
        key = self._cache_key("TEXT", url)
        if key not in self._cache:
            response = await self.request("GET", url, api=api, headers=headers)
            response.raise_for_status()
            self._cache[key] = decode_body(response.content, response.headers.get("content-type", ""))
        return self._cache[key]

    async def fetch_bytes(self, url: str, *, limit: int, headers: dict[str, str] | None = None,
                          retries: int | None = None) -> Fetched:
        attempts = (self.config.retries if retries is None else retries) + 1
        for attempt in range(attempts):
            retry_after = None
            try:
                async with self._semaphore(url):
                    async with self.client.stream("GET", url, headers=self._headers(False, headers)) as response:
                        if response.status_code in RETRY_STATUSES and attempt + 1 < attempts:
                            retry_after = response.headers.get("Retry-After") or ""
                        else:
                            body = bytearray()
                            truncated = False
                            async for chunk in response.aiter_bytes():
                                body.extend(chunk)
                                if len(body) >= limit:
                                    truncated = True
                                    break
                            return Fetched(response.status_code, response.headers, bytes(body[:limit]),
                                           str(response.url), truncated)
            except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError):
                if attempt + 1 >= attempts:
                    raise
            await asyncio.sleep(self._delay(attempt, retry_after))
        raise SourceUnavailable(f"{url}: исчерпаны попытки")

# ---------------------------------------------------------------------------
# Проверка PDF
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PdfCheck:
    ok: bool
    url: str
    reason: str = ""
    final_url: str = ""
    size: int | None = None
    fingerprint: str = ""
    content: bytes | None = field(default=None, repr=False, compare=False)


def _content_range_total(value: str) -> int | None:
    match = re.search(r"/(\d+)\s*$", value or "")
    return int(match.group(1)) if match else None


class PdfValidator:
    """Проверка, что ссылка ведёт на настоящий PDF > 100 КБ, а не на 404 или HTML-заглушку."""

    def __init__(self, http: Http, config: ParserConfig, concurrency: int = 4) -> None:
        self.http = http
        self.config = config
        self._cache: dict[tuple[str, bool], PdfCheck] = {}
        self._limit = asyncio.Semaphore(concurrency)

    async def check(self, url: str, *, full: bool = False) -> PdfCheck:
        key = (url_key(url) or url, full)
        if key not in self._cache:
            async with self._limit:
                self._cache[key] = await self._check(url, full)
        return self._cache[key]

    async def _check(self, url: str, full: bool) -> PdfCheck:
        reason = blocked_reason(url)
        if reason:
            return PdfCheck(False, url, reason)
        try:
            head = await self.http.request("HEAD", url, retries=1)
        except (httpx.HTTPError, SourceUnavailable):
            head = None
        if head is not None:
            if head.status_code in {404, 410}:
                return PdfCheck(False, url, f"HTTP {head.status_code}")
            if head.status_code >= 400 and head.status_code not in {400, 403, 405, 406, 501}:
                return PdfCheck(False, url, f"HTTP {head.status_code}")
            # Размер и сигнатуру проверяем по телу ответа: HEAD у многих серверов врёт.
        limit = self.config.max_pdf_bytes if full else PROBE_BYTES
        headers = {"Accept": "application/pdf,*/*;q=0.8"}
        if not full:
            headers["Range"] = f"bytes=0-{PROBE_BYTES - 1}"
        try:
            fetched = await self.http.fetch_bytes(url, limit=limit, headers=headers, retries=1)
        except (httpx.HTTPError, SourceUnavailable) as error:
            return PdfCheck(False, url, f"нет ответа ({type(error).__name__})")
        if fetched.status not in {200, 206}:
            return PdfCheck(False, url, f"HTTP {fetched.status}")
        content_type = fetched.headers.get("content-type", "").lower()
        start = fetched.body[:1024].lstrip(b"\xef\xbb\xbf \t\r\n\x00")
        if not start.startswith(b"%PDF-"):
            looks_html = "html" in content_type or b"<html" in fetched.body[:2048].lower() or \
                b"<!doctype" in fetched.body[:2048].lower()
            return PdfCheck(False, url, "HTML-страница вместо PDF" if looks_html else "файл не является PDF")
        total = None
        if fetched.status == 206:
            total = _content_range_total(fetched.headers.get("content-range", ""))
        if total is None and fetched.status == 200:
            length = fetched.headers.get("content-length", "")
            if length.isdigit() and "content-encoding" not in fetched.headers:
                total = int(length)
        if total is None and not fetched.truncated:
            total = len(fetched.body)
        size = total if total is not None else len(fetched.body)
        if size < MIN_PDF_BYTES:
            return PdfCheck(False, url, f"PDF слишком мал ({size} байт)")
        fingerprint = hashlib.sha256(fetched.body[:FINGERPRINT_BYTES]).hexdigest()[:32] + f":{size}"
        content = fetched.body if full and not fetched.truncated else None
        return PdfCheck(True, url, "", fetched.final_url, size, fingerprint, content)


@dataclass
class FrontMatter:
    authors: list[str] = field(default_factory=list)
    publisher: str = ""
    year: int | None = None
    isbn: str = ""
    udc: str = ""
    bbk: str = ""
    edn: str = ""
    doi: str = ""
    grif: str = ""
    label: str = ""
    title: str = ""


_ISBN = re.compile(r"ISBN[\s:‑-]*((?:97[89][\s‑–-]?)?\d[\d\s‑–-]{8,16}[\dXxХх])")
_UDC = re.compile(r"УДК[\s:]*([0-9][0-9.:()\-+/\[\]=\"' ]{1,40}[0-9)\]])")
_BBK = re.compile(r"ББК[\s:]*([0-9][0-9A-Za-zА-Яа-я.\-()/ ]{1,30}[0-9A-Za-zА-Яа-я)])")
_EDN = re.compile(r"\bEDN[\s:]*([A-Z]{6})\b")
_DOI_TEXT = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>,;]+)")
_GRIF = re.compile(
    r"(?:Рекомендовано|Допущено|Утверждено|Рекомендован[оа]?)[^.]{0,220}?"
    r"(?:Ф?УМО|учебно-методическ\w+\s+объединени\w+|Министерств\w+|редакционно-издательск\w+|учен\w+\s+совет\w*|"
    r"методическ\w+\s+совет\w*)",
    re.IGNORECASE | re.DOTALL,
)
_AUTHOR_INITIALS = re.compile(r"\b([А-ЯЁ]\.\s?(?:[А-ЯЁ]\.\s?)?[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?)")
_AUTHOR_SURNAME_FIRST = re.compile(r"\b([А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?,?\s[А-ЯЁ]\.\s?(?:[А-ЯЁ]\.)?)")
_YEAR_CONTEXT = re.compile(r"(?:©|\(c\)|\bгод\b|\bг\.|,)\s*((?:19[5-9]|20[0-4])\d)\b|\b((?:19[5-9]|20[0-4])\d)\s*г\.")
_NON_AUTHORS = {"москва", "санкт-петербург", "новосибирск", "россия", "издательство", "кафедра"}


def detect_publisher(text: str) -> str:
    lowered = normalise(text)
    for name, pattern in PUBLISHERS:
        if re.search(pattern, lowered, re.IGNORECASE):
            return name
    return ""


def kind_label(text: str) -> str:
    lowered = normalise(text)
    for marker, label in (("учебное пособие", "Учебное пособие"), ("учебник", "Учебник"),
                          ("курс лекций", "Курс лекций"), ("конспект лекций", "Курс лекций"),
                          ("монография", "Монография"), ("сборник задач", "Задачник"),
                          ("практикум", "Практикум"), ("методические указания", "Методическое пособие")):
        if marker in lowered:
            return label
    return ""


def _plausible_year(value: int | None) -> int | None:
    return value if value and 1900 <= value <= date.today().year else None


def parse_front_matter(text: str) -> FrontMatter:
    """Библиографические признаки с титульного листа и оборота (ISBN, УДК, ББК, гриф, авторы, год)."""
    result = FrontMatter()
    if not text:
        return result
    for match in _ISBN.finditer(text):
        isbn = valid_isbn(match.group(1))
        if isbn:
            result.isbn = isbn
            break
    if match := _UDC.search(text):
        result.udc = clean(match.group(1)).strip(" .:")
    if match := _BBK.search(text):
        result.bbk = clean(match.group(1)).strip(" .:")
    if match := _EDN.search(text):
        result.edn = match.group(1)
    if match := _DOI_TEXT.search(text):
        result.doi = normalise_doi(match.group(1))
    if match := _GRIF.search(text):
        result.grif = clean(match.group(0))[:240]
    result.publisher = detect_publisher(text[:12000])
    years = [int(a or b) for a, b in _YEAR_CONTEXT.findall(text[:12000])]
    years = [year for year in years if _plausible_year(year)]
    if years:
        result.year = Counter(years).most_common(1)[0][0]
    head = text[:3000]
    authors: list[str] = []
    for pattern in (_AUTHOR_INITIALS, _AUTHOR_SURNAME_FIRST):
        for match in pattern.finditer(head):
            name = clean(match.group(1)).rstrip(",")
            if normalise(surname(name)) in _NON_AUTHORS:
                continue
            if all(surname(name) != surname(known) for known in authors):
                authors.append(name)
        if authors:
            break
    result.authors = authors[:6]
    result.label = kind_label(text[:6000])
    return result


def pdf_front_text(content: bytes, pages: int = 5) -> tuple[str, dict[str, str]]:
    """Текст первых страниц PDF и метаданные /Title, /Author (pypdf)."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(content))
        metadata = reader.metadata or {}
        info = {"title": clean(getattr(metadata, "title", "") or metadata.get("/Title", "") or ""),
                "author": clean(getattr(metadata, "author", "") or metadata.get("/Author", "") or "")}
        chunks = []
        for index in range(min(pages, len(reader.pages))):
            chunks.append(reader.pages[index].extract_text() or "")
        return "\n".join(chunks), info
    except Exception as error:  # noqa: BLE001 - pypdf бросает много разных исключений на битых файлах
        logger.debug("PDF не разобран: %s", error)
        return "", {}

# ---------------------------------------------------------------------------
# Разбор HTML (stdlib)
# ---------------------------------------------------------------------------

_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
_BLOCK = {"p", "div", "li", "tr", "td", "h1", "h2", "h3", "h4", "h5", "section", "article", "dd", "dt", "ul", "ol"}


@dataclass
class Page:
    meta: dict[str, list[str]]
    title: str
    h1: str
    text: str
    labels: list[str]
    links: list[tuple[str, str]]
    items: list[tuple[str, list[tuple[str, str]]]]
    jsonld: list[Any]

    def first(self, *names: str) -> str:
        for name in names:
            for value in self.meta.get(name.lower(), []):
                if clean(value):
                    return clean(value)
        return ""

    def all(self, name: str) -> list[str]:
        return [clean(value) for value in self.meta.get(name.lower(), []) if clean(value)]


class _PageParser(HTMLParser):
    MAX_TEXT = 500_000

    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.meta: dict[str, list[str]] = {}
        self.title_parts: list[str] = []
        self.h1_parts: list[str] = []
        self.text_parts: list[str] = []
        self.text_length = 0
        self.labels: list[str] = []
        self.links: list[tuple[str, str]] = []
        self.items: list[tuple[str, list[tuple[str, str]]]] = []
        self.jsonld: list[Any] = []
        self._skip: str | None = None
        self._script: list[str] | None = None
        self._in_title = False
        self._h1 = 0
        self._li: list[tuple[list[str], list[tuple[str, str]]]] = []
        self._link: tuple[str, list[str]] | None = None
        self._label: list[list[Any]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key.lower(): (value or "") for key, value in attrs}
        if self._skip:
            return
        if tag == "meta":
            key = (values.get("name") or values.get("property") or values.get("itemprop") or "").strip().lower()
            if key and values.get("content"):
                self.meta.setdefault(key, []).append(values["content"])
            return
        if tag in {"script", "style", "noscript", "template"}:
            self._skip = tag
            self._script = [] if tag == "script" and "ld+json" in values.get("type", "").lower() else None
            return
        if tag in _VOID:
            if tag == "br":
                self._data("\n")
            return
        if tag == "title":
            self._in_title = True
        if tag == "h1":
            self._h1 += 1
        for entry in self._label:
            if entry[0] == tag:
                entry[1] += 1
        classes = values.get("class", "").lower().split()
        if any("label" in name or "badge" in name for name in classes):
            self._label.append([tag, 1, [values.get("title", "")]])
        if tag == "li":
            self._li.append(([], []))
        if tag == "a":
            self._link = (urljoin(self.base_url, values.get("href", "").strip()), [])
        if tag in _BLOCK:
            self._data("\n")

    def handle_endtag(self, tag: str) -> None:
        if self._skip:
            if tag == self._skip:
                if self._script is not None:
                    try:
                        self.jsonld.append(json.loads("".join(self._script)))
                    except ValueError:
                        pass
                self._skip, self._script = None, None
            return
        if tag == "title":
            self._in_title = False
        if tag == "h1" and self._h1:
            self._h1 -= 1
        if tag == "a" and self._link is not None:
            href, parts = self._link
            self._link = None
            text = clean(" ".join(parts))
            if href:
                self.links.append((href, text))
                if self._li:
                    self._li[-1][1].append((href, text))
        if tag == "li" and self._li:
            parts, links = self._li.pop()
            self.items.append((clean(" ".join(parts)), links))
        for entry in self._label:
            if entry[0] == tag:
                entry[1] -= 1
        while self._label and self._label[-1][1] <= 0:
            text = clean(" ".join(self._label.pop()[2]))
            if text:
                self.labels.append(text)
        if tag in _BLOCK:
            self._data("\n")

    def handle_data(self, data: str) -> None:
        if self._skip:
            if self._script is not None:
                self._script.append(data)
            return
        self._data(data)

    def _data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
        if self._h1:
            self.h1_parts.append(data)
        if self._li:
            self._li[-1][0].append(data)
        if self._link is not None:
            self._link[1].append(data)
        for entry in self._label:
            entry[2].append(data)
        if self.text_length < self.MAX_TEXT:
            self.text_parts.append(data)
            self.text_length += len(data)


def parse_html(text: str, base_url: str = "") -> Page:
    parser = _PageParser(base_url)
    try:
        parser.feed(text or "")
        parser.close()
    except Exception as error:  # noqa: BLE001 - HTMLParser на битой разметке
        logger.debug("HTML разобран частично: %s", error)
    while parser._li:
        parts, links = parser._li.pop()
        parser.items.append((clean(" ".join(parts)), links))
    raw = "".join(parser.text_parts)
    lines = [clean(line) for line in raw.splitlines()]
    return Page(parser.meta, clean(" ".join(parser.title_parts)), clean(" ".join(parser.h1_parts)),
                "\n".join(line for line in lines if line), parser.labels, parser.links, parser.items, parser.jsonld)


def jsonld_objects(page: Page, kinds: set[str]) -> list[dict]:
    found: list[dict] = []

    def walk(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, dict):
            kind = node.get("@type")
            kind_set = {kind} if isinstance(kind, str) else set(kind or [])
            if kind_set & kinds:
                found.append(node)
            for key in ("@graph", "mainEntity", "itemListElement"):
                if key in node:
                    walk(node[key])

    walk(page.jsonld)
    return found

# ---------------------------------------------------------------------------
# Словарь терминов и расширение запросов
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TermEntry:
    triggers: tuple[str, ...]
    ru: tuple[str, ...]
    en: tuple[str, ...]
    broad_ru: tuple[str, ...] = ()
    broad_en: tuple[str, ...] = ()

    def matches(self, text_stems: frozenset[str]) -> bool:
        return any(set(phrase_stems(trigger)) <= text_stems for trigger in self.triggers)


_RL_BROAD = (("обучение с подкреплением",), ("reinforcement learning",))
_ML_BROAD = (("машинное обучение",), ("machine learning",))
_DL_BROAD = (("глубокое обучение", "нейронные сети"), ("deep learning",))
_AG_BROAD = (("аналитическая геометрия",), ("analytic geometry",))
_LA_BROAD = (("линейная алгебра",), ("linear algebra",))
_MA_BROAD = (("математический анализ",), ("mathematical analysis",))
_PR_BROAD = (("теория вероятностей",), ("probability theory",))
_ST_BROAD = (("математическая статистика",), ("mathematical statistics",))
_DM_BROAD = (("дискретная математика", "алгоритмы"), ("discrete mathematics",))
_AL_BROAD = (("алгоритмы и структуры данных",), ("algorithms and data structures",))

TERMS: tuple[TermEntry, ...] = (
    TermEntry(("временных различий", "временных разностей", "td-обучение", "temporal difference", "td learning"),
              ("метод временных различий", "метод временных разностей", "TD-обучение", "Q-обучение", "SARSA",
               "темпоральных различий"),
              ("temporal difference learning", "TD learning", "Q-learning", "SARSA"), *_RL_BROAD),
    TermEntry(("обучение с подкреплением", "reinforcement learning", "марковский процесс принятия решений"),
              ("обучение с подкреплением", "марковский процесс принятия решений", "Q-обучение", "градиент стратегии"),
              ("reinforcement learning", "Markov decision process", "policy gradient"), *_ML_BROAD),
    TermEntry(("q-обучение", "q-learning", "sarsa", "deep q-network"),
              ("Q-обучение", "глубокая Q-сеть"), ("Q-learning", "deep Q-network"), *_RL_BROAD),
    TermEntry(("кривые второго порядка", "коники", "конические сечения"),
              ("кривые второго порядка", "эллипс гипербола парабола", "канонические уравнения кривых второго порядка",
               "конические сечения"), ("conic sections", "quadratic curves"), *_AG_BROAD),
    TermEntry(("поверхности второго порядка", "квадрики"),
              ("поверхности второго порядка", "эллипсоид гиперболоид параболоид", "квадрики"),
              ("quadric surfaces",), *_AG_BROAD),
    TermEntry(("прямая и плоскость", "прямые и плоскости", "уравнение плоскости", "уравнение прямой"),
              ("уравнения прямой и плоскости", "прямая и плоскость в пространстве"),
              ("lines and planes in space",), *_AG_BROAD),
    TermEntry(("векторная алгебра", "скалярное произведение", "векторное произведение", "смешанное произведение"),
              ("векторная алгебра", "скалярное и векторное произведение векторов"), ("vector algebra",), *_AG_BROAD),
    TermEntry(("матрицы", "определители", "определитель", "ранг матрицы", "обратная матрица"),
              ("матрицы и определители", "ранг матрицы", "обратная матрица"), ("matrices and determinants",),
              *_LA_BROAD),
    TermEntry(("собственные значения", "собственные векторы", "собственных значений", "собственных векторов"),
              ("собственные значения и собственные векторы", "характеристический многочлен", "спектральное разложение"),
              ("eigenvalues and eigenvectors", "spectral decomposition"), *_LA_BROAD),
    TermEntry(("системы линейных уравнений", "системы линейных алгебраических уравнений", "метод гаусса"),
              ("системы линейных алгебраических уравнений", "метод Гаусса", "правило Крамера"),
              ("systems of linear equations", "Gaussian elimination"), *_LA_BROAD),
    TermEntry(("линейные пространства", "векторные пространства", "линейные операторы", "линейные отображения"),
              ("линейные пространства", "линейные операторы", "базис и размерность"),
              ("vector spaces", "linear operators"), *_LA_BROAD),
    TermEntry(("квадратичные формы",), ("квадратичные формы", "приведение квадратичной формы к каноническому виду"),
              ("quadratic forms",), *_LA_BROAD),
    TermEntry(("линейная регрессия", "метод наименьших квадратов", "регрессионный анализ"),
              ("линейная регрессия", "метод наименьших квадратов", "гребневая регрессия"),
              ("linear regression", "least squares", "ridge regression"), *_ML_BROAD),
    TermEntry(("логистическая регрессия", "линейные классификаторы"),
              ("логистическая регрессия", "линейные классификаторы"), ("logistic regression", "linear classifiers"),
              *_ML_BROAD),
    TermEntry(("градиентный спуск", "градиентного спуска", "методы оптимизации", "стохастический градиент"),
              ("градиентный спуск", "стохастический градиентный спуск", "методы оптимизации"),
              ("gradient descent", "stochastic gradient descent"), *_ML_BROAD),
    TermEntry(("нейронные сети", "нейронных сетей", "персептрон", "обратного распространения", "нейросети"),
              ("искусственные нейронные сети", "многослойный персептрон", "метод обратного распространения ошибки"),
              ("artificial neural networks", "backpropagation", "multilayer perceptron"), *_DL_BROAD),
    TermEntry(("сверточные нейронные сети", "сверточных нейронных сетей", "сверточные сети"),
              ("сверточные нейронные сети", "распознавание изображений"), ("convolutional neural networks",),
              *_DL_BROAD),
    TermEntry(("рекуррентные нейронные сети", "рекуррентных нейронных сетей", "lstm"),
              ("рекуррентные нейронные сети", "LSTM"), ("recurrent neural networks", "LSTM"), *_DL_BROAD),
    TermEntry(("трансформер", "трансформеры", "механизм внимания", "transformer", "attention mechanism"),
              ("архитектура трансформер", "механизм внимания"), ("transformer architecture", "attention mechanism"),
              *_DL_BROAD),
    TermEntry(("деревья решений", "случайный лес", "градиентный бустинг", "бустинг", "ансамбли моделей"),
              ("деревья решений", "случайный лес", "градиентный бустинг", "ансамбли моделей"),
              ("decision trees", "random forest", "gradient boosting"), *_ML_BROAD),
    TermEntry(("опорных векторов", "svm", "ядерные методы"),
              ("метод опорных векторов", "ядерные методы"), ("support vector machine", "kernel methods"), *_ML_BROAD),
    TermEntry(("кластеризация", "кластерный анализ", "k-средних", "кластеризации"),
              ("кластеризация", "кластерный анализ", "метод k-средних"), ("clustering", "k-means"), *_ML_BROAD),
    TermEntry(("снижение размерности", "понижение размерности", "главных компонент", "pca"),
              ("метод главных компонент", "снижение размерности"),
              ("principal component analysis", "dimensionality reduction"), *_ML_BROAD),
    TermEntry(("байесовские", "байесовский", "наивный байес", "байесовского"),
              ("байесовский вывод", "наивный байесовский классификатор"), ("Bayesian inference", "naive Bayes"),
              *_ML_BROAD),
    TermEntry(("переобучение", "кросс-валидация", "перекрестная проверка", "регуляризация", "оценка качества моделей"),
              ("переобучение", "кросс-валидация", "регуляризация моделей"),
              ("overfitting", "cross-validation", "regularization"), *_ML_BROAD),
    TermEntry(("обработка естественного языка", "nlp", "векторные представления слов", "эмбеддинги",
               "языковые модели"),
              ("обработка естественного языка", "векторные представления слов", "языковые модели"),
              ("natural language processing", "word embeddings", "language models"), *_ML_BROAD),
    TermEntry(("компьютерное зрение", "распознавание изображений", "обнаружение объектов"),
              ("компьютерное зрение", "распознавание изображений", "обнаружение объектов"),
              ("computer vision", "object detection", "image recognition"), *_DL_BROAD),
    TermEntry(("марковские цепи", "цепи маркова", "марковских цепей", "марковские процессы"),
              ("цепи Маркова", "марковские процессы"), ("Markov chains",), *_PR_BROAD),
    TermEntry(("случайные величины", "законы распределения", "функция распределения", "случайных величин"),
              ("случайные величины", "законы распределения случайных величин"),
              ("random variables", "probability distributions"), *_PR_BROAD),
    TermEntry(("проверка гипотез", "статистических гипотез", "критерий согласия", "доверительные интервалы"),
              ("проверка статистических гипотез", "критерии согласия", "доверительные интервалы"),
              ("statistical hypothesis testing", "confidence intervals"), *_ST_BROAD),
    TermEntry(("предел функции", "пределы", "производная", "дифференцирование", "непрерывность"),
              ("предел функции", "производная и дифференциал"), ("limits and derivatives",), *_MA_BROAD),
    TermEntry(("интеграл", "интегралы", "интегрирование", "интегральное исчисление"),
              ("определённый интеграл", "неопределённый интеграл"), ("integral calculus",), *_MA_BROAD),
    TermEntry(("числовые ряды", "степенные ряды", "ряды фурье", "функциональные ряды"),
              ("числовые ряды", "степенные ряды", "ряды Фурье"), ("series convergence", "Fourier series"),
              *_MA_BROAD),
    TermEntry(("теория графов", "графы", "кратчайшие пути", "обход графа", "алгоритмы на графах"),
              ("теория графов", "алгоритмы на графах", "поиск кратчайших путей"),
              ("graph theory", "graph algorithms", "shortest paths"), *_DM_BROAD),
    TermEntry(("сортировка", "алгоритмы сортировки", "сложность алгоритмов", "асимптотическая сложность"),
              ("алгоритмы сортировки", "вычислительная сложность алгоритмов"),
              ("sorting algorithms", "computational complexity"), *_AL_BROAD),
    TermEntry(("динамическое программирование",), ("динамическое программирование",), ("dynamic programming",),
              *_AL_BROAD),
    TermEntry(("реляционная модель", "нормализация", "sql", "реляционные базы данных", "транзакции"),
              ("реляционные базы данных", "нормализация отношений", "язык SQL"),
              ("relational databases", "database normalization", "SQL"), ("базы данных",), ("databases",)),
    TermEntry(("компиляторы", "синтаксический анализ", "формальные грамматики", "лексический анализ"),
              ("компиляторы", "синтаксический анализ", "формальные грамматики"),
              ("compilers", "parsing", "formal grammars"), ("теория формальных языков",), ("formal languages",)),
    TermEntry(("операционные системы", "процессы и потоки", "управление памятью", "планирование процессов"),
              ("операционные системы", "процессы и потоки", "управление памятью"),
              ("operating systems", "process scheduling", "memory management"), ("операционные системы",),
              ("operating systems",)),
    TermEntry(("объектно-ориентированное программирование", "ооп", "наследование", "полиморфизм"),
              ("объектно-ориентированное программирование", "наследование и полиморфизм"),
              ("object-oriented programming",), ("программирование",), ("programming",)),
    TermEntry(("анализ данных", "разведочный анализ", "визуализация данных", "интеллектуальный анализ данных"),
              ("интеллектуальный анализ данных", "разведочный анализ данных"),
              ("data mining", "exploratory data analysis"), *_ML_BROAD),
)

_IT_MARKERS = ("машинн", "нейрон", "данных", "программир", "информатик", "алгоритм", "подкреплен", "интеллект",
               "компьютер", "сети", "операционн", "компилятор", "глубокое обучение", "learning", "data", "python",
               "обработка", "зрение")
_MATH_MARKERS = ("алгебр", "геометри", "математическ", "анализ", "дифференциальн", "интеграл", "вероятност",
                 "статистик", "дискретн", "топологи", "теория чисел", "уравнени", "численные методы", "кривые",
                 "поверхност", "матриц", "ряды", "логика", "математика")


def classify_area(topic: str, discipline: str) -> Literal["math", "it"]:
    text = normalise(f"{discipline} {topic}")
    if any(marker in text for marker in _IT_MARKERS):
        return "it"
    if any(marker in text for marker in _MATH_MARKERS):
        return "math"
    return "it"

# ---------------------------------------------------------------------------
# Каталог фундаментальных учебников (только метаданные, без ссылок)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CanonicalBook:
    key: str
    authors: tuple[str, ...]
    title: str
    publisher: str
    triggers: tuple[str, ...]

    def matches(self, text_stems: frozenset[str]) -> bool:
        return any(set(phrase_stems(trigger)) <= text_stems for trigger in self.triggers)

    @property
    def surnames(self) -> str:
        return " ".join(surname(author).capitalize() for author in self.authors[:2])


CANONICAL_BOOKS: tuple[CanonicalBook, ...] = (
    CanonicalBook("sutton-barto", ("Р. С. Саттон", "Э. Г. Барто"), "Обучение с подкреплением", "ДМК Пресс",
                  ("обучение с подкреплением", "временных различий", "временных разностей", "q-обучение",
                   "марковский процесс принятия решений", "reinforcement learning", "temporal difference")),
    CanonicalBook("goodfellow", ("Я. Гудфеллоу", "И. Бенджио", "А. Курвилль"), "Глубокое обучение", "ДМК Пресс",
                  ("глубокое обучение", "нейронные сети", "нейронных сетей", "сверточные нейронные сети",
                   "рекуррентные нейронные сети", "обратного распространения", "deep learning")),
    CanonicalBook("nikolenko", ("С. И. Николенко", "А. А. Кадурин", "Е. О. Архангельская"),
                  "Глубокое обучение. Погружение в мир нейронных сетей", "Питер",
                  ("глубокое обучение", "нейронные сети", "нейронных сетей", "сверточные нейронные сети")),
    CanonicalBook("haykin", ("С. Хайкин",), "Нейронные сети: полный курс", "Вильямс",
                  ("нейронные сети", "нейронных сетей", "персептрон", "обратного распространения")),
    CanonicalBook("flach", ("П. Флах",), "Машинное обучение. Наука и искусство построения алгоритмов, "
                  "которые извлекают знания из данных", "ДМК Пресс",
                  ("машинное обучение", "деревья решений", "линейная регрессия", "кластеризация",
                   "логистическая регрессия", "опорных векторов")),
    CanonicalBook("james", ("Г. Джеймс", "Д. Уиттон", "Т. Хасти", "Р. Тибширани"),
                  "Введение в статистическое обучение с примерами на языке R", "ДМК Пресс",
                  ("статистическое обучение", "линейная регрессия", "кросс-валидация", "регуляризация",
                   "деревья решений", "опорных векторов")),
    CanonicalBook("cormen", ("Т. Кормен", "Ч. Лейзерсон", "Р. Ривест", "К. Штайн"), "Алгоритмы: построение и анализ",
                  "Вильямс", ("алгоритмы", "сортировка", "теория графов", "кратчайшие пути",
                              "динамическое программирование", "сложность алгоритмов", "структуры данных")),
    CanonicalBook("knuth", ("Д. Кнут",), "Искусство программирования", "Вильямс",
                  ("алгоритмы сортировки", "сортировка", "комбинаторные алгоритмы")),
    CanonicalBook("ilyin-poznyak-ag", ("В. А. Ильин", "Э. Г. Позняк"), "Аналитическая геометрия", "Физматлит",
                  ("аналитическая геометрия", "кривые второго порядка", "поверхности второго порядка",
                   "прямая и плоскость", "векторная алгебра")),
    CanonicalBook("ilyin-poznyak-la", ("В. А. Ильин", "Э. Г. Позняк"), "Линейная алгебра", "Физматлит",
                  ("линейная алгебра", "матрицы", "определители", "линейные пространства", "собственные значения",
                   "системы линейных уравнений", "квадратичные формы")),
    CanonicalBook("beklemishev", ("Д. В. Беклемишев",), "Курс аналитической геометрии и линейной алгебры", "Физматлит",
                  ("аналитическая геометрия", "линейная алгебра", "кривые второго порядка", "квадратичные формы")),
    CanonicalBook("akopyan-zaslavsky", ("А. В. Акопян", "А. А. Заславский"), "Геометрические свойства кривых второго порядка",
                  "МЦНМО", ("кривые второго порядка", "коники", "конические сечения")),
    CanonicalBook("kostrikin", ("А. И. Кострикин",), "Введение в алгебру", "МЦНМО",
                  ("алгебра", "линейная алгебра", "многочлены", "группы")),
    CanonicalBook("vinberg", ("Э. Б. Винберг",), "Курс алгебры", "МЦНМО",
                  ("алгебра", "линейная алгебра", "многочлены", "квадратичные формы")),
    CanonicalBook("zorich", ("В. А. Зорич",), "Математический анализ", "МЦНМО",
                  ("математический анализ", "предел функции", "производная", "интеграл", "числовые ряды")),
    CanonicalBook("fichtenholz", ("Г. М. Фихтенгольц",), "Курс дифференциального и интегрального исчисления",
                  "Физматлит", ("математический анализ", "интеграл", "производная", "числовые ряды")),
    CanonicalBook("shiryaev", ("А. Н. Ширяев",), "Вероятность", "МЦНМО",
                  ("теория вероятностей", "случайные величины", "марковские цепи", "предельные теоремы")),
    CanonicalBook("gmurman", ("В. Е. Гмурман",), "Теория вероятностей и математическая статистика", "Юрайт",
                  ("теория вероятностей", "математическая статистика", "случайные величины", "проверка гипотез")),
    CanonicalBook("vereshchagin-shen", ("Н. К. Верещагин", "А. Шень"),
                  "Лекции по математической логике и теории алгоритмов", "МЦНМО",
                  ("математическая логика", "теория алгоритмов", "вычислимые функции")),
    CanonicalBook("shen-programming", ("А. Шень",), "Программирование: теоремы и задачи", "МЦНМО",
                  ("программирование", "алгоритмы", "сортировка", "рекурсия")),
    CanonicalBook("aho", ("А. Ахо", "М. Лам", "Р. Сети", "Дж. Ульман"),
                  "Компиляторы: принципы, технологии и инструментарий", "Вильямс",
                  ("компиляторы", "синтаксический анализ", "лексический анализ", "формальные грамматики")),
    CanonicalBook("tanenbaum-os", ("Э. Таненбаум", "Х. Бос"), "Современные операционные системы", "Питер",
                  ("операционные системы", "процессы и потоки", "управление памятью")),
    CanonicalBook("date", ("К. Дж. Дейт",), "Введение в системы баз данных", "Вильямс",
                  ("базы данных", "реляционная модель", "sql", "нормализация", "транзакции")),
    CanonicalBook("yablonsky", ("С. В. Яблонский",), "Введение в дискретную математику", "Высшая школа",
                  ("дискретная математика", "теория графов", "булевы функции")),
    CanonicalBook("gavrilov-sapozhenko", ("Г. П. Гаврилов", "А. А. Сапоженко"),
                  "Задачи и упражнения по дискретной математике", "Физматлит",
                  ("дискретная математика", "теория графов", "булевы функции")),
)


def catalog_match(title: str, authors: Sequence[str]) -> CanonicalBook | None:
    key = title_key(title)
    names = {surname(author) for author in authors}
    for book in CANONICAL_BOOKS:
        book_key = title_key(book.title)
        same_author = bool(names & {surname(author) for author in book.authors})
        if similar_titles(title, book.title) and (same_author or not names):
            return book
        if book_key and key.startswith(book_key) and same_author:
            return book
    return None

# ---------------------------------------------------------------------------
# Профиль темы
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TopicProfile:
    topic: str
    discipline: str
    area: Literal["math", "it"]
    topic_stems: frozenset[str]
    discipline_stems: frozenset[str]
    phrases: tuple[tuple[str, ...], ...]
    broad_phrases: tuple[tuple[str, ...], ...]
    queries: tuple[str, ...]
    article_queries_ru: tuple[str, ...]
    article_queries_en: tuple[str, ...]
    textbook_queries: tuple[str, ...]
    catalog: tuple[CanonicalBook, ...]


def _unique(values: Iterable[str]) -> list[str]:
    seen: dict[str, str] = {}
    for value in values:
        text = clean(value)
        if text and normalise(text) not in seen:
            seen[normalise(text)] = text
    return list(seen.values())


def build_profile(topic: str, discipline: str, extra_terms: Sequence[str] = ()) -> TopicProfile:
    topic, discipline = clean(topic), clean(discipline)
    topic_set = stem_set(topic)
    discipline_set = stem_set(discipline)
    topic_entries = [entry for entry in TERMS if entry.matches(topic_set)]
    context_entries = [entry for entry in TERMS if entry not in topic_entries and entry.matches(discipline_set)]
    dictionary_ru = _unique(term for entry in topic_entries for term in entry.ru)
    dictionary_en = _unique(term for entry in topic_entries for term in entry.en)
    llm_ru = _unique(term for term in extra_terms if has_cyrillic(term))
    llm_en = _unique(term for term in extra_terms if clean(term) and not has_cyrillic(term))
    ru_terms = _unique([*dictionary_ru[:2], *llm_ru[:2], *dictionary_ru[2:], *llm_ru[2:]])
    en_terms = _unique([*dictionary_en[:2], *llm_en[:2], *dictionary_en[2:], *llm_en[2:]])
    broad_ru = _unique([*(term for entry in topic_entries for term in entry.broad_ru),
                        *(term for entry in context_entries for term in entry.ru[:1]), discipline])
    broad_en = _unique([*(term for entry in topic_entries for term in entry.broad_en),
                        *(term for entry in context_entries for term in entry.en[:1])])
    broad_ru = [term for term in broad_ru if normalise(term) != normalise(topic)] or [discipline]
    general = broad_ru[0] if broad_ru else discipline
    exact = f'"{topic}" {general}'.strip()
    terminology = " ".join(_unique([*ru_terms[:2], *en_terms[:3]])) or f"{topic} {discipline}"
    generalised = f"{general} учебное пособие pdf"
    article_ru = _unique([topic, *ru_terms[:3]])[:4]
    article_en = _unique([*en_terms[:3], *broad_en[:1]])[:4]
    textbook_queries = _unique([f"{topic} учебное пособие", f"{general} учебное пособие", f"{discipline} учебник"])
    phrases = tuple(dict.fromkeys(phrase for term in [*ru_terms, *en_terms] if (phrase := phrase_stems(term))))
    broad = tuple(dict.fromkeys(phrase for term in [*broad_ru, *broad_en] if (phrase := phrase_stems(term))))
    joint = topic_set | discipline_set | frozenset(s for term in broad_ru for s in stems(term))
    catalog = tuple(book for book in CANONICAL_BOOKS if book.matches(topic_set) or book.matches(joint))
    return TopicProfile(topic, discipline, classify_area(topic, discipline), topic_set, discipline_set, phrases,
                        broad, (exact, terminology, generalised), tuple(article_ru), tuple(article_en),
                        tuple(textbook_queries), catalog[:4])

# ---------------------------------------------------------------------------
# Кандидаты и результат
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    kind: SourceKind
    title: str
    origin: str
    source_url: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str = ""
    publisher: str = ""
    abstract: str = ""
    body: str = ""
    keywords: list[str] = field(default_factory=list)
    language: str = ""
    doi: str = ""
    edn: str = ""
    udc: str = ""
    bbk: str = ""
    isbn: str = ""
    volume: str = ""
    issue: str = ""
    pages: str = ""
    citations: int | None = None
    affiliations: list[str] = field(default_factory=list)
    labels: set[str] = field(default_factory=set)
    venue_type: str = ""
    peer_reviewed: bool = False
    field_of_study: str = ""
    pdf_candidates: list[str] = field(default_factory=list)
    pdf_url: str = ""
    pdf_size: int | None = None
    pdf_fingerprint: str = ""
    grif: str = ""
    label: str = ""
    ebs: str = ""
    catalog_key: str = ""
    enrich: set[str] = field(default_factory=set)
    relevance: float = 0.0
    score: float = 0.0
    breakdown: dict[str, float] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    reason: str = ""

    def text(self) -> str:
        return " ".join([self.title, self.abstract, " ".join(self.keywords), self.body[:6000]])


@dataclass(frozen=True)
class LiteratureSource:
    title: str
    authors: list[str]
    type: SourceKind
    year: int | None
    publisher_or_journal: str
    verification_badge: str
    pdf_url: str | None
    source_url: str
    relevance_score: float
    doi: str = ""
    isbn: str = ""
    udc: str = ""
    language: str = ""
    origin: str = ""
    evidence: tuple[str, ...] = ()
    score_breakdown: dict[str, float] = field(default_factory=dict)

    def to_dict(self, *, extended: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "title": self.title, "authors": list(self.authors), "type": self.type, "year": self.year,
            "publisher_or_journal": self.publisher_or_journal, "verification_badge": self.verification_badge,
            "pdf_url": self.pdf_url, "source_url": self.source_url, "relevance_score": self.relevance_score,
        }
        if extended:
            data.update({"doi": self.doi or None, "isbn": self.isbn or None, "udc": self.udc or None,
                         "language": self.language or None, "origin": self.origin,
                         "evidence": list(self.evidence), "score_breakdown": dict(self.score_breakdown)})
        return data


@dataclass
class TopicLiterature:
    discipline: str
    topic: str
    queries: list[str]
    sources: list[LiteratureSource]
    unavailable: list[str] = field(default_factory=list)
    rejected: list[dict[str, str]] = field(default_factory=list)

    @property
    def textbooks(self) -> list[LiteratureSource]:
        return [source for source in self.sources if source.type == "textbook"]

    @property
    def articles(self) -> list[LiteratureSource]:
        return [source for source in self.sources if source.type == "article"]

    @property
    def meets_requirement(self) -> bool:
        return len(self.textbooks) >= 1 or len(self.articles) >= 2

    def to_dict(self, *, extended: bool = True) -> dict[str, Any]:
        data = {"discipline": self.discipline, "topic": self.topic, "meets_requirement": self.meets_requirement,
                "sources": [source.to_dict(extended=extended) for source in self.sources]}
        if extended:
            data.update({"queries": self.queries, "unavailable_sources": self.unavailable,
                         "rejected": self.rejected[:50]})
        return data

# ---------------------------------------------------------------------------
# Адаптеры источников
# ---------------------------------------------------------------------------


@dataclass
class SearchContext:
    http: Http
    config: ParserConfig
    profile: TopicProfile
    validator: PdfValidator
    yandex_budget: list[int]


def _int(value: Any) -> int | None:
    try:
        number = int(str(value).strip()[:4]) if value is not None and str(value).strip() else None
    except ValueError:
        return None
    return _plausible_year(number)


def _abstract_from_index(index: dict[str, list[int]] | None) -> str:
    if not index:
        return ""
    positions = sorted((position, word) for word, places in index.items() for position in places)
    return clean(" ".join(word for _, word in positions))[:1500]


async def search_openalex(ctx: SearchContext, query: str, *, language: str | None = "ru",
                          kind: SourceKind = "article") -> list[Candidate]:
    types = "article|review" if kind == "article" else "book|book-chapter"
    filters = [f"type:{types}", "is_oa:true"]
    if language:
        filters.append(f"language:{language}")
    params = {
        "search": query, "filter": ",".join(filters), "per_page": "25",
        "select": "id,doi,display_name,publication_year,language,type,authorships,primary_location,"
                  "best_oa_location,open_access,cited_by_count,biblio,primary_topic,abstract_inverted_index",
    }
    if ctx.config.contact_email:
        params["mailto"] = ctx.config.contact_email
    data = await ctx.http.get_json(OPENALEX_WORKS, params=params)
    candidates = []
    for work in data.get("results") or []:
        title = clean(work.get("display_name"))
        if not title:
            continue
        doi = normalise_doi(work.get("doi") or "")
        location = work.get("primary_location") or {}
        source = location.get("source") or {}
        best = work.get("best_oa_location") or {}
        access = work.get("open_access") or {}
        pdfs = [url for url in (best.get("pdf_url"), location.get("pdf_url"),
                                access.get("oa_url") if str(access.get("oa_url") or "").lower().endswith(".pdf") else None)
                if url]
        landing = best.get("landing_page_url") or location.get("landing_page_url") or \
            (f"https://doi.org/{doi}" if doi else work.get("id") or "")
        authorships = work.get("authorships") or []
        authors = [clean((item.get("author") or {}).get("display_name")) for item in authorships[:8]]
        affiliations = [clean(inst.get("display_name")) for item in authorships for inst in item.get("institutions") or []]
        biblio = work.get("biblio") or {}
        pages = "–".join(str(page) for page in (biblio.get("first_page"), biblio.get("last_page")) if page)
        topic = work.get("primary_topic") or {}
        work_type = str(work.get("type") or "")
        candidate_kind: SourceKind = "textbook" if work_type in {"book", "book-chapter"} else "article"
        venue_type = str(source.get("type") or "")
        candidates.append(Candidate(
            kind=candidate_kind, title=title[:500], origin="OpenAlex", source_url=landing,
            authors=[author for author in authors if author], year=_int(work.get("publication_year")),
            venue=clean(source.get("display_name")), publisher=clean(source.get("host_organization_name")),
            abstract=_abstract_from_index(work.get("abstract_inverted_index")), language=work.get("language") or "",
            doi=doi, volume=str(biblio.get("volume") or ""), issue=str(biblio.get("issue") or ""), pages=pages,
            citations=work.get("cited_by_count"), affiliations=[name for name in affiliations if name][:8],
            venue_type=venue_type, field_of_study=clean((topic.get("field") or {}).get("display_name")),
            peer_reviewed=candidate_kind == "article" and venue_type in {"journal", "conference"},
            pdf_candidates=_unique(pdfs),
        ))
    return candidates


def mathnet_pdf_url(doi: str) -> str:
    match = re.fullmatch(r"10\.4213/([a-z]+)(\d+)", doi or "")
    if not match:
        return ""
    return (f"https://www.mathnet.ru/php/getFT.phtml?jrnid={match.group(1)}&paperid={match.group(2)}"
            f"&what=fullt&option_lang=rus")


async def search_crossref(ctx: SearchContext, query: str, *, prefix: str | None = None,
                          russian_only: bool = True, kind: SourceKind = "article") -> list[Candidate]:
    types = ["type:journal-article", "type:proceedings-article"] if kind == "article" else \
        ["type:book", "type:monograph", "type:edited-book", "type:reference-book"]
    filters = [*types]
    if prefix:
        filters.append(f"prefix:{prefix}")
    params = {
        "query.bibliographic": query, "filter": ",".join(filters), "rows": "20",
        "select": "DOI,title,container-title,issued,author,is-referenced-by-count,volume,issue,page,link,ISSN,ISBN,"
                  "publisher,type,abstract,language,update-to,URL",
    }
    if ctx.config.contact_email:
        params["mailto"] = ctx.config.contact_email
    data = await ctx.http.get_json(CROSSREF_WORKS, params=params)
    candidates = []
    for work in (data.get("message") or {}).get("items") or []:
        title = clean(next(iter(work.get("title") or []), ""))
        doi = normalise_doi(work.get("DOI") or "")
        if not title or not doi:
            continue
        if any("retract" in str(update.get("type", "")).lower() for update in work.get("update-to") or []):
            continue
        language = str(work.get("language") or "")
        if russian_only and not has_cyrillic(title) and language != "ru":
            continue
        issued = ((work.get("issued") or {}).get("date-parts") or [[None]])[0]
        authors = [clean(" ".join(person.get(key) or "" for key in ("given", "family")))
                   for person in (work.get("author") or [])[:8]]
        affiliations = [clean(item.get("name")) for person in work.get("author") or []
                        for item in person.get("affiliation") or []]
        pdfs = [link.get("URL") for link in work.get("link") or []
                if "pdf" in str(link.get("content-type") or "").lower() and link.get("URL")]
        origin = next((name for doi_prefix, name in TRUSTED_DOI_PREFIXES.items() if doi.startswith(doi_prefix + "/")),
                      "Crossref")
        if doi.startswith("10.4213/") and (pdf := mathnet_pdf_url(doi)):
            pdfs.append(pdf)
        work_type = str(work.get("type") or "")
        venue = clean(next(iter(work.get("container-title") or []), ""))
        is_book = work_type in {"book", "monograph", "edited-book", "reference-book"}
        isbn = next((valid for raw in work.get("ISBN") or [] if (valid := valid_isbn(raw))), "")
        candidates.append(Candidate(
            kind="textbook" if is_book else "article", title=title[:500], origin=origin,
            source_url="https://doi.org/" + quote(doi, safe="/"), authors=[a for a in authors if a],
            year=_int(issued[0] if issued else None), venue=venue, publisher=clean(work.get("publisher")),
            abstract=strip_tags(work.get("abstract") or "")[:1500], language=language or ("ru" if has_cyrillic(title) else ""),
            doi=doi, volume=str(work.get("volume") or ""), issue=str(work.get("issue") or ""),
            pages=str(work.get("page") or ""), citations=work.get("is-referenced-by-count"),
            affiliations=[a for a in affiliations if a][:8], isbn=isbn,
            venue_type="journal" if work_type == "journal-article" else ("conference" if work_type == "proceedings-article" else work_type),
            peer_reviewed=(work_type in {"journal-article", "proceedings-article"} and bool(venue)) or origin != "Crossref",
            pdf_candidates=_unique(pdfs),
        ))
    return candidates


async def search_semantic_scholar(ctx: SearchContext, query: str) -> list[Candidate]:
    if not ctx.config.semantic_scholar_enabled:
        return []
    headers = {"x-api-key": ctx.config.semantic_scholar_api_key} if ctx.config.semantic_scholar_api_key else None
    params = {"query": query, "limit": "20",
              "fields": "title,year,authors,venue,publicationVenue,externalIds,openAccessPdf,citationCount,abstract,"
                        "fieldsOfStudy,publicationTypes,journal,url"}
    data = await ctx.http.get_json(SEMANTIC_SCHOLAR_SEARCH, params=params, headers=headers)
    candidates = []
    for paper in data.get("data") or []:
        title = clean(paper.get("title"))
        if not title:
            continue
        doi = normalise_doi((paper.get("externalIds") or {}).get("DOI") or "")
        venue = paper.get("publicationVenue") or {}
        journal = paper.get("journal") or {}
        pdf = (paper.get("openAccessPdf") or {}).get("url")
        types = {str(kind).lower() for kind in paper.get("publicationTypes") or []}
        candidates.append(Candidate(
            kind="article", title=title[:500], origin="Semantic Scholar",
            source_url=f"https://doi.org/{doi}" if doi else clean(paper.get("url")),
            authors=[clean(a.get("name")) for a in (paper.get("authors") or [])[:8] if a.get("name")],
            year=_int(paper.get("year")), venue=clean(journal.get("name") or paper.get("venue") or venue.get("name")),
            abstract=clean(paper.get("abstract"))[:1500], language="ru" if has_cyrillic(title) else "en", doi=doi,
            volume=clean(journal.get("volume")), pages=clean(journal.get("pages")),
            citations=paper.get("citationCount"), venue_type=clean(venue.get("type")).lower(),
            peer_reviewed=bool(types & {"journalarticle", "conference"}) or clean(venue.get("type")).lower() in {"journal", "conference"},
            field_of_study=", ".join(paper.get("fieldsOfStudy") or []), pdf_candidates=[pdf] if pdf else [],
        ))
    return candidates


async def search_cyberleninka(ctx: SearchContext, query: str) -> list[Candidate]:
    payload = {"mode": "articles", "q": query, "size": 10, "from": 0}
    headers = {"Referer": f"{CYBERLENINKA}/search?q={quote(query)}", "Origin": CYBERLENINKA,
               "Accept": "application/json"}
    data = await ctx.http.post_json(CYBERLENINKA_SEARCH, payload, headers=headers)
    if not isinstance(data, dict):
        raise SourceUnavailable("КиберЛенинка вернула неожиданный ответ")
    candidates = []
    for item in data.get("articles") or []:
        link = clean(item.get("link"))
        title = strip_tags(item.get("name") or "")
        if not title or not link.startswith("/article/"):
            continue
        url = CYBERLENINKA + link
        authors = item.get("authors") or []
        if isinstance(authors, str):
            authors = [part for part in re.split(r"\s*,\s*", authors) if part]
        candidates.append(Candidate(
            kind="article", title=title[:500], origin="КиберЛенинка", source_url=url,
            authors=[strip_tags(str(author)) for author in authors[:8]], year=_int(item.get("year")),
            venue=strip_tags(item.get("journal") or ""), abstract=strip_tags(item.get("annotation") or "")[:1500],
            language="ru" if has_cyrillic(title) else "", venue_type="journal",
            pdf_candidates=[url + "/pdf"], enrich={"cyberleninka"},
        ))
    return candidates


_KNOWN_LABELS = {"вак": "ВАК", "rsci": "RSCI", "ринц": "РИНЦ", "scopus": "Scopus", "web of science": "Web of Science",
                 "wos": "Web of Science", "esci": "ESCI", "doaj": "DOAJ", "erih plus": "ERIH PLUS"}
_FIELD_LINE = re.compile(r"Область наук[^:\n]*[:\n]\s*([^\n]{3,90})")


def labels_from(texts: Iterable[str]) -> set[str]:
    found = set()
    for text in texts:
        lowered = normalise(text)
        for marker, name in _KNOWN_LABELS.items():
            if re.search(rf"(?<![a-zа-я]){re.escape(marker)}(?![a-zа-я])", lowered):
                found.add(name)
    return found


async def enrich_cyberleninka(ctx: SearchContext, candidate: Candidate) -> None:
    page = parse_html(await ctx.http.get_text(candidate.source_url), candidate.source_url)
    candidate.title = page.first("citation_title") or candidate.title
    authors = page.all("citation_author")
    if authors:
        candidate.authors = authors[:8]
    candidate.venue = page.first("citation_journal_title") or candidate.venue
    candidate.year = _int(page.first("citation_publication_date", "citation_date")) or candidate.year
    candidate.volume = page.first("citation_volume") or candidate.volume
    candidate.issue = page.first("citation_issue") or candidate.issue
    first, last = page.first("citation_firstpage"), page.first("citation_lastpage")
    if first:
        candidate.pages = f"{first}–{last}" if last else first
    candidate.doi = normalise_doi(page.first("citation_doi")) or candidate.doi
    keywords = page.first("citation_keywords", "keywords")
    if keywords:
        candidate.keywords = [clean(word) for word in re.split(r"[;,]", keywords) if clean(word)][:15]
    candidate.abstract = candidate.abstract or page.first("description", "og:description")[:1500]
    pdf = page.first("citation_pdf_url")
    if pdf:
        candidate.pdf_candidates = _unique([urljoin(candidate.source_url, pdf), *candidate.pdf_candidates])
    candidate.labels |= labels_from(page.labels)
    front = parse_front_matter(page.text[:20000])
    candidate.udc = candidate.udc or front.udc
    candidate.bbk = candidate.bbk or front.bbk
    candidate.edn = candidate.edn or front.edn
    candidate.doi = candidate.doi or front.doi
    if match := _FIELD_LINE.search(page.text):
        candidate.field_of_study = clean(match.group(1))
    indexed = candidate.labels & {"ВАК", "RSCI", "РИНЦ", "Scopus", "Web of Science", "ESCI"}
    candidate.peer_reviewed = bool(indexed) or bool(candidate.doi)


_MCCME_TAIL = re.compile(
    r"\s*(?:\(\s*c\d?\s*\)|©|\(\s*(?:pdf|djvu|ps|html|tex)\b|\[|—\s*М\.|\bМ\.\s*:|\bМ\.,|\bМЦНМО\b|\bИзд).*$",
    re.IGNORECASE,
)


def parse_mccme_entry(text: str) -> tuple[list[str], str, str, int | None]:
    """Разбор строки каталога МЦНМО «А. В. Акопян. Геометрия в картинках. (c2) М.: МЦНМО, 2011. (pdf, 0.5M)»."""
    remaining = clean(text)
    authors: list[str] = []
    while True:
        match = re.match(r"^([А-ЯЁ]\.\s?(?:[А-ЯЁ]\.\s?)?[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?)\s*(?:,|и\b)?\s*", remaining)
        if not match:
            break
        authors.append(clean(match.group(1)))
        remaining = remaining[match.end():]
    remaining = remaining.lstrip(".,: ")
    title = _MCCME_TAIL.sub("", remaining).strip(" .,;:")
    years = [int(year) for year in re.findall(r"\b((?:19[5-9]|20[0-4])\d)\b", text)]
    publisher = detect_publisher(text) or "МЦНМО"
    return authors, title[:300], publisher, _plausible_year(max(years)) if years else None


async def search_mccme(ctx: SearchContext) -> list[Candidate]:
    page = parse_html(await ctx.http.get_text(MCCME_FREE_BOOKS), MCCME_FREE_BOOKS)
    candidates = []
    for text, links in page.items:
        pdf = next((href for href, label in links
                    if urlsplit(href).path.lower().endswith(".pdf") or "pdf" in normalise(label)), "")
        if not pdf or url_extension(pdf) in {".zip", ".djvu", ".ps", ".gz"} or len(text) < 10:
            continue
        authors, title, publisher, year = parse_mccme_entry(text)
        if not title or not authors:
            continue
        candidates.append(Candidate(
            kind="textbook", title=title, origin="МЦНМО", source_url=MCCME_FREE_BOOKS, authors=authors,
            year=year, publisher=publisher, language="ru", venue_type="publisher-library",
            pdf_candidates=[pdf], label=kind_label(text) or "Учебное пособие",
        ))
    return candidates


@dataclass(frozen=True)
class WebHit:
    url: str
    domain: str
    title: str
    mime: str
    passages: tuple[str, ...]


def parse_yandex_xml(raw: bytes) -> list[WebHit]:
    root = ElementTree.fromstring(raw)
    error = root.find(".//error")
    if error is not None:
        if error.get("code") == "15":
            return []
        raise SourceUnavailable(f"Yandex Search: {clean(error.text)} (код {error.get('code')})")
    hits = []
    for doc in root.iter("doc"):
        url = clean(doc.findtext("url"))
        if not url:
            continue
        title_node = doc.find("title")
        title = clean("".join(title_node.itertext())) if title_node is not None else ""
        passages = tuple(clean("".join(node.itertext())) for node in doc.iter("passage"))
        headline = clean(doc.findtext("headline"))
        hits.append(WebHit(url, clean(doc.findtext("domain")) or host_of(url), title,
                           clean(doc.findtext("mime-type")).lower(), passages + ((headline,) if headline else ())))
    return hits


async def yandex_web_search(ctx: SearchContext, text: str) -> list[WebHit]:
    if not ctx.config.yandex_search_enabled or ctx.yandex_budget[0] <= 0:
        return []
    ctx.yandex_budget[0] -= 1
    body = {
        "query": {"searchType": "SEARCH_TYPE_RU", "queryText": text[:400], "familyMode": "FAMILY_MODE_STRICT",
                  "page": "0", "fixTypoMode": "FIX_TYPO_MODE_OFF"},
        "groupSpec": {"groupMode": "GROUP_MODE_FLAT", "groupsOnPage": "20", "docsInGroup": "1"},
        "maxPassages": "2", "l10n": "LOCALIZATION_RU", "folderId": ctx.config.yandex_folder_id,
        "responseFormat": "FORMAT_XML",
    }
    headers = {"Authorization": f"Api-Key {ctx.config.yandex_search_api_key}"}
    response = await ctx.http.request("POST", YANDEX_WEB_SEARCH, json=body, api=True, headers=headers)
    response.raise_for_status()
    raw = response.json().get("rawData") or ""
    try:
        xml = base64.b64decode(raw, validate=True)
    except (ValueError, TypeError):
        xml = raw.encode("utf-8")
    if not xml.lstrip().startswith(b"<"):
        xml = raw.encode("utf-8")
    return parse_yandex_xml(xml)


def site_queries(base: str, domains: Sequence[str], *, pdf: bool) -> list[str]:
    queries, chunk = [], []
    suffix = " mime:pdf" if pdf else ""
    for domain in domains:
        trial = f"{base} ({' | '.join(f'site:{d}' for d in [*chunk, domain])}){suffix}"
        if len(trial) > 400 and chunk:
            queries.append(f"{base} ({' | '.join(f'site:{d}' for d in chunk)}){suffix}")
            chunk = []
        chunk.append(domain)
    if chunk:
        queries.append(f"{base} ({' | '.join(f'site:{d}' for d in chunk)}){suffix}")
    return queries


async def search_yandex_textbooks(ctx: SearchContext) -> list[Candidate]:
    if not ctx.config.yandex_search_enabled:
        return []
    profile = ctx.profile
    requests: list[tuple[str, bool]] = []
    for book in profile.catalog[:2]:
        requests += [(query, False) for query in site_queries(f"{book.title} {book.surnames}", list(EBS_DOMAINS), pdf=False)[:1]]
    for query in profile.textbook_queries[:2]:
        requests += [(item, True) for item in site_queries(query, EDU_DOMAINS, pdf=True)[:1]]
    results = await asyncio.gather(*(yandex_web_search(ctx, query) for query, _ in requests), return_exceptions=True)
    candidates: list[Candidate] = []
    seen: set[str] = set()
    for result in results:
        if isinstance(result, BaseException):
            logger.warning("Yandex Search: %s", result)
            continue
        for hit in result:
            key = url_key(hit.url)
            host = host_of(hit.url)
            if not key or key in seen or blocked_reason(hit.url):
                continue
            seen.add(key)
            title = re.sub(r"\s+[—|-]\s+[^—|-]{2,40}$", "", hit.title).strip() or hit.title
            ebs_name = next((name for domain, name in EBS_DOMAINS.items() if domain_in(host, [domain])), "")
            if ebs_name:
                candidates.append(Candidate(kind="textbook", title=title, origin=f"ЭБС {ebs_name}", source_url=hit.url,
                                            body=" ".join(hit.passages), ebs=ebs_name, venue_type="ebs",
                                            enrich={"ebs"}))
            elif domain_in(host, EDU_DOMAINS) and (hit.mime == "pdf" or url_extension(hit.url) == ".pdf"):
                candidates.append(Candidate(kind="textbook", title=title, origin=f"Репозиторий {host}",
                                            source_url=hit.url, body=" ".join(hit.passages), venue_type="repository",
                                            pdf_candidates=[hit.url], enrich={"pdf_front"}))
    return candidates


async def enrich_ebs(ctx: SearchContext, candidate: Candidate) -> None:
    page = parse_html(await ctx.http.get_text(candidate.source_url), candidate.source_url)
    books = jsonld_objects(page, {"Book"})
    if books:
        book = books[0]
        candidate.title = clean(book.get("name")) or candidate.title
        authors = book.get("author") or []
        authors = authors if isinstance(authors, list) else [authors]
        candidate.authors = [clean(a.get("name") if isinstance(a, dict) else a) for a in authors if a][:6] or candidate.authors
        publisher = book.get("publisher")
        candidate.publisher = clean(publisher.get("name") if isinstance(publisher, dict) else publisher) or candidate.publisher
        candidate.isbn = valid_isbn(str(book.get("isbn") or "")) or candidate.isbn
        candidate.year = _int(book.get("datePublished")) or candidate.year
        candidate.abstract = clean(book.get("description"))[:1500] or candidate.abstract
    candidate.title = page.first("citation_title", "og:title") or page.h1 or candidate.title
    candidate.authors = candidate.authors or page.all("citation_author")[:6] or page.all("book:author")[:6]
    candidate.isbn = candidate.isbn or valid_isbn(page.first("citation_isbn", "book:isbn"))
    candidate.year = candidate.year or _int(page.first("citation_publication_date", "book:release_date"))
    candidate.publisher = candidate.publisher or page.first("citation_publisher")
    front = parse_front_matter(page.text[:30000])
    candidate.isbn = candidate.isbn or front.isbn
    candidate.year = candidate.year or front.year
    candidate.publisher = candidate.publisher or front.publisher
    candidate.grif = candidate.grif or front.grif
    candidate.udc, candidate.bbk = candidate.udc or front.udc, candidate.bbk or front.bbk
    candidate.authors = candidate.authors or front.authors
    candidate.label = candidate.label or kind_label(page.text[:6000])
    candidate.body = page.text[:6000]
    candidate.language = "ru" if has_cyrillic(candidate.title) else candidate.language


async def enrich_pdf_front(ctx: SearchContext, candidate: Candidate) -> None:
    for url in candidate.pdf_candidates[:2]:
        check = await ctx.validator.check(url, full=True)
        if not check.ok:
            candidate.reason = check.reason
            continue
        candidate.pdf_url, candidate.pdf_size, candidate.pdf_fingerprint = url, check.size, check.fingerprint
        if check.content is None:
            return
        text, info = pdf_front_text(check.content)
        front = parse_front_matter(text)
        meta_title = info.get("title", "")
        if meta_title and len(meta_title) > 8 and not re.search(r"microsoft|untitled|\.docx?|\.tex|без имени",
                                                                meta_title, re.IGNORECASE):
            candidate.title = meta_title
        candidate.authors = candidate.authors or front.authors or \
            [clean(name) for name in re.split(r"[;,]| and ", info.get("author", "")) if len(clean(name)) > 3][:6]
        candidate.publisher = candidate.publisher or front.publisher
        candidate.year = candidate.year or front.year
        candidate.isbn, candidate.udc, candidate.bbk = front.isbn, front.udc, front.bbk
        candidate.grif, candidate.edn = front.grif, front.edn
        candidate.doi = candidate.doi or front.doi
        candidate.label = candidate.label or front.label
        candidate.body = clean(text)[:8000]
        candidate.language = "ru" if has_cyrillic(text[:3000] or candidate.title) else "en"
        return

# ---------------------------------------------------------------------------
# Релевантность, верификация, скоринг
# ---------------------------------------------------------------------------

_NEGATIVE = frozenset(stem(word) for word in NEGATIVE_WORDS)
_POSITIVE = frozenset(stem(word) for word in POSITIVE_WORDS)


def relevance(profile: TopicProfile, candidate: Candidate) -> float:
    if candidate.kind == "article" and profile.phrases:
        texts = [candidate.title, candidate.abstract, *candidate.keywords, candidate.body[:6000]]
        phrases = (*profile.phrases, phrase_stems(profile.topic))
        streams = [tuple(stems(text)) for text in texts if text]
        if not any(phrase and stream[offset:offset + len(phrase)] == phrase
                   for stream in streams for phrase in phrases for offset in range(len(stream))):
            return 0.0
    title = stem_set(candidate.title)
    body = stem_set(" ".join([candidate.abstract, " ".join(candidate.keywords), candidate.body[:6000]]))
    everything = title | body
    core = profile.topic_stems
    cover_title = len(core & title) / len(core) if core else 0.0
    cover_all = len(core & everything) / len(core) if core else 0.0
    phrase_title = any(set(phrase) <= title for phrase in profile.phrases)
    phrase_body = any(set(phrase) <= everything for phrase in profile.phrases)
    topical = max(cover_title, 0.85 * cover_all, 0.9 if phrase_title else 0.0, 0.7 if phrase_body else 0.0)
    # Первая «широкая» формулировка — ближайшая к теме (для TD это «обучение с подкреплением»).
    for position, phrase in enumerate(profile.broad_phrases):
        specific = position == 0
        if set(phrase) <= title:
            topical = max(topical, (0.75 if specific else 0.6) if candidate.kind == "textbook" else 0.4)
        elif set(phrase) <= everything:
            topical = max(topical, (0.55 if specific else 0.5) if candidate.kind == "textbook" else 0.35)
    disc = profile.discipline_stems
    discipline_cover = len(disc & everything) / len(disc) if disc else 0.0
    return round(min(1.0, 0.8 * topical + 0.2 * discipline_cover), 3)


def domain_rejection(candidate: Candidate) -> str:
    field_text = normalise(candidate.field_of_study)
    allowed = ("computer", "mathemat", "engineering", "decision", "physics", "компьютер", "математ", "информац",
               "техни", "физик", "механик")
    if field_text and any(marker in field_text for marker in FORBIDDEN_FIELDS) \
            and not any(marker in field_text for marker in allowed):
        return f"нецелевая область знаний: {candidate.field_of_study}"
    words = stem_set(" ".join([candidate.title, candidate.abstract, " ".join(candidate.keywords)]))
    negative = words & _NEGATIVE
    positive = words & _POSITIVE
    if len(negative) >= 2 and len(negative) > len(positive):
        return "нецелевая предметная область (медицина/экономика/право/педагогика)"
    if stem_set(candidate.title) & _NEGATIVE and not stem_set(candidate.title) & _POSITIVE:
        return "в названии нецелевая предметная область"
    return ""


def article_evidence(candidate: Candidate) -> list[str]:
    evidence = []
    if candidate.doi:
        evidence.append(f"DOI {candidate.doi}")
    elif candidate.edn:
        evidence.append(f"EDN {candidate.edn}")
    if candidate.udc:
        evidence.append(f"УДК {candidate.udc}")
    elif candidate.bbk:
        evidence.append(f"ББК {candidate.bbk}")
    if candidate.venue and candidate.year and (candidate.volume or candidate.issue or candidate.pages):
        details = ", ".join(part for part in (
            f"т. {candidate.volume}" if candidate.volume else "", f"№ {candidate.issue}" if candidate.issue else "",
            f"с. {candidate.pages}" if candidate.pages else "") if part)
        evidence.append(f"«{candidate.venue}», {candidate.year}, {details}")
    if (candidate.citations or 0) > 0:
        evidence.append(f"цитирований: {candidate.citations}")
    elif any(ACADEMIC_AFFILIATION.search(name) for name in candidate.affiliations):
        evidence.append("аффилиация: " + next(name for name in candidate.affiliations if ACADEMIC_AFFILIATION.search(name)))
    return evidence


def textbook_evidence(candidate: Candidate) -> list[str]:
    evidence = []
    if candidate.authors:
        evidence.append("авторы: " + ", ".join(candidate.authors[:3]))
    if candidate.publisher:
        evidence.append(f"издательство: {candidate.publisher}")
    if candidate.year:
        evidence.append(f"год: {candidate.year}")
    if candidate.isbn:
        evidence.append(f"ISBN {format_isbn(candidate.isbn)}")
    if candidate.grif:
        evidence.append("гриф: " + candidate.grif[:120])
    if candidate.udc or candidate.bbk:
        evidence.append(("УДК " + candidate.udc) if candidate.udc else ("ББК " + candidate.bbk))
    if candidate.catalog_key:
        evidence.append("фундаментальный учебник из каталога")
    if candidate.ebs:
        evidence.append(f"карточка ЭБС {candidate.ebs}")
    return evidence


def verification_failure(candidate: Candidate) -> str:
    if candidate.kind == "article":
        if not candidate.peer_reviewed:
            return "не подтверждена публикация в рецензируемом издании"
        if len(article_evidence(candidate)) < 2:
            return "менее двух подтверждающих метаданных (DOI/EDN, УДК, выходные данные, цитирования/аффилиация)"
        return ""
    if not candidate.authors:
        return "у учебника не указаны авторы"
    if not ((candidate.publisher and candidate.year) or candidate.isbn or candidate.grif):
        return "нет издательства и года, ISBN или грифа"
    if not candidate.pdf_url and not candidate.ebs and not candidate.doi:
        return "нет проверенного PDF или карточки ЭБС"
    return ""


def score(candidate: Candidate, profile: TopicProfile) -> dict[str, float]:
    current = date.today().year
    recency = 0.0
    if profile.area == "math" or candidate.catalog_key:
        recency = 0.2
    elif candidate.year and MODERN_SINCE <= candidate.year <= current:
        recency = 0.2
    elif candidate.year and 2010 <= candidate.year < MODERN_SINCE:
        recency = 0.1
    parts = {"pdf": 0.3 if candidate.pdf_url else 0.0, "verification": 0.3,
             "relevance": round(0.2 * candidate.relevance, 3), "recency": recency}
    parts["total"] = round(min(1.0, sum(parts.values())), 3)
    return parts


def badge(candidate: Candidate) -> str:
    parts: list[str] = []
    if candidate.kind == "article":
        indexes = [name for name in ("ВАК", "RSCI", "РИНЦ", "Scopus", "Web of Science", "ESCI") if name in candidate.labels]
        if indexes:
            parts.append("Журнал " + " / ".join(indexes))
        elif candidate.origin in TRUSTED_DOI_PREFIXES.values():
            parts.append(f"Рецензируемый журнал • {candidate.origin}")
        elif candidate.venue_type == "conference":
            parts.append("Труды конференции")
        else:
            parts.append("Рецензируемый журнал")
        if candidate.doi:
            parts.append(f"DOI: {candidate.doi}")
        elif candidate.edn:
            parts.append(f"EDN: {candidate.edn}")
        if candidate.udc:
            parts.append(f"УДК {candidate.udc}")
        if (candidate.citations or 0) > 0:
            parts.append(f"цит.: {candidate.citations}")
    else:
        parts.append(candidate.label or "Учебник")
        if candidate.publisher:
            parts.append(f"Изд. {candidate.publisher}")
        if candidate.isbn:
            parts.append(f"ISBN {format_isbn(candidate.isbn)}")
        if candidate.grif:
            parts.append("Гриф")
        if candidate.ebs:
            parts.append(f"ЭБС {candidate.ebs}")
    if candidate.pdf_url:
        parts.append("PDF")
    return " • ".join(parts)


def merge_candidates(primary: Candidate, other: Candidate) -> Candidate:
    """Объединить сведения об одном издании, найденном в разных источниках."""
    for name in ("venue", "publisher", "abstract", "body", "doi", "edn", "udc", "bbk", "isbn", "volume", "issue",
                 "pages", "venue_type", "field_of_study", "grif", "label", "ebs", "catalog_key", "language"):
        if not getattr(primary, name) and getattr(other, name):
            setattr(primary, name, getattr(other, name))
    primary.year = primary.year or other.year
    primary.authors = primary.authors or other.authors
    primary.keywords = primary.keywords or other.keywords
    primary.citations = max(primary.citations or 0, other.citations or 0) or primary.citations or other.citations
    primary.affiliations = _unique([*primary.affiliations, *other.affiliations])[:8]
    primary.labels |= other.labels
    primary.peer_reviewed = primary.peer_reviewed or other.peer_reviewed
    primary.pdf_candidates = _unique([*primary.pdf_candidates, *other.pdf_candidates])
    primary.enrich |= other.enrich
    if other.origin not in primary.origin:
        primary.origin = f"{primary.origin} + {other.origin}"
    if "cyberleninka" in other.enrich and "cyberleninka.ru" not in primary.source_url and not primary.doi:
        primary.source_url = other.source_url
    return primary


def same_work(first: Candidate, second: Candidate) -> bool:
    if first.doi and first.doi == second.doi:
        return True
    shared_pages = {"", url_key(MCCME_FREE_BOOKS)}
    keys = {url_key(first.source_url), url_key(first.pdf_url)} - shared_pages
    if keys & ({url_key(second.source_url), url_key(second.pdf_url)} - shared_pages):
        return True
    if first.pdf_fingerprint and first.pdf_fingerprint == second.pdf_fingerprint:
        return True
    if similar_titles(first.title, second.title):
        return True
    left, right = title_key(first.title), title_key(second.title)
    shorter, longer = sorted((left, right), key=len)
    authors = {surname(a) for a in first.authors} & {surname(a) for a in second.authors}
    return len(shorter) >= 12 and longer.startswith(shorter) and bool(authors - {""})


def deduplicate(candidates: Iterable[Candidate], *, merge: bool = False) -> list[Candidate]:
    kept: list[Candidate] = []
    for candidate in candidates:
        twin = next((item for item in kept if same_work(item, candidate)), None)
        if twin is None:
            kept.append(candidate)
        elif merge and twin.kind == candidate.kind:
            merge_candidates(twin, candidate)
    return kept


def to_source(candidate: Candidate) -> LiteratureSource:
    venue = candidate.venue if candidate.kind == "article" else (candidate.publisher or candidate.venue)
    return LiteratureSource(
        title=candidate.title, authors=list(candidate.authors), type=candidate.kind, year=candidate.year,
        publisher_or_journal=venue, verification_badge=badge(candidate), pdf_url=candidate.pdf_url or None,
        source_url=candidate.source_url, relevance_score=candidate.score, doi=candidate.doi,
        isbn=format_isbn(candidate.isbn) if candidate.isbn else "", udc=candidate.udc,
        language=candidate.language or ("ru" if has_cyrillic(candidate.title) else "en"), origin=candidate.origin,
        evidence=tuple(article_evidence(candidate) if candidate.kind == "article" else textbook_evidence(candidate)),
        score_breakdown=dict(candidate.breakdown),
    )

# ---------------------------------------------------------------------------
# Оркестратор
# ---------------------------------------------------------------------------

LlmExpander = Callable[[str, str], Awaitable[list[str]]]


class LiteratureParser:
    """Подбор литературы. Использование: `async with LiteratureParser() as parser: await parser.find(...)`."""

    def __init__(self, config: ParserConfig | None = None, *, transport: httpx.AsyncBaseTransport | None = None,
                 llm_expander: LlmExpander | None = None) -> None:
        self.config = config or ParserConfig.from_env()
        self.http = Http(self.config, transport)
        self.validator = PdfValidator(self.http, self.config)
        self.llm_expander = llm_expander if self.config.use_llm else None

    async def __aenter__(self) -> "LiteratureParser":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self.http.aclose()

    async def find_many(self, items: Sequence[tuple[str, str]]) -> list[TopicLiterature]:
        """items — пары (тема, дисциплина)."""
        limit = asyncio.Semaphore(self.config.topic_concurrency)

        async def run(topic: str, discipline: str) -> TopicLiterature:
            async with limit:
                try:
                    return await asyncio.wait_for(self.find(topic, discipline), self.config.source_timeout * 2)
                except (TimeoutError, httpx.HTTPError, SourceUnavailable, ValueError):
                    return TopicLiterature(discipline, topic, [], [], unavailable=["Поиск литературы"])

        return list(await asyncio.gather(*(run(topic, discipline) for topic, discipline in items)))

    async def find(self, topic: str, discipline: str) -> TopicLiterature:
        unavailable: list[str] = []
        extra: list[str] = []
        if self.llm_expander is not None:
            try:
                extra = list(await asyncio.wait_for(self.llm_expander(topic, discipline), 30))
            except Exception as error:  # noqa: BLE001 - расширение запросов необязательно
                logger.warning("LLM-расширение запросов недоступно: %s", error)
                unavailable.append("LLM: расширение запросов")
        profile = build_profile(topic, discipline, extra)
        ctx = SearchContext(self.http, self.config, profile, self.validator, [self.config.yandex_max_requests])
        rejected: list[dict[str, str]] = []
        found = await self._gather(self._russian_tasks(ctx), unavailable)
        accepted = await self._evaluate(ctx, found, rejected)
        selected = self._select(accepted)
        if self.config.english_fallback and sum(c.kind == "article" for c in selected) < 2:
            more = await self._gather(self._english_tasks(ctx), unavailable)
            accepted = deduplicate(sorted([*accepted, *await self._evaluate(ctx, more, rejected)],
                                          key=self._rank, reverse=True))
            selected = self._select(accepted)
        queries = _unique([*profile.queries, *profile.article_queries_ru, *profile.article_queries_en,
                           *profile.textbook_queries])
        return TopicLiterature(discipline, topic, queries, [to_source(c) for c in selected],
                               _unique(unavailable), rejected)

    def _russian_tasks(self, ctx: SearchContext) -> list[tuple[str, Awaitable[list[Candidate]]]]:
        profile = ctx.profile
        tasks: list[tuple[str, Awaitable[list[Candidate]]]] = []
        for query in profile.article_queries_ru[:3]:
            tasks.append(("OpenAlex", search_openalex(ctx, query, language="ru")))
            tasks.append(("Crossref", search_crossref(ctx, query)))
            tasks.append(("КиберЛенинка", search_cyberleninka(ctx, query)))
        prefixes = (*MATH_PREFIXES, *IT_PREFIXES) if profile.area == "math" else (*IT_PREFIXES, *MATH_PREFIXES)
        for prefix in prefixes:
            for query in profile.article_queries_ru[:2]:
                tasks.append((TRUSTED_DOI_PREFIXES[prefix], search_crossref(ctx, query, prefix=prefix)))
        for query in profile.textbook_queries[:2]:
            tasks.append(("OpenAlex: книги", search_openalex(ctx, query, language="ru", kind="textbook")))
        tasks.append(("МЦНМО", search_mccme(ctx)))
        if self.config.yandex_search_enabled:
            tasks.append(("Yandex Search: вузы и ЭБС", search_yandex_textbooks(ctx)))
        if self.config.semantic_scholar_enabled:
            for query in profile.article_queries_ru[:1]:
                tasks.append(("Semantic Scholar", search_semantic_scholar(ctx, query)))
        return tasks

    def _english_tasks(self, ctx: SearchContext) -> list[tuple[str, Awaitable[list[Candidate]]]]:
        tasks: list[tuple[str, Awaitable[list[Candidate]]]] = []
        for query in ctx.profile.article_queries_en[:2]:
            tasks.append(("OpenAlex (en)", search_openalex(ctx, query, language="en")))
            tasks.append(("Crossref (en)", search_crossref(ctx, query, russian_only=False)))
            if self.config.semantic_scholar_enabled:
                tasks.append(("Semantic Scholar", search_semantic_scholar(ctx, query)))
        return tasks

    async def _gather(self, tasks: list[tuple[str, Awaitable[list[Candidate]]]],
                      unavailable: list[str]) -> list[Candidate]:
        async def guarded(label: str, task: Awaitable[list[Candidate]]) -> list[Candidate]:
            try:
                return await asyncio.wait_for(task, self.config.source_timeout)
            except (httpx.HTTPError, asyncio.TimeoutError, SourceUnavailable, ValueError, KeyError, TypeError,
                    ElementTree.ParseError) as error:
                logger.info("%s недоступен: %s", label, error)
                unavailable.append(label)
                return []

        results = await asyncio.gather(*(guarded(label, task) for label, task in tasks))
        return [candidate for batch in results for candidate in batch]

    @staticmethod
    def _rank(candidate: Candidate) -> tuple:
        return (candidate.score, candidate.language == "ru" or has_cyrillic(candidate.title),
                bool(candidate.pdf_url), candidate.year or 0)

    async def _evaluate(self, ctx: SearchContext, candidates: list[Candidate],
                        rejected: list[dict[str, str]]) -> list[Candidate]:
        profile = ctx.profile

        def reject(candidate: Candidate, reason: str) -> None:
            rejected.append({"title": candidate.title[:200], "url": candidate.source_url,
                             "origin": candidate.origin, "reason": reason})

        screened = []
        for candidate in candidates:
            reason = blocked_reason(candidate.source_url) or (
                "внутренний файл без библиографического описания"
                if looks_like_filename(candidate.title) and is_internal_file(candidate.title) else "")
            candidate.pdf_candidates = [url for url in candidate.pdf_candidates if not blocked_reason(url)]
            if not reason:
                reason = domain_rejection(candidate)
            if not reason:
                candidate.relevance = relevance(profile, candidate)
                if candidate.relevance < 0.5 and not (candidate.enrich & {"pdf_front", "ebs"} and candidate.relevance >= 0.3):
                    reason = f"низкая релевантность теме ({candidate.relevance:.2f})"
            if reason:
                reject(candidate, reason)
            else:
                screened.append(candidate)
        screened.sort(key=lambda c: (c.relevance, c.language == "ru", bool(c.doi), c.citations or 0), reverse=True)
        unique = deduplicate(screened, merge=True)
        articles = [c for c in unique if c.kind == "article"][:self.config.articles_to_verify]
        textbooks = [c for c in unique if c.kind == "textbook"][:self.config.textbooks_to_verify]
        await asyncio.gather(*(self._enrich(ctx, c) for c in [*articles, *textbooks]))
        accepted = []
        for candidate in [*articles, *textbooks]:
            book = catalog_match(candidate.title, candidate.authors)
            if book and candidate.kind == "textbook":
                candidate.catalog_key = book.key
                candidate.publisher = candidate.publisher or book.publisher
                candidate.authors = candidate.authors or list(book.authors)
            reason = domain_rejection(candidate)
            if not reason:
                candidate.relevance = relevance(profile, candidate)
                if candidate.relevance < 0.5:
                    reason = f"низкая релевантность теме ({candidate.relevance:.2f})"
            reason = reason or verification_failure(candidate)
            if not reason:
                candidate.breakdown = score(candidate, profile)
                candidate.score = candidate.breakdown["total"]
                if candidate.score < self.config.min_score:
                    reason = f"рейтинг {candidate.score:.2f} < {self.config.min_score}"
            if reason:
                reject(candidate, reason)
            else:
                accepted.append(candidate)
        accepted.sort(key=self._rank, reverse=True)
        return deduplicate(accepted)

    async def _enrich(self, ctx: SearchContext, candidate: Candidate) -> None:
        try:
            if "cyberleninka" in candidate.enrich:
                await enrich_cyberleninka(ctx, candidate)
            if "ebs" in candidate.enrich:
                await enrich_ebs(ctx, candidate)
            if "pdf_front" in candidate.enrich:
                await enrich_pdf_front(ctx, candidate)
        except (httpx.HTTPError, SourceUnavailable) as error:
            logger.info("Страница %s не обогащена: %s", candidate.source_url, error)
        if not candidate.pdf_url:
            for url in candidate.pdf_candidates[:3]:
                check = await ctx.validator.check(url)
                if check.ok:
                    candidate.pdf_url, candidate.pdf_size, candidate.pdf_fingerprint = url, check.size, check.fingerprint
                    break
        if (self.config.deep_pdf_check and candidate.kind == "article" and candidate.pdf_url
                and len(article_evidence(candidate)) < 2 and (candidate.pdf_size or 0) <= 15 * 1024 * 1024):
            check = await ctx.validator.check(candidate.pdf_url, full=True)
            if check.ok and check.content:
                text, _ = pdf_front_text(check.content, pages=2)
                front = parse_front_matter(text)
                candidate.udc = candidate.udc or front.udc
                candidate.edn = candidate.edn or front.edn
                candidate.doi = candidate.doi or front.doi

    def _select(self, accepted: list[Candidate]) -> list[Candidate]:
        textbooks = [c for c in accepted if c.kind == "textbook"][:self.config.max_textbooks]
        articles = [c for c in accepted if c.kind == "article"][:self.config.max_articles]
        return [*textbooks, *articles]


async def find_literature(topic: str, discipline: str, *, config: ParserConfig | None = None,
                          transport: httpx.AsyncBaseTransport | None = None,
                          llm_expander: LlmExpander | None = None) -> TopicLiterature:
    async with LiteratureParser(config, transport=transport, llm_expander=llm_expander) as parser:
        return await parser.find(topic, discipline)


def yandex_llm_expander(settings: dict[str, str] | None = None) -> LlmExpander:
    """Расширение запросов академической моделью University в Yandex AI Studio."""
    schema = {"type": "object", "properties": {"terms": {"type": "array", "items": {"type": "string"}}},
              "required": ["terms"], "additionalProperties": False}
    instructions = (
        "Ты библиограф технического вуза. Для темы рабочей программы и дисциплины верни до восьми "
        "академических поисковых терминов: русские синонимы, англоязычные термины и ключевые методы темы. "
        "Без названий вузов, авторов и общих слов вроде «учебник» или «курс»."
    )

    async def expand(topic: str, discipline: str) -> list[str]:
        def call() -> str:
            from run_rpd import open_yandex, require_response_text, yandex_settings
            current = settings or yandex_settings()
            model = current.get("YANDEX_MODEL_URI_UNIVERSITY") or f"gpt://{current['YANDEX_FOLDER_ID']}/aliceai-llm-flash"
            with open_yandex(current, timeout=30) as client:
                response = client.responses.create(
                    model=model, instructions=instructions,
                    input=json.dumps({"topic": topic, "discipline": discipline}, ensure_ascii=False),
                    text={"format": {"type": "json_schema", "name": "literature_terms", "schema": schema, "strict": True}},
                    max_output_tokens=300, store=False,
                )
                return require_response_text(response)

        data = json.loads(await asyncio.to_thread(call))
        return [clean(term) for term in data.get("terms", []) if isinstance(term, str) and 2 <= len(clean(term)) <= 80][:8]

    return expand

# ---------------------------------------------------------------------------
# Интеграция с программой курса (domain.schemas)
# ---------------------------------------------------------------------------

LEARNING_TYPES = {"Books", "Crossref", "arXiv", "documentation", "Literature"}


def bibliographic_description(source: LiteratureSource) -> str:
    authors = ", ".join(source.authors[:4]) + (" и др." if len(source.authors) > 4 else "")
    head = f"{authors}. {source.title}" if authors else source.title
    if source.type == "article":
        text = f"{head} // {source.publisher_or_journal}" if source.publisher_or_journal else head
        text += f". — {source.year}" if source.year else ""
        text += f". — DOI: {source.doi}" if source.doi else ""
    else:
        text = head + (f". — {source.publisher_or_journal}" if source.publisher_or_journal else "")
        text += f", {source.year}" if source.year else ""
        text += f". — ISBN {source.isbn}" if source.isbn else ""
    return clean(text) + "."


def literature_record(source: LiteratureSource):
    """SourceRecord типа Literature для реестра источников курса."""
    from domain.schemas import SourceRecord

    lines = [f"Название: {source.title}", f"Библиографическое описание: {bibliographic_description(source)}",
             f"Тип: {'учебник' if source.type == 'textbook' else 'научная статья'}",
             f"Подтверждение: {source.verification_badge}", f"Страница: {source.source_url}"]
    if source.doi:
        lines.append(f"DOI: {source.doi}")
    if source.isbn:
        lines.append(f"ISBN: {source.isbn}")
    if source.pdf_url:
        lines.append(f"PDF: {source.pdf_url}")
    lines.append(f"Оценка релевантности: {source.relevance_score:.2f}")
    content = "\n".join(lines)
    digest = hashlib.sha256(content.encode()).hexdigest()
    key = "Literature-" + hashlib.sha256(f"Literature\n{source.source_url}\n{digest}".encode()).hexdigest()[:20]
    return SourceRecord(source_id=key, source_type="Literature", url=source.source_url, title=source.title[:500],
                        retrieved_at=datetime.now(timezone.utc), raw_hash=digest, content=content)


def _record_field(record: Any, name: str) -> str:
    return str(record.get(name, "") if isinstance(record, dict) else getattr(record, name, "") or "")


def is_junk_reference(reference: dict, sources: dict[str, Any]) -> bool:
    record = sources.get(reference.get("source_id"))
    if record is None:
        return False
    if _record_field(record, "source_type") not in LEARNING_TYPES:
        return True
    return is_internal_file(_record_field(record, "title")) or bool(blocked_reason(_record_field(record, "url")))


def clean_references(references: list[dict], sources: dict[str, Any]) -> list[dict]:
    result: list[dict] = []
    titles: list[str] = []
    urls: set[str] = set()
    for reference in references:
        if is_junk_reference(reference, sources) or any(r["source_id"] == reference["source_id"] for r in result):
            continue
        record = sources.get(reference["source_id"])
        title = _record_field(record, "title") if record is not None else reference.get("locator", "")
        url = url_key(_record_field(record, "url")) if record else ""
        if (url and url in urls) or any(similar_titles(title, seen) for seen in titles):
            continue
        if url:
            urls.add(url)
        titles.append(title)
        result.append(reference)
    return result


def literature_reference(source: LiteratureSource, record: Any) -> dict:
    locator = source.title if len(source.title) >= 5 else bibliographic_description(source)
    return {"source_ids": [record.source_id], "source_id": record.source_id, "locator": locator,
            "bibliographic_excerpt": bibliographic_description(source)}


async def enrich_course_program(program: dict, sources: dict[str, Any], *, parser: LiteratureParser | None = None,
                                llm_expander: LlmExpander | None = None) -> tuple[dict, list[Any], list[TopicLiterature]]:
    """Прикрепить к темам курса проверенную литературу и убрать мусорные/дублирующиеся ссылки.

    Возвращает обновлённую программу, новые SourceRecord и результаты поиска по темам.
    """
    updated = copy.deepcopy(program)
    subjects = updated.get("subjects", [])
    reinforcement_learning = any("подкреплением" in subject["title"].casefold() for subject in subjects)
    shared_topics = Counter(clean(topic["title"]).casefold() for subject in subjects for topic in subject.get("topics", []))
    curated = {}
    for subject in subjects:
        title = clean(subject["title"])
        references = references_for_discipline(title, reinforcement_learning=reinforcement_learning)
        if references:
            curated[title] = [LiteratureSource(ref.title, list(ref.authors), ref.kind, ref.year, ref.publisher,
                                               "Библиографическая запись автора или издателя", ref.pdf, ref.url, 1.0,
                                               doi=ref.doi) for ref in references]

    def lookup_title(topic: str, discipline: str) -> str:
        broad = topic == f"Основные понятия дисциплины «{discipline}»" or shared_topics[clean(topic).casefold()] > 1
        return clean(discipline if broad else topic)

    pairs = list(dict.fromkeys((lookup_title(topic["title"], subject["title"]), clean(subject["title"]))
                               for subject in subjects for topic in subject.get("topics", [])))
    # Книжный каталог закрывает общие и повторяющиеся темы. По конкретным новым темам работает поиск.
    multi_topic = {clean(subject["title"]) for subject in subjects if len(subject.get("topics", [])) > 1}
    search_pairs = [(topic, discipline) for topic, discipline in pairs
                    if discipline not in curated or (topic != discipline and discipline in multi_topic)]
    own = parser is None
    results = []
    if search_pairs:
        active = parser or LiteratureParser(llm_expander=llm_expander)
        try:
            results = await active.find_many(search_pairs)
        finally:
            if own:
                await active.aclose()
    by_pair = {(result.topic, result.discipline): result for result in results}
    records: dict[str, Any] = {}
    registry = dict(sources)

    def attach(found: Sequence[LiteratureSource]) -> list[dict]:
        refs = []
        for item in found:
            record = literature_record(item)
            records[record.source_id] = record
            registry[record.source_id] = record
            refs.append(literature_reference(item, record))
        return refs

    for subject in subjects:
        subject_sources: list[LiteratureSource] = []
        discipline = clean(subject["title"])
        for topic in subject.get("topics", []):
            target = lookup_title(topic["title"], subject["title"])
            result = by_pair.get((target, discipline))
            found = result.sources if result else curated.get(discipline, []) if target == discipline else []
            subject_sources.extend(found)
            merged = clean_references([*attach(found), *topic.get("learning_references", [])], registry)
            topic["learning_references"] = merged or topic.get("learning_references", [])
            for assignment in topic.get("assignments", []):
                kept = clean_references(assignment.get("learning_references", []), registry)
                assignment["learning_references"] = kept or attach(found[:1]) or assignment.get("learning_references", [])
        kept = clean_references([*attach(curated.get(discipline, [])), *subject.get("learning_references", [])], registry)
        if not kept:
            best = max(subject_sources, key=lambda item: (item.type == "textbook", item.relevance_score),
                       default=None)
            kept = attach([best]) if best is not None else []
        subject["learning_references"] = kept or subject.get("learning_references", [])
    return updated, list(records.values()), results

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEMO_TOPICS: tuple[tuple[str, str], ...] = (
    ("Обучение на основе временных различий", "Машинное обучение"),
    ("Кривые второго порядка", "Аналитическая геометрия"),
    ("Собственные значения и собственные векторы", "Линейная алгебра"),
    ("Сверточные нейронные сети", "Глубокое обучение"),
    ("Проверка статистических гипотез", "Теория вероятностей и математическая статистика"),
)


def topics_from_json(path: Path) -> list[tuple[str, str]]:
    """Темы из course.json (экспорт кабинета) или JSON РПД (final_schema.json)."""
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    pairs: list[tuple[str, str]] = []
    program = data.get("program", data) if isinstance(data, dict) else {}
    for subject in program.get("subjects", []) if isinstance(program, dict) else []:
        for topic in subject.get("topics", []):
            pairs.append((topic["title"], subject["title"]))
    if not pairs and isinstance(data, dict):
        discipline = (data.get("discipline") or {}).get("name") or data.get("title") or ""
        for block in data.get("curriculum") or []:
            for topic in block.get("topics") or []:
                title = topic.get("title") if isinstance(topic, dict) else topic
                if title:
                    pairs.append((str(title), str(discipline)))
    return list(dict.fromkeys(pairs))


def _print_result(result: TopicLiterature) -> None:
    status = "OK" if result.meets_requirement else "НЕ ВЫПОЛНЕНО"
    print(f"\n=== {result.discipline} → {result.topic} [{status}] "
          f"учебников: {len(result.textbooks)}, статей: {len(result.articles)}")
    for source in result.sources:
        print(f"  [{source.type}] {source.relevance_score:.2f} {source.title} ({source.year or 'год не указан'})")
        print(f"      {source.verification_badge}")
        print(f"      {source.pdf_url or source.source_url}")
    if result.unavailable:
        print("  Недоступные источники: " + ", ".join(result.unavailable))


async def _run_cli(args: argparse.Namespace) -> int:
    if args.json_file:
        pairs = topics_from_json(args.json_file)
    elif args.topic:
        if not args.discipline:
            raise SystemExit("Для --topic укажите --discipline.")
        pairs = [(topic, args.discipline) for topic in args.topic]
    else:
        pairs = list(DEMO_TOPICS)
    if args.limit:
        pairs = pairs[: args.limit]
    config = ParserConfig.from_env()
    expander = yandex_llm_expander() if args.llm else None
    async with LiteratureParser(config, llm_expander=expander) as parser:
        results = await parser.find_many(pairs)
    for result in results:
        _print_result(result)
    if args.output:
        args.output.write_text(json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
        print(f"\nJSON сохранён: {args.output.resolve()}")
    failed = [r for r in results if not r.meets_requirement]
    print(f"\nИтого тем: {len(results)}, выполнено правило комплектации: {len(results) - len(failed)}")
    return 0 if not failed else 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Подобрать проверенную литературу к темам РПД.")
    parser.add_argument("--discipline", help="Дисциплина, например «Машинное обучение»")
    parser.add_argument("--topic", action="append", help="Тема (можно несколько раз)")
    parser.add_argument("--json-file", type=Path, help="course.json из ZIP-экспорта или JSON РПД")
    parser.add_argument("--limit", type=int, default=0, help="Ограничить число тем")
    parser.add_argument("--output", type=Path, help="Сохранить результат в JSON")
    parser.add_argument("--llm", action="store_true", help="Расширять запросы через Yandex AI Studio")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent / ".env", encoding="utf-8-sig", interpolate=False)
    except ImportError:
        pass
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    return asyncio.run(_run_cli(args))


if __name__ == "__main__":
    sys.exit(main())
