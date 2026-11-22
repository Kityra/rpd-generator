"""Чтение загружаемых источников: полный текст TXT/DOCX/PDF и выгрузки вакансий JSON/CSV."""

import csv
import io
import json
from dataclasses import dataclass
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

from pypdf import PdfReader
from pypdf.errors import PdfReadError


MAX_BYTES = 20 * 1024 * 1024
MAX_PAGES = 300
ATTACHMENT_MAX_MB = 10
ATTACHMENT_MAX_CHARS = 3000
WORD = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def read_document(name: str, data: bytes) -> str:
    """Полный текст документа. DOCX — по абзацам и строкам таблиц (ячейки через « | »)."""
    if not data or len(data) > MAX_BYTES:
        raise ValueError(f"{name}: нужен непустой файл до 20 МБ.")
    suffix = name.lower().rsplit(".", 1)[-1]
    try:
        if suffix in ("txt", "md"):
            return data.decode("utf-8-sig")
        if suffix == "docx":
            with ZipFile(BytesIO(data)) as archive:
                root = ElementTree.fromstring(archive.read("word/document.xml"))
            lines = []
            body = root.find(f"{WORD}body")
            for node in body if body is not None else []:
                if node.tag == f"{WORD}p":
                    lines.append("".join(text.text or "" for text in node.iter(f"{WORD}t")))
                elif node.tag == f"{WORD}tbl":
                    for row in node.iter(f"{WORD}tr"):
                        for cell in row.iter(f"{WORD}tc"):
                            # Каждый абзац ячейки — отдельная строка: так разбор ПС видит пункты списков.
                            for paragraph in cell.iter(f"{WORD}p"):
                                lines.append("".join(text.text or "" for text in paragraph.iter(f"{WORD}t")))
            return "\n".join(lines)
        if suffix == "pdf":
            reader = PdfReader(BytesIO(data))
            if reader.is_encrypted:
                raise ValueError("PDF защищён паролем")
            if len(reader.pages) > MAX_PAGES:
                raise ValueError(f"в PDF больше {MAX_PAGES} страниц")
            return "\n".join(page.extract_text() or "" for page in reader.pages)
    except (BadZipFile, KeyError, ElementTree.ParseError, UnicodeDecodeError) as error:
        raise ValueError(f"{name}: не удалось прочитать файл ({type(error).__name__}).") from error
    raise ValueError(f"{name}: поддерживаются TXT, DOCX и PDF.")


def read_attachment(name: str, data: bytes) -> dict[str, str]:
    """Короткая выдержка из вложения к правке курса или ответу эксперта."""
    name = Path(name.replace("\\", "/")).name
    suffix = Path(name).suffix.lower()
    if not data or len(data) > ATTACHMENT_MAX_MB * 1024**2:
        raise ValueError(f"{name}: допустим непустой файл размером до {ATTACHMENT_MAX_MB} МБ.")
    try:
        if suffix == ".txt":
            text = data.decode("utf-8-sig")
        elif suffix == ".docx":
            with ZipFile(BytesIO(data)) as archive:
                info = archive.getinfo("word/document.xml")
                if info.file_size > 4 * 1024**2:
                    raise ValueError("слишком большой текст документа")
                root = ElementTree.fromstring(archive.read(info))
            text = "\n".join("".join(node.itertext()) for node in root.iter(f"{WORD}t"))
        elif suffix == ".pdf":
            reader = PdfReader(BytesIO(data))
            if reader.is_encrypted or len(reader.pages) > 50:
                raise ValueError("используйте PDF без пароля и размером до 50 страниц")
            chunks = []
            for page in reader.pages:
                chunks.append((page.extract_text() or "")[:ATTACHMENT_MAX_CHARS])
                if sum(map(len, chunks)) >= ATTACHMENT_MAX_CHARS:
                    break
            text = "\n".join(chunks)
        else:
            raise ValueError("доступные форматы — TXT, DOCX и PDF")
    except (BadZipFile, KeyError, ElementTree.ParseError, PdfReadError, ValueError, OSError) as error:
        raise ValueError(f"{name}: не удалось прочитать файл. {error}") from error
    return {"name": name[:180], "text": text.replace("\x00", "").strip()[:ATTACHMENT_MAX_CHARS]}


@dataclass(frozen=True)
class VacancyRow:
    employer: str
    title: str
    text: str
    published_at: date | None
    url: str = ""
    region: str = ""


_FIELDS = {"employer": ("employer", "company", "работодатель", "компания"),
           "title": ("title", "name", "вакансия", "название"),
           "text": ("text", "description", "описание", "текст"),
           "duties": ("duties", "responsibilities", "обязанности"),
           "requirements": ("requirements", "требования"),
           "published_at": ("published_at", "date", "published", "дата", "дата публикации"),
           "url": ("url", "link", "ссылка"), "region": ("region", "area", "регион", "город")}


def _date(value) -> date | None:
    if not value:
        return None
    text = str(value).strip()[:10]
    for pattern in ("%Y-%m-%d", "%d.%m.%Y"):
        try:
            return datetime.strptime(text, pattern).date()
        except ValueError:
            continue
    raise ValueError(f"Дата «{value}» не распознана: используйте ГГГГ-ММ-ДД или ДД.ММ.ГГГГ.")


def _pick(row: dict, field: str):
    lowered = {str(key).strip().casefold(): value for key, value in row.items()}
    return next((lowered[name] for name in _FIELDS[field] if lowered.get(name) not in (None, "")), None)


def _as_lines(value) -> str:
    """Список пунктов: JSON-массив или строка CSV с пунктами через «|»."""
    if isinstance(value, str) and "|" in value:
        value = [part.strip() for part in value.split("|") if part.strip()]
    if isinstance(value, list):
        return "\n".join(f"- {item}" for item in value)
    return str(value or "")


def parse_vacancies(name: str, data: bytes) -> list[VacancyRow]:
    """JSON (список или {"vacancies": [...]}) или CSV: employer, title, text | duties + requirements, published_at, url."""
    text = data.decode("utf-8-sig")
    if name.lower().endswith(".json"):
        payload = json.loads(text)
        rows = payload.get("vacancies", payload.get("items", [])) if isinstance(payload, dict) else payload
    else:
        rows = list(csv.DictReader(io.StringIO(text), delimiter=";" if text.count(";") > text.count(",") else ","))
    result = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"Строка {index}: ожидается объект с полями вакансии.")
        employer, title = _pick(row, "employer"), _pick(row, "title")
        if not employer or not title:
            raise ValueError(f"Строка {index}: нужны работодатель (employer) и название (title).")
        if isinstance(employer, dict):
            employer = employer.get("name") or ""
        body = _pick(row, "text") or ""
        duties, requirements = _pick(row, "duties"), _pick(row, "requirements")
        if duties:
            body += "\nОбязанности:\n" + _as_lines(duties)
        if requirements:
            body += "\nТребования:\n" + _as_lines(requirements)
        if not str(body).strip():
            raise ValueError(f"Строка {index}: нет текста вакансии (text или duties/requirements).")
        result.append(VacancyRow(employer=" ".join(str(employer).split()), title=" ".join(str(title).split()),
                                 text=f"{title}\n{str(body).strip()}", published_at=_date(_pick(row, "published_at")),
                                 url=str(_pick(row, "url") or ""), region=str(_pick(row, "region") or "")))
    if not result:
        raise ValueError("В файле нет вакансий.")
    return result
