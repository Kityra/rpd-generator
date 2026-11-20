"""Преобразование разрешённой выгрузки в карточки вакансий."""

import hashlib
from datetime import datetime, timezone
from html.parser import HTMLParser
import re
from urllib.parse import urljoin, urlsplit, urlunsplit


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = None

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"} and not self.hidden:
            self.hidden = tag
        if tag in {"p", "br", "li", "div"}:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if self.hidden == tag:
            self.hidden = None
        if tag in {"p", "li", "div"}:
            self.parts.append(" ")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def text(value, limit=8000):
    if isinstance(value, dict):
        value = value.get("name") or value.get("title")
    if isinstance(value, list):
        return "; ".join(filter(None, (text(item) for item in value)))[:limit] or None
    if not isinstance(value, str):
        return None
    parser = PlainText()
    parser.feed(value)
    parser.close()
    return " ".join("".join(parser.parts).split())[:limit] or None


def tags(value):
    if isinstance(value, str):
        value = re.split(r"[,;\n]", value)
    by_key = {}
    for item in value if isinstance(value, list) else []:
        if label := text(item, 120):
            by_key.setdefault(label.casefold(), label)
    return list(by_key.values())[:60]


def timestamp(value):
    if not isinstance(value, str):
        return None
    try:
        date = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return date.replace(tzinfo=date.tzinfo or timezone.utc).astimezone(timezone.utc).isoformat()
    except (ValueError, OverflowError):
        return None


STACK = ("Python", "JavaScript", "TypeScript", "Java", "Kotlin", "Swift", "C#", "C++", "Golang",
         "SQL", "PostgreSQL", "MySQL", "Redis", "Docker", "Kubernetes", "Git", "Linux", "React",
         "Vue", "Django", "FastAPI", "Terraform", "Jenkins", "1С", "Kafka", "Spark")
ROLE = re.compile(
    r"программист|разработчик|тестировщик|системн\w+ (?:аналитик|администратор)|"
    r"информационн\w+ безопасност|аналитик данных|техническ\w+ поддержк|"
    r"\b(?:devops|sre|frontend|front-end|backend|back-end|fullstack|developer|"
    r"qa|data scientist|data engineer|software engineer|it support)\b", re.I
)


def vacancy_url(value):
    if not isinstance(value, str) or not value.strip():
        return None
    parsed = urlsplit(urljoin("https://www.rabota.ru/", value.strip()))
    host = parsed.hostname or ""
    if (parsed.scheme != "https" or parsed.username or parsed.password or parsed.port not in (None, 443)
            or not (host == "rabota.ru" or host.endswith(".rabota.ru"))
            or not parsed.path.startswith("/vacancy/") or not parsed.path[len("/vacancy/"):].strip("/")):
        return None
    return urlunsplit(("https", host, parsed.path.rstrip("/"), "", ""))


def amount(value):
    if type(value) is int:
        return value if 0 <= value < 10**12 else None
    if isinstance(value, str):
        digits = "".join(value.split())
        if re.fullmatch(r"[0-9]{1,12}", digits):
            return int(digits)
    return None


def salary(value):
    if isinstance(value, str):
        note = text(value, 500)
        return {"text": note} if note else None
    if not isinstance(value, dict):
        return None
    low = amount(value.get("from", value.get("minimum")))
    high = amount(value.get("to", value.get("maximum")))
    note = text(value.get("text"), 500)
    currency = text(value.get("currency"), 8)
    currency = currency.upper() if currency else None
    if currency in {"РУБ", "RUR", "₽"}:
        currency = "RUB"
    if low is not None and high is not None and low > high:
        low = high = None
    if low is None and high is None and note is None:
        return None
    return {"minimum": low, "maximum": high, "currency": currency, "text": note,
            "period": text(value.get("period"), 30),
            "gross": value.get("gross") if type(value.get("gross")) is bool else None}


def normalise(raw):
    if not isinstance(raw, dict):
        raise ValueError("Карточка должна быть объектом")
    title = text(raw.get("title") or raw.get("name"), 500)
    if not title:
        raise ValueError("Отсутствует название")
    url = vacancy_url(raw.get("url") or raw.get("vacancy_url"))
    source_id = raw.get("id", raw.get("source_id", raw.get("vacancy_id")))
    source_id = str(source_id).strip() if type(source_id) in (str, int) else ""
    if not source_id and not url:
        raise ValueError("Требуется ID или ссылка")
    fields = {name: text(raw.get(name)) for name in
              ("description", "requirements", "responsibilities", "conditions", "experience",
               "employment_type", "schedule")}
    skills = tags(raw.get("skills") or raw.get("key_skills"))
    technology_list = tags(raw.get("technologies") or raw.get("stack") or raw.get("tools"))
    searchable = " ".join(filter(None, [title, fields["description"], fields["requirements"],
                                         fields["responsibilities"], *skills, *technology_list]))
    inferred = [name for name in STACK if re.search(r"(?<![\w+#])" + re.escape(name) + r"(?![\w+#])",
                                                    searchable, re.I)]
    category = text(raw.get("category")) or ""
    if not (ROLE.search(title) or inferred or category.casefold() in {"it", "ит", "it / интернет / телеком"}):
        return None
    return {
        "source": "rabota.ru", "source_id": source_id or "url:" + hashlib.sha256(url.encode()).hexdigest(),
        "url": url, "title": title, "employer": text(raw.get("employer") or raw.get("company"), 500),
        "region": text(raw.get("region") or raw.get("city"), 500), **fields,
        "skills": skills, "technologies": tags([*inferred, *technology_list]),
        "salary": salary(raw.get("salary") or raw.get("compensation")),
        "remote": raw.get("remote") if type(raw.get("remote")) is bool else None,
        "published_at": timestamp(raw.get("published_at")), "updated_at": timestamp(raw.get("updated_at")),
    }
