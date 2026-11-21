"""Работа с текстом без LLM: нормализация цитат, ключи формулировок, глагол в 3-м лице, словарь областей."""

import re

from orchestration.market_analysis import SkillDictionary
from orchestration.semantic_check import concepts, tokens

from activity.config import load_areas, load_config


_PUNCT = str.maketrans({"«": '"', "»": '"', "„": '"', "“": '"', "”": '"', "‘": "'", "’": "'", "`": "'",
                        "—": "-", "–": "-", "‑": "-", " ": " ", " ": " ", "ё": "е", "Ё": "Е"})


def normalize(text: str) -> str:
    """Регистр, пробелы, кавычки, тире и ё/е не влияют на сравнение цитаты с источником."""
    return " ".join((text or "").translate(_PUNCT).casefold().split())


def quote_problem(quote: str, content: str, config: dict | None = None) -> str | None:
    """Проверка цитаты кодом (раздел 7.2). None — цитата дословно есть в тексте источника."""
    limits = (config or load_config())["extraction"]
    value = normalize(quote).strip(" .,;:…\"'")
    if not value:
        return "нет цитаты"
    if len(value) < limits["min_quote_chars"]:
        return f"цитата короче {limits['min_quote_chars']} символов"
    if len(quote) > limits["max_quote_chars"] * 1.2:
        return f"цитата длиннее {limits['max_quote_chars']} символов"
    if value not in normalize(content):
        return "цитаты нет в тексте источника дословно"
    return None


def terms(text: str) -> set[str]:
    """Понятия текста плюс короткие латинские сокращения (CI, CD, Go, ML), которые concepts() пропускает."""
    short = {token for token in tokens(text) if len(token) == 2 and token.isascii() and token.isalpha()}
    return concepts(text) | short


def object_concepts(text: str) -> set[str]:
    """Понятия формулировки без первого слова (глагола или отглагольного существительного)."""
    words = (text or "").split(maxsplit=1)
    return terms(words[1]) if len(words) > 1 else set()


def canonical_key(text: str, level: str = "task", config: dict | None = None) -> str:
    """Ключ формулировки: основа глагола + понятия объекта. «Разрабатывает API» и «Разработка API» совпадают."""
    if level != "task":
        return "k:" + " ".join(sorted(concepts(text)))
    words = tokens(text)
    if not words:
        return "t:"
    size = (config or load_config())["canonical"]["verb_prefix_len"]
    return f"t:{words[0][:size]}:" + " ".join(sorted(object_concepts(text)))


def jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left and right else 0.0


# Неправильные глаголы, которые не сводятся к правилам ниже.
_IRREGULAR = {"писать": "пишет", "вести": "ведёт", "брать": "берёт", "искать": "ищет", "мочь": "может",
              "печь": "печёт", "нести": "несёт", "расти": "растёт", "давать": "даёт", "вставать": "встаёт",
              "создавать": "создаёт", "передавать": "передаёт", "задавать": "задаёт", "сдавать": "сдаёт",
              "узнавать": "узнаёт", "признавать": "признаёт", "доставать": "достаёт", "описать": "опишет",
              "написать": "напишет", "переписать": "перепишет", "спать": "спит", "гнать": "гонит",
              "жить": "живёт", "пить": "пьёт", "бить": "бьёт", "шить": "шьёт", "лить": "льёт", "быть": "есть",
              "есть": "ест", "хотеть": "хочет", "смотреть": "смотрит", "видеть": "видит", "зависеть": "зависит",
              "вертеть": "вертит", "терпеть": "терпит", "держать": "держит", "слышать": "слышит",
              "дышать": "дышит", "решать": "решает", "проводить": "проводит"}
_INFINITIVE = re.compile(r"^[а-яё-]+(?:ть|ти|чь)(?:ся)?$", re.I)


def is_infinitive(word: str) -> bool:
    return bool(_INFINITIVE.match(word.casefold()))


def third_person(word: str) -> str:
    """Инфинитив → 3-е лицо единственного числа: «разрабатывать» → «разрабатывает»."""
    lower = word.casefold()
    reflexive = lower.endswith("ся")
    base = lower[:-2] if reflexive else lower
    if base in _IRREGULAR:
        result = _IRREGULAR[base]
    elif base.endswith(("овать", "евать")) and not base.endswith(("ывать", "ивать")):
        result = base[:-5] + "ует"
    elif base.endswith(("ать", "ять", "еть")):
        result = base[:-2] + "ет"
    elif base.endswith("ить"):
        result = base[:-3] + "ит"
    else:
        return word
    if reflexive:
        result += "ся"
    return result[:1].upper() + result[1:] if word[:1].isupper() else result


def as_task(line: str) -> list[str]:
    """Строка обязанности из вакансии → одна или две задачи с глаголом в 3-м лице.

    «Разрабатывать и поддерживать микросервисы» → «Разрабатывает микросервисы», «Поддерживает микросервисы».
    Если строка не начинается с глагола, задача не выделяется.
    """
    text = " ".join(line.strip(" \t-•–—*·;.,").split())
    words = text.split()

    def verb(word: str) -> str | None:
        # Существительные на -ть («безопасность») правило не меняет — это не глагол.
        converted = third_person(word) if is_infinitive(word) else word
        return converted.capitalize() if converted.casefold() != word.casefold() else None

    first = verb(words[0]) if words else None
    if first is None:
        return []
    second = verb(words[2]) if len(words) > 3 and words[1] == "и" else None
    if second is not None:
        rest = " ".join(words[3:])
        return [f"{first} {rest}", f"{second} {rest}"]
    return [" ".join([first, *words[1:]])]


class AreaDictionary:
    """Межпрофессиональные области знаний: имя области ищется в тексте по синонимам (config/knowledge_areas.yaml)."""

    def __init__(self, areas: list[dict] | None = None):
        self.areas = {area["name"]: area for area in (areas or load_areas())}
        self.dictionary = SkillDictionary([[name, *area.get("synonyms", [])] for name, area in self.areas.items()])

    def detect(self, text: str) -> list[str]:
        return sorted(self.dictionary.detect(text))

    def kind(self, name: str) -> str:
        return self.areas[name]["kind"]
