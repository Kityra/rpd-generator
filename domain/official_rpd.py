"""Conservative scope check for official materials and the final export gate."""

import re
import unicodedata
from typing import Literal, TypedDict


class RpdLookup(TypedDict):
    status: Literal["live", "snapshot", "empty", "unavailable", "unconfigured"]
    reason: str


def _normalized(text: str) -> str:
    return " " + " ".join(re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold().replace("ё", "е"))) + " "


def matches_scope(title: str, text: str, *, institution: str, year: str, program: str, discipline: str) -> bool:
    parts = (_normalized(title), _normalized(text))
    mandatory = (institution, year, discipline)
    if not all(value.strip() for value in mandatory):
        return False
    return all(any(query in part for part in parts)
               for query in (_normalized(value) for value in (*mandatory, program) if value.strip()))


def course_subject_matches(meta: dict, subject: dict, title: str, text: str) -> bool:
    return matches_scope(title, text, institution=meta.get("institution", ""), year=str(meta.get("year", "")),
                         program=meta.get("program") or meta.get("title", ""), discipline=subject["title"])


def audit_block_reason(audit: dict | None) -> str:
    if not audit:
        return "Официальные материалы ещё не проверены."
    missing = audit.get("missing", [])
    if missing:
        return "Нет подтверждённых материалов РПД для дисциплин: " + ", ".join(missing) + "."
    return ""


def course_audit(meta: dict, subjects: list[dict], sources, lookup: RpdLookup | None) -> dict:
    official = [source for source in sources if source.source_type == "RPD"]
    confirmed, missing = {}, []
    for subject in subjects:
        ids = [source.source_id for source in official
               if course_subject_matches(meta, subject, source.title, source.content)]
        if ids:
            confirmed[subject["id"]] = ids
        else:
            missing.append(subject["title"])
    return {"lookup": lookup, "subject_sources": confirmed, "missing": missing}
