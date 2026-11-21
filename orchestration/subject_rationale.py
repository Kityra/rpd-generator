"""Explain inclusion of each discipline using only curated database material.

Обоснование проверяется дважды: кодом (цитата не служебная, содержит понятия названия дисциплины,
объяснение опирается на цитату) и вызовом-судьёй на быстрой модели (supports / weak / unrelated).
Дисциплина без подтверждённого обоснования получает rationale_check.status ≠ supports — «требует проверки».
"""

import asyncio
import logging
from textwrap import wrap
from typing import Literal

from openai import APIResponseValidationError, APIStatusError
from pydantic import Field, ValidationError, ValidationInfo, model_validator

from domain.schemas import Model, SUBJECT_RATIONALE_SOURCE_TYPES, SourceRecord, SubjectRationale
from orchestration import prompts
from orchestration.clients import EvidenceClient, ModelClient
from orchestration.semantic_check import concepts, excerpt_supports_title, rationale_problem, title_coverage


LOG = logging.getLogger(__name__)
UNAVAILABLE = (APIResponseValidationError, APIStatusError, RuntimeError, ValidationError)
BATCH_SIZE = 4


class DisciplineRationale(Model):
    subject_id: str
    explanation: str | None
    excerpt_ids: list[str] = Field(max_length=3)


class SubjectRationaleBatch(Model):
    items: list[DisciplineRationale]

    @model_validator(mode="after")
    def check_subjects(self, info: ValidationInfo) -> "SubjectRationaleBatch":
        choices = (info.context or {})["subject_excerpts"]
        seen = set()
        for item in self.items:
            if item.subject_id not in choices or item.subject_id in seen:
                raise ValueError("Скопируйте один subject_id для каждой дисциплины из запроса.")
            if not set(item.excerpt_ids) <= choices[item.subject_id]:
                raise ValueError("Цитируйте только фрагменты, переданные для этой дисциплины.")
            if len(set(item.excerpt_ids)) != len(item.excerpt_ids):
                raise ValueError("Не повторяйте фрагменты в excerpt_ids.")
            if bool(item.explanation) != bool(item.excerpt_ids) or (
                item.explanation is not None and not 20 <= len(item.explanation.strip()) <= 900
            ):
                raise ValueError("Нужно объяснение длиной 20–900 символов с цитатой или пустой результат.")
            seen.add(item.subject_id)
        return self


class RationaleVerdict(Model):
    subject_id: str
    verdict: Literal["supports", "weak", "unrelated"]
    reason: str = Field(max_length=400)


class RationaleVerdicts(Model):
    items: list[RationaleVerdict]

    @model_validator(mode="after")
    def check_subjects(self, info: ValidationInfo) -> "RationaleVerdicts":
        expected = set((info.context or {}).get("judged_subjects", ()))
        if {item.subject_id for item in self.items} - expected:
            raise ValueError("Скопируйте subject_id из запроса.")
        return self


def _excerpts(record: SourceRecord, title: str, terms: set[str]) -> list[str]:
    """Фрагменты записи, относящиеся к дисциплине: содержат понятия названия, не служебные."""
    chunks = wrap(" ".join(record.content.split()), width=340, break_long_words=False, break_on_hyphens=False)
    useful = [chunk for chunk in chunks if 15 <= len(chunk) <= 400 and excerpt_supports_title(title, chunk)]
    return sorted(useful, key=lambda chunk: len(concepts(chunk) & terms), reverse=True)[:2]


def checked(status: str, reason: str, *, judge: bool) -> dict:
    return {"status": status, "reason": reason, "judge": judge}


async def explain_subjects(program: dict, registry: dict[str, SourceRecord], evidence: EvidenceClient,
                          llm: ModelClient, owner: str) -> tuple[dict, list[SourceRecord]]:
    source_limit = asyncio.Semaphore(4)
    model_limit = asyncio.Semaphore(2)
    database = {key: source for key, source in registry.items()
                if source.source_type in SUBJECT_RATIONALE_SOURCE_TYPES}
    subjects = program["subjects"]

    def existing(subject):
        """Сохранённое обоснование остаётся, если его цитаты по-прежнему проходят проверку кодом."""
        rationale = subject.get("inclusion_rationale")
        if rationale and rationale_problem(subject["title"], rationale["explanation"],
                                           [item["excerpt"] for item in rationale["evidence"]]):
            return None
        return rationale

    async def gather_sources(subject, *, retry):
        topics = ", ".join(topic["title"] for topic in subject["topics"])
        title = subject["title"]
        query = title if retry else f"Дисциплина «{title}». Темы: {topics}."[:900]
        role = "subject_rationale_extended" if retry else "subject_rationale"
        async with source_limit:
            try:
                found = await evidence.search(role, query, owner_scope=owner)
            except UNAVAILABLE as error:
                LOG.warning("Поиск материалов для обоснования дисциплины: %s", type(error).__name__)
                found = []
        # Сохраняем уже использованный материал курса вместе с результатом поиска по дисциплине.
        previous = (database[key] for key in [*subject["justification"]["source_ids"], *subject["source_ids"]]
                    if key in database)
        candidates = {record.source_id: record for record in [*found, *previous]
                      if record.source_type in SUBJECT_RATIONALE_SOURCE_TYPES}
        # Записи, где понятия дисциплины встречаются чаще, идут первыми.
        ordered = sorted(candidates.items(), key=lambda item: title_coverage(title, item[1].content), reverse=True)
        return dict(ordered[:7])

    async def judge(rows: list[dict]) -> dict[str, RationaleVerdict] | None:
        if not rows:
            return {}
        try:
            async with model_limit:
                verdicts = await llm.generate(RationaleVerdicts, prompts.RATIONALE_JUDGE,
                                              {"course": program["title"], "disciplines": rows},
                                              {"judged_subjects": [row["subject_id"] for row in rows]}, flash=True)
        except UNAVAILABLE as error:
            LOG.warning("Проверка обоснований судьёй недоступна: %s", type(error).__name__)
            return None
        return {item.subject_id: item for item in verdicts.items}

    async def explain_batch(batch, *, retry=False):
        found = await asyncio.gather(*(gather_sources(subject, retry=retry) for subject in batch))
        sources = {key: source for records in found for key, source in records.items()}
        disciplines, choices, excerpts, subject_ids = [], {}, {}, {}
        for number, (subject, records) in enumerate(zip(batch, found, strict=True), 1):
            key = f"D{number}"
            topics = [topic["title"] for topic in subject["topics"]]
            terms = concepts(" ".join([subject["title"], *topics]))
            fragments = []
            for record in records.values():
                for text in _excerpts(record, subject["title"], terms):
                    excerpt_key = f"E{len(excerpts) + 1}"
                    excerpts[excerpt_key] = {"source_id": record.source_id, "excerpt": text}
                    fragments.append({"excerpt_id": excerpt_key, "index": record.source_type,
                                      "title": record.title, "text": text})
            if fragments:
                disciplines.append({"subject_id": key, "title": subject["title"], "topics": topics, "fragments": fragments})
                choices[key] = {fragment["excerpt_id"] for fragment in fragments}
                subject_ids[key] = subject["id"]
        if not disciplines:
            return {}, {}, {}
        try:
            async with model_limit:
                report = await llm.generate(SubjectRationaleBatch, prompts.SUBJECT_RATIONALE,
                                            {"course": program["title"], "disciplines": disciplines},
                                            {"subject_excerpts": choices}, flash=not retry)
        except UNAVAILABLE as error:
            LOG.warning("Обоснование состава курса не сформировано для группы дисциплин: %s", type(error).__name__)
            return {}, {}, {}
        titles = {row["subject_id"]: row["title"] for row in disciplines}
        candidates, checks = {}, {}
        for item in report.items:
            if not item.excerpt_ids:
                continue
            evidence_items = [excerpts[key] for key in item.excerpt_ids]
            source_ids = list(dict.fromkeys(fragment["source_id"] for fragment in evidence_items))
            try:
                rationale = SubjectRationale.model_validate(
                    {"explanation": item.explanation, "source_ids": source_ids, "evidence": evidence_items},
                    context={"sources": sources, "rationale_subject": titles[item.subject_id]})
            except ValidationError as error:
                checks[subject_ids[item.subject_id]] = checked("unrelated", error.errors()[0]["msg"], judge=False)
                continue
            candidates[item.subject_id] = rationale.model_dump(mode="json")
        by_text = {(item["source_id"], item["excerpt"]): excerpt_id for excerpt_id, item in excerpts.items()}
        rows = [{"subject_id": key, "title": titles[key], "explanation": value["explanation"],
                 "fragments": [{"excerpt_id": by_text.get((e["source_id"], e["excerpt"])),
                                "title": sources[e["source_id"]].title, "text": e["excerpt"]} for e in value["evidence"]]}
                for key, value in candidates.items()]
        verdicts = await judge(rows)
        rationales = {}
        for key, rationale in candidates.items():
            subject_id = subject_ids[key]
            if verdicts is None:
                rationales[subject_id] = rationale
                checks[subject_id] = checked("supports", "Проверено кодом; судья недоступен.", judge=False)
                continue
            verdict = verdicts.get(key)
            if verdict is None:
                checks[subject_id] = checked("weak", "Судья не оценил обоснование.", judge=True)
            elif verdict.verdict == "supports":
                rationales[subject_id] = rationale
                checks[subject_id] = checked("supports", verdict.reason, judge=True)
            else:
                checks[subject_id] = checked(verdict.verdict, verdict.reason, judge=True)
        used = {key: sources[key] for rationale in rationales.values() for key in rationale["source_ids"]}
        return rationales, used, checks

    def requests(batch_subjects, *, retry=False):
        return asyncio.gather(*(explain_batch(batch_subjects[index:index + BATCH_SIZE], retry=retry)
                                for index in range(0, len(batch_subjects), BATCH_SIZE)))

    targets = [subject for subject in subjects if not existing(subject)]
    batches = await requests(targets)
    found = {key for items, _, _ in batches for key in items}
    missing = [subject for subject in targets if subject["id"] not in found]
    if missing:
        batches.extend(await requests(missing, retry=True))
    rationales = {key: value for items, _, _ in batches for key, value in items.items()}
    checks = {}
    for _, _, items in batches:
        for key, value in items.items():
            if key not in checks or value["status"] == "supports":
                checks[key] = value
    additions = {key: value for _, used, _ in batches for key, value in used.items() if key not in registry}

    def updated(subject):
        rationale = rationales.get(subject["id"], existing(subject))
        if rationale is not None and subject["id"] not in rationales:
            check = subject.get("rationale_check") or checked("supports", "Сохранённое обоснование.", judge=False)
        elif rationale is not None:
            check = checks[subject["id"]]
        else:
            check = checks.get(subject["id"]) or checked("missing", "В базе не найдено подтверждающих фрагментов.",
                                                         judge=False)
            if check["status"] == "supports":
                check = checked("missing", "Обоснование не подтверждено.", judge=False)
        return {**subject, "inclusion_rationale": rationale, "rationale_check": check}

    return {**program, "subjects": [updated(subject) for subject in subjects]}, list(additions.values())
