"""Explain inclusion of each discipline using only curated database material."""

import asyncio
import logging
from textwrap import wrap

from openai import APIResponseValidationError, APIStatusError
from pydantic import Field, ValidationError, ValidationInfo, model_validator

from domain.schemas import Model, SUBJECT_RATIONALE_SOURCE_TYPES, SourceRecord, SubjectRationale
from orchestration import prompts
from orchestration.clients import EvidenceClient, ModelClient
from orchestration.draft import keywords


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


def _terms(text: str) -> set[str]:
    return {word[:5] for word in keywords(text)}


def _excerpts(record: SourceRecord, terms: set[str]) -> list[str]:
    chunks = wrap(" ".join(record.content.split()), width=340, break_long_words=False, break_on_hyphens=False)
    useful = [chunk for chunk in chunks if 15 <= len(chunk) <= 400]
    return sorted(useful, key=lambda chunk: len(_terms(chunk) & terms), reverse=True)[:2]


async def explain_subjects(program: dict, registry: dict[str, SourceRecord], evidence: EvidenceClient,
                          llm: ModelClient, owner: str) -> tuple[dict, list[SourceRecord]]:
    source_limit = asyncio.Semaphore(4)
    model_limit = asyncio.Semaphore(2)
    database = {key: source for key, source in registry.items()
                if source.source_type in SUBJECT_RATIONALE_SOURCE_TYPES}
    subjects = program["subjects"]

    async def gather_sources(subject):
        topics = ", ".join(topic["title"] for topic in subject["topics"])
        query = f"Дисциплина «{subject['title']}». Темы: {topics}."[:900]
        async with source_limit:
            try:
                found = await evidence.search("subject_rationale", query, owner_scope=owner)
            except UNAVAILABLE as error:
                LOG.warning("Поиск материалов для обоснования дисциплины: %s", type(error).__name__)
                found = []
        # Сохраняем уже использованный материал курса вместе с результатом поиска по дисциплине.
        previous = (database[key] for key in [*subject["justification"]["source_ids"], *subject["source_ids"]]
                    if key in database)
        candidates = {record.source_id: record for record in [*found, *previous]
                      if record.source_type in SUBJECT_RATIONALE_SOURCE_TYPES}
        return dict(list(candidates.items())[:7])

    async def explain_batch(batch):
        found = await asyncio.gather(*(gather_sources(subject) for subject in batch))
        sources = {key: source for records in found for key, source in records.items()}
        disciplines, choices, excerpts, subject_ids = [], {}, {}, {}
        for number, (subject, records) in enumerate(zip(batch, found, strict=True), 1):
            key = f"D{number}"
            topics = [topic["title"] for topic in subject["topics"]]
            terms = _terms(" ".join([subject["title"], *topics]))
            fragments = []
            for record in records.values():
                for text in _excerpts(record, terms):
                    excerpt_key = f"E{len(excerpts) + 1}"
                    excerpts[excerpt_key] = {"source_id": record.source_id, "excerpt": text}
                    fragments.append({"excerpt_id": excerpt_key, "index": record.source_type,
                                      "title": record.title, "text": text})
            if fragments:
                disciplines.append({"subject_id": key, "title": subject["title"], "topics": topics, "fragments": fragments})
                choices[key] = {fragment["excerpt_id"] for fragment in fragments}
                subject_ids[key] = subject["id"]
        if not disciplines:
            return {}, {}
        try:
            async with model_limit:
                report = await llm.generate(SubjectRationaleBatch, prompts.SUBJECT_RATIONALE,
                                            {"course": program["title"], "disciplines": disciplines},
                                            {"subject_excerpts": choices}, flash=True)
        except UNAVAILABLE as error:
            LOG.warning("Обоснование состава курса не сформировано для группы дисциплин: %s", type(error).__name__)
            return {}, {}
        rationales = {}
        for item in report.items:
            if not item.excerpt_ids:
                continue
            evidence = [excerpts[key] for key in item.excerpt_ids]
            source_ids = list(dict.fromkeys(fragment["source_id"] for fragment in evidence))
            rationale = SubjectRationale.model_validate({"explanation": item.explanation, "source_ids": source_ids,
                                                         "evidence": evidence}, context={"sources": sources})
            rationales[subject_ids[item.subject_id]] = rationale.model_dump(mode="json")
        used = {key: sources[key] for rationale in rationales.values() for key in rationale["source_ids"]}
        return rationales, used

    batches = await asyncio.gather(*(explain_batch(subjects[index:index + BATCH_SIZE])
                                     for index in range(0, len(subjects), BATCH_SIZE)))
    rationales = {key: value for items, _ in batches for key, value in items.items()}
    additions = {key: value for _, used in batches for key, value in used.items() if key not in registry}
    return {**program, "subjects": [{**subject, "inclusion_rationale": rationales.get(subject["id"])}
                                     for subject in subjects]}, list(additions.values())
