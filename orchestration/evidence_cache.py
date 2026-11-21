"""Persist checked source material around live calls with retry/circuit breaking."""

from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import logging
from typing import Callable

import httpx
from openai import APIStatusError
from pydantic import Field, model_validator
from sqlalchemy.exc import SQLAlchemyError

from domain.schemas import Model, SourceRecord, SourceType
from orchestration.resilience import ExternalUnavailable
from storage.evidence_snapshots import snapshot_store


LOG = logging.getLogger(__name__)
STORE_ERRORS = (SQLAlchemyError, OSError, ImportError, ValueError, TypeError)
ACADEMIC_TTL = timedelta(days=7)
MARKET_TTL = timedelta(days=1)
LITERATURE_TTL = timedelta(hours=6)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _key(parts: tuple) -> str:
    return _digest(json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def source_record(kind: SourceType, url: str, title: str, content: str, *,
                  retrieved_at: datetime | None = None, relevance_score: float | None = None,
                  parent_document_id: str | None = None, publication_year: int | None = None) -> SourceRecord:
    digest = _digest(content)
    key = kind + "-" + _digest(f"{kind}\n{url}\n{digest}")[:20]
    return SourceRecord(source_id=key, source_type=kind, url=url, title=title,
                        retrieved_at=retrieved_at or _now(), raw_hash=digest, content=content,
                        relevance_score=relevance_score, parent_document_id=parent_document_id,
                        publication_year=publication_year)


class SnapshotEvidence(Model):
    source: SourceRecord
    details: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_provenance(self):
        if self.source.retrieved_at.utcoffset() is None or self.source.raw_hash != _digest(self.source.content):
            raise ValueError("Не удалось проверить дату или хэш источника.")
        return self


def search_evidence(namespace: tuple[str, ...], parameters: tuple, fetch: Callable[[], tuple[SnapshotEvidence, ...]],
                    *, ttl: timedelta) -> tuple[tuple[SnapshotEvidence, ...], datetime | None]:
    key = _key(("evidence-v1", namespace, parameters))
    access_key = _key(("evidence-access-v1", namespace))
    started_at = _now()
    try:
        records = fetch()
    except ExternalUnavailable:
        cached = None
        try:
            row = snapshot_store().read(key, access_key, _now(), ttl)
            if row is not None:
                payload, taken_at = row
                if not isinstance(payload, list):
                    raise ValueError("Неверный формат снимка.")
                cached = tuple(SnapshotEvidence.model_validate(item) for item in payload), taken_at
        except STORE_ERRORS as error:
            LOG.warning("Не удалось прочитать снимок %s (%s).", namespace[0], type(error).__name__)
        if cached is not None:
            return cached
        raise
    except (APIStatusError, httpx.HTTPStatusError) as error:
        if error.response.status_code in {401, 403, 404}:
            denied_at = _now()
            # Keep the access record longer than snapshots from older in-flight calls.
            _save(namespace[0], access_key, denied_at, denied_at, ACADEMIC_TTL + timedelta(minutes=5), None)
        raise

    received_at = _now()
    _save(namespace[0], key, started_at, received_at, ttl, [item.model_dump(mode="json") for item in records])
    return records, None


def _save(source: str, key: str, started_at: datetime, received_at: datetime,
          ttl: timedelta, payload: list[dict] | None) -> None:
    try:
        snapshot_store().write(key, started_at, received_at, received_at + ttl, payload)
    except STORE_ERRORS as error:
        LOG.warning("Не удалось записать снимок %s (%s).", source, type(error).__name__)
