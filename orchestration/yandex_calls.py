"""Shared Yandex AI Studio calls for the CLI and course graph."""

from datetime import datetime
from hashlib import sha256
from urllib.parse import quote

from openai import APIResponseValidationError, APIStatusError

from domain.official_rpd import RpdLookup
from domain.schemas import SourceRecord, SourceType
from orchestration.evidence_cache import ACADEMIC_TTL, MARKET_TTL, SnapshotEvidence, search_evidence, source_record
from orchestration.resilience import EXTERNAL, ExternalUnavailable


class InvalidIndexResponse(ValueError):
    """The index response cannot be used as evidence or saved as a snapshot."""


def _scope(settings: dict[str, str]) -> str:
    return sha256(f"{settings['YANDEX_FOLDER_ID']}\0{settings.get('YANDEX_API_KEY', '')}".encode()).hexdigest()


def model_response(client, settings, *, model: str, budget: float = 120, **kwargs):
    return EXTERNAL.call(("Yandex", "responses", _scope(settings), model),
                         lambda timeout: client.responses.create(model=model, timeout=timeout, **kwargs),
                         budget=budget, read=budget)


def search_fragments(client, settings, index: SourceType, store_id: str, query: str, count: int, *,
                     scope: tuple[str, ...] = ("local",), budget: float = 30
                     ) -> tuple[tuple[SourceRecord, ...], datetime | None]:
    service = ("Yandex", "vector_search", _scope(settings), index, store_id)

    def fetch():
        result = EXTERNAL.call(service, lambda timeout: client.vector_stores.search(
            store_id, query=query, max_num_results=count, timeout=timeout), budget=budget, read=min(15, budget))
        fragments = []
        if not isinstance(result.data, list):
            raise InvalidIndexResponse("Индекс вернул некорректный список фрагментов.")
        for hit in result.data:
            text = "\n".join(part.text for part in hit.content if part.type == "text" and part.text)
            if text.strip():
                if not hit.file_id or not hit.filename:
                    raise InvalidIndexResponse("Во фрагменте индекса нет ссылки на файл.")
                url = f"https://ai.api.cloud.yandex.net/v1/files/{quote(hit.file_id, safe='')}"
                fragments.append(SnapshotEvidence(source=source_record(index, url, hit.filename, text[:6000])))
        return tuple(fragments)

    if index == "Interview":
        return tuple(item.source for item in fetch()), None
    ttl = MARKET_TTL if index in {"HH", "Priklad"} else ACADEMIC_TTL
    fragments, taken_at = search_evidence(service, (scope, query, count), fetch, ttl=ttl)
    return tuple(item.source for item in fragments), taken_at


def lookup_rpd(client, settings, query: str, count: int, **options
               ) -> tuple[tuple[SourceRecord, ...], datetime | None, RpdLookup]:
    store_id = settings.get("YANDEX_VECTOR_STORE_ID_RPD")
    if not store_id:
        return (), None, {"status": "unconfigured", "reason": "Индекс РПД не настроен."}
    try:
        fragments, taken_at = search_fragments(client, settings, "RPD", store_id, query, count, **options)
    except (APIStatusError, APIResponseValidationError, ExternalUnavailable, InvalidIndexResponse) as error:
        status = f" (HTTP {error.status_code})" if isinstance(error, APIStatusError) else ""
        return (), None, {"status": "unavailable", "reason": f"Индекс РПД недоступен{status}, пригодного снимка нет."}
    if not fragments:
        if taken_at is not None:
            return (), None, {"status": "unavailable", "reason": "В сохранённом снимке РПД нет материалов по запросу."}
        return (), None, {"status": "empty", "reason": "Индекс РПД не вернул материалов по запросу."}
    return fragments, taken_at, {"status": "snapshot" if taken_at is not None else "live", "reason": ""}
