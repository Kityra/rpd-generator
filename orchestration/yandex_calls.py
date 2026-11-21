"""Shared Yandex AI Studio calls for the CLI and course graph."""

from datetime import datetime
from hashlib import sha256
from urllib.parse import quote

from domain.schemas import SourceRecord, SourceType
from orchestration.evidence_cache import ACADEMIC_TTL, MARKET_TTL, SnapshotEvidence, search_evidence, source_record
from orchestration.resilience import EXTERNAL, ExternalUnavailable


class InvalidIndexResponse(ValueError):
    """The index response cannot be used as evidence or saved as a snapshot."""


def _scope(settings: dict[str, str]) -> str:
    return sha256(f"{settings['YANDEX_FOLDER_ID']}\0{settings.get('YANDEX_API_KEY', '')}".encode()).hexdigest()


MODEL_CIRCUIT_COOLDOWN = 10.0
STREAM_READ_TIMEOUT = 60.0


def collect_stream(events):
    """Итоговый объект ответа из потока событий Responses API."""
    if not hasattr(events, "__iter__") and hasattr(events, "output_text"):
        return events  # сервер вернул обычный ответ вместо потока
    final = None
    for event in events:
        if getattr(event, "type", "") in {"response.completed", "response.incomplete", "response.failed"}:
            final = event.response
    if final is None:
        raise RuntimeError("Поток ответа модели оборвался без итогового события.")
    return final


def model_response(client, settings, *, model: str, budget: float = 120, stream: bool = False,
                   request_key: str | None = None, **kwargs):
    """Вызов модели. При stream=True ответ читается потоком: лимит чтения действует на каждый фрагмент,
    поэтому длинный ответ не обрывается, пока модель продолжает писать, а общий срок задаёт budget."""
    if stream:
        def request(timeout):
            return collect_stream(client.responses.create(model=model, timeout=timeout, stream=True, **kwargs))
        read = min(STREAM_READ_TIMEOUT, budget)
    else:
        def request(timeout):
            return client.responses.create(model=model, timeout=timeout, **kwargs)
        read = budget
    return EXTERNAL.call(("Yandex", "responses", _scope(settings), model), request, budget=budget, read=read,
                         request_key=request_key, count_timeouts=request_key is None,
                         cooldown=MODEL_CIRCUIT_COOLDOWN if request_key else None, wait_if_open=request_key is not None)


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
                score = getattr(hit, "score", None)
                score = float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) else None
                fragments.append(SnapshotEvidence(source=source_record(index, url, hit.filename, text[:6000],
                                                                       relevance_score=score)))
        return tuple(fragments)

    if index == "Interview":
        return tuple(item.source for item in fetch()), None
    ttl = MARKET_TTL if index in {"HH", "Priklad"} else ACADEMIC_TTL
    try:
        fragments, taken_at = search_evidence(service, (scope, query, count), fetch, ttl=ttl)
    except ExternalUnavailable as error:
        if index == "RPD":
            raise ExternalUnavailable("Индекс РПД недоступен, пригодного снимка нет.") from error
        raise
    if index == "RPD" and not fragments:
        detail = "В снимке РПД" if taken_at is not None else "В индексе РПД"
        raise ExternalUnavailable(f"{detail} нет материалов по этому запросу.")
    return tuple(item.source for item in fragments), taken_at
