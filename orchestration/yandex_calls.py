"""Shared Yandex AI Studio calls for the CLI and course graph."""

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256

from orchestration.resilience import EXTERNAL, ExternalUnavailable


@dataclass(frozen=True)
class VectorFragment:
    file_id: str
    title: str
    text: str


class InvalidIndexResponse(ValueError):
    """The index response cannot be used as evidence or saved as a snapshot."""


def _scope(settings: dict[str, str]) -> str:
    return sha256(f"{settings['YANDEX_FOLDER_ID']}\0{settings.get('YANDEX_API_KEY', '')}".encode()).hexdigest()


def model_response(client, settings, *, model: str, budget: float = 120, **kwargs):
    return EXTERNAL.call(("Yandex", "responses", _scope(settings), model),
                         lambda timeout: client.responses.create(model=model, timeout=timeout, **kwargs),
                         budget=budget, read=budget)


def search_fragments(client, settings, index: str, store_id: str, query: str, count: int, *, budget: float = 30
                     ) -> tuple[tuple[VectorFragment, ...], datetime | None]:
    service = ("Yandex", "vector_search", _scope(settings), index, store_id)
    key = (service, sha256(query.encode()).hexdigest(), count)

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
                fragments.append(VectorFragment(hit.file_id, hit.filename, text[:6000]))
        return tuple(fragments)

    hours = 24 if index in {"HH", "Priklad"} else 168
    fragments, taken_at = EXTERNAL.search(key, fetch, max_age=hours * 3600)
    if index == "RPD" and taken_at is not None and not fragments:
        raise ExternalUnavailable("Индекс РПД недоступен, подтверждающих материалов в снимке нет.")
    return fragments, taken_at
