import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from domain.schemas import SourceRecord
from run_rpd import ScientificSearch, collect_recent_articles, open_yandex, require_response_text


logger = logging.getLogger(__name__)
Output = TypeVar("Output", bound=BaseModel)


class ModelClient(Protocol):
    async def generate(self, output: type[Output], prompt: str, payload: dict[str, Any],
                       context: dict[str, Any], flash: bool = False) -> Output:
        """Выполнить реальный структурированный вызов с проверкой."""


class EvidenceClient(Protocol):
    async def search(self, role: str, query: str) -> list[SourceRecord]:
        """Получить проверяемые записи из реальных источников."""


def output_schema(node: Any) -> Any:
    if isinstance(node, list):
        return [output_schema(item) for item in node]
    if not isinstance(node, dict):
        return node
    result = {key: output_schema(value) for key, value in node.items()
              if key not in {"default", "minLength", "minItems", "minimum", "maximum", "pattern", "format"}}
    if "properties" in result:
        result["required"] = list(result["properties"])
        result["additionalProperties"] = False
    return result


class YandexModelClient:
    def __init__(self, settings: dict[str, str]) -> None:
        self.settings = settings

    async def generate(self, output: type[Output], prompt: str, payload: dict[str, Any],
                       context: dict[str, Any], flash: bool = False) -> Output:
        return await asyncio.to_thread(self._generate, output, prompt, payload, context, flash)

    def _generate(self, output: type[Output], prompt: str, payload: dict[str, Any],
                  context: dict[str, Any], flash: bool) -> Output:
        model = (f"gpt://{self.settings['YANDEX_FOLDER_ID']}/aliceai-llm-flash" if flash
                 else self.settings["YANDEX_MODEL_URI"])
        request = payload
        with open_yandex(self.settings) as client:
            for attempt in range(2):
                response = client.responses.create(
                    model=model, instructions=prompt,
                    input=json.dumps(request, ensure_ascii=False),
                    text={"format": {"type": "json_schema", "name": output.__name__,
                                     "schema": output_schema(output.model_json_schema()), "strict": True}},
                    max_output_tokens=8192 if flash else 16384, store=False,
                )
                raw = require_response_text(response)
                try:
                    return output.model_validate_json(raw, context=context)
                except ValidationError as error:
                    logger.warning("Ошибка схемы %s, попытка %s", output.__name__, attempt + 1)
                    if attempt == 1:
                        raise
                    request = {**payload, "previous_output": raw, "repair": error.errors(
                        include_input=False, include_context=False, include_url=False)}
        raise RuntimeError("Не удалось проверить структурированный ответ.")


def source_record(kind: str, url: str, title: str, content: str) -> SourceRecord:
    digest = hashlib.sha256(content.encode()).hexdigest()
    key = hashlib.sha256(f"{kind}\n{url}\n{digest}".encode()).hexdigest()
    return SourceRecord(source_id=key, source_type=kind, url=url, title=title,
                        retrieved_at=datetime.now(timezone.utc), raw_hash=digest, content=content)


class YandexEvidenceClient:
    def __init__(self, settings: dict[str, str]) -> None:
        self.settings = settings

    async def search(self, role: str, query: str) -> list[SourceRecord]:
        return await asyncio.to_thread(self._search, role, query)

    def _search(self, role: str, query: str) -> list[SourceRecord]:
        indexes = {
            "business": ("Priklad", "HH"), "books": ("Books",),
            "university": ("RPD", "Books"), "interview": ("Interview",),
            "factcheck": ("Priklad", "HH", "Books"), "science": (),
        }
        if role not in indexes:
            raise ValueError(f"Неизвестная роль поиска {role}.")
        records = []
        with open_yandex(self.settings) as client:
            for index in indexes[role]:
                store = self.settings.get(f"YANDEX_VECTOR_STORE_ID_{index.upper()}")
                if not store:
                    raise ValueError(f"Для {role} требуется индекс {index}.")
                result = client.vector_stores.search(store, query=query, max_num_results=4, timeout=30)
                for hit in result.data:
                    text = "\n".join(part.text for part in hit.content if part.type == "text" and part.text)
                    if text.strip():
                        records.append(source_record(index, f"https://ai.api.cloud.yandex.net/v1/files/{hit.file_id}",
                                                     hit.filename, text[:6000]))
            if role == "science":
                def plan(prompt: str, subject: str) -> str:
                    response = client.responses.create(
                        model=f"gpt://{self.settings['YANDEX_FOLDER_ID']}/aliceai-llm-flash",
                        instructions=prompt, input=subject, store=False, max_output_tokens=384,
                        text={"format": {"type": "json_schema", "name": "search", "strict": True,
                                         "schema": output_schema(ScientificSearch.model_json_schema())}})
                    return require_response_text(response)
                for article in collect_recent_articles(query, plan, logger.warning):
                    records.append(source_record("arXiv" if article.status == "arxiv_candidate" else "Crossref",
                                                 article.url, article.title, str(article.as_science_source(include_abstract=True))))
        return records
