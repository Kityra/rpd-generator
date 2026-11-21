import asyncio
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from openai import APIResponseValidationError, APIStatusError
from pydantic import BaseModel, ValidationError, ValidationInfo, model_validator

from domain.schemas import (ActivityAspect, ActivityProcess, ActivityTask, AppliedActivityModel, CourseProgramModel,
                            EvidenceResults, LearningReference, Model, SourceRecord, SubjectPlan, TopicAssignments)
from orchestration.call_log import CALL_STATS, GENERATION_NOTES
from orchestration.evidence_cache import source_record
from orchestration.relevance_filter import DEFAULT_MIN_SCORE, dropped_notice, filter_fragments
from orchestration.resilience import ExternalUnavailable, snapshot_notice
from orchestration.yandex_calls import InvalidIndexResponse, model_response, search_fragments
from run_rpd import ScientificSearch, collect_recent_articles, open_yandex, require_response_text


logger = logging.getLogger(__name__)
Output = TypeVar("Output", bound=BaseModel)


class TaskList(Model):
    tasks: list[ActivityTask]


class NoActivityEvidence(ValueError):
    """По доступным материалам не выделено ни одной рабочей задачи."""


class ActivityStructure(Model):
    processes: list[ActivityProcess]
    aspects: list[ActivityAspect]

    @model_validator(mode="after")
    def known_tasks(self, info: ValidationInfo):
        valid = set(info.context["task_ids"])
        if any(set(part.task_ids) - valid for part in [*self.processes, *self.aspects]):
            raise ValueError("Копируй идентификаторы задач в точности из tasks.")
        return self


class ModelClient(Protocol):
    async def generate(self, output: type[Output], prompt: str, payload: dict[str, Any],
                       context: dict[str, Any], flash: bool = False) -> Output:
        """Выполнить реальный структурированный вызов с проверкой."""


class EvidenceClient(Protocol):
    async def search(self, role: str, query: str, *, owner_scope: str = "") -> list[SourceRecord]:
        """Получить проверяемые записи из реальных источников."""


def output_schema(node: Any) -> Any:
    if isinstance(node, list):
        return [output_schema(item) for item in node]
    if not isinstance(node, dict):
        return node
    result = {key: output_schema(value) for key, value in node.items()
              if key not in {"default", "minLength", "minItems", "minimum", "maximum", "pattern", "format"}}
    if "properties" in result:
        # Поля, которые заполняет только сервер (происхождение, служебные пометки), модели не передаются.
        result["properties"] = {key: value for key, value in result["properties"].items()
                                if not (isinstance(value, dict) and value.get("server_only"))}
        result["required"] = list(result["properties"])
        result["additionalProperties"] = False
    return result


def grounded_schema(output, context):
    original = output.model_json_schema()
    schema = output_schema(original)
    allowed = [key for key, source in context.get("sources", {}).items() if source.source_type != "Interview"]
    if context.get("allowed_source_ids") is not None:
        # Модель цитирует только источники, переданные ей в запросе (короткая схема и короткий вход).
        allowed = [key for key in context["allowed_source_ids"] if key in allowed]
    for item, original_item in zip([schema, *schema.get("$defs", {}).values()],
                                   [original, *original.get("$defs", {}).values()], strict=True):
        props = item.get("properties", {})
        if allowed and "source_ids" in props:
            props["source_ids"]["items"]["enum"] = allowed
            minimum = original_item["properties"]["source_ids"].get("minItems")
            if minimum is not None:
                props["source_ids"]["minItems"] = minimum
        if allowed and "source_id" in props:
            props["source_id"]["enum"] = allowed
        if "task_ids" in props and context.get("task_ids"):
            props["task_ids"]["items"]["enum"] = context["task_ids"]
        if "activity_task_ids" in props and context.get("activity"):
            props["activity_task_ids"]["items"]["enum"] = [task.id for task in context["activity"].tasks]
        if "excerpt_ids" in props and context.get("subject_excerpts"):
            choices = context["subject_excerpts"]
            props["subject_id"]["enum"] = list(choices)
            props["excerpt_ids"]["items"]["enum"] = sorted(set().union(*choices.values()))
    return schema


def _env(settings: dict[str, str], name: str, default: str) -> str:
    value = settings.get(name)
    if value is None or value == "":
        value = os.environ.get(name, "")
    return value.strip() or default


@dataclass(frozen=True)
class GenerationOptions:
    temperature: float = 0.2
    draft_max_output_tokens: int = 12000
    stream_long_responses: bool = True

    @classmethod
    def from_settings(cls, settings: dict[str, str]) -> "GenerationOptions":
        temperature = float(_env(settings, "YANDEX_TEMPERATURE", "0.2"))
        if not 0 <= temperature <= 1:
            raise ValueError("YANDEX_TEMPERATURE должна быть в диапазоне 0–1.")
        tokens = int(_env(settings, "YANDEX_DRAFT_MAX_OUTPUT_TOKENS", "12000"))
        stream = _env(settings, "YANDEX_STREAM_LONG_RESPONSES", "1").casefold() not in {"0", "false", "no"}
        return cls(temperature, max(1024, tokens), stream)

    def as_dict(self) -> dict:
        return {"temperature": self.temperature, "draft_max_output_tokens": self.draft_max_output_tokens,
                "stream_long_responses": self.stream_long_responses}


class InputTooLarge(RuntimeError):
    """Запрос не помещается в контекст модели (лимит входных токенов)."""


# Оценка по реальному отказу Yandex: 217 тыс. символов (вход + схема + инструкции) ≈ 64 тыс. токенов.
CHARS_PER_TOKEN = 2.6


def estimate_tokens(*texts: str) -> int:
    return int(sum(len(text) for text in texts) / CHARS_PER_TOKEN) + 1


def _compact_program(program: dict) -> dict:
    return {"title": program.get("title"), "subjects": [
        {"title": subject.get("title"), "topics": [topic.get("title") for topic in subject.get("topics", [])]}
        for subject in program.get("subjects", [])]}


def fit_request(request: dict, prompt: str, schema: dict, limit: int) -> tuple[dict, list[str]]:
    """Ужать запрос до лимита входных токенов: сжать программу, укоротить и сократить источники.

    Возвращает запрос и список сделанных сокращений; если ужать не удалось — InputTooLarge.
    """
    schema_text = json.dumps(schema, ensure_ascii=False)

    def size(value):
        return estimate_tokens(prompt, schema_text, json.dumps(value, ensure_ascii=False, default=list))

    steps = []
    current = request
    if size(current) <= limit:
        return current, steps
    for key in ("course_program", "baseline", "revision_baseline"):
        if isinstance(current.get(key), dict) and current[key].get("subjects"):
            current = {**current, key: _compact_program(current[key])}
            steps.append(f"{key}: только названия дисциплин и тем")
    for length, count in ((1200, None), (600, None), (600, 20), (300, 12), (300, 6)):
        if size(current) <= limit:
            break
        if isinstance(current.get("sources"), list):
            sources = current["sources"][:count] if count else current["sources"]
            current = {**current, "sources": [{**item, "content": str(item.get("content", ""))[:length]}
                                              if isinstance(item, dict) else item for item in sources]}
            steps.append(f"источники: {len(sources)} шт. по {length} символов")
    tokens = size(current)
    if tokens > limit:
        raise InputTooLarge(f"Запрос ≈{tokens} токенов не помещается в лимит модели {limit} даже после сокращения.")
    return current, steps


def request_budget(input_chars: int, max_output_tokens: int, *, flash: bool = False) -> float:
    """Лимит времени от размера запроса: чтение входа (~2000 символов/с) и генерация (~40 токенов/с)."""
    expected = 15 + input_chars / 2000 + max_output_tokens / 40
    return float(min(600, max(30 if flash else 60, expected)))


def normalize_blocks(data, context: dict):
    """Механическое согласование блоков программы до проверки схемой (смысл не меняется):
    фундаментальный блок всегда защищён; в первом черновике (без baseline) все блоки новые."""
    first_draft = not context.get("baseline")

    def walk(node):
        if isinstance(node, list):
            return [walk(item) for item in node]
        if not isinstance(node, dict):
            return node
        result = {key: walk(value) for key, value in node.items()}
        if "change_status" in result and "justification" in result:
            if result.get("is_fundamental"):
                result["locked"] = True
            if first_draft:
                result["change_status"], result["previous_value"] = "added", None
        return result

    return walk(data)


def parse_output(output, raw: str, context: dict):
    """Проверка ответа модели; для программы курса — после механического согласования блоков."""
    if output not in (CourseProgramModel, TopicAssignments):
        return output.model_validate_json(raw, context=context)
    try:
        data = json.loads(raw)
    except ValueError:
        return output.model_validate_json(raw, context=context)  # ошибка json_invalid для повторного запроса
    return output.model_validate(normalize_blocks(data, context), context=context)


def drop_unverified_references(output, raw: str, error: ValidationError, context: dict):
    """Если не подтвердились только дословные учебные ссылки, убрать их и принять остальной ответ."""
    problems = error.errors(include_input=False, include_context=False, include_url=False)
    if not problems or not all("learning_references" in item["loc"] for item in problems):
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    dropped = 0

    def clean(node):
        nonlocal dropped
        if isinstance(node, list):
            return [clean(item) for item in node]
        if not isinstance(node, dict):
            return node
        result = {key: clean(value) for key, value in node.items()}
        if isinstance(result.get("learning_references"), list):
            kept = []
            for reference in result["learning_references"]:
                try:
                    LearningReference.model_validate(reference, context=context)
                except (ValidationError, KeyError, TypeError):
                    dropped += 1
                else:
                    kept.append(reference)
            result["learning_references"] = kept
        return result

    try:
        return output.model_validate(normalize_blocks(clean(data), context), context=context), dropped
    except ValidationError:
        return None


class YandexModelClient:
    def __init__(self, settings: dict[str, str]) -> None:
        self.settings = settings

    async def generate(self, output: type[Output], prompt: str, payload: dict[str, Any],
                       context: dict[str, Any], flash: bool = False, temperature: float | None = None) -> Output:
        if output is AppliedActivityModel:
            return await asyncio.to_thread(self._activity, prompt, payload, context)
        return await asyncio.to_thread(self._generate, output, prompt, payload, context, flash, temperature)

    def _activity(self, prompt, payload, context):
        task_list = self._generate(TaskList, prompt + " Сначала выдели только подтверждённые рабочие задачи.",
                                   payload, context, False)
        if not task_list.tasks:
            raise NoActivityEvidence("Отсутствуют подтверждённые задачи.")
        tasks = []
        seen = set()
        for task in task_list.tasks:
            title = " ".join(task.title.casefold().split())
            if title and title not in seen:
                seen.add(title)
                key = "t-" + hashlib.sha256(title.encode()).hexdigest()[:10]
                tasks.append(task.model_copy(update={"id": key}).model_dump(mode="json"))
        structure = {"processes": [], "aspects": []}
        if tasks:
            short_ids = {f"T{i}": task["id"] for i, task in enumerate(tasks, 1)}
            catalog = [{**task, "id": key} for key, task in zip(short_ids, tasks, strict=True)]
            parts = self._generate(ActivityStructure,
                prompt + " Сгруппируй переданные задачи в реальные процессы и аспекты. Копируй task_ids из каталога tasks.",
                {**payload, "tasks": catalog}, {**context, "task_ids": list(short_ids)}, False)
            structure = parts.model_dump(mode="json")
            for part in [*structure["processes"], *structure["aspects"]]:
                part["task_ids"] = [short_ids[key] for key in part["task_ids"]]
        return AppliedActivityModel.model_validate({"tasks": tasks, **structure, "source_ids": []}, context=context)

    def options(self) -> GenerationOptions:
        return GenerationOptions.from_settings(self.settings)

    def model_for(self, output: type[BaseModel], flash: bool) -> str:
        if flash:
            return self.settings.get("YANDEX_MODEL_URI_FLASH") or f"gpt://{self.settings['YANDEX_FOLDER_ID']}/aliceai-llm-flash"
        if output in (CourseProgramModel, SubjectPlan):
            return self.settings.get("YANDEX_MODEL_URI_COURSE_DESIGN") or self.settings["YANDEX_MODEL_URI"]
        return self.settings["YANDEX_MODEL_URI"]

    def describe(self) -> dict:
        """Модели и параметры генерации — сохраняются в версии курса."""
        options = self.options()
        return {"design_model": self.model_for(CourseProgramModel, False),
                "default_model": self.model_for(Model, False), "flash_model": self.model_for(Model, True),
                **options.as_dict()}

    def _generate(self, output: type[Output], prompt: str, payload: dict[str, Any],
                  context: dict[str, Any], flash: bool, temperature: float | None = None) -> Output:
        model = self.model_for(output, flash)
        options = self.options()
        # Экстрактор модели деятельности требует temperature = 0 (детерминизм), остальные этапы — из настроек.
        temperature = options.temperature if temperature is None else temperature
        request = payload
        draft = "batch_subjects" in payload
        max_tokens = options.draft_max_output_tokens if draft else (8192 if flash else 16384)
        stream = options.stream_long_responses and draft
        stats = CALL_STATS.get()
        if stats is not None:
            stats.model = model
            stats.parameters = {"temperature": temperature, "max_output_tokens": max_tokens, "stream": stream}
        notes = GENERATION_NOTES.get()
        schema = grounded_schema(output, context)
        # Входной лимит модели минус запас на ответ не нужен: Yandex ограничивает именно вход.
        limit = int(int(_env(self.settings, "YANDEX_INPUT_TOKEN_LIMIT", "32768")) * 0.9)
        with open_yandex(self.settings) as client:
            for attempt in range(2):
                if stats is not None:
                    stats.attempts += 1
                request, shortened = fit_request(request, prompt, schema, limit)
                if shortened:
                    logger.warning("%s: запрос сокращён до лимита модели (%s).", output.__name__, "; ".join(shortened))
                body = json.dumps(request, ensure_ascii=False)
                arguments = dict(
                    model=model, instructions=prompt, input=body,
                    text={"format": {"type": "json_schema", "name": output.__name__,
                                     "schema": schema, "strict": True}},
                    max_output_tokens=max_tokens, temperature=temperature, store=False,
                    budget=request_budget(len(prompt) + len(body), max_tokens, flash=flash),
                    request_key=hashlib.sha256(f"{output.__name__}\0{body}".encode()).hexdigest()[:24],
                )
                try:
                    response = model_response(client, self.settings, stream=stream, **arguments)
                except APIStatusError as error:
                    if not stream or error.status_code not in {400, 404, 405, 422, 501}:
                        raise
                    logger.warning("Потоковый ответ не поддержан (HTTP %s), повтор без потока.", error.status_code)
                    stream = False
                    response = model_response(client, self.settings, **arguments)
                if stats is not None:
                    stats.add_usage(getattr(response, "usage", None))
                if getattr(response, "status", None) == "failed":
                    message = str(getattr(getattr(response, "error", None), "message", "") or "")
                    if "input tokens" in message:
                        raise InputTooLarge(f"Запрос больше контекста модели: {message[-80:]}")
                    raise RuntimeError(f"Модель вернула ошибку: {message[-300:] or 'без описания'}")
                raw = require_response_text(response)
                try:
                    return parse_output(output, raw, context)
                except ValidationError as error:
                    repaired = drop_unverified_references(output, raw, error, context)
                    if repaired is not None:
                        result, dropped = repaired
                        logger.warning("%s: удалено неподтверждённых учебных ссылок — %s", output.__name__, dropped)
                        if notes is not None:
                            notes.append(f"Удалены неподтверждённые учебные ссылки — {dropped}. "
                                         "Учебные источники добавит этап подбора литературы.")
                        return result
                    logger.warning("Ошибка схемы %s, попытка %s", output.__name__, attempt + 1)
                    if attempt == 1:
                        raise
                    request = {**payload, "previous_output": raw, "repair": error.errors(
                        include_input=False, include_context=False, include_url=False)}
        raise RuntimeError("Не удалось проверить структурированный ответ.")


class YandexEvidenceClient:
    def __init__(self, settings: dict[str, str]) -> None:
        self.settings = settings

    async def search(self, role: str, query: str, *, owner_scope: str = "") -> list[SourceRecord]:
        return await asyncio.to_thread(self._search, role, query, owner_scope)

    def _search(self, role: str, query: str, owner_scope: str) -> list[SourceRecord]:
        indexes = {
            "business": ("Priklad", "HH"), "books": ("Books",),
            "university": ("RPD", "Books"), "interview": ("Interview",),
            "factcheck": ("Priklad", "HH", "Books"), "science": (),
            "subject_rationale": ("Books", "RPD"),
            "subject_rationale_extended": ("Books", "RPD", "Priklad"),
        }
        if role not in indexes:
            raise ValueError(f"Неизвестная роль поиска {role}.")
        records = EvidenceResults()

        def note(message: str) -> None:
            records.notices.append(message)
            logger.warning("%s", message)

        min_score = float(_env(self.settings, "VECTOR_MIN_SCORE", str(DEFAULT_MIN_SCORE)))
        with open_yandex(self.settings) as client:
            rationale_search = role in {"subject_rationale", "subject_rationale_extended"}
            for index in indexes[role]:
                store = self.settings.get(f"YANDEX_VECTOR_STORE_ID_{index.upper()}")
                required = index == "RPD" and role == "university"
                if not store:
                    if required:
                        raise ValueError("Обязательный индекс РПД не настроен.")
                    note(f"Индекс {index} не настроен.")
                    continue
                try:
                    fragments, taken_at = search_fragments(
                        client, self.settings, index, store, query,
                        {"subject_rationale": 2, "subject_rationale_extended": 10}.get(role, 4),
                        scope=("portal", owner_scope, role), budget=12 if rationale_search else 30,
                    )
                except (APIStatusError, APIResponseValidationError, ExternalUnavailable, InvalidIndexResponse) as error:
                    if required:
                        reason = str(error) if isinstance(error, ExternalUnavailable) else type(error).__name__
                        logger.warning("Поиск в обязательном индексе РПД остановлен: %s", reason)
                        raise
                    note(f"Индекс {index} недоступен.")
                    continue
                if taken_at is not None:
                    note(snapshot_notice(index, taken_at))
                if index != "Interview":
                    kept, dropped = filter_fragments(fragments, min_score)
                    if required and not kept:
                        # Обязательный индекс РПД: при низкой близости оставляем содержательные фрагменты с пометкой.
                        kept, _ = filter_fragments(fragments, float("-inf"))
                        if kept:
                            note(f"Индекс {index}: близость найденных фрагментов ниже порога {min_score:.2f}, "
                                 "проверьте их соответствие курсу.")
                    if dropped:
                        note(dropped_notice(index, dropped))
                    fragments = kept
                records.extend(fragments)
            if role == "science":
                def plan(prompt: str, subject: str) -> str:
                    response = model_response(client, self.settings,
                        model=f"gpt://{self.settings['YANDEX_FOLDER_ID']}/aliceai-llm-flash",
                        instructions=prompt, input=subject, store=False, max_output_tokens=384, budget=20,
                        text={"format": {"type": "json_schema", "name": "search", "strict": True,
                                         "schema": output_schema(ScientificSearch.model_json_schema())}})
                    return require_response_text(response)
                for article in collect_recent_articles(query, plan, note):
                    records.append(source_record("arXiv" if article.status == "arxiv_candidate" else "Crossref",
                                                 article.url, article.title, article.as_science_source(
                                                     include_abstract=True, include_snapshot_notice=False)["text"],
                                                 retrieved_at=article.retrieved_at))
        return records
