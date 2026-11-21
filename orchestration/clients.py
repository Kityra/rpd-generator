import asyncio
import hashlib
import json
import logging
from typing import Any, Protocol, TypeVar

from openai import APIResponseValidationError, APIStatusError
from pydantic import BaseModel, ValidationError, ValidationInfo, model_validator

from domain.schemas import (ActivityAspect, ActivityProcess, ActivityTask, AppliedActivityModel, CourseProgramModel,
                            EvidenceResults, Model, SourceRecord, SubjectPlan)
from orchestration.evidence_cache import source_record
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
        result["required"] = list(result["properties"])
        result["additionalProperties"] = False
    return result


def grounded_schema(output, context):
    original = output.model_json_schema()
    schema = output_schema(original)
    allowed = [key for key, source in context.get("sources", {}).items() if source.source_type != "Interview"]
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


class YandexModelClient:
    def __init__(self, settings: dict[str, str]) -> None:
        self.settings = settings

    async def generate(self, output: type[Output], prompt: str, payload: dict[str, Any],
                       context: dict[str, Any], flash: bool = False) -> Output:
        if output is AppliedActivityModel:
            return await asyncio.to_thread(self._activity, prompt, payload, context)
        return await asyncio.to_thread(self._generate, output, prompt, payload, context, flash)

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

    def _generate(self, output: type[Output], prompt: str, payload: dict[str, Any],
                  context: dict[str, Any], flash: bool) -> Output:
        if flash:
            model = f"gpt://{self.settings['YANDEX_FOLDER_ID']}/aliceai-llm-flash"
        elif output in (CourseProgramModel, SubjectPlan):
            model = self.settings.get("YANDEX_MODEL_URI_COURSE_DESIGN") or self.settings["YANDEX_MODEL_URI"]
        else:
            model = self.settings["YANDEX_MODEL_URI"]
        request = payload
        draft_batch = "batch_subjects" in payload
        with open_yandex(self.settings) as client:
            for attempt in range(2):
                response = model_response(client, self.settings,
                    model=model, instructions=prompt,
                    input=json.dumps(request, ensure_ascii=False),
                    text={"format": {"type": "json_schema", "name": output.__name__,
                                     "schema": grounded_schema(output, context), "strict": True}},
                    max_output_tokens=4096 if draft_batch else (8192 if flash else 16384), store=False,
                    budget=60 if draft_batch else (45 if flash else 120),
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
