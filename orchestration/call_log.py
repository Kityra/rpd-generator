"""Журнал вызовов модели: этап, модель, длительность, токены и итог каждого вызова."""

import logging
import re
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Awaitable, Callable

import httpx
from openai import APIConnectionError, APITimeoutError
from pydantic import ValidationError

from orchestration.resilience import ExternalUnavailable


LOG = logging.getLogger(__name__)
OUTCOME_LABELS = {
    "ok": "успешно", "validation_error": "ответ не прошёл проверку схемы", "timeout": "истекло время ожидания",
    "circuit_open": "запросы к модели временно приостановлены", "moderated": "ответ ограничен модерацией",
    "truncated": "ответ модели обрезан по длине", "unavailable": "сервис модели недоступен", "error": "ошибка вызова",
    "input_too_large": "запрос больше контекста модели",
    "network_error": "сетевая ошибка",
}


@dataclass
class CallScope:
    course_id: str | None = None
    run_id: str | None = None
    stage: str = ""
    subject: str | None = None
    subject_id: str | None = None


@dataclass
class CallStats:
    """Заполняется клиентом модели. Объект общий для потока to_thread и вызывающей задачи."""
    model: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    attempts: int = 0
    parameters: dict = field(default_factory=dict)

    def add_usage(self, usage: Any) -> None:
        if usage is None:
            return
        for name in ("input_tokens", "output_tokens"):
            value = getattr(usage, name, None)
            if isinstance(value, int):
                setattr(self, name, (getattr(self, name) or 0) + value)


CALL_SCOPE: ContextVar[CallScope] = ContextVar("model_call_scope", default=CallScope())
CALL_STATS: ContextVar[CallStats | None] = ContextVar("model_call_stats", default=None)
# Пометки клиента модели для текущей дисциплины (например, удалённые неподтверждённые ссылки).
GENERATION_NOTES: ContextVar[list[str] | None] = ContextVar("generation_notes", default=None)

_SECRET = re.compile(r"(?i)\b(api[-_ ]?key|bearer|authorization|token|secret|password)\b\s*[:=]?\s*\S+")
_LONG_TOKEN = re.compile(r"\b[A-Za-z0-9_\-]{32,}\b")


def sanitize_error(error: BaseException | str | None, limit: int = 500) -> str | None:
    """Текст ошибки без ключей, токенов и длинных идентификаторов."""
    if error is None:
        return None
    text = str(error) if not isinstance(error, BaseException) else f"{type(error).__name__}: {error}"
    text = _SECRET.sub(lambda match: f"{match.group(1)}=***", text)
    text = _LONG_TOKEN.sub("***", text)
    text = " ".join(text.split())
    return text[:limit]


def _moderated(error: BaseException) -> bool:
    return type(error).__name__ == "YandexModeratedResponse"


def classify_outcome(error: BaseException | None) -> str:
    if error is None:
        return "ok"
    if _moderated(error):
        return "moderated"
    if type(error).__name__ == "InputTooLarge":
        return "input_too_large"
    if isinstance(error, ValidationError):
        return "validation_error"
    if isinstance(error, ExternalUnavailable):
        if "приостановлены" in str(error):
            return "circuit_open"
        cause = error.__cause__
        if isinstance(cause, (TimeoutError, httpx.TimeoutException, APITimeoutError)):
            return "timeout"
        if isinstance(cause, (httpx.NetworkError, APIConnectionError)):
            return "network_error"
        return "unavailable"
    if isinstance(error, (TimeoutError, httpx.TimeoutException, APITimeoutError)):
        return "timeout"
    if isinstance(error, (httpx.NetworkError, APIConnectionError)):
        return "network_error"
    if isinstance(error, RuntimeError) and "max_output_tokens" in str(error):
        return "truncated"
    return "error"


def failure_note(error: BaseException) -> str:
    """Краткая причина для пометки «собрано шаблоном»."""
    outcome = classify_outcome(error)
    return f"{OUTCOME_LABELS[outcome]} ({outcome}): {sanitize_error(error, 240)}"


class RecordingModelClient:
    """Оборачивает клиент модели и пишет журнал каждого вызова generate()."""

    def __init__(self, inner, record: Callable[[dict], Awaitable[None]] | None, *, clock=monotonic):
        self.inner = inner
        self.record = record
        self.clock = clock

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def describe(self) -> dict | None:
        """Модель и параметры генерации, если клиент их сообщает."""
        describe = getattr(type(self.inner), "describe", None)
        return describe(self.inner) if callable(describe) else None

    async def generate(self, output, prompt, payload, context, **options):
        flash = bool(options.get("flash"))
        stats = CallStats()
        token = CALL_STATS.set(stats)
        started = self.clock()
        error: BaseException | None = None
        try:
            return await self.inner.generate(output, prompt, payload, context, **options)
        except BaseException as caught:
            error = caught
            raise
        finally:
            CALL_STATS.reset(token)
            if not isinstance(error, (KeyboardInterrupt, SystemExit)):
                await self._save(output, stats, started, error, flash)

    async def _save(self, output, stats: CallStats, started: float, error: BaseException | None, flash: bool) -> None:
        if self.record is None:
            return
        scope = CALL_SCOPE.get()
        outcome = classify_outcome(error)
        entry = {
            "course_id": scope.course_id, "run_id": scope.run_id, "stage": scope.stage or "unknown",
            "subject": scope.subject, "subject_id": scope.subject_id,
            "output": getattr(output, "__name__", str(output)),
            "model": stats.model or ("flash" if flash else "default"),
            "duration_ms": int(max(0.0, self.clock() - started) * 1000),
            "input_tokens": stats.input_tokens, "output_tokens": stats.output_tokens,
            "attempts": stats.attempts or 1, "outcome": outcome,
            "error": (f"{type(error).__name__}: {OUTCOME_LABELS[outcome]}"
                      if scope.stage == "llm_judge" and error else sanitize_error(error)),
            "parameters": stats.parameters or None,
        }
        try:
            await self.record(entry)
        except Exception as failure:  # журнал не должен останавливать генерацию
            LOG.warning("Не удалось записать журнал вызова модели (%s).", type(failure).__name__)
