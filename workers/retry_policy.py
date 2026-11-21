"""Classify command failures independently of the HTTP client's own retries."""

from dataclasses import dataclass
from datetime import timedelta
from random import uniform

import httpx
from openai import APIResponseValidationError, APIStatusError, APITimeoutError
from pydantic import ValidationError
from sqlalchemy.exc import DBAPIError

from orchestration.resilience import ExternalUnavailable, is_temporary, retry_after
from run_rpd import YandexModeratedResponse


@dataclass(frozen=True)
class Failure:
    code: str
    retryable: bool = False
    server_delay: float = 0

    def delay(self, attempt: int) -> timedelta:
        base = min(300, 30 * 2 ** (attempt - 1))
        return timedelta(seconds=max(uniform(base, base * 1.25), min(86_400, self.server_delay)))


def classify_failure(error: Exception) -> Failure:
    if isinstance(error, (APIStatusError, httpx.HTTPStatusError)):
        status = error.response.status_code
        if status == 429:
            return Failure("rate_limited", True, retry_after(error))
        if is_temporary(error):
            return Failure("upstream_unavailable", True, retry_after(error))
        if status in {401, 403}:
            return Failure("authorization")
        return Failure("invalid_configuration" if 400 <= status < 500 else "processing_failed")
    if isinstance(error, ExternalUnavailable):
        cause = error.__cause__
        if isinstance(cause, Exception) and is_temporary(cause):
            return classify_failure(cause)
        return Failure("upstream_unavailable", True)
    if isinstance(error, (TimeoutError, httpx.TimeoutException, APITimeoutError)):
        return Failure("timeout", True)
    if is_temporary(error):
        return Failure("network", True)
    if isinstance(error, DBAPIError):
        sqlstate = getattr(error.orig, "sqlstate", None) or getattr(error.orig, "pgcode", "") or ""
        sqlite_code = getattr(error.orig, "sqlite_errorcode", 0)
        if (error.connection_invalidated or sqlstate.startswith("08") or sqlstate in {"40001", "40P01", "55P03"}
                or sqlite_code & 0xFF in {5, 6}):
            return Failure("database_unavailable", True)
    if isinstance(error, (APIResponseValidationError, ValidationError)):
        return Failure("invalid_response")
    if isinstance(error, YandexModeratedResponse):
        return Failure("moderation")
    if isinstance(error, PermissionError):
        return Failure("authorization")
    if isinstance(error, (ValueError, KeyError)):
        return Failure("invalid_configuration")
    return Failure("processing_failed")
