"""Bounded calls and last successful snapshots for external evidence providers."""

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import logging
from random import uniform
from threading import BoundedSemaphore, Lock
from time import monotonic, sleep
from typing import Callable, Hashable, TypeVar, cast

import httpx
from openai import APIConnectionError, APIStatusError


T = TypeVar("T")
Service = tuple[str, ...]
LOG = logging.getLogger(__name__)
TRANSIENT_STATUS = {429, 500, 502, 503, 504}
_POOL = ThreadPoolExecutor(max_workers=16, thread_name_prefix="external-api")
_SLOTS = BoundedSemaphore(16)


def _bounded_call(request: Callable[[], T], remaining: float) -> T:
    # A synchronous SDK call cannot be interrupted safely. Limit outstanding work,
    # and do not start a second attempt while a timed-out request may still be active.
    if not _SLOTS.acquire(blocking=False):
        raise TimeoutError("Все места для внешних запросов заняты.")
    try:
        future = _POOL.submit(request)
    except BaseException:
        _SLOTS.release()
        raise
    future.add_done_callback(lambda _: _SLOTS.release())
    return future.result(timeout=remaining)


def is_temporary(error: Exception) -> bool:
    if isinstance(error, (APIStatusError, httpx.HTTPStatusError)):
        return error.response.status_code in TRANSIENT_STATUS
    return isinstance(error, (APIConnectionError, httpx.TimeoutException, httpx.NetworkError,
                              httpx.ProxyError, httpx.RemoteProtocolError, TimeoutError))


def retry_after(error: Exception) -> float:
    if not isinstance(error, (APIStatusError, httpx.HTTPStatusError)):
        return 0
    value = error.response.headers.get("Retry-After", "").strip()
    try:
        if value.isascii() and value.isdecimal():
            return float(value)
        when = parsedate_to_datetime(value)
        return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return 0


class ExternalUnavailable(RuntimeError):
    """A provider exhausted its time/attempt budget or its circuit is open."""


@dataclass
class _Circuit:
    failures: int = 0
    open_until: float | None = None
    probing: bool = False
    generation: int = 0


class Resilience:
    def __init__(self, *, clock=monotonic, wait=sleep, jitter=uniform, invoke=_bounded_call,
                 threshold: int = 3, cooldown: float = 30, cache_size: int = 256):
        self.clock = clock
        self.wait = wait
        self.jitter = jitter
        self.invoke = invoke
        self.threshold = threshold
        self.cooldown = cooldown
        self.cache_size = cache_size
        self._lock = Lock()
        self._circuits: dict[Service, _Circuit] = {}
        self._snapshots: OrderedDict[Hashable, tuple[float, datetime, object]] = OrderedDict()

    def call(self, service: Service, request: Callable[[httpx.Timeout], T], *, budget: float, read: float) -> T:
        deadline = self.clock() + budget
        name = "/".join(service[:2])
        with self._lock:
            circuit = self._circuits.setdefault(service, _Circuit())
            probe = circuit.open_until is not None
            if probe and (circuit.probing or self.clock() < circuit.open_until):
                raise ExternalUnavailable(f"{name}: запросы временно приостановлены.")
            circuit.probing = probe
            generation = circuit.generation

        server_delay = 0.0
        try:
            for attempt in range(1 if probe else 3):
                remaining = deadline - self.clock()
                if remaining <= 0:
                    raise TimeoutError("Исчерпан бюджет внешнего запроса.")
                timeout = httpx.Timeout(min(read, remaining), connect=min(3, remaining))
                try:
                    result = self.invoke(lambda: request(timeout), remaining)
                    if self.clock() >= deadline:
                        raise TimeoutError("Исчерпан бюджет внешнего запроса.")
                    break
                except Exception as error:
                    if not is_temporary(error):
                        raise
                    server_delay = retry_after(error)
                    delay = max(server_delay, self.jitter(0.25, 0.5) * 2 ** attempt)
                    if isinstance(error, TimeoutError) or probe or attempt == 2 or delay >= deadline - self.clock():
                        raise
                    self.wait(delay)
        except Exception as error:
            with self._lock:
                if generation == circuit.generation:
                    circuit.probing = False
                    if is_temporary(error):
                        circuit.failures += 1
                        if probe or circuit.failures >= self.threshold or server_delay > 0:
                            circuit.open_until = self.clock() + max(self.cooldown, server_delay)
                            circuit.generation += 1
                    else:
                        circuit.failures = 0
                        circuit.open_until = None
            if is_temporary(error):
                LOG.warning("%s временно недоступен (%s).", name, type(error).__name__)
                raise ExternalUnavailable(f"{name}: временный сбой внешнего сервиса.") from error
            raise
        else:
            with self._lock:
                if generation == circuit.generation:
                    circuit.failures = 0
                    circuit.open_until = None
                    circuit.probing = False
            return result

    def search(self, key: Hashable, fetch: Callable[[], tuple[T, ...]], *, max_age: float
               ) -> tuple[tuple[T, ...], datetime | None]:
        started = self.clock()
        try:
            records = fetch()
        except ExternalUnavailable:
            with self._lock:
                snapshot = self._snapshots.get(key)
                if snapshot is None or self.clock() - snapshot[0] > max_age:
                    raise
                self._snapshots.move_to_end(key)
                return cast(tuple[T, ...], snapshot[2]), snapshot[1]
        except (APIStatusError, httpx.HTTPStatusError) as error:
            if error.response.status_code in {401, 403, 404}:
                with self._lock:
                    self._snapshots.pop(key, None)
            raise
        with self._lock:
            previous = self._snapshots.get(key)
            if previous is None or previous[0] <= started:
                self._snapshots[key] = (started, datetime.now(timezone.utc), records)
                self._snapshots.move_to_end(key)
                if len(self._snapshots) > self.cache_size:
                    self._snapshots.popitem(last=False)
        return records, None


EXTERNAL = Resilience()


def snapshot_notice(source: str, taken_at: datetime) -> str:
    return f"{source}: использован сохранённый снимок от {taken_at:%Y-%m-%d %H:%M} UTC."
