"""Bounded calls and a separate circuit breaker for each external operation."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import logging
from random import uniform
from threading import BoundedSemaphore, Lock
from time import monotonic, sleep
from typing import Callable, TypeVar

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
    failed_requests: set = field(default_factory=set)
    open_until: float | None = None
    probing: bool = False
    generation: int = 0


class Resilience:
    def __init__(self, *, clock=monotonic, wait=sleep, jitter=uniform, invoke=_bounded_call,
                 threshold: int = 3, cooldown: float = 30):
        self.clock = clock
        self.wait = wait
        self.jitter = jitter
        self.invoke = invoke
        self.threshold = threshold
        self.cooldown = cooldown
        self._lock = Lock()
        self._circuits: dict[Service, _Circuit] = {}

    def call(self, service: Service, request: Callable[[httpx.Timeout], T], *, budget: float, read: float,
             request_key: str | None = None, count_timeouts: bool = True, cooldown: float | None = None,
             wait_if_open: bool = False) -> T:
        """Вызов с повторами и выключателем.

        request_key — сбои одного и того же запроса считаются один раз: выключатель открывается,
        когда подряд упали `threshold` разных запросов, а не один долгий запрос в нескольких попытках.
        count_timeouts=False — исчерпание собственного бюджета времени не считается сбоем сервиса
        (долгий ответ модели зависит от размера запроса, а не от состояния провайдера).
        wait_if_open — если пауза закончится раньше половины бюджета, запрос дожидается её, а не падает.
        """
        deadline = self.clock() + budget
        name = "/".join(service[:2])
        pause = self.cooldown if cooldown is None else cooldown
        while True:
            with self._lock:
                circuit = self._circuits.setdefault(service, _Circuit())
                probe = circuit.open_until is not None
                delay = (circuit.open_until - self.clock()) if probe else 0.0
                if probe and (circuit.probing or delay > 0):
                    if not (wait_if_open and not circuit.probing and 0 < delay <= (deadline - self.clock()) / 2):
                        raise ExternalUnavailable(f"{name}: запросы временно приостановлены.")
                else:
                    circuit.probing = probe
                    generation = circuit.generation
                    break
            self.wait(delay)

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
                    own_timeout = isinstance(error, TimeoutError) and not count_timeouts
                    if is_temporary(error) and not own_timeout:
                        if request_key is None:
                            circuit.failures += 1
                        else:
                            circuit.failed_requests.add(request_key)
                            circuit.failures = len(circuit.failed_requests)
                        if probe or circuit.failures >= self.threshold or (server_delay > 0 and request_key is None):
                            circuit.open_until = self.clock() + max(pause, server_delay)
                            circuit.generation += 1
                            circuit.failed_requests.clear()
                    elif not is_temporary(error):
                        circuit.failures = 0
                        circuit.failed_requests.clear()
                        circuit.open_until = None
            if is_temporary(error):
                LOG.warning("%s временно недоступен (%s).", name, type(error).__name__)
                raise ExternalUnavailable(f"{name}: временный сбой внешнего сервиса.") from error
            raise
        else:
            with self._lock:
                if generation == circuit.generation:
                    circuit.failures = 0
                    circuit.failed_requests.clear()
                    circuit.open_until = None
                    circuit.probing = False
            return result

EXTERNAL = Resilience()


def snapshot_notice(source: str, taken_at: datetime) -> str:
    return f"{source}: использован сохранённый снимок от {taken_at:%Y-%m-%d %H:%M} UTC."
