"""Точный поиск по полным файлам ролевых справочников Yandex и по локальному файлу учётных записей."""
import asyncio
import hashlib
from collections.abc import Callable, Iterator
from pathlib import Path
from threading import Lock
from dataclasses import dataclass, field

from argon2 import PasswordHasher
from openai import OpenAI

from auth.user_repository import CurrentUser


ROLES = ("teacher", "expert", "program_head", "admin")
# Старые имена ролей сохраняют прежние права и идентификаторы пользователей.
ROLE_NAMES = {"teacher": "teacher", "преподаватель": "teacher", "methodologist": "teacher", "методист курса": "teacher",
              "expert": "expert", "эксперт": "expert",
              "program_head": "program_head", "руководитель": "program_head", "руководитель программы": "program_head",
              "методист": "program_head", "admin": "admin", "администратор": "admin"}


class DirectoryError(RuntimeError):
    """Справочник недоступен или имеет неоднозначные записи."""


@dataclass(repr=False)
class Account:
    user: CurrentUser
    password_hash: str = field(repr=False)


def accounts_from_text(text: str, role: str | None) -> Iterator[Account]:
    """Записи «Логин: … / Пароль: … / Имя: …». Если role не задана, роль берётся из поля «Роль» записи."""
    # Текст перед первой парой полей является заголовком файла.
    row = {}
    hasher = PasswordHasher()

    def account() -> Account:
        login = row.get("login", "").strip().casefold()
        if not login or bool(row.get("password")) == bool(row.get("hash")):
            raise DirectoryError("Неполная или неоднозначная запись пользователя.")
        kind = role or ROLE_NAMES.get(row.get("role", "").strip().casefold())
        if kind not in ROLES:
            raise DirectoryError(f"У записи {login} не указана или не распознана роль.")
        key = hashlib.sha256(login.encode()).hexdigest()
        return Account(CurrentUser(id=f"{kind}:{key}", email=login, role=kind, display_name=row.get("name") or login),
                       row.get("hash") or hasher.hash(row["password"]))

    fields = {"логин": "login", "login": "login", "email": "login", "почта": "login",
              "пароль": "password", "password": "password", "password_hash": "hash", "хеш пароля": "hash",
              "имя": "name", "фио": "name", "display_name": "name", "роль": "role", "role": "role"}
    for line in text.splitlines():
        label, sep, value = line.partition(":")
        key = fields.get(label.strip().casefold())
        if not sep or not key:
            continue
        if key == "login" and "login" in row:
            yield account()
            row = {}
        if key in row:
            raise DirectoryError("Повторяется поле в записи пользователя.")
        row[key] = value.strip()
    if not row:
        raise DirectoryError("В файле не распознаны записи пользователей.")
    yield account()


class AccountDirectory:
    """Общая логика справочников: поиск по логину и ID, списки ролей."""

    def __init__(self) -> None:
        self.accounts: dict[str, Account] = {}
        self.by_id: dict[str, Account] = {}
        self.loaded = False
        self.lock = Lock()

    def _read(self) -> dict[str, Account]:
        raise NotImplementedError

    def _initialize(self) -> None:
        with self.lock:
            if not self.loaded:
                self.accounts = self._read()
                self.by_id = {entry.user.id: entry for entry in self.accounts.values()}
                self.loaded = True

    async def initialize(self) -> None:
        if not self.loaded:
            await asyncio.to_thread(self._initialize)

    async def get_by_email(self, email: str) -> CurrentUser | None:
        await self.initialize()
        item = self.accounts.get(email.strip().casefold())
        return item.user if item else None

    async def get_by_id(self, user_id: str) -> CurrentUser | None:
        await self.initialize()
        item = self.by_id.get(user_id)
        return item.user if item else None

    async def password_hash(self, user_id: str) -> str:
        await self.initialize()
        item = self.by_id.get(user_id)
        return item.password_hash if item else ""

    async def _role(self, role: str) -> list[CurrentUser]:
        await self.initialize()
        return sorted((item.user for item in self.accounts.values() if item.user.role == role),
                      key=lambda user: user.display_name)

    async def list_experts(self) -> list[CurrentUser]:
        return await self._role("expert")

    async def list_teachers(self) -> list[CurrentUser]:
        return await self._role("teacher")


class YandexUserRepository(AccountDirectory):
    def __init__(self, client_factory: Callable[[], OpenAI], teachers_id: str, experts_id: str,
                 heads_id: str = "") -> None:
        super().__init__()
        if not teachers_id or not experts_id or teachers_id == experts_id:
            raise DirectoryError("Укажите разные TEACHERS_INDEX_ID и EXPERTS_INDEX_ID.")
        if heads_id and heads_id in (teachers_id, experts_id):
            raise DirectoryError("PROGRAM_HEADS_INDEX_ID должен отличаться от справочников методистов и экспертов.")
        self.client_factory = client_factory
        self.indexes = {"teacher": teachers_id, "expert": experts_id}
        if heads_id:
            self.indexes["program_head"] = heads_id

    def _read(self) -> dict[str, Account]:
        accounts = {}
        with self.client_factory() as client:
            for role, index in self.indexes.items():
                for entry in client.vector_stores.files.list(vector_store_id=index):
                    if entry.status != "completed":
                        raise DirectoryError("Файл справочника ещё не готов.")
                    raw = client.files.content(entry.id).content.decode("utf-8-sig")
                    for item in accounts_from_text(raw, role):
                        if item.user.email in accounts:
                            raise DirectoryError("Логин неоднозначен в справочниках пользователей.")
                        accounts[item.user.email] = item
        return accounts


class LocalUserRepository(AccountDirectory):
    """Локальный файл учётных записей для разработки: роль указывается в каждой записи."""

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = Path(path)

    def _read(self) -> dict[str, Account]:
        try:
            text = self.path.read_text(encoding="utf-8-sig")
        except OSError as error:
            raise DirectoryError(f"Не удалось прочитать LOCAL_USERS_FILE ({self.path.name}).") from error
        accounts = {}
        for item in accounts_from_text(text, None):
            if item.user.email in accounts:
                raise DirectoryError("Логин повторяется в локальном файле учётных записей.")
            accounts[item.user.email] = item
        return accounts


class CombinedUserRepository:
    """Несколько справочников: логин должен быть однозначным во всех."""

    def __init__(self, repositories: list) -> None:
        self.repositories = repositories

    async def _first(self, method: str, value: str):
        found = [user for repo in self.repositories if (user := await getattr(repo, method)(value)) is not None]
        if len(found) > 1:
            raise DirectoryError("Логин неоднозначен: он есть в нескольких справочниках.")
        return found[0] if found else None

    async def get_by_email(self, email: str) -> CurrentUser | None:
        return await self._first("get_by_email", email)

    async def get_by_id(self, user_id: str) -> CurrentUser | None:
        return await self._first("get_by_id", user_id)

    async def password_hash(self, user_id: str) -> str:
        for repo in self.repositories:
            if await repo.get_by_id(user_id) is not None:
                return await repo.password_hash(user_id)
        return ""

    async def list_experts(self) -> list[CurrentUser]:
        return [user for repo in self.repositories for user in await repo.list_experts()]

    async def list_teachers(self) -> list[CurrentUser]:
        return [user for repo in self.repositories for user in await repo.list_teachers()]
