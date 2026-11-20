"""Точный поиск по полным файлам двух ролевых справочников Yandex."""
import asyncio
import hashlib
from collections.abc import Callable, Iterator
from threading import Lock
from dataclasses import dataclass, field

from argon2 import PasswordHasher
from openai import OpenAI

from auth.user_repository import CurrentUser


class DirectoryError(RuntimeError):
    """Справочник недоступен или имеет неоднозначные записи."""


@dataclass(repr=False)
class Account:
    user: CurrentUser
    password_hash: str = field(repr=False)


def accounts_from_text(text: str, role: str) -> Iterator[Account]:
    # Текст перед первой парой полей является заголовком файла.
    row = {}
    hasher = PasswordHasher()

    def account() -> Account:
        login = row.get("login", "").strip().casefold()
        if not login or bool(row.get("password")) == bool(row.get("hash")):
            raise DirectoryError("Неполная или неоднозначная запись пользователя.")
        key = hashlib.sha256(login.encode()).hexdigest()
        return Account(CurrentUser(id=f"{role}:{key}", email=login, role=role, display_name=row.get("name") or login),
                       row.get("hash") or hasher.hash(row["password"]))

    fields = {"логин": "login", "login": "login", "email": "login", "почта": "login",
              "пароль": "password", "password": "password", "password_hash": "hash", "хеш пароля": "hash",
              "имя": "name", "фио": "name", "display_name": "name"}
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


class YandexUserRepository:
    def __init__(self, client_factory: Callable[[], OpenAI], teachers_id: str, experts_id: str) -> None:
        if not teachers_id or not experts_id or teachers_id == experts_id:
            raise DirectoryError("Укажите разные TEACHERS_INDEX_ID и EXPERTS_INDEX_ID.")
        self.client_factory = client_factory
        self.indexes = {"teacher": teachers_id, "expert": experts_id}
        self.accounts = {}
        self.by_id = {}
        self.loaded = False
        self.lock = Lock()

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

    async def list_experts(self) -> list[CurrentUser]:
        await self.initialize()
        return sorted((item.user for item in self.accounts.values() if item.user.role == "expert"),
                      key=lambda user: user.display_name)
