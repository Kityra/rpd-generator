from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, field_validator
from sqlalchemy import MetaData, Table, func, select
from sqlalchemy.ext.asyncio import AsyncEngine

from auth.settings import Settings


class CurrentUser(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    email: str
    role: Literal["teacher", "expert", "admin"]
    display_name: str

    @field_validator("role", mode="before")
    @classmethod
    def unify_methodologist_role(cls, value):
        # Keep legacy user IDs unchanged so saved ownership and sessions still resolve.
        return "teacher" if value == "program_head" else value


class UserRepository(Protocol):
    async def get_by_email(self, email: str) -> CurrentUser | None:
        """Найти существующую учётную запись."""

    async def get_by_id(self, user_id: str) -> CurrentUser | None:
        """Прочитать пользователя по устойчивому ID."""

    async def password_hash(self, user_id: str) -> str:
        """Прочитать существующий хеш для серверной проверки."""

    async def list_experts(self) -> list[CurrentUser]:
        """Выбрать существующих экспертов."""

    async def list_teachers(self) -> list[CurrentUser]:
        """Выбрать существующих методистов."""


class SqlUserRepository:
    def __init__(self, engine: AsyncEngine, settings: Settings) -> None:
        self.engine = engine
        self.settings = settings
        self.tables: dict[str, Table] = {}
        self.role_table: Table | None = None

    async def initialize(self) -> None:
        s = self.settings
        if not s.user_database_url:
            raise ValueError("Укажите USER_DATABASE_URL существующей БД пользователей.")
        names = [s.users_table] if s.user_schema_mode != "split" else [s.teachers_table, s.experts_table]
        async with self.engine.connect() as connection:
            for name in names:
                table = await connection.run_sync(lambda sync, name=name: Table(
                    name, MetaData(), schema=s.user_db_schema, autoload_with=sync))
                for column in (s.id_column, s.email_column, s.password_hash_column, s.display_name_column):
                    if column not in table.c:
                        raise ValueError(f"Нет колонки {column} в {name}.")
                self.tables[name] = table
            if s.user_schema_mode == "related":
                self.role_table = await connection.run_sync(lambda sync: Table(
                    s.user_roles_table, MetaData(), schema=s.user_db_schema, autoload_with=sync))

    def _statement(self, table: Table):
        s = self.settings
        if s.user_schema_mode == "related":
            return select(table, self.role_table.c[s.user_role_value_column].label("_role")).join(
                self.role_table, table.c[s.id_column] == self.role_table.c[s.user_role_user_id_column])
        return select(table)

    def _user(self, name: str, row: dict) -> CurrentUser:
        s = self.settings
        if s.user_schema_mode == "split":
            raw = "teacher" if name == s.teachers_table else "expert"
        else:
            raw = str(row["_role" if s.user_schema_mode == "related" else s.role_column])
        role = s.role_map.get(raw)
        if role is None:
            raise ValueError(f"Роль {raw} отсутствует в ROLE_MAP.")
        return CurrentUser(id=f"{name}:{row[s.id_column]}", email=str(row[s.email_column]),
                           display_name=str(row[s.display_name_column] or ""), role=role)

    async def _find(self, email: str | None = None, user_id: str | None = None) -> list[tuple[str, dict]]:
        result = []
        async with self.engine.connect() as connection:
            for name, table in self.tables.items():
                statement = self._statement(table)
                if user_id is not None:
                    namespace, raw = user_id.split(":", 1)
                    if name != namespace:
                        continue
                    column = table.c[self.settings.id_column]
                    statement = statement.where(column == column.type.python_type(raw))
                if email is not None:
                    statement = statement.where(func.lower(table.c[self.settings.email_column]) == email.strip().lower())
                rows = (await connection.execute(statement)).mappings().all()
                result.extend((name, dict(row)) for row in rows)
        return result

    async def get_by_email(self, email: str) -> CurrentUser | None:
        rows = await self._find(email=email)
        if len(rows) > 1:
            raise ValueError("Неоднозначный логин или несколько ролей. Уточните отображение БД.")
        return self._user(*rows[0]) if rows else None

    async def get_by_id(self, user_id: str) -> CurrentUser | None:
        rows = await self._find(user_id=user_id)
        if len(rows) > 1:
            raise ValueError("У пользователя несколько ролей. Требуется правило выбора роли.")
        return self._user(*rows[0]) if rows else None

    async def password_hash(self, user_id: str) -> str:
        rows = await self._find(user_id=user_id)
        if len(rows) != 1:
            raise PermissionError("Учётная запись недоступна.")
        return str(rows[0][1][self.settings.password_hash_column])

    async def list_experts(self) -> list[CurrentUser]:
        return [user for name, row in await self._find() if (user := self._user(name, row)).role == "expert"]

    async def list_teachers(self) -> list[CurrentUser]:
        return [user for name, row in await self._find() if (user := self._user(name, row)).role == "teacher"]
