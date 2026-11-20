from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8-sig", extra="ignore")

    app_database_url: str = "sqlite+aiosqlite:///./data/application.sqlite"
    user_database_url: str = ""
    teachers_index_id: str = ""
    experts_index_id: str = ""
    yandex_folder_id: str = ""
    yandex_api_key: SecretStr | None = None
    user_schema_mode: Literal["single", "split", "related"] = "single"
    users_table: str = "users"
    teachers_table: str = "teachers"
    experts_table: str = "experts"
    user_db_schema: str | None = None
    id_column: str = "id"
    email_column: str = "email"
    password_hash_column: str = "password_hash"
    display_name_column: str = "display_name"
    role_column: str = "role"
    user_roles_table: str = "user_roles"
    user_role_user_id_column: str = "user_id"
    user_role_value_column: str = "role"
    role_map: dict[str, str] = Field(default_factory=lambda: {
        "teacher": "teacher", "expert": "expert", "admin": "admin",
    })
    hash_scheme: str = "auto"
    allow_rehash_on_login: bool = False
    session_signing_key: SecretStr
    app_env: Literal["development", "production"] = "development"
    checkpoint_url: str = "data/checkpoints.sqlite"
    public_origin: str = "http://localhost:8501"
    business_coverage_threshold: float = Field(default=0.6, ge=0, le=1)
    science_coverage_threshold: float = Field(default=0.6, ge=0, le=1)
    max_interview_rounds: int = Field(default=3, ge=1, le=10)
    interview_skill_directory: str = "skills/interview_methods"

    @model_validator(mode="after")
    def validate_auth(self) -> "Settings":
        if len(self.session_signing_key.get_secret_value()) < 32:
            raise ValueError("SESSION_SIGNING_KEY должен содержать минимум 32 символа.")
        if self.allow_rehash_on_login:
            raise ValueError("Rehash требует отдельного подтверждённого legacy-адаптера записи.")
        if self.app_env == "production" and not self.public_origin.startswith("https://"):
            raise ValueError("Для production требуется HTTPS PUBLIC_ORIGIN.")
        return self
