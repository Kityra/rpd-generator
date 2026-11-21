from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url

from storage.database import DEFAULT_APP_DATABASE_URL, LEGACY_CHECKPOINT_PATH, postgres_dsn


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8-sig", extra="ignore",
                                      hide_input_in_errors=True)

    app_database_url: str = Field(default=DEFAULT_APP_DATABASE_URL, repr=False)
    user_database_url: str = ""
    teachers_index_id: str = ""
    experts_index_id: str = ""
    # Справочник руководителей образовательных программ (третий файл в Yandex AI Studio, формат как у методистов).
    program_heads_index_id: str = ""
    # Локальный файл учётных записей для разработки и демо (только при APP_ENV=development). Формат — local_users.example.txt.
    local_users_file: str = ""
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
        "teacher": "teacher", "methodologist": "teacher", "expert": "expert", "admin": "admin", "program_head": "program_head",
    })
    hash_scheme: str = "auto"
    allow_rehash_on_login: bool = False
    session_signing_key: SecretStr
    app_env: Literal["development", "production"] = "development"
    checkpoint_url: str = Field(default="", repr=False)
    public_origin: str = "http://localhost:8501"
    business_coverage_threshold: float = Field(default=0.6, ge=0, le=1)
    science_coverage_threshold: float = Field(default=0.6, ge=0, le=1)
    gap_weak_weight: float = Field(default=0.4, ge=0, le=1)
    gap_freshness_half_life_years: float = Field(default=5, gt=0)
    gap_target_independent_sources: int = Field(default=3, ge=1)
    gap_covered_threshold: float = Field(default=0.7, gt=0, le=1)
    boundary_conflict_threshold: float = Field(default=0.6, gt=0, le=1)
    boundary_violation_threshold: float = Field(default=0.5, gt=0, le=1)
    interview_skill_directory: str = "skills/interview_methods"
    max_invalid_redirects: int = Field(default=2, ge=0, le=10)
    literature_parser_enabled: bool = True
    literature_use_llm: bool = False
    # Повтор генерации дисциплины при таймауте или паузе выключателя, секунды через запятую.
    draft_retry_delays: str = "30,120,300"
    draft_concurrency: int = Field(default=3, ge=1, le=16)
    public_vacancies_enabled: bool = True
    employer_careers_enabled: bool = True

    def retry_delays(self) -> list[float]:
        return [float(part) for part in self.draft_retry_delays.split(",") if part.strip()]

    @model_validator(mode="after")
    def validate_auth(self) -> "Settings":
        backend = make_url(self.app_database_url).get_backend_name()
        if not self.checkpoint_url:
            self.checkpoint_url = (postgres_dsn(self.app_database_url) if backend in {"postgres", "postgresql"}
                                   else LEGACY_CHECKPOINT_PATH)
        if len(self.session_signing_key.get_secret_value()) < 32:
            raise ValueError("SESSION_SIGNING_KEY должен содержать минимум 32 символа.")
        if self.allow_rehash_on_login:
            raise ValueError("Rehash требует отдельного подтверждённого legacy-адаптера записи.")
        if self.app_env == "production" and not self.public_origin.startswith("https://"):
            raise ValueError("Для production требуется HTTPS PUBLIC_ORIGIN.")
        return self
