import logging
from pathlib import Path

import streamlit as st
from openai import APIError
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import async_sessionmaker

from activity.documents import ATTACHMENT_MAX_MB, read_attachment
from auth.password_verifier import ExistingPasswordVerifier
from auth.session import AuthService, require_role
from auth.settings import Settings
from auth.user_repository import SqlUserRepository
from auth.yandex_users import CombinedUserRepository, DirectoryError, LocalUserRepository, YandexUserRepository
from apps.presentation import portal_brand
from run_rpd import open_yandex
from storage.database import DatabaseSetupError, ensure_schema, make_engine, run_async


@st.cache_resource(ttl=30, show_spinner=False)
def directory(teacher_id, expert_id, heads_id, folder, _key):
    return YandexUserRepository(lambda: open_yandex({"YANDEX_FOLDER_ID": folder, "YANDEX_API_KEY": _key}, timeout=15),
                                teacher_id, expert_id, heads_id)


@st.cache_resource(ttl=30, show_spinner=False)
def local_directory(path: str, _modified: float):
    return LocalUserRepository(path)


def local_users(settings: Settings) -> LocalUserRepository | None:
    """Локальный файл учётных записей — только для разработки и демо (APP_ENV=development)."""
    if not settings.local_users_file or settings.app_env != "development":
        return None
    path = Path(settings.local_users_file)
    if not path.is_file():
        raise DirectoryError(f"Файл LOCAL_USERS_FILE не найден: {path}. Скопируйте local_users.example.txt.")
    return local_directory(str(path.resolve()), path.stat().st_mtime)


def composer(label, key):
    msg = st.chat_input(label, key=key, accept_file="multiple", file_type=["txt", "pdf", "docx"],
                        max_upload_size=ATTACHMENT_MAX_MB)
    if not msg:
        return None
    if len(msg.files) > 3 or len(msg.text) > 16000:
        raise ValueError("Одно сообщение может содержать до 16 000 символов и трёх файлов.")
    files = [read_attachment(f.name, f.getvalue()) for f in msg.files]
    if any(not f["text"] for f in files):
        raise ValueError("В одном из файлов отсутствует читаемый текст.")
    return msg.text.strip(), files


async def authenticated(roles, render, account):
    settings = Settings()
    engine = make_engine(settings.app_database_url)
    user_engine = None
    try:
        repositories = []
        if settings.teachers_index_id or settings.experts_index_id:
            if not settings.yandex_api_key or not settings.yandex_folder_id:
                raise DirectoryError("Для входа заполните YANDEX_API_KEY и YANDEX_FOLDER_ID.")
            repositories.append(directory(settings.teachers_index_id, settings.experts_index_id,
                                          settings.program_heads_index_id, settings.yandex_folder_id,
                                          settings.yandex_api_key.get_secret_value()))
        elif settings.user_database_url:
            user_engine = make_engine(settings.user_database_url)
            sql_users = SqlUserRepository(user_engine, settings)
            await sql_users.initialize()
            repositories.append(sql_users)
        if (local := local_users(settings)) is not None:
            repositories.append(local)
        if not repositories:
            raise DirectoryError("Укажите TEACHERS_INDEX_ID и EXPERTS_INDEX_ID в .env "
                                 "(или LOCAL_USERS_FILE для локальной разработки).")
        users = repositories[0] if len(repositories) == 1 else CombinedUserRepository(repositories)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        auth = AuthService(settings, users, ExistingPasswordVerifier(settings.hash_scheme), factory)
        actor = await require_role(auth, roles, account=account)
        await render(actor, settings, factory, users)
    finally:
        if user_engine is not None:
            await user_engine.dispose()
        await engine.dispose()


def portal(title, roles, render):
    with st.container(key="portal-header", horizontal=True, horizontal_alignment="distribute",
                      vertical_alignment="center", gap="small"):
        portal_brand()
        account = st.container(key="portal-account", horizontal=True, horizontal_alignment="right",
                               vertical_alignment="center", width="content", gap="small")
    st.title(title)
    try:
        ensure_schema(Settings().app_database_url)
        run_async(authenticated(roles, render, account))
    except (DirectoryError, DatabaseSetupError) as error:
        st.error(str(error))
    except ValidationError:
        st.error("Проверьте настройки приложения в .env, включая SESSION_SIGNING_KEY.")
    except (SQLAlchemyError, OSError) as error:
        cause = getattr(error, "orig", None)
        logging.error("Хранилище кабинета: %s (%s)", type(error).__name__, getattr(cause, "sqlstate", type(cause).__name__))
        st.error("База приложения недоступна. Проверьте APP_DATABASE_URL в .env и подключение к базе.")
    except (APIError, UnicodeError):
        st.error("Не удалось прочитать справочник пользователей в Yandex Cloud. Проверьте доступ и формат файла.")
    except ValueError as error:
        st.error(str(error))
