import asyncio
import logging
from dataclasses import asdict

import streamlit as st
from openai import APIError
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import async_sessionmaker

from auth.password_verifier import ExistingPasswordVerifier
from auth.session import AuthService, require_role
from auth.settings import Settings
from auth.user_repository import SqlUserRepository
from auth.yandex_users import DirectoryError, YandexUserRepository
from apps.presentation import portal_brand
from run_rpd import open_yandex
from storage.database import DatabaseSetupError, ensure_schema, make_engine
from web_preview import read_attachment


@st.cache_resource(ttl=30, show_spinner=False)
def directory(teacher_id, expert_id, folder, _key):
    repo = YandexUserRepository(lambda: open_yandex({"YANDEX_FOLDER_ID": folder, "YANDEX_API_KEY": _key}, timeout=15),
                                teacher_id, expert_id)
    return repo


def composer(label, key):
    msg = st.chat_input(label, key=key, accept_file="multiple", file_type=["txt", "pdf", "docx"], max_upload_size=10)
    if not msg:
        return None
    if len(msg.files) > 3 or len(msg.text) > 16000:
        raise ValueError("Одно сообщение может содержать до 16 000 символов и трёх файлов.")
    files = [asdict(read_attachment(f.name, f.getvalue())) for f in msg.files]
    if any(not f["text"] for f in files):
        raise ValueError("В одном из файлов отсутствует читаемый текст.")
    return msg.text.strip(), files


async def authenticated(roles, render):
    settings = Settings()
    engine = make_engine(settings.app_database_url)
    user_engine = None
    try:
        if settings.teachers_index_id or settings.experts_index_id:
            if not settings.yandex_api_key or not settings.yandex_folder_id:
                raise DirectoryError("Для входа заполните YANDEX_API_KEY и YANDEX_FOLDER_ID.")
            users = directory(settings.teachers_index_id, settings.experts_index_id, settings.yandex_folder_id,
                              settings.yandex_api_key.get_secret_value())
        elif settings.user_database_url:
            user_engine = make_engine(settings.user_database_url)
            users = SqlUserRepository(user_engine, settings)
            await users.initialize()
        else:
            raise DirectoryError("Укажите TEACHERS_INDEX_ID и EXPERTS_INDEX_ID в .env.")
        factory = async_sessionmaker(engine, expire_on_commit=False)
        auth = AuthService(settings, users, ExistingPasswordVerifier(settings.hash_scheme), factory)
        actor = await require_role(auth, roles)
        await render(actor, settings, factory, users)
    finally:
        if user_engine is not None:
            await user_engine.dispose()
        await engine.dispose()


def portal(title, roles, render):
    portal_brand()
    st.title(title)
    try:
        ensure_schema(Settings().app_database_url)
        asyncio.run(authenticated(roles, render))
    except (DirectoryError, DatabaseSetupError) as error:
        st.error(str(error))
    except ValidationError:
        st.error("Проверьте настройки приложения в .env, включая SESSION_SIGNING_KEY.")
    except SQLAlchemyError as error:
        cause = getattr(error, "orig", None)
        logging.error("Хранилище кабинета: %s (%s)", type(error).__name__, getattr(cause, "sqlite_errorname", type(cause).__name__))
        st.error("База приложения недоступна. Проверьте APP_DATABASE_URL в .env и доступ к файлу базы.")
    except (APIError, UnicodeError):
        st.error("Не удалось прочитать справочник пользователей в Yandex Cloud. Проверьте доступ и формат файла.")
    except (ValueError, PermissionError) as error:
        st.error(str(error))
    except OSError as error:
        logging.error("Файл базы приложения: %s", error)
        st.error("Не удалось открыть файл базы приложения. Проверьте APP_DATABASE_URL в .env и права на папку data.")
