import asyncio
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import streamlit as st
from sqlalchemy.ext.asyncio import async_sessionmaker
from streamlit.components.v1 import declare_component

from auth.settings import Settings
from orchestration.clients import YandexModelClient
from run_rpd import yandex_settings
from storage.database import DatabaseSetupError, ensure_schema, make_engine
from storage.profession_boundaries import submit_boundary_turn, validate_boundary_invite


read_invite_fragment = declare_component("boundary_invite", path=str(Path(__file__).with_name("boundary_link")))


def _token_from_input(value: str) -> str:
    value = value.strip()
    if "#" not in value:
        return value
    try:
        return parse_qs(urlsplit(value).fragment).get("invite", [""])[0]
    except ValueError:
        return ""


async def guest_interview(settings: Settings, token: str) -> None:
    engine = make_engine(settings.app_database_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        boundary = await validate_boundary_invite(factory, token)
        if boundary is None:
            st.error("Приглашение недоступно или истекло.")
            return
        st.subheader(f"{boundary['primary_profession']} и {boundary['adjacent_profession']}")
        if boundary["interview_completed"]:
            if note := st.session_state.pop("boundary_clarification", None):
                st.info(note)
            st.success("Интервью завершено. Методист получил результаты.")
            return
        st.caption(f"Реплика {boundary['turn_count'] + 1} из 3")
        st.write("Укажите, какие из этих пунктов относятся к каждой роли, какие общие и какой результат передаётся между ролями.")
        for item in boundary["overlapping_items"]:
            st.text(f"• {item}")
        if note := st.session_state.pop("boundary_clarification", None):
            st.info(note)
        with st.form(f"boundary-{boundary['source_interview_id']}-{boundary['turn_count']}"):
            answer = st.text_area("Ваш ответ", max_chars=16000)
            finish = st.checkbox("Завершить интервью после этого ответа")
            sent = st.form_submit_button("Передать ответ методисту", type="primary")
        if sent:
            try:
                _, note = await submit_boundary_turn(factory, token, answer, YandexModelClient(yandex_settings()),
                                                     settings, expected_turn=boundary["turn_count"], finish=finish)
            except (PermissionError, ValueError) as error:
                st.warning(str(error))
            else:
                st.session_state["boundary_clarification"] = note
                st.rerun()
    finally:
        await engine.dispose()


def main():
    st.title("Интервью по границам профессий")
    token = read_invite_fragment(default=None, key="boundary_invite_fragment")
    if not token:
        pasted = st.text_input("Гостевое приглашение", type="password",
                               help="Вставьте ссылку целиком или код в конце ссылки.")
        token = _token_from_input(pasted)
    if not token:
        return
    try:
        settings = Settings()
        ensure_schema(settings.app_database_url)
        asyncio.run(guest_interview(settings, token))
    except (DatabaseSetupError, ValueError):
        st.error("Интервью временно недоступно.")
