"""Shared visual elements and saved workflow states for both portals."""

import asyncio
from html import escape
from pathlib import Path

import streamlit as st
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from domain.source_weights import SourceWeights, course_source_weights
from storage.database import make_engine, run_async
from storage.models import Course
from storage.portal import WEIGHT_EDITABLE_STATUSES, WEIGHT_REVIEW_STATUSES, save_source_weights


STATUS = {
    "queued": "В очереди", "queued_audit": "В очереди на аудит",
    "agents_running": "Идёт анализ источников", "gap_analysis": "Проверяем полноту данных",
    "designing": "Формируется программа", "collecting_literature": "Подбираем учебные материалы",
    "applying_source_weights": "Применяем веса источников",
    "waiting_for_expert": "Ждёт интервью эксперта", "answer_queued": "Ответ эксперта принят",
    "waiting_for_review": "Черновик на проверке", "review_queued": "Решение принято в обработку",
    "approved": "Актуален", "pending_update_review": "Обновление на проверке", "failed": "Работа приостановлена",
}
STEPS = ("Источники", "Модель деятельности", "Программа", "Материалы", "Настройка весов", "Проверка", "Утверждение")
POSITIONS = {
    "queued": 0, "queued_audit": 0, "agents_running": 0,
    "gap_analysis": 1, "waiting_for_expert": 1, "answer_queued": 1,
    "designing": 2, "collecting_literature": 3,
    "applying_source_weights": 4,
    "waiting_for_review": 5, "pending_update_review": 5, "review_queued": 5, "approved": 6,
}

WEIGHT_LABELS = {
    "interview": "Интервью эксперта", "books": "Books · книги и учебники",
    "business": "Business · рынок труда", "science": "Science · статьи и документация",
    "university": "University · вузовские РПД",
}
def portal_style():
    st.html(Path(__file__).with_name("portal.css"))


def portal_brand():
    portal_style()
    st.html('<div class="portal-brand"><span class="portal-mark" aria-hidden="true"></span>'
            '<span>Проектирование программ и курсов</span></div>')


def status_badge(status: str) -> str:
    kind = ("ready" if status == "approved" else "attention" if status == "failed"
            else "waiting" if status in {"waiting_for_expert", "waiting_for_review", "pending_update_review"}
            else "progress")
    return f'<span class="portal-badge portal-badge--{kind}">{escape(STATUS.get(status, status))}</span>'


def progress_html(status: str, *, auto_refresh: bool = False, weights_link: bool = False) -> str:
    position = POSITIONS.get(status)
    steps = []
    for index, label in enumerate(STEPS):
        current = position == index and status != "approved"
        complete = status == "approved" or position is not None and index < position
        state = "current" if current else "complete" if complete else "pending"
        marker = ' aria-current="step"' if current else ""
        label = f'<a href="#course-source-weights">{label}</a>' if weights_link and index == 4 else f'<span>{label}</span>'
        steps.append(f'<li class="portal-step portal-step--{state}"{marker}>'
                     f'<span class="portal-step-line" aria-hidden="true"></span>{label}</li>')
    refresh_note = '<p class="portal-progress-note">Статус обновляется автоматически</p>' if auto_refresh else ""
    return ('<section class="portal-progress" aria-label="Подготовка курса">'
            '<div class="portal-progress-head"><strong>Подготовка курса</strong>'
            f'<span role="status">{status_badge(status)}</span></div>'
            f'<ol class="portal-steps">{"".join(steps)}</ol>{refresh_note}</section>')


async def _read_course_state(url: str, course_id: str, owner_id: str):
    engine = make_engine(url)
    try:
        async with engine.connect() as conn:
            result = await conn.execute(select(Course.status, Course.updated_at).where(
                Course.id == course_id, Course.owner_user_id == owner_id,
            ))
            return result.one_or_none()
    finally:
        await engine.dispose()


@st.fragment(run_every=4)
def follow_course(url: str, course: Course):
    # The full page already read the status in its async renderer.
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        return
    try:
        current = run_async(_read_course_state(url, course.id, course.owner_user_id))
    except (SQLAlchemyError, OSError):
        return
    if current != (course.status, course.updated_at):
        st.rerun()


def course_progress(course: Course, *, database_url: str | None = None):
    st.html(progress_html(course.status, auto_refresh=database_url is not None, weights_link=True))
    if database_url is not None:
        follow_course(database_url, course)


async def source_weights_panel(actor, factory, course: Course, *, invitation_id: str | None = None):
    weights = course_source_weights(course.course_meta)
    editable = (bool(invitation_id) and course.status == "waiting_for_expert" if actor.role == "expert"
                else course.status in WEIGHT_EDITABLE_STATUSES)
    note_key = f"source-weights-note-{course.id}"
    if note := st.session_state.pop(note_key, None):
        st.success(note)
    st.html('<div id="course-source-weights"></div>')
    with st.expander("Настройка весов"):
        st.caption("По умолчанию у всех групп 1,0. При расхождении программа опирается на источник с большим весом. "
                   "При весе 0 источник используется для проверки или когда других подтверждений нет.")
        if not editable:
            st.write(" · ".join(f"{label} — {getattr(weights, key):g}" for key, label in WEIGHT_LABELS.items()))
            return
        form_key = f"source-weights-{course.id}-" + "-".join(f"{value:g}" for value in weights.model_dump().values())
        with st.form(form_key):
            values = {key: st.slider(label, 0.0, 1.0, getattr(weights, key), 0.05)
                      for key, label in WEIGHT_LABELS.items()}
            submit = st.form_submit_button("Сохранить и пересчитать черновик" if course.status in WEIGHT_REVIEW_STATUSES
                                           else "Сохранить веса")
        if course.status == "approved":
            st.caption("Веса будут применены при следующей пересборке курса.")
        if submit:
            try:
                recalculating = await save_source_weights(factory, actor, course.id, SourceWeights(**values),
                                                          previous=weights, invitation_id=invitation_id)
            except (ValueError, PermissionError) as error:
                st.warning(str(error))
            else:
                st.session_state[note_key] = ("Веса сохранены. Черновик поставлен в очередь на пересчёт."
                                              if recalculating else "Веса сохранены.")
                st.rerun()
