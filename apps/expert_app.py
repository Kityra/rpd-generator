import hashlib

import streamlit as st
from sqlalchemy import select

from activity.llm import build_extractor as extractor
from apps.common import composer, portal
from apps.expert_survey import survey
from apps.presentation import course_progress
from auth.session import aware
from storage.models import InterviewMessage, now
from storage.portal import assigned_interviews
from storage.workflow import SqlWorkflowStore


async def render(actor, settings, factory, users):
    st.sidebar.text(actor.display_name)
    survey_tab, interview_tab = st.tabs(["Анкета модели деятельности", "Интервью по курсам"])
    with survey_tab:
        await survey(actor, factory, extractor(factory))
    with interview_tab:
        await interviews(actor, factory)


async def interviews(actor, factory):
    token = st.query_params.get("token")
    async with factory() as db:
        if token:
            digest = hashlib.sha256(token.encode()).hexdigest()
            target = (await db.execute(assigned_interviews(actor, token_hash=digest))).first()
            if not target or (target[0].status == "active" and aware(target[0].expires_at) <= now()):
                st.error("Приглашение недоступно или срок его действия истёк.")
                return
            st.session_state["expert_course"] = target[2].id
            del st.query_params["token"]
        rows = (await db.execute(assigned_interviews(actor))).all()
        rows = [(inv, interview, course) for inv, interview, course in rows
                if inv.status == "used" or aware(inv.expires_at) > now()]
        if not rows:
            st.info("У вас пока нет приглашений на интервью.")
            st.button("Обновить")
            return
        courses = {c.id: c.course_meta["title"] for _, _, c in rows}
        if st.session_state.get("expert_course") not in courses:
            st.session_state["expert_course"] = next(iter(courses))
        chosen = st.selectbox("Интервью по курсу", list(courses), format_func=courses.get, key="expert_course")
        chat = [r for r in reversed(rows) if r[2].id == chosen]
        inv, interview, course = chat[-1]
        course_progress(course)
        with st.container(key="expert-dialog"):
            st.subheader("Диалог с интервьюером")
            for _, entry, _ in chat:
                with st.chat_message("assistant"):
                    st.write(entry.gap_report.get("prompt"))
                messages = (await db.scalars(select(InterviewMessage).where(InterviewMessage.interview_id == entry.id)
                                            .order_by(InterviewMessage.created_at))).all()
                for message in messages:
                    with st.chat_message("user"):
                        st.write(message.content)
                        if message.attachments:
                            st.caption(", ".join(f["name"] for f in message.attachments))
    with st.container():
        if inv.status == "active" and interview.status == "invited":
            pending_key = f"draft-answer-{interview.id}"
            reply = composer("Ваш ответ и материалы к вопросу", f"answer-{interview.id}")
            if reply:
                st.session_state[pending_key] = reply
            pending = st.session_state.get(pending_key)
            if pending:
                content, files = pending
                with st.chat_message("user"):
                    st.write(content)
                    if files:
                        st.caption(", ".join(f["name"] for f in files))
            if st.button("Завершить интервью и передать в курс", type="primary", disabled=not pending):
                content, files = pending
                await SqlWorkflowStore(factory).submit_interview("", actor, interview.interrupt_id, inv.id,
                            content or "Ответ в приложенных материалах.", files, invitation_id=inv.id)
                del st.session_state[pending_key]
                st.rerun()
        else:
            st.success("Ответ сохранён. Преподаватель увидит результаты проверки в карточке курса.")
        st.button("Обновить интервью", type="tertiary")


def main():
    portal("Кабинет эксперта", {"expert"}, render)


if __name__ == "__main__":
    main()
