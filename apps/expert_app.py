import hashlib

import streamlit as st
from sqlalchemy import select

from activity.llm import build_extractor as extractor
from apps.common import composer, portal
from apps.expert_survey import survey
from apps.presentation import course_progress
from auth.session import aware
from orchestration.interviewer_turn import CATEGORY_SLOT
from storage.models import InterviewMessage, now
from storage.portal import assigned_interviews
from storage.workflow import SqlWorkflowStore


async def render(actor, settings, factory, users):
    st.sidebar.text(actor.display_name)
    survey_tab, interview_tab = st.tabs(["Анкета модели деятельности", "Интервью по курсам"])
    with survey_tab:
        await survey(actor, factory, extractor(factory))
    with interview_tab:
        await interviews(actor, factory, database_url=settings.app_database_url)


async def interviews(actor, factory, *, database_url=None):
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
        course_progress(course, database_url=database_url)
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
            decisions, valid = profession_decisions(interview)
            report = interview.gap_report
            more = report.get("interview_can_continue", report.get("interview_turn", 1) < report.get("interview_max_turns", 1))
            continue_interview = more and st.checkbox("Запросить следующий вопрос", key=f"more-{interview.id}")
            label = "Ответить и продолжить интервью" if continue_interview else "Завершить интервью и передать в курс"
            if st.button(label, type="primary", disabled=not valid or not (pending or decisions)):
                content, files = pending or ("", [])
                await SqlWorkflowStore(factory).submit_interview(
                    "", actor, interview.interrupt_id, inv.id,
                    content or ("Ответ в приложенных материалах." if files else "Решения по профилю профессии."),
                    files, invitation_id=inv.id, profession_decisions=decisions, continue_interview=continue_interview)
                st.session_state.pop(pending_key, None)
                st.rerun()
        else:
            st.success("Ответ сохранён. Методист увидит результаты проверки в карточке курса.")
            verification = interview.gap_report.get("verification", {})
            if (verification.get("interviewer_result", {}).get("classification") == "nonsense_or_offtopic"
                    and not verification.get("redirected")):
                st.info("Оставшиеся поля переданы методисту.")
        st.button("Обновить интервью", type="tertiary")


def profession_decisions(interview):
    report = interview.gap_report
    if not report.get("profession_profile"):
        return [], True
    st.caption(f"Реплика №{report['interview_turn']} · до {report['interview_max_turns']} содержательных ответов")
    labels = {None: "Без решения", "confirm": "Подтвердить", "edit": "Исправить", "reject": "Отклонить"}
    allowed = set(report["review_candidate_ids"])
    decisions, valid = [], True
    for claim in report["profession_profile"]["claims"]:
        if claim["claim_id"] not in allowed:
            continue
        key = f"{interview.id}-{claim['claim_id']}"
        with st.expander(claim["statement"]):
            for ref in claim["evidence"]:
                st.text(ref["quote_or_metric"])
            action = st.selectbox("Решение по утверждению", list(labels), format_func=labels.get, key=key)
            statement = st.text_input("Уточнённая формулировка", key=f"edit-{key}", max_chars=2000).strip() if action == "edit" else None
            if action:
                valid = valid and (action != "edit" or bool(statement))
                decisions.append({"claim_id": claim["claim_id"], "action": action, "statement": statement})
    categories = {"work_task": "Рабочая задача", "duty_area": "Обязанность", "work_artifact": "Результат работы",
                  "tool_stack": "Инструмент", "domain_constraint": "Условие или критерий качества"}
    if "requested_slots" in report:
        categories = {category: label for category, label in categories.items()
                      if CATEGORY_SLOT[category] in report["requested_slots"]}
    if not categories:
        return decisions, valid
    category = st.selectbox("Что вы подтверждаете", list(categories), format_func=categories.get,
                            key=f"expert-category-{interview.id}")
    tasks = st.text_area("Подтверждаемые сведения", placeholder="Одно утверждение на строку. Назовите действие и результат.",
                         key=f"expert-tasks-{interview.id}", max_chars=16000)
    for index, line in enumerate(filter(None, (item.strip() for item in tasks.splitlines())), 1):
        decisions.append({"claim_id": str(index), "action": "add", "statement": line, "category": category})
        valid = valid and len(line) <= 2000
    if len(decisions) > 40:
        valid = False
        st.warning("За одну реплику можно передать до 40 решений и новых задач.")
    return decisions, valid


def main():
    portal("Кабинет эксперта", {"expert"}, render)


if __name__ == "__main__":
    main()
