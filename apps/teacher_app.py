from uuid import uuid4
from urllib.parse import urlencode

import streamlit as st
from sqlalchemy import select

from apps.common import composer, portal
from domain.export import HOURS, block_html, course_export_block, export_course
from storage.models import Course, CourseVersion, Interview, Notification, SourceRow
from storage.portal import acknowledge, create_course, decide, recheck_official_rpd, retry_command
from storage.workflow import SqlWorkflowStore


STATUS = {"queued": "В очереди", "queued_audit": "В очереди на аудит", "agents_running": "Идёт анализ источников",
          "gap_analysis": "Проверяем полноту данных", "designing": "Формируется программа",
          "waiting_for_expert": "Ждёт интервью эксперта", "answer_queued": "Ответ эксперта принят",
          "waiting_for_review": "Черновик на проверке", "review_queued": "Решение принято в обработку",
          "approved": "Актуален", "pending_update_review": "Требует проверки после автообновления",
          "failed": "Работа приостановлена"}


async def new_course(actor, factory):
    with st.form("new_course"):
        title = st.text_input("Курс или профессия", max_chars=300)
        institution = st.text_input("Вуз", max_chars=300)
        year = st.text_input("Учебный год", placeholder="2026/2027", max_chars=20)
        manual = st.text_area("Обязательные дисциплины", placeholder="По одному предмету в строке. Можно оставить пустым.")
        auto = st.form_submit_button("✨ Добавить предметы автоматически")
        submit = st.form_submit_button("Сформировать курс")
    if auto or submit:
        if not all(s.strip() for s in (title, institution, year)):
            st.error("Заполните название курса, вуз и учебный год.")
            return
        subjects = list(dict.fromkeys(s.strip() for s in manual.splitlines() if s.strip()))
        course_id = st.session_state.setdefault("new_request_id", str(uuid4()))
        await create_course(factory, actor, course_id, {"title": title.strip(), "institution": institution.strip(),
                            "year": year.strip(), "expand_subjects": auto or not subjects}, subjects)
        st.session_state.update(selected_course=course_id, screen="course")
        del st.session_state["new_request_id"]
        st.rerun()


def workload_form(subjects, version_id, previous):
    loads = {}
    with st.expander("Учебная нагрузка по дисциплинам"):
        st.caption("Отметьте дисциплины с утверждёнными часами. Сумму часов и ЗЕ приложение рассчитает при экспорте.")
        for subject in subjects:
            key = f"load-{version_id}-{subject['id']}"
            saved = previous.get(subject["id"], {})
            if st.checkbox(subject["title"], key=key, value=bool(saved)):
                fields = {}
                for field, label in HOURS.items():
                    fields[field] = st.number_input(label, min_value=0, max_value=10000, value=saved.get(field, 0), key=f"{key}-{field}")
                rate = st.number_input("Часов в одной ЗЕ", min_value=1, max_value=1000,
                                       value=saved.get("hours_per_credit", None), key=key + "-rate")
                loads[subject["id"]] = {**fields, "hours_per_credit": rate}
    return loads


async def course_detail(actor, settings, factory, users, course, notices):
    st.header(course.course_meta["title"])
    st.write(STATUS.get(course.status, course.status))
    for notice in notices:
        if notice.course_id == course.id:
            st.info(notice.content)
    if any(n.course_id == course.id and n.read_at is None for n in notices):
        await acknowledge(factory, actor, course.id)
    async with factory() as db:
        registry = {r.source_id: r.record for r in (await db.scalars(select(SourceRow).where(SourceRow.course_id == course.id))).all()}
        interviews = (await db.scalars(select(Interview).where(Interview.course_id == course.id)
                                      .order_by(Interview.created_at.desc()))).all()
        active = await db.get(CourseVersion, course.active_version_id) if course.active_version_id else None
        draft = await db.get(CourseVersion, course.draft_version_id) if course.draft_version_id else None
    if interviews:
        for interview in interviews:
            claims = interview.gap_report.get("verification", {}).get("claims", [])
            if claims:
                with st.expander("Результаты проверки интервью"):
                    for claim in claims:
                        st.write(claim["claim"], "—", claim["verification_status"])
                        st.caption(claim["explanation"])
                        for key in claim["source_ids"]:
                            if key in registry:
                                st.write(registry[key]["title"])
    if course.status == "waiting_for_expert" and interviews:
        current = interviews[0]
        st.subheader("Вопрос эксперту")
        st.write(current.gap_report["prompt"])
        experts = await users.list_experts()
        if experts:
            choice = st.selectbox("Пригласить эксперта", experts, format_func=lambda e: e.display_name)
            if st.button("Сформировать ссылку-приглашение"):
                token = await SqlWorkflowStore(factory).invite(current.id, choice.id, actor, users)
                st.session_state[f"invite-{current.id}"] = settings.public_origin.rstrip("/") + "/expert?" + urlencode({"token": token})
            if link := st.session_state.get(f"invite-{current.id}"):
                st.code(link, language=None)
                st.caption("Передайте ссылку выбранному эксперту. Приглашение также появится в его кабинете.")
        else:
            st.info("В справочнике нет экспертов.")
    version = draft or active
    activity = version.content.get("activity") if version else None
    if not activity and interviews:
        activity = interviews[0].gap_report.get("activity")
    if activity:
        with st.expander("Модель прикладной деятельности"):
            for task in activity.get("tasks", []):
                st.write(task["title"])
                if task.get("artifacts"):
                    st.caption("Результаты: " + ", ".join(task["artifacts"]))

    async def show_official_check(item):
        if reason := course_export_block(course.course_meta, item.content, registry):
            st.warning(f"Официальная проверка версии {item.revision} приостановлена. {reason}")
            audit = item.content.get("official_audit") or {}
            lookups = [audit.get("lookup"), *audit.get("lookups", {}).values()]
            for detail in dict.fromkeys(entry["reason"] for entry in lookups if entry and entry.get("reason")):
                st.caption(detail)
            if st.button("Перепроверить материалы РПД", key=f"official-rpd-{item.id}"):
                await recheck_official_rpd(factory, actor, course.id, item.id)
                st.info("Проверка поставлена в очередь. Результат появится после обновления статуса.")
            return False
        for note in (item.content.get("official_audit") or {}).get("notices", []):
            st.caption(note)
        return True

    export_available = False
    if version:
        export_available = await show_official_check(version)
        program = version.content["program"]
        st.subheader(f"Программа курса · версия {version.revision}")
        for subject in program["subjects"]:
            with st.expander(subject["title"]):
                st.html(block_html(subject, registry))
        audit = program.get("business_audit")
        if audit:
            with st.expander("Оценка курса индустрией"):
                st.metric("Соответствие требованиям рынка", f"{audit['market_fit_score']:.0%}")
                st.write(audit["evidence_summary"])
                for metric in audit["metrics"]:
                    st.write(metric["name"], metric["value"], metric["unit"], metric["region"])
                if audit["unavailable_metrics"]:
                    st.caption("Нет подтверждённых измерений — " + ", ".join(audit["unavailable_metrics"]))
    if draft and course.status in {"waiting_for_review", "pending_update_review"}:
        update = bool(active)
        loads = workload_form(draft.content["program"]["subjects"], draft.id, active.content.get("workloads", {}) if active else {})
        if st.button("Утвердить изменения" if update else "Утвердить курс", type="primary"):
            await decide(factory, actor, course.id, draft.id, "approve", workloads=loads)
            st.rerun()
        if update and st.button("Отклонить изменения"):
            await decide(factory, actor, course.id, draft.id, "reject")
            st.rerun()
        with st.container():
            change = composer("Что изменить или уточнить", f"edit-{draft.id}")
        if change:
            await decide(factory, actor, course.id, draft.id, "revise", *change)
            st.rerun()
    if active:
        if draft:
            export_available = await show_official_check(active)
        if export_available:
            st.download_button(f"Скачать утверждённую версию {active.revision} · ZIP",
                               export_course(course.course_meta, active, registry), f"course-v{active.revision}.zip", "application/zip")
        if course.next_audit_at:
            st.caption(f"Следующий аудит — {course.next_audit_at:%d.%m.%Y}")
    if course.status == "failed" and st.button("Повторить обработку"):
        await retry_command(factory, actor, course.id)
        st.rerun()
    st.button("Обновить статус")


async def render(actor, settings, factory, users):
    async with factory() as db:
        courses = (await db.scalars(select(Course).where(Course.owner_user_id == actor.id).order_by(Course.updated_at.desc()))).all()
        notices = (await db.scalars(select(Notification).where(Notification.user_id == actor.id)
                                    .order_by(Notification.created_at.desc()))).all()
    count = sum(n.read_at is None for n in notices)
    if requested := st.query_params.get("course"):
        if any(course.id == requested for course in courses):
            st.session_state.update(screen="course", selected_course=requested)
        del st.query_params["course"]
    st.sidebar.write(actor.display_name)
    if st.sidebar.button(f"Ваши курсы ({count})" if count else "Ваши курсы"):
        st.session_state.screen = "list"
    if st.sidebar.button("Создать курс"):
        st.session_state.screen = "create"
    screen = st.session_state.get("screen", "list")
    if screen == "create":
        await new_course(actor, factory)
        return
    if screen == "course":
        current = next((c for c in courses if c.id == st.session_state.get("selected_course")), None)
        if current:
            await course_detail(actor, settings, factory, users, current, notices)
            return
    st.subheader("Ваши курсы")
    if not courses:
        st.info("Курсов пока нет. Выберите «Создать курс» в меню.")
    for course in courses:
        if st.button(course.course_meta["title"], key=f"open-{course.id}"):
            st.session_state.update(screen="course", selected_course=course.id)
            st.rerun()
        st.caption(STATUS.get(course.status, course.status))
        for notice in notices:
            if notice.course_id == course.id and notice.read_at is None:
                st.info(notice.content)


def main():
    portal("Кабинет преподавателя", {"teacher", "admin"}, render)


if __name__ == "__main__":
    main()
