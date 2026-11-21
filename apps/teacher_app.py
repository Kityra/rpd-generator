from uuid import uuid4
from urllib.parse import urlencode

import streamlit as st
from sqlalchemy import select

from apps.common import composer, portal
from domain.export import HOURS, block_html, export_course, rationale_report_html, source_html
from storage.models import Course, CourseVersion, Interview, Notification, SourceRow
from storage.portal import acknowledge, create_course, decide, finish_without_answer, retry_command
from storage.workflow import SqlWorkflowStore


STATUS = {"queued": "В очереди", "queued_audit": "В очереди на аудит", "agents_running": "Идёт анализ источников",
          "gap_analysis": "Проверяем полноту данных", "designing": "Формируется программа",
          "collecting_literature": "University подбирает литературу по темам",
          "waiting_for_expert": "Ждёт интервью эксперта", "answer_queued": "Ответ эксперта принят",
          "waiting_for_review": "Черновик на проверке", "review_queued": "Решение принято в обработку",
          "approved": "Актуален", "pending_update_review": "Требует проверки после автообновления",
          "failed": "Работа приостановлена"}

GAP_LABELS = {"missing_tasks": "рабочие задачи и процессы", "missing_stack": "применяемые инструменты",
              "missing_artifacts": "результаты работы", "missing_quality_criteria": "критерии качества",
              "conflicting_evidence": "подтверждение по независимым источникам"}

VERIFICATION_LABELS = {"verified": "Подтверждено", "needs_clarification": "Не подтверждено",
                       "rejected_hallucination": "Противоречит источникам"}


def interview_results(interviews, registry):
    reports = [interview.gap_report["verification"] for interview in reversed(interviews)
               if "verification" in interview.gap_report]
    if not reports:
        return
    with st.expander("Проверка ответов эксперта"):
        st.caption("В программах курса используются подтверждённые сведения.")
        for index, report in enumerate(reports, 1):
            st.markdown(f"**Ответ {index}**")
            if not report["claims"]:
                reason = ("Проверку ответа не удалось завершить." if report.get("assessment") == "unavailable"
                          else "В ответе не выделено проверяемых утверждений.")
                st.warning(reason)
            for claim in report["claims"]:
                status = claim["verification_status"]
                st.write(f"**{VERIFICATION_LABELS[status]}**")
                st.write(claim["claim"])
                st.caption(claim["explanation"])
                for key in claim["source_ids"]:
                    if key in registry:
                        st.write(registry[key]["title"])


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
    if course.status == "waiting_for_expert" and interviews:
        current = interviews[0]
        if st.button("Сформировать черновик по имеющимся данным", type="primary"):
            await finish_without_answer(factory, actor, course.id, current.id)
            st.rerun()
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
    if version:
        program = version.content["program"]
        label = "Черновик РПД" if draft else "Программа курса"
        st.subheader(f"{label} · версия {version.revision}")
        gap_types = version.content.get("evidence_gaps", {}).get("gap_types", [])
        if gap_types:
            gaps = ", ".join(GAP_LABELS.get(kind, kind) for kind in gap_types)
            st.warning(f"Требуют проверки преподавателя — {gaps}.")
        for notice in version.content.get("source_notices", []):
            st.caption(notice)
        audit = program.get("business_audit")
        vacancy_ids = version.content.get("vacancy_source_ids", [])
        with st.expander("Отчёт по курсу"):
            st.html(rationale_report_html(program, registry))
            if audit or vacancy_ids:
                st.markdown("## Оценка курса индустрией")
                if audit:
                    st.metric("Покрытие рабочих задач курсом (оценка модели)", f"{audit['market_fit_score']:.0%}")
                    st.write(audit["evidence_summary"])
                    for metric in audit["metrics"]:
                        st.write(metric["name"], metric["value"], metric["unit"], metric["region"])
                    if audit["unavailable_metrics"]:
                        st.caption("Нет подтверждённых измерений — " + ", ".join(audit["unavailable_metrics"]))
                if vacancy_ids:
                    st.write("Вакансии из выборки")
                    for key in vacancy_ids:
                        if record := registry.get(key):
                            st.html(source_html(record))
                            st.caption("\n".join(record["content"].splitlines()[:2]))
        for index, subject in enumerate(program["subjects"]):
            with st.expander(subject["title"], expanded=index == 0):
                st.html(block_html(subject, registry))
    interview_results(interviews, registry)
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
