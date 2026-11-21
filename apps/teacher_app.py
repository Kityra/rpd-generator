from uuid import uuid4
from urllib.parse import urlencode

import streamlit as st
from sqlalchemy import select

from apps.common import composer, portal
from apps.presentation import course_progress, status_badge
from domain.export import block_html, export_course, rationale_report_html, source_html
from domain.workload import HOURS, MAX_HOURS, proposal_from_loads, propose_workload
from storage.models import Course, CourseVersion, Interview, Notification, SourceRow
from storage.portal import acknowledge, create_course, decide, finish_without_answer, retry_command
from storage.workflow import SqlWorkflowStore


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
    st.header("Новый курс")
    with st.form("new_course"):
        title = st.text_input("Курс или профессия", max_chars=300)
        institution = st.text_input("Вуз", max_chars=300)
        year = st.text_input("Учебный год", placeholder="2026/2027", max_chars=20)
        manual = st.text_area("Обязательные дисциплины", placeholder="По одному предмету в строке. Можно оставить пустым.")
        auto = st.form_submit_button("Добавить предметы автоматически", type="primary")
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


def workload_form(subjects, version_id, proposal):
    loads = {}
    with st.expander("Учебная нагрузка по дисциплинам"):
        if proposal["origin"] == "default":
            st.caption("Черновой ориентир для всего курса — 60 з.е. по 36 акад. ч. Сверьте объём с учебным планом.")
        else:
            st.caption("Черновая нагрузка. Проверьте значения перед утверждением курса.")
        rate = st.number_input("Академических часов в 1 з.е.", min_value=1, max_value=1000,
                               value=proposal["hours_per_credit"], key=f"workload-rate-{version_id}")
        for subject in subjects:
            key = f"load-{version_id}-{subject['id']}"
            saved = proposal["loads"][subject["id"]]
            st.text(subject["title"])
            columns = st.columns(len(HOURS))
            fields = {field: column.number_input(label, min_value=0, max_value=MAX_HOURS,
                                                  value=saved.get(field, 0), key=f"{key}-{field}")
                      for (field, label), column in zip(HOURS.items(), columns, strict=True)}
            total = sum(fields.values())
            st.caption(f"{total} акад. ч. · {total / rate:.2f} з.е.")
            loads[subject["id"]] = {**fields, "hours_per_credit": rate}
        total = sum(row[field] for row in loads.values() for field in HOURS)
        st.caption(f"Итого по курсу — {total} акад. ч. · {total / rate:.2f} з.е.")
    return loads


def course_context(meta):
    return " · ".join(str(meta[key]) for key in ("institution", "year") if meta.get(key))


async def expert_actions(actor, settings, factory, users, course, interview):
    st.subheader("Интервью эксперта")
    with st.chat_message("assistant"):
        st.write(interview.gap_report["prompt"])
    experts = await users.list_experts()
    if experts:
        choice = st.selectbox("Пригласить эксперта", experts, format_func=lambda e: e.display_name)
        if st.button("Сформировать ссылку-приглашение", width="stretch"):
            token = await SqlWorkflowStore(factory).invite(interview.id, choice.id, actor, users)
            st.session_state[f"invite-{interview.id}"] = settings.public_origin.rstrip("/") + "/expert?" + urlencode({"token": token})
        if link := st.session_state.get(f"invite-{interview.id}"):
            st.code(link, language=None)
            st.caption("Передайте ссылку выбранному эксперту. Приглашение также появится в его кабинете.")
    else:
        st.info("В справочнике нет экспертов.")


def report_panel(version, registry, interviews):
    program = version.content["program"]
    st.html(rationale_report_html(program, registry))
    audit = program.get("business_audit")
    vacancy_ids = version.content.get("vacancy_source_ids", [])
    if audit or vacancy_ids:
        st.subheader("Оценка курса индустрией")
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
    for notice in version.content.get("source_notices", []):
        st.caption(notice)
    interview_results(interviews, registry)


def course_workspace(version, registry, interviews, can_review, active):
    subjects = version.content["program"]["subjects"]
    labels = ["Программа", "Отчёт по курсу"]
    if can_review:
        labels.append("Учебная нагрузка")
    tabs = st.tabs(labels)
    with tabs[0]:
        for index, subject in enumerate(subjects):
            with st.expander(subject["title"], expanded=index == 0):
                st.html(block_html(subject, registry))
    with tabs[1]:
        report_panel(version, registry, interviews)
    if not can_review:
        return None, None
    previous = version.content.get("workload_proposal")
    if not previous and active:
        previous = proposal_from_loads(active.content.get("workloads", {}))
    proposal = propose_workload(subjects, previous)
    with tabs[2]:
        return workload_form(subjects, version.id, proposal), proposal


async def review_actions(actor, factory, course, draft, active, loads, proposal):
    st.subheader("Правки и утверждение")
    with st.container():
        change = composer("Что изменить или уточнить", f"edit-{draft.id}")
    if change:
        await decide(factory, actor, course.id, draft.id, "revise", *change,
                     workloads=loads if loads != proposal["loads"] else None)
        st.rerun()
    if st.button("Утвердить изменения" if active else "Утвердить курс", type="primary", width="stretch"):
        await decide(factory, actor, course.id, draft.id, "approve", workloads=loads)
        st.rerun()
    if st.button("Обновить материалы и обоснования", width="stretch"):
        await decide(factory, actor, course.id, draft.id, "refresh_materials")
        st.rerun()
    if active and st.button("Отклонить изменения", width="stretch"):
        await decide(factory, actor, course.id, draft.id, "reject")
        st.rerun()


async def course_detail(actor, settings, factory, users, course, notices):
    st.header(course.course_meta["title"])
    if details := course_context(course.course_meta):
        st.caption(details)
    course_progress(course, database_url=settings.app_database_url)
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
    version = draft or active
    can_review = bool(draft and course.status in {"waiting_for_review", "pending_update_review"})
    waiting = bool(interviews and course.status == "waiting_for_expert")
    has_actions = waiting or can_review or active or (version and course.status == "failed")

    with st.container(key="course-workspace"):
        main, side = st.columns([1.85, 1], gap="medium") if has_actions else (st.container(), None)
        with main.container(key="course-program"):
            if version:
                label = "Черновик РПД" if draft else "Программа курса"
                st.subheader(f"{label} · версия {version.revision}")
                gap_types = version.content.get("evidence_gaps", {}).get("gap_types", [])
                if gap_types:
                    gaps = ", ".join(GAP_LABELS.get(kind, kind) for kind in gap_types)
                    st.warning(f"Требуют проверки преподавателя — {gaps}.")
                loads, proposal = course_workspace(version, registry, interviews, can_review, active)
            else:
                st.subheader("Программа курса")
                if waiting:
                    st.write("Программа появится после ответа эксперта или по имеющимся данным, если вы продолжите без интервью.")
                elif course.status == "failed":
                    if st.button("Повторить обработку", type="primary"):
                        await retry_command(factory, actor, course.id)
                        st.rerun()
                else:
                    st.write("Программа появится здесь после подготовки.")
                interview_results(interviews, registry)
        if side is not None:
            with side.container(key="course-actions"):
                if waiting:
                    st.subheader("Продолжить подготовку")
                    if st.button("Сформировать черновик по имеющимся данным", type="primary", width="stretch"):
                        await finish_without_answer(factory, actor, course.id, interviews[0].id)
                        st.rerun()
                    st.caption("Сборка продолжится без интервью. Оставшиеся пробелы будут отмечены в черновике.")
                    await expert_actions(actor, settings, factory, users, course, interviews[0])
                if can_review:
                    await review_actions(actor, factory, course, draft, active, loads, proposal)
                if active:
                    st.download_button(f"Скачать утверждённую версию {active.revision} · ZIP",
                                       export_course(course.course_meta, active, registry), f"course-v{active.revision}.zip",
                                       "application/zip", width="stretch")
                    if course.next_audit_at:
                        st.caption(f"Следующий аудит — {course.next_audit_at:%d.%m.%Y}")
                if version and course.status == "failed" and st.button("Повторить обработку", type="primary"):
                    await retry_command(factory, actor, course.id)
                    st.rerun()


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
    st.sidebar.text(actor.display_name)
    if st.sidebar.button(f"Ваши курсы ({count})" if count else "Ваши курсы", width="stretch"):
        st.session_state.screen = "list"
    if st.sidebar.button("Создать курс", type="primary", icon=":material/add:", width="stretch"):
        st.session_state.screen = "create"
    screen = st.session_state.get("screen", "list")
    if courses and screen != "list":
        st.sidebar.markdown("**Курсы**")
        for course in courses:
            selected = screen == "course" and course.id == st.session_state.get("selected_course")
            if st.sidebar.button(course.course_meta["title"], key=f"nav-{course.id}",
                                 type="secondary" if selected else "tertiary", width="stretch"):
                st.session_state.update(screen="course", selected_course=course.id)
                st.rerun()
    if screen == "create":
        await new_course(actor, factory)
        return
    if screen == "course":
        current = next((c for c in courses if c.id == st.session_state.get("selected_course")), None)
        if current:
            await course_detail(actor, settings, factory, users, current, notices)
            return
    st.header("Ваши курсы")
    if not courses:
        st.info("Курсов пока нет. Выберите «Создать курс» в меню.")
    with st.container(key="course-workspace"):
        for offset in range(0, len(courses), 2):
            for column, course in zip(st.columns(2, gap="medium"), courses[offset:offset + 2]):
                with column.container(border=True):
                    st.html(status_badge(course.status))
                    st.subheader(course.course_meta["title"])
                    if info := course_context(course.course_meta):
                        st.caption(info)
                    if st.button("Открыть курс", key=f"open-{course.id}", icon=":material/arrow_forward:"):
                        st.session_state.update(screen="course", selected_course=course.id)
                        st.rerun()
                    for notice in notices:
                        if notice.course_id == course.id and notice.read_at is None:
                            st.info(notice.content)


def main():
    portal("Кабинет преподавателя", {"teacher", "admin"}, render)


if __name__ == "__main__":
    main()
