from uuid import uuid4
from urllib.parse import urlencode

import streamlit as st
from sqlalchemy import select

from activity.rpd_bridge import origin_note
from apps.common import composer, portal
from apps.teacher_programs import knowledge_screen, my_disciplines
from apps.presentation import course_progress, status_badge
from domain.export import (OUTLINE_LABEL, block_html, composition_html, export_course, generation_summary,
                           rationale_report_html, source_html, subject_origin)
from pydantic import ValidationError

from domain.export import LEVEL_LABELS
from orchestration.call_log import OUTCOME_LABELS
from orchestration.market_analysis import course_market, parse_market_file
from domain.workload import (HOURS, MAX_HOURS, WorkloadParams, load_norms, proposal_from_loads, propose_workload,
                             topic_hours, validate_workload)
from storage.models import Course, CourseVersion, Interview, Notification, SourceRow
from storage.portal import (REBUILD_STATUSES, acknowledge, create_course, decide, finish_without_answer, rebuild_course,
                            retry_command, save_market)
from storage.workflow import SqlWorkflowStore, model_call_failures


GAP_LABELS = {"missing_tasks": "рабочие задачи и процессы", "missing_stack": "применяемые инструменты",
              "missing_artifacts": "результаты работы", "missing_quality_criteria": "критерии качества",
              "conflicting_evidence": "подтверждение по независимым источникам"}

VERIFICATION_LABELS = {"verified": "Подтверждено", "needs_clarification": "Не подтверждено",
                       "rejected_hallucination": "Противоречит источникам"}


def interview_results(interviews, registry):
    answered = [interview for interview in reversed(interviews) if "verification" in interview.gap_report]
    if not answered:
        return
    with st.expander("Проверка ответов эксперта"):
        st.caption("В программах курса используются подтверждённые сведения.")
        for index, interview in enumerate(answered, 1):
            report = interview.gap_report["verification"]
            received = f" · получен {interview.submitted_at:%d.%m.%Y}" if interview.submitted_at else ""
            st.markdown(f"**Ответ {index}**{received}")
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
        profession = st.text_input("Профессия выпускника", max_chars=200,
                                   placeholder="Если отличается от названия курса, например «Аналитик данных»")
        manual = st.text_area("Обязательные дисциплины", placeholder="По одному предмету в строке. Можно оставить пустым.")
        norms = load_norms()
        st.markdown("**Учебная нагрузка** — сверьте с учебным планом")
        programs = list(norms["program_types"])
        left, right = st.columns(2)
        program_type = left.selectbox("Тип программы", programs, index=programs.index(norms["defaults"]["program_type"]),
                                      format_func=lambda key: norms["program_types"][key]["label"])
        unit = right.radio("Объём задан в", ["з.е.", "часах"], horizontal=True)
        volume = left.number_input("Общий объём курса", min_value=1, max_value=MAX_HOURS,
                                   value=int(norms["program_types"][program_type]["total_credits"]))
        rate = right.number_input("Академических часов в 1 з.е.", min_value=1, max_value=1000,
                                  value=int(norms["defaults"]["hours_per_credit"]))
        semesters = left.number_input("Число семестров", min_value=1, max_value=12, value=int(norms["defaults"]["semesters"]))
        forms = list(norms["assessment_forms"])
        form = right.selectbox("Форма аттестации по дисциплинам", forms, index=forms.index(norms["defaults"]["assessment_form"]),
                               format_func=lambda key: norms["assessment_forms"][key]["label"])
        auto = st.form_submit_button("Добавить предметы автоматически", type="primary")
        submit = st.form_submit_button("Сформировать курс")
    if auto or submit:
        if not all(s.strip() for s in (title, institution, year)):
            st.error("Заполните название курса, вуз и учебный год.")
            return
        subjects = list(dict.fromkeys(s.strip() for s in manual.splitlines() if s.strip()))
        course_id = st.session_state.setdefault("new_request_id", str(uuid4()))
        settings = {"program_type": program_type, "hours_per_credit": int(rate), "semesters": int(semesters),
                    "assessment_form": form}
        settings["total_credits" if unit == "з.е." else "total_hours"] = int(volume)
        meta = {"title": title.strip(), "institution": institution.strip(), "year": year.strip(),
                "expand_subjects": auto or not subjects, "workload": settings}
        if profession.strip():
            meta["profession"] = profession.strip()
        await create_course(factory, actor, course_id, meta, subjects)
        st.session_state.update(selected_course=course_id, screen="course")
        del st.session_state["new_request_id"]
        st.rerun()


def workload_form(subjects, version_id, proposal, params):
    loads = {}
    norms = load_norms()
    forms = list(norms["assessment_forms"])
    with st.expander("Учебная нагрузка по дисциплинам"):
        if not params.explicit:
            st.caption(f"Параметры нагрузки курса не заданы: использован ориентир {params.total_hours} акад. ч "
                       f"по {params.hours_per_credit} ч в з.е. Сверьте объём с учебным планом.")
        else:
            st.caption(f"{norms['program_types'][params.program_type]['label']} · объём курса {params.total_hours} акад. ч "
                       f"({params.total_hours / params.hours_per_credit:g} з.е.) · семестров — {params.semesters}. "
                       "Часы рассчитаны программой по настройкам config/workload_norms.yaml.")
        rate = st.number_input("Академических часов в 1 з.е.", min_value=1, max_value=1000,
                               value=proposal["hours_per_credit"], key=f"workload-rate-{version_id}")
        for subject in subjects:
            key = f"load-{version_id}-{subject['id']}"
            saved = proposal["loads"][subject["id"]]
            st.text(subject["title"])
            if subject.get("importance") and subject["importance"] != "standard":
                st.caption(f"Важность по оценке модели — {subject['importance']}: {subject.get('importance_reason') or ''}")
            columns = st.columns(len(HOURS))
            fields = {field: column.number_input(label, min_value=0, max_value=MAX_HOURS,
                                                  value=saved.get(field, 0), key=f"{key}-{field}")
                      for (field, label), column in zip(HOURS.items(), columns, strict=True)}
            left, right = st.columns(2)
            semester = left.number_input("Семестр", min_value=1, max_value=max(12, params.semesters),
                                         value=int(saved.get("semester") or 1), key=f"{key}-semester")
            current = saved.get("assessment_form") or params.assessment_form
            form = right.selectbox("Форма аттестации", forms, index=forms.index(current) if current in forms else 0,
                                   format_func=lambda item: norms["assessment_forms"][item]["label"], key=f"{key}-form")
            total = sum(fields.values())
            st.caption(f"{total} акад. ч. · {total / rate:.2f} з.е.")
            row = {**saved, **fields, "hours_per_credit": rate, "semester": int(semester), "assessment_form": form}
            if row != saved:
                row["topics"] = topic_hours(subject, row)
                if total % rate == 0:
                    row["credits"] = total // rate
                else:
                    row.pop("credits", None)
            loads[subject["id"]] = row
        total = sum(row[field] for row in loads.values() for field in HOURS)
        st.caption(f"Итого по курсу — {total} акад. ч. · {total / rate:.2f} з.е.")
        issues = validate_workload(subjects, {**proposal, "loads": loads, "hours_per_credit": rate, "total_hours": total},
                                   params)
        if issues:
            st.warning("Проверьте нагрузку до утверждения:\n\n" + "\n".join(f"- {issue}" for issue in issues))
    return loads


def generation_panel(program, failures):
    summary = generation_summary(program)
    if not summary["outline"] and not summary["unknown"] and not failures:
        return
    total = summary["model"] + summary["outline"] + summary["unknown"]
    text = f"Собрано моделью — {summary['model']} из {total}, шаблоном — {summary['outline']}."
    if summary["unknown"]:
        text += f" Для {summary['unknown']} происхождение не записано (старая версия)."
    (st.warning if summary["outline"] else st.caption)(text)
    if summary["reasons"] or failures:
        with st.expander("Почему дисциплины собраны шаблоном"):
            for reason, count in summary["reasons"]:
                st.write(f"{reason} — {count}")
            if failures:
                st.caption("Сбои вызовов модели по журналу")
                for stage, outcome, count in failures:
                    st.write(f"{stage}: {OUTCOME_LABELS.get(outcome, outcome)} — {count}")


def market_report(analysis):
    st.subheader("Навыки рынка и покрытие курсом")
    if analysis.status == "no_data":
        for item in analysis.warnings:
            st.info(item)
        return
    details = [analysis.source, f"выборка — {analysis.sample_size} вакансий" if analysis.sample_size else None,
               f"период {analysis.period}" if analysis.period else None, analysis.region or "регион не указан"]
    st.caption(" · ".join(item for item in details if item))
    if analysis.market_fit is not None:
        st.metric("Покрытие критичных навыков курсом (расчёт)", f"{analysis.market_fit:.0%}")
    for item in analysis.warnings:
        st.warning(item)
    st.dataframe([{"Навык": row.skill,
                   "Доля вакансий": f"{row.share:.0%}" if row.share is not None else "",
                   "Вакансий": row.vacancies, "Вес": row.weight, "Критичный": row.critical,
                   "Покрытие": LEVEL_LABELS[row.level] + (" (вручную)" if row.manual else ""),
                   "Дисциплины": ", ".join(row.subjects), "Приоритет": row.priority} for row in analysis.skills],
                 hide_index=True)
    if analysis.uncovered:
        st.write("Не покрыты курсом: " + ", ".join(analysis.uncovered))
    if analysis.weak_subjects:
        st.write("Слабо связаны с рынком и задачами: " + ", ".join(analysis.weak_subjects))


async def market_actions(actor, factory, course, analysis, subjects):
    with st.expander("Рыночные данные курса"):
        st.caption("JSON или CSV с навыками (skill, weight, vacancies) или вакансиями (title, skills). "
                   "Шаблоны — в папке templates/ репозитория. Цифры считает программа, модель их не задаёт.")
        upload = st.file_uploader("Загрузить файл", type=["json", "csv"], key=f"market-{course.id}")
        if upload is not None and st.button("Сохранить рыночные данные", key=f"market-save-{course.id}"):
            try:
                dataset = parse_market_file(upload.name, upload.getvalue())
            except (ValueError, ValidationError) as error:
                st.error(f"Не удалось прочитать файл: {error}")
            else:
                await save_market(factory, actor, course.id, dataset=dataset.model_dump(mode="json"))
                st.rerun()
        if (course.course_meta.get("market") or {}).get("dataset") and st.button(
                "Удалить загруженные данные", key=f"market-drop-{course.id}"):
            await save_market(factory, actor, course.id, dataset=None)
            st.rerun()
        if analysis.status == "ok" and analysis.skills:
            st.caption("Ручная правка: какие дисциплины покрывают навык.")
            overrides = dict((course.course_meta.get("market") or {}).get("overrides") or {})
            titles = [subject["title"] for subject in subjects]
            skill = st.selectbox("Навык", [row.skill for row in analysis.skills], key=f"market-skill-{course.id}")
            row = next(row for row in analysis.skills if row.skill == skill)
            chosen = st.multiselect("Дисциплины", titles, default=[title for title in row.subjects if title in titles],
                                    key=f"market-subjects-{course.id}-{skill}")
            left, right = st.columns(2)
            if left.button("Сохранить соответствие", key=f"market-override-{course.id}"):
                await save_market(factory, actor, course.id, overrides={**overrides, skill: chosen})
                st.rerun()
            if skill in overrides and right.button("Вернуть расчёт", key=f"market-reset-{course.id}"):
                overrides.pop(skill)
                await save_market(factory, actor, course.id, overrides=overrides)
                st.rerun()


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


def report_panel(version, registry, interviews, analysis=None):
    program = version.content["program"]
    if analysis is not None:
        market_report(analysis)
    if composition := composition_html(version.content.get("composition_check")):
        st.html(composition)
    st.html(rationale_report_html(program, registry))
    audit = program.get("business_audit")
    vacancy_ids = version.content.get("vacancy_source_ids", [])
    if audit or vacancy_ids:
        st.subheader("Оценка курса индустрией")
    if audit:
        st.metric("Мнение модели о покрытии рабочих задач", f"{audit['market_fit_score']:.0%}",
                  help="Оценка модели, а не расчёт. Расчёт кодом — в разделе «Навыки рынка».")
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


def course_workspace(version, registry, interviews, can_review, active, meta=None, analysis=None):
    subjects = version.content["program"]["subjects"]
    labels = ["Программа", "Отчёт по курсу"]
    if can_review:
        labels.append("Учебная нагрузка")
    tabs = st.tabs(labels)
    with tabs[0]:
        for index, subject in enumerate(subjects):
            label = subject["title"]
            if subject_origin(subject) == "outline":
                label += f" · {OUTLINE_LABEL}"
            with st.expander(label, expanded=index == 0):
                st.html(block_html(subject, registry))
    with tabs[1]:
        report_panel(version, registry, interviews, analysis)
    if not can_review:
        return None, None
    previous = version.content.get("workload_proposal")
    if not previous and active:
        previous = proposal_from_loads(active.content.get("workloads", {}))
    params = WorkloadParams.from_meta(meta)
    proposal = propose_workload(subjects, previous, params)
    with tabs[2]:
        return workload_form(subjects, version.id, proposal, params), proposal


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
    (st.success if (course.course_meta.get("activity_model") or {}).get("tasks") else st.caption)(
        origin_note(course.course_meta))
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
                generation_panel(version.content["program"], await model_call_failures(factory, course.id))
                analysis = course_market(course.course_meta, version.content["program"], registry,
                                         version.content.get("vacancy_source_ids", []), version.content.get("activity"))
                loads, proposal = course_workspace(version, registry, interviews, can_review, active, course.course_meta,
                                                   analysis)
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
                if version:
                    await market_actions(actor, factory, course, analysis, version.content["program"]["subjects"])
                if version and course.status in REBUILD_STATUSES:
                    summary = generation_summary(version.content["program"])
                    if summary["outline"] or summary["unknown"]:
                        st.caption("Часть дисциплин собрана шаблоном. Пересборка сформирует все дисциплины моделью; "
                                   "утверждённая версия останется действующей до утверждения новой.")
                    if st.button("Пересобрать курс моделью", width="stretch", key=f"rebuild-{course.id}"):
                        await rebuild_course(factory, actor, course.id)
                        st.rerun()
                if active:
                    market = course_market(course.course_meta, active.content["program"], registry,
                                           active.content.get("vacancy_source_ids", []), active.content.get("activity"))
                    st.download_button(f"Скачать утверждённую версию {active.revision} · ZIP",
                                       export_course(course.course_meta, active, registry, market.model_dump(mode="json")),
                                       f"course-v{active.revision}.zip", "application/zip", width="stretch")
                    if course.last_approved_at:
                        st.caption(f"Утверждён — {course.last_approved_at:%d.%m.%Y}")
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
    if st.sidebar.button("Мои дисциплины в программах", icon=":material/school:", width="stretch",
                         type="secondary" if st.session_state.get("screen") == "programs" else "tertiary"):
        st.session_state.screen = "programs"
        st.rerun()
    if st.sidebar.button("Карта знаний", icon=":material/hub:", width="stretch",
                         type="secondary" if st.session_state.get("screen") == "map" else "tertiary"):
        st.session_state.screen = "map"
        st.rerun()
    screen = st.session_state.get("screen", "list")
    if screen == "programs":
        await my_disciplines(actor, factory)
        return
    if screen == "map":
        await knowledge_screen(factory)
        return
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
