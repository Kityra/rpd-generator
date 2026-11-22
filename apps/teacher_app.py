from datetime import date
from html import escape
from types import SimpleNamespace
from urllib.parse import urlencode
from uuid import uuid4

import streamlit as st
from sqlalchemy import select

from activity.rpd_bridge import origin_note
from apps.boundary_review import boundary_review
from apps.common import composer, portal
from apps.head_app import render_management
from apps.teacher_programs import knowledge_screen, my_disciplines
from apps.presentation import course_progress, status_badge
from domain.export import (OUTLINE_LABEL, block_html, composition_html, export_course, gap_coverage_html, generation_summary,
                           profession_counts, rationale_report_html, source_html, subject_origin)
from pydantic import ValidationError

from domain.skill_mapper import TOOL_COLUMN_NOTE, competence_label, tool_labels
from orchestration.call_log import OUTCOME_LABELS
from orchestration.course_lifespan import INSUFFICIENT_EVIDENCE, LIFESPAN_UNAVAILABLE_NOTE, SUPPORT_FEED_UNAVAILABLE
from orchestration.market_analysis import course_market, parse_market_file, with_competences
from storage.models import Course, CourseVersion, Interview, Notification, SourceRow
from storage.portal import (REBUILD_STATUSES, acknowledge, create_course, decide, delete_course, finish_without_answer,
                            rebuild_course, retry_command, save_market)
from storage.workflow import SqlWorkflowStore, model_call_failures
from storage.profession_boundaries import course_boundaries, public_boundary


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
            if report.get("interviewer_result", {}).get("classification") == "nonsense_or_offtopic":
                st.caption("Интервьюер запросил конкретный пример." if report.get("redirected")
                           else "Оставшиеся поля переданы методисту.")
                continue
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
        auto = st.form_submit_button("Добавить предметы автоматически", type="primary")
        submit = st.form_submit_button("Сформировать курс")
    if auto or submit:
        if not all(s.strip() for s in (title, institution, year)):
            st.error("Заполните название курса, вуз и учебный год.")
            return
        subjects = list(dict.fromkeys(s.strip() for s in manual.splitlines() if s.strip()))
        course_id = st.session_state.setdefault("new_request_id", str(uuid4()))
        meta = {"title": title.strip(), "institution": institution.strip(), "year": year.strip(),
                "expand_subjects": auto or not subjects}
        if profession.strip():
            meta["profession"] = profession.strip()
        await create_course(factory, actor, course_id, meta, subjects)
        st.session_state.update(selected_course=course_id, screen="course")
        del st.session_state["new_request_id"]
        st.rerun()


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
        with st.expander("Почему дисциплины собраны шаблоном" if summary["outline"] else "Подробности проверок"):
            for reason, count in summary["reasons"]:
                st.write(f"{reason} — {count}")
            if failures:
                st.caption("Сбои вызовов модели по журналу")
                for stage, outcome, count in failures:
                    name = {"llm_judge": "Модель-судья", "lifespan_assessment": "Прогноз актуальности"}.get(stage, stage)
                    st.write(f"{name}: {OUTCOME_LABELS.get(outcome, outcome)} — {count}")


def market_report(analysis):
    st.subheader("Навыки рынка и покрытие курсом")
    if analysis.status == "no_data":
        for item in analysis.warnings:
            st.info(item)
        return
    analysis = with_competences(analysis, {})
    details = [analysis.source, f"выборка — {analysis.sample_size} вакансий" if analysis.sample_size else None,
               f"период {analysis.period}" if analysis.period else None, analysis.region or "регион не указан"]
    st.caption(" · ".join(item for item in details if item))
    if analysis.market_fit is not None:
        st.metric("Покрытие критичных компетенций курсом (расчёт)", f"{analysis.market_fit:.0%}")
    for item in analysis.warnings:
        st.warning(item)
    st.dataframe([{"Компетенция": row.competence_name,
                   "Инструменты": ", ".join(tool_labels(row)) or "—",
                   "Доля вакансий": f"{row.share:.0%}" if row.share is not None else "",
                   "Вакансий": row.vacancies, "Вес": row.weight, "Критичный": row.critical,
                   "Покрытие": "покрыт темами" if row.level == "full" else "не покрыт",
                   "Дисциплина": (", ".join(row.subjects) or "—") if row.level == "full" else "—"}
                  for row in analysis.competences], hide_index=True,
                 column_config={"Критичный": st.column_config.CheckboxColumn("Критичный")})
    if any(not tool_labels(row) or row.expert_tools for row in analysis.competences):
        st.caption(TOOL_COLUMN_NOTE)
    uncovered = [competence_label(row) for row in analysis.competences if row.level != "full"]
    if uncovered:
        st.write("Не покрыты курсом: " + ", ".join(uncovered))
    if analysis.weak_subjects:
        st.write("Слабо связаны с рынком и задачами: " + ", ".join(analysis.weak_subjects))


async def market_actions(actor, factory, course, analysis, subjects):
    with st.expander("Загрузка рыночных данных и настройка соответствий"):
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


def lifespan_unavailable_reason(content: dict) -> str:
    if "lifespan_assessment" not in content:
        return "Эта версия сохранена до появления прогноза актуальности."
    prefix = f"{LIFESPAN_UNAVAILABLE_NOTE} ("
    notices = [content["lifespan_note"]] if content.get("lifespan_note") else content.get("source_notices", [])
    for notice in notices:
        if notice == INSUFFICIENT_EVIDENCE:
            return ("Горизонт пока нельзя рассчитать. Нужна дата окончания поддержки версии из программы "
                    "или заметное изменение доли навыка в сопоставимых выборках вакансий.")
        if notice == SUPPORT_FEED_UNAVAILABLE:
            return "Не удалось получить дату окончания поддержки. Официальный источник временно недоступен."
        if notice == f"{LIFESPAN_UNAVAILABLE_NOTE}: нет рыночных или научных источников.":
            return "Для расчёта не нашлось рыночных или научных источников по курсу."
        if notice.startswith(prefix) and notice.endswith(")."):
            outcome = notice.removeprefix(prefix).removesuffix(").")
            if outcome != "ok" and (reason := OUTCOME_LABELS.get(outcome)):
                return f"Прогноз недоступен. {reason.capitalize()}."
    return "Прогноз недоступен. Причина для этой версии не сохранилась."


async def lifespan_actions(actor, factory, course, content, registry):
    with st.container(border=True, key=f"lifespan-{course.id}"):
        st.subheader("Прогноз актуальности курса")
        if assessment := content.get("lifespan_assessment"):
            years = f"{assessment['estimated_lifespan_years']:g}".replace(".", ",")
            if assessment.get("calculation_method"):
                st.metric("Горизонт до проверки программы, лет", years)
                review_by = date.fromisoformat(assessment["review_by"])
                st.caption(f"Проверить к {review_by:%d.%m.%Y}.")
            else:
                st.metric("Горизонт без пересмотра программы, лет", years)
                st.caption("Сохранённая оценка модели. Расчёт по датам и выборкам появится при обновлении версии.")
            with st.expander("Основания прогноза"):
                st.markdown("**Ключевые факты по рынку и технологиям**")
                items = "".join(f"<li>{escape(point)}</li>" for point in assessment["hr_digest_points"])
                st.html(f"<ul>{items}</ul>")
                evidence = "".join(source_html(registry[key]) for key in dict.fromkeys(assessment["source_ids"])
                                   if key in registry)
                if evidence:
                    st.html(f"<details><summary>Источники</summary>{evidence}</details>")
        else:
            st.caption(lifespan_unavailable_reason(content))

        if actor.role != "teacher" or actor.id != course.owner_user_id:
            return
        confirm = f"confirm-course-deletion-{course.id}"
        if not st.session_state.get(confirm):
            if not st.button("Удалить курс", key=f"delete-{course.id}", width="stretch"):
                return
            st.session_state[confirm] = True
        st.caption("Курс и его версии будут архивированы. Приглашения экспертов станут недоступны.")
        if st.button("Подтвердить удаление", key=f"delete-confirm-{course.id}", width="stretch"):
            try:
                await delete_course(factory, course.id, actor)
            except (PermissionError, ValueError) as error:
                st.warning(str(error))
            else:
                st.session_state.pop(confirm, None)
                st.session_state.pop("selected_course", None)
                st.session_state.screen = "list"
                st.rerun()
        if st.button("Оставить курс", key=f"delete-cancel-{course.id}", width="stretch"):
            st.session_state.pop(confirm, None)
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
    if profile := version.content.get("profession_profile"):
        rejected = [claim for claim in profile["claims"] if claim["verification_status"] == "rejected"]
        if rejected:
            with st.expander("Утверждения, отклонённые экспертом"):
                for claim in rejected:
                    st.text(claim["statement"])
                    st.caption(claim["verification_reason"])
    if coverage := gap_coverage_html(version.content.get("evidence_gaps"), include_methodology=False):
        st.html(coverage)
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
        if not notice.startswith(LIFESPAN_UNAVAILABLE_NOTE):
            st.caption(notice)
    interview_results(interviews, registry)


async def course_workspace(version, registry, interviews, analysis, *, actor, factory, course):
    subjects = version.content["program"]["subjects"]
    tabs = st.tabs(["Программа", "Отчёт по курсу"])
    with tabs[0]:
        for index, subject in enumerate(subjects):
            label = subject["title"]
            if subject_origin(subject) == "outline":
                label += f" · {OUTLINE_LABEL}"
            with st.expander(label, expanded=index == 0):
                st.html(block_html(subject, registry))
    with tabs[1]:
        report_panel(version, registry, interviews, analysis)
        await market_actions(actor, factory, course, analysis, subjects)


async def review_actions(actor, factory, course, draft, active):
    st.subheader("Правки и утверждение")
    with st.container():
        change = composer("Что изменить или уточнить", f"edit-{draft.id}")
    if change:
        await decide(factory, actor, course.id, draft.id, "revise", *change)
        st.rerun()
    if st.button("Утвердить изменения" if active else "Утвердить курс", type="primary", width="stretch"):
        await decide(factory, actor, course.id, draft.id, "approve")
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
        boundaries = [public_boundary(row) for row in await course_boundaries(db, course.id)]
    version = draft or active
    await boundary_review(actor, settings, factory, course.id, boundaries)
    can_review = bool(draft and course.status in {"waiting_for_review", "pending_update_review"})
    waiting = bool(interviews and course.status == "waiting_for_expert")
    has_actions = waiting or version is not None

    with st.container(key="course-workspace"):
        main, side = st.columns([1.85, 1], gap="medium") if has_actions else (st.container(), None)
        with main.container(key="course-program"):
            if version:
                label = "Черновик" if draft else "Программа курса"
                st.subheader(f"{label} · версия {version.revision}")
                gap_types = version.content.get("evidence_gaps", {}).get("gap_types", [])
                if gap_types:
                    gaps = ", ".join(GAP_LABELS.get(kind, kind) for kind in gap_types)
                    st.warning(f"Требуют проверки методиста — {gaps}.")
                generation_panel(version.content["program"], await model_call_failures(factory, course.id))
                if profile := version.content.get("profession_profile"):
                    counts = profession_counts(profile)
                    st.caption(f"Профиль профессии · подтверждено {counts['verified']} · "
                               f"требует проверки методиста {counts['needs_review']} · отклонено {counts['rejected']}")
                analysis = course_market(course.course_meta, version.content["program"], registry,
                                         version.content.get("vacancy_source_ids", []), version.content.get("activity"),
                                         mappings=version.content.get("skill_mappings"), saved=version.content.get("market_analysis"),
                                         profile=version.content.get("profession_profile"))
                await course_workspace(version, registry, interviews, analysis, actor=actor, factory=factory, course=course)
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
                    report = interviews[0].gap_report
                    profile = report.get("profession_profile") or {}
                    choices = {claim["claim_id"]: claim["statement"] for claim in profile.get("claims", [])
                               if claim["verification_status"] == "needs_review"}
                    selected = st.multiselect("Включить в черновик для проверки", list(choices),
                                              format_func=choices.get, key=f"profile-check-{interviews[0].id}") if choices else []
                    manual_tasks = []
                    handoff = report.get("methodologist_handoff")
                    if handoff:
                        st.info(report["prompt"])
                    if profile.get("sample_adequacy") == "empty_external_data" or report.get("requires_manual_input"):
                        st.info("Укажите известные рабочие задачи." if handoff else
                                "Для профиля нет внешних данных. Укажите известные вам рабочие задачи или пригласите эксперта.")
                        template = st.text_area("Рабочие задачи методиста", key=f"manual-{interviews[0].id}",
                                                placeholder="Одна задача на строку\nДействие → объект → проверяемый результат",
                                                max_chars=16000)
                        manual_tasks = template.splitlines()
                    if st.button("Сформировать черновик по имеющимся данным", type="primary", width="stretch"):
                        try:
                            await finish_without_answer(factory, actor, course.id, interviews[0].id,
                                                        acknowledged_claim_ids=selected, manual_tasks=manual_tasks)
                        except ValueError as error:
                            st.warning(str(error))
                        else:
                            st.rerun()
                    st.caption("Сборка продолжится без интервью. Оставшиеся пробелы будут отмечены в черновике.")
                    if not handoff:
                        await expert_actions(actor, settings, factory, users, course, interviews[0])
                if can_review:
                    await review_actions(actor, factory, course, draft, active)
                if version:
                    await lifespan_actions(actor, factory, course, version.content, registry)
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
                                           active.content.get("vacancy_source_ids", []), active.content.get("activity"),
                                           mappings=active.content.get("skill_mappings"), saved=active.content.get("market_analysis"),
                                           profile=active.content.get("profession_profile"))
                    export_version = SimpleNamespace(status=active.status, revision=active.revision,
                        content={key: value for key, value in active.content.items()
                                 if key not in {"lifespan_assessment", "lifespan_market_sample", "lifespan_note"}})
                    st.download_button(f"Скачать утверждённую версию {active.revision} · ZIP",
                                       export_course(course.course_meta, export_version, registry, market.model_dump(mode="json")),
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
        courses = (await db.scalars(select(Course).where(Course.owner_user_id == actor.id, Course.status != "cancelled")
                                    .order_by(Course.updated_at.desc()))).all()
        notices = (await db.scalars(select(Notification).where(Notification.user_id == actor.id)
                                    .order_by(Notification.created_at.desc()))).all()
    count = sum(n.read_at is None for n in notices)
    if requested := st.query_params.get("profession"):
        st.session_state.update(screen="management", profession_id=requested, head_screen="model")
        del st.query_params["profession"]
    if requested := st.query_params.get("course"):
        if any(course.id == requested for course in courses):
            st.session_state.update(screen="course", selected_course=requested)
        del st.query_params["course"]
    st.sidebar.text(actor.display_name)
    screen = st.session_state.get("screen", "list")
    if st.sidebar.button(f"Ваши курсы ({count})" if count else "Ваши курсы", icon=":material/folder_open:",
                         type="secondary" if screen == "list" else "tertiary", width="stretch"):
        st.session_state.screen = "list"
        st.rerun()
    if st.sidebar.button("Создать курс", type="primary", icon=":material/add:", width="stretch"):
        st.session_state.screen = "create"
        st.rerun()
    if st.sidebar.button("Мои дисциплины в программах", icon=":material/school:", width="stretch",
                         type="secondary" if screen == "programs" else "tertiary"):
        st.session_state.screen = "programs"
        st.rerun()
    if st.sidebar.button("Карта знаний", icon=":material/hub:", width="stretch",
                         type="secondary" if screen == "map" else "tertiary"):
        st.session_state.screen = "map"
        st.rerun()
    if st.sidebar.button("Управление программами", icon=":material/account_tree:", width="stretch",
                         type="secondary" if screen == "management" else "tertiary"):
        st.session_state.screen = "management"
        st.rerun()
    screen = st.session_state.get("screen", "list")
    if screen == "management":
        await render_management(actor, factory, users)
        return
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
                with column.container(border=True, key=f"portal-card-course-{course.id}"):
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
    portal("Кабинет методиста", {"teacher", "admin"}, render)


if __name__ == "__main__":
    main()
