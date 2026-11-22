"""Кабинет методиста: дисциплины, закреплённые руководителем программы, и карта знаний."""

from uuid import uuid4

import streamlit as st

from activity import store
from activity.program import PART_LABELS, analyse
from activity.rpd_bridge import course_snapshot
from apps.activity_ui import demo_banner, status_chip, trend_chip
from apps.head_app import knowledge_map
from storage.portal import create_course


async def program_state(factory, program) -> tuple[dict, dict]:
    async with factory() as db:
        data, contents = await store.program_input(db, program)
    return analyse(data, contents), data.models


async def my_disciplines(actor, factory) -> None:
    st.header("Мои дисциплины в программах")
    st.caption("Руководитель программы закрепил за вами дисциплины. Здесь видно, какие проверенные задачи профессии "
               "покрывает ваша дисциплина. По ним строится РПД.")
    async with factory() as db:
        rows = await store.teacher_disciplines(db, actor.id)
    if not rows:
        st.info("За вами пока не закреплены дисциплины программ. Курс можно создать вручную — «Создать курс».")
        return
    states = {}
    for discipline, program in rows:
        if program.id not in states:
            states[program.id] = await program_state(factory, program)
        analysis, models = states[program.id]
        snapshot = course_snapshot(models, analysis["links"].get(discipline.id, {}),
                                   program={"id": program.id, "name": program.name},
                                   discipline={"id": discipline.id, "name": discipline.name})
        with st.container(border=True, key=f"portal-card-discipline-{discipline.id}"):
            demo_banner(program.is_demo)
            st.subheader(discipline.name)
            st.caption(f"Программа «{program.name}» · {discipline.zet:g} ЗЕТ · семестр {discipline.semester or '—'} · "
                       f"{PART_LABELS.get(discipline.part, discipline.part)}")
            if snapshot["tasks"]:
                st.write(f"Закреплено проверенных задач профессии: {len(snapshot['tasks'])}.")
                for task in snapshot["tasks"]:
                    with st.expander(f"{task['title']} · {task['profession']}"):
                        st.html(status_chip(task["status"]) + trend_chip(task["trend"]))
                        st.write(f"Покрытие: {task['coverage']} — {task['coverage_reason']}")
                        for line in task["reason"][1:]:
                            st.write(f"— {line}")
                        if task["tools"] or task["knowledge"]:
                            st.caption("Знания и инструменты: " + "; ".join([*task["knowledge"], *task["tools"]][:10]))
            else:
                st.warning("Дисциплина не покрывает ни одной проверенной задачи профессии. Добавьте в её содержание "
                           "темы и результаты по задачам модели или обсудите с руководителем программы. РПД можно "
                           "создать по прежней схеме: задачи предложит модель по источникам курса.")
            if discipline.course_id:
                if st.button("Открыть курс", key=f"open-course-{discipline.id}", icon=":material/arrow_forward:"):
                    st.session_state.update(screen="course", selected_course=discipline.course_id)
                    st.rerun()
                continue
            with st.form(f"rpd-{discipline.id}"):
                left, right = st.columns(2)
                institution = left.text_input("Вуз", value=program.institution, max_chars=300)
                year = right.text_input("Учебный год", value=program.academic_year, placeholder="2026/2027", max_chars=20)
                label = "Создать РПД по модели деятельности" if snapshot["tasks"] else "Создать РПД"
                if st.form_submit_button(label, type="primary"):
                    if not institution.strip() or not year.strip():
                        st.error("Заполните вуз и учебный год.")
                    else:
                        await create_rpd(actor, factory, discipline, program, snapshot, institution, year)
                        st.rerun()


async def create_rpd(actor, factory, discipline, program, snapshot: dict, institution: str, year: str) -> None:
    """Курс из одной дисциплины программы: задачи деятельности берутся из модели профессии (раздел 7.14)."""
    meta = {"title": discipline.name, "institution": institution.strip(), "year": year.strip(), "expand_subjects": False,
            "program": {"id": program.id, "name": program.name, "discipline_id": discipline.id}}
    if snapshot["professions"]:
        meta["profession"] = ", ".join(item["name"] for item in snapshot["professions"])
    if snapshot["tasks"]:
        meta["activity_model"] = snapshot
    course_id = str(uuid4())
    await create_course(factory, actor, course_id, meta, [discipline.name])
    await store.link_course(factory, discipline.id, course_id)
    st.session_state.update(screen="course", selected_course=course_id)


async def knowledge_screen(factory) -> None:
    st.header("Карта знаний")
    st.caption("Выберите свою область знаний и посмотрите, в каких профессиях и реальных задачах она нужна. "
               "Каждая задача — готовая идея для проекта, с источником.")
    async with factory() as db:
        models = await store.latest_contents(db)
    knowledge_map(models)
