"""Анкета эксперта по модели деятельности (раздел 7.5): одна задача — одна карточка, два раунда Дельфи."""

import time
from collections import Counter

import streamlit as st
from sqlalchemy import select

from activity import store
from activity.experts import COMPANY_SIZE_LABELS, DEMAND_LABELS, FORECAST_LABELS, ROLE_KIND_LABELS, VERDICT_LABELS
from apps.activity_ui import quotes_html
from storage.activity_models import ActivityExpert, ActivityItem, ExpertVote, Profession


async def survey(actor, factory, extractor) -> None:
    async with factory() as db:
        rows = (await db.execute(select(ActivityExpert, Profession).join(Profession, Profession.id == ActivityExpert.profession_id)
                                 .where(ActivityExpert.user_id == actor.id).order_by(Profession.name))).all()
    if not rows:
        st.info("Вас пока не пригласили оценить модель деятельности. Приглашение отправляет методист.")
        return
    names = {expert.id: profession.name for expert, profession in rows}
    if st.session_state.get("survey_expert") not in names:
        st.session_state["survey_expert"] = next(iter(names))
    chosen = st.selectbox("Профессия", list(names), format_func=names.get, key="survey_expert")
    expert, profession = next((e, p) for e, p in rows if e.id == chosen)
    submitted = set(expert.submitted_rounds or [])
    current = (profession.meta or {}).get("delphi_round", 1)
    round_number = 1 if 1 not in submitted else 2 if current == 2 and 2 not in submitted else None
    if round_number is None:
        st.success("Спасибо! Ваши ответы сохранены. Если методист запустит следующий раунд, анкета появится здесь.")
        return
    cards = await store.expert_cards(factory, expert.id, round_number)
    if not cards:
        st.info("Карточек для оценки пока нет: модель ещё собирается из источников.")
        return
    async with factory() as db:
        items = {item.id: item for item in (await db.scalars(select(ActivityItem).where(ActivityItem.id.in_(cards)))).all()}
        own = {vote.item_id for vote in (await db.scalars(select(ExpertVote).where(
            ExpertVote.expert_id == expert.id, ExpertVote.round == round_number))).all()}
        version = await store.latest_version(db, profession.id)
        colleagues = (await db.scalars(select(ExpertVote).where(ExpertVote.round == 1, ExpertVote.item_id.in_(cards),
                                                                ExpertVote.expert_id != expert.id))).all() \
            if round_number == 2 else []
    content = version.content if version else {"tasks": {}, "functions": []}
    pending = [key for key in cards if key in items and key not in own]
    done = len(cards) - len(pending)
    st.subheader(f"Раунд {round_number} · «{profession.name}»")
    st.progress(done / len(cards), text=f"Карточка {min(done + 1, len(cards))} из {len(cards)} · около "
                                         f"{max(1, round(len(pending) * 0.5))} мин осталось")
    if round_number == 1:
        st.caption("Отвечайте по своему опыту. Ответы коллег и статусы задач в первом раунде не показываются.")
    else:
        st.caption("Второй раунд: спорные задачи и задачи, предложенные экспертами. Видно распределение ответов коллег "
                   "(обезличено). Можно изменить свой ответ.")
    if pending:
        await card(factory, expert, items[pending[0]], content, round_number, colleagues)
    else:
        await final_form(actor, factory, expert, round_number, extractor)


async def card(factory, expert, item, content: dict, round_number: int, colleagues: list) -> None:
    started = st.session_state.setdefault(f"card-start-{expert.id}-{item.id}-{round_number}", time.monotonic())
    task = content["tasks"].get(item.id)
    function = next((f["name"] for f in content.get("functions", []) if task and f["id"] == task["function_id"]), None)
    with st.container(border=True, key="survey-card"):
        st.markdown(f"### {item.name}")
        if function and function != "Прочее (требует разметки)":
            st.caption(f"Функция: {function}")
        quotes = task["quotes"][:2] if task else []
        if quotes:
            st.html(quotes_html(quotes))
        if round_number == 2:
            votes = Counter(vote.verdict for vote in colleagues if vote.item_id == item.id)
            if votes:
                st.markdown("**Ответы коллег в первом раунде**")
                st.dataframe([{"Ответ": VERDICT_LABELS[key], "Экспертов": votes.get(key, 0)} for key in VERDICT_LABELS],
                             hide_index=True, width="stretch")
            comments = (task or {}).get("metrics", {}).get("experts", {}).get("comments", [])
            for comment in comments[:5]:
                st.caption(f"Комментарий коллеги: «{comment}»")
        key = f"{expert.id}-{item.id}-{round_number}"
        verdict = st.radio(f"Это часть работы специалиста?", list(VERDICT_LABELS), format_func=VERDICT_LABELS.get,
                           index=None, key=f"verdict-{key}")
        yes = verdict in ("regular", "sometimes")
        importance = st.select_slider("Насколько важно для новичка в первые 2 года?", options=[1, 2, 3, 4, 5], value=3,
                                      disabled=not yes, key=f"importance-{key}",
                                      help="1 — почти не нужно, 5 — без этого не работают")
        forecast = st.radio("Через 3 года эта задача…", list(FORECAST_LABELS), format_func=FORECAST_LABELS.get, index=None,
                            horizontal=True, disabled=verdict == "dont_know", key=f"forecast-{key}")
        wording = st.radio("Формулировка точная?", ["yes", "no"], format_func={"yes": "да", "no": "нет"}.get, index=0,
                           horizontal=True, key=f"wording-{key}")
        suggestion = st.text_input("Ваша формулировка", key=f"suggestion-{key}") if wording == "no" else ""
        comment = st.text_input("Комментарий (необязательно)", key=f"comment-{key}")
        if st.button("Ответить и дальше", type="primary", disabled=verdict is None, width="stretch", key=f"next-{key}"):
            await store.save_vote(factory, expert.id, item.id, round_number, {
                "verdict": verdict, "importance": importance if yes else None,
                "forecast": forecast if verdict != "dont_know" else None, "wording_ok": wording == "yes",
                "wording_suggestion": suggestion, "comment": comment,
                "elapsed_ms": int((time.monotonic() - started) * 1000)})
            st.rerun()


async def final_form(actor, factory, expert, round_number: int, extractor) -> None:
    st.success("Все карточки раунда оценены. Осталось несколько вопросов о роли.")
    with st.form(f"final-{expert.id}-{round_number}"):
        profile = {}
        if round_number == 1:
            profile["missing_text"] = st.text_area(
                "Чего не хватает в списке?",
                help="Задачи, которые специалист делает, но которых не было в карточках. По одной в строке, "
                     "лучше с глаголом: «Проверять код, сгенерированный ИИ-ассистентами».")
            profile["role_forecast_text"] = st.text_input("Во что превратится эта роль через 3–5 лет?")
            profile["role_demand"] = st.radio("Спрос на эту роль через 3–5 лет", list(DEMAND_LABELS),
                                              format_func=DEMAND_LABELS.get, horizontal=True, index=1)
            st.markdown("**Коротко о вас** (в отчётах только агрегаты)")
            left, right = st.columns(2)
            profile["years"] = left.number_input("Стаж в профессии, лет", 0, 60, int(expert.years or 3))
            profile["role_kind"] = right.selectbox("Роль", list(ROLE_KIND_LABELS), format_func=ROLE_KIND_LABELS.get)
            profile["company_size"] = left.selectbox("Размер компании", list(COMPANY_SIZE_LABELS),
                                                     format_func=COMPANY_SIZE_LABELS.get)
            profile["company_kind"] = right.text_input("Тип компании", placeholder="продуктовая, аутсорс, банк…")
            profile["region"] = st.text_input("Регион")
        if st.form_submit_button("Завершить раунд", type="primary"):
            with st.spinner("Сохраняем ответы"):
                result = await store.finish_round(factory, actor, expert.id, round_number, profile, extractor)
            if result.get("missing"):
                st.toast("Ваши предложения добавлены в модель и попадут во второй раунд")
            st.rerun()
