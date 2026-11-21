"""Кабинет руководителя образовательной программы (и методиста): модели деятельности, эксперты, программа."""

from collections import Counter
from html import escape

import streamlit as st
from sqlalchemy import select

from activity import store
from activity.config import load_config
from activity.documents import parse_vacancies, read_document
from activity.engine import SOURCE_TYPE_LABELS, STATUS_LABELS, TREND_LABELS, percent, plural
from activity.experts import DEMAND_LABELS, Vote, changed_share
from activity.llm import build_extractor as extractor
from activity.program import PART_LABELS, RULE_LABELS, SIGNAL_LABELS, find_profession, retention, signal_lines
from activity.report import (CONFIDENCE_LABELS, DECISION_LABELS, DISCLAIMER, justification_document, model_document,
                             model_json, program_document, program_json)
from apps.activity_ui import (demo_banner, kpis, needed_hint, quotes_html, status_bar, task_detail, task_label)
from apps.common import portal
from storage.activity_models import (ActivityExpert, ActivityMergeLog, ActivitySource, ExpertVote, Profession, Program,
                                     ProgramDiscipline, Recommendation)


SCREENS = {"professions": ("Профессии", ":material/work:"), "model": ("Модель деятельности", ":material/account_tree:"),
           "sources": ("Источники", ":material/library_books:"), "experts": ("Эксперты", ":material/groups:"),
           "program": ("Моя программа", ":material/school:"), "signal": ("Сигнал по профилю", ":material/trending_down:"),
           "map": ("Карта знаний", ":material/hub:"), "queue": ("Очередь методиста", ":material/rule:"),
           "export": ("Экспорт", ":material/download:")}
DOC_TYPES = ("employer_doc", "expert", "forecast", "science", "rpd")


async def professions_list(factory):
    async with factory() as db:
        rows = (await db.scalars(select(Profession).order_by(Profession.is_demo.desc(), Profession.name))).all()
        versions = {row.id: await store.latest_version(db, row.id) for row in rows}
        experts = Counter((await db.scalars(select(ActivityExpert.profession_id))).all())
    return rows, versions, experts


def choose_profession(rows) -> Profession | None:
    if not rows:
        st.info("Профессий пока нет. Создайте её на экране «Профессии».")
        return None
    ids = [row.id for row in rows]
    if st.session_state.get("profession_id") not in ids:
        st.session_state["profession_id"] = ids[0]
    names = {row.id: row.name + (" · ДЕМО" if row.is_demo else "") for row in rows}
    chosen = st.selectbox("Профессия", ids, format_func=names.get, key="profession_id")
    return next(row for row in rows if row.id == chosen)


async def screen_professions(actor, factory):
    st.header("Профессии")
    st.caption("Одна проверенная модель деятельности на профессию — её видят руководитель программы, методист и преподаватели.")
    rows, versions, experts = await professions_list(factory)
    for offset in range(0, len(rows), 2):
        for column, row in zip(st.columns(2, gap="medium"), rows[offset:offset + 2]):
            version = versions[row.id]
            with column.container(border=True):
                st.subheader(row.name)
                details = [row.region or "регион не указан", f"горизонт {row.horizon_years} г.",
                           ", ".join(row.standard_codes or []) or "ПС не указан"]
                st.caption(" · ".join(details) + (" · ДЕМО: синтетические данные" if row.is_demo else ""))
                if version:
                    statuses = version.counts.get("statuses", {})
                    tasks = sum(count for key, count in statuses.items() if key != "control")
                    st.html(status_bar(statuses))
                    st.write(f"Версия {version.number} на {version.as_of:%d.%m.%Y}: подтверждено "
                             f"{statuses.get('confirmed', 0) + statuses.get('confirmed_early', 0)} из {tasks} задач, "
                             f"спорных {statuses.get('disputed', 0)}. Экспертов: приглашено {experts.get(row.id, 0)}, "
                             f"надёжных {version.counts.get('experts', 0)}.")
                else:
                    st.write("Модель ещё не собрана: загрузите источники.")
                if st.button("Открыть модель", key=f"open-{row.id}", icon=":material/arrow_forward:"):
                    st.session_state.update(profession_id=row.id, head_screen="model")
                    st.rerun()
    with st.expander("Новая профессия", expanded=not rows, icon=":material/add:"):
        with st.form("new-profession"):
            name = st.text_input("Название", placeholder="Например, «Backend-разработчик»", max_chars=300)
            left, right = st.columns(2)
            region = left.text_input("Регион", placeholder="Россия")
            horizon = right.number_input("Горизонт прогноза, лет", 1, 10, 3)
            codes = st.text_input("Коды профстандартов через запятую", placeholder="06.001")
            if st.form_submit_button("Создать профессию", type="primary"):
                profession_id = await store.create_profession(factory, actor, name, region, int(horizon), codes.split(","))
                st.session_state.update(profession_id=profession_id, head_screen="sources")
                st.rerun()


async def screen_model(actor, factory):
    rows, _, _ = await professions_list(factory)
    profession = choose_profession(rows)
    if profession is None:
        return
    demo_banner(profession.is_demo)
    async with factory() as db:
        version = await store.latest_version(db, profession.id)
    head, action = st.columns([3, 1], vertical_alignment="bottom")
    head.header(f"Модель деятельности «{profession.name}»")
    if action.button("Пересчитать модель", icon=":material/refresh:", width="stretch",
                     help="Пересчёт по текущим источникам, голосам и config/activity_model.yaml"):
        with st.spinner("Считаем метрики, статусы и тренды"):
            await store.link_claims(factory, profession.id, actor.id)
            await store.build_version(factory, profession.id)
        st.rerun()
    if version is None:
        st.info("Модель ещё не собрана. Загрузите источники на экране «Источники».")
        return
    content = version.content
    tasks = content["tasks"]
    statuses = Counter(task["status"] for task in tasks.values())
    experts = content["experts"]
    st.caption(f"Версия {version.number} на {content['as_of']} · правила {content['config_hash'][:10]} · "
               f"данные {content['input_hash'][:10]}. Статусы и тренды считает код, LLM только извлекает утверждения.")
    kpis([(statuses["confirmed"] + statuses["confirmed_early"], "подтверждено"), (statuses["preliminary"], "предварительно"),
          (statuses["disputed"], "спорно"), (statuses["rejected"], "отклонено"), (statuses["hypothesis"], "гипотезы"),
          (f"{experts['reliable']} / {len(experts['excluded'])}", "экспертов надёжных / исключено"),
          (content["passport"]["employers_now"], "работодателей за год")])
    labels = ["Модель", "Спорные", "Отклонено", "Гипотезы", "Отброшено при извлечении", "Исключённые эксперты",
              "Паспорт данных", "Изменения"]
    tabs = st.tabs(labels)
    with tabs[0]:
        chosen = st.pills("Статус", ["confirmed", "confirmed_early", "preliminary", "disputed", "rejected", "hypothesis"],
                            selection_mode="multi", default=["confirmed", "confirmed_early", "preliminary", "disputed"],
                            format_func=STATUS_LABELS.get, key=f"status-filter-{profession.id}")
        trends = st.pills("Тренд", list(TREND_LABELS), selection_mode="multi", format_func=TREND_LABELS.get,
                             key=f"trend-filter-{profession.id}")
        focus = st.query_params.get("task")
        for function in content["functions"]:
            members = [tasks[key] for key in function["tasks"] if tasks[key]["status"] in (chosen or STATUS_LABELS)
                       and (not trends or tasks[key]["trend"] in trends)]
            if not members:
                continue
            code = f" · {function['code']}" if function.get("code") else ""
            st.markdown(f"#### {function['name']}{code}")
            st.caption(f"{len(members)} {plural(len(members), 'задача', 'задачи', 'задач')} · Σ приоритета "
                       f"подтверждённых и предварительных — {function['priority']}")
            for task in members:
                with st.expander(task_label(task), expanded=task["id"] == focus):
                    task_detail(task, content)
    for tab, status in zip(tabs[1:4], ("disputed", "rejected", "hypothesis")):
        with tab:
            members = [task for task in tasks.values() if task["status"] == status]
            if status == "disputed":
                with st.expander("Почему 10 или 50 экспертов", icon=":material/help:"):
                    needed_hint()
            if not members:
                st.info("Таких задач нет.")
            for task in sorted(members, key=lambda task: -task["priority"]):
                with st.expander(task_label(task)):
                    task_detail(task, content)
    with tabs[4]:
        st.caption("Утверждения, которые не участвуют в расчётах: цитаты нет в источнике дословно, задача другой роли.")
        rows = content["discarded_claims"]
        if rows:
            st.dataframe([{"Утверждение": row["text"], "Цитата экстрактора": row["quote"], "Причина": row["reason"],
                           "Источник": row["source_title"], "Экстрактор": row["extractor"]} for row in rows],
                         hide_index=True, width="stretch")
        else:
            st.info("Отброшенных утверждений нет.")
    with tabs[5]:
        st.write(f"Исключено {len(experts['excluded'])} из {experts['total']} экспертов. Голоса исключённых хранятся, "
                 "но в расчёт не входят. Персональные данные не показываются.")
        if experts["excluded_reason_counts"]:
            st.caption("Причины: " + "; ".join(f"{key} — {value}" for key, value in experts["excluded_reason_counts"].items()))
        for item in experts["excluded"]:
            with st.container(border=True):
                st.markdown(f"**{item['label']}** · {item['segment']}")
                for reason in item["reasons"]:
                    st.write(f"— {reason}")
                checks = item["checks"]
                st.caption(f"Ловушки «чужих» задач: ошибок {checks['foreign']['errors']} из {checks['foreign']['answered']}; "
                           f"дубли: расхождений {checks['duplicates']['mismatched']} из {checks['duplicates']['pairs']}; "
                           f"медиана {checks['median_seconds']} с на карточку; одинаковых ответов "
                           f"{percent(checks['same_answers']['share'])} из {checks['same_answers']['cards']}.")
    with tabs[6]:
        passport = content["passport"]
        st.dataframe([{"Тип источника": SOURCE_TYPE_LABELS.get(kind, kind), "Источников": item["count"],
                       "Независимых": item["origins"], "Период": f"{item['from']} — {item['to']}"}
                      for kind, item in passport["sources"].items()], hide_index=True, width="stretch")
        claims = passport["claims"]
        st.write(f"Утверждений {claims['total']}: прошли проверку цитаты {claims['verified']}, отброшено {claims['rejected']}, "
                 f"привязано к пунктам модели {claims['linked']}. Окно рынка {passport['window'][0]} — {passport['window'][1]}.")
        segments = experts["segments"]
        st.write("Сегменты надёжных экспертов: " + "; ".join(f"{key} — {value}" for field in ("role_kind", "company_size")
                                                           for key, value in segments[field].items()))
        for item in passport["limitations"]:
            st.warning(item)
    with tabs[7]:
        diff = content.get("diff") or {}
        if not diff:
            st.info("Это первая версия модели.")
        else:
            names = {key: task["name"] for key, task in tasks.items()}
            st.write(f"Новых задач: {len(diff['added'])}, удалено: {len(diff['removed'])}.")
            for change in diff["status"]:
                st.write(f"«{names.get(change['id'], change['id'])}»: {STATUS_LABELS[change['from']]} → {STATUS_LABELS[change['to']]}")
            for change in diff["trend"]:
                st.write(f"«{names.get(change['id'], change['id'])}»: тренд {TREND_LABELS[change['from']]} → {TREND_LABELS[change['to']]}")


async def run_extraction(factory, actor, profession, source_ids: list[str]):
    llm = extractor(factory)
    notes = []
    progress = st.progress(0.0, text="Извлекаем утверждения")
    for index, source_id in enumerate(source_ids, 1):
        result = await store.extract_source(factory, source_id, profession.name, llm)
        notes.extend(result.get("notes") or [])
        progress.progress(index / len(source_ids), text=f"Обработано источников: {index} из {len(source_ids)}")
    progress.progress(1.0, text="Связываем формулировки и пересчитываем модель")
    await store.link_claims(factory, profession.id, actor.id)
    await store.build_version(factory, profession.id)
    for note in dict.fromkeys(notes):
        st.warning(note)


async def screen_sources(actor, factory):
    rows, _, _ = await professions_list(factory)
    profession = choose_profession(rows)
    if profession is None:
        return
    demo_banner(profession.is_demo)
    st.header("Источники")
    st.caption("Каждое утверждение хранится с дословной цитатой. Профстандарт и вакансии разбираются правилами, свободный "
               "текст — LLM-экстрактором (temperature 0, кэш по хэшу), и каждую цитату проверяет код.")
    if extractor(factory) is None:
        st.info("LLM-экстрактор не настроен (.env): свободный текст без списков задач не будет разобран, правила работают.")
    tab_standard, tab_vacancies, tab_documents = st.tabs(["Профстандарт", "Вакансии", "Документы и интервью"])
    with tab_standard, st.form("standard"):
        st.caption("Файл ПС (DOCX, PDF, TXT) или вставленный текст трудовых функций: «Трудовые действия», "
                   "«Необходимые умения», «Необходимые знания». Пример формата — fixtures/activity_demo/standard_06_001.txt.")
        upload = st.file_uploader("Файл профстандарта", type=["docx", "pdf", "txt"])
        text = st.text_area("или текст", height=140)
        left, middle, right = st.columns(3)
        code = left.text_input("Код ПС", placeholder="06.001")
        year = middle.number_input("Год утверждения", 2000, 2100, 2022)
        approved = right.date_input("Дата утверждения", value=None)
        if st.form_submit_button("Загрузить и разобрать", type="primary"):
            content = read_document(upload.name, upload.getvalue()) if upload else text
            source_id, _ = await store.add_source(
                factory, actor, profession.id, "standard", f"ПС {code}".strip() if code else (upload.name if upload else "Профстандарт"),
                content, origin_key=f"Минтруд России: ПС {code}", standard_code=code.strip() or None,
                standard_year=int(year), published_at=approved)
            await run_extraction(factory, actor, profession, [source_id])
            st.toast("Профстандарт разобран")
    with tab_vacancies, st.form("vacancies"):
        st.caption("JSON или CSV: employer, title, published_at (ГГГГ-ММ-ДД), url, а также text или duties + requirements. "
                   "Шаблон — templates/activity_vacancies.csv. Доля считается по работодателям, а не по вакансиям.")
        upload = st.file_uploader("Выгрузка вакансий", type=["json", "csv"])
        platform = st.text_input("Площадка или источник выгрузки", placeholder="HH, Работа.ру, карьерный сайт")
        if st.form_submit_button("Загрузить вакансии", type="primary") and upload:
            created = []
            for row in parse_vacancies(upload.name, upload.getvalue()):
                source_id, new = await store.add_source(
                    factory, actor, profession.id, "vacancy", f"{row.title} — {row.employer}", row.text,
                    origin_key=row.employer, url=row.url, published_at=row.published_at,
                    meta={"platform": platform.strip(), "region": row.region})
                if new:
                    created.append(source_id)
            if created:
                await run_extraction(factory, actor, profession, created)
            st.toast(f"Новых вакансий: {len(created)}")
    with tab_documents, st.form("documents"):
        st.caption("Документы работодателей, интервью с представителями профессии, прогнозы, публикации, РПД других вузов.")
        kind = st.selectbox("Тип источника", DOC_TYPES, format_func=SOURCE_TYPE_LABELS.get)
        upload = st.file_uploader("Файл", type=["docx", "pdf", "txt", "md"])
        text = st.text_area("или текст", height=140)
        left, middle, right = st.columns(3)
        title = left.text_input("Название")
        origin = middle.text_input("Организация или автор", help="Один работодатель или издатель — один независимый голос.")
        published = right.date_input("Дата документа", value=None)
        url = st.text_input("Ссылка (если есть)")
        if st.form_submit_button("Загрузить и разобрать", type="primary"):
            content = read_document(upload.name, upload.getvalue()) if upload else text
            source_id, _ = await store.add_source(factory, actor, profession.id, kind, title or (upload.name if upload else ""),
                                                  content, origin_key=origin or title, url=url, published_at=published)
            await run_extraction(factory, actor, profession, [source_id])
            st.toast("Документ разобран")
    async with factory() as db:
        sources = (await db.scalars(select(ActivitySource).where(ActivitySource.profession_id == profession.id)
                                    .order_by(ActivitySource.type, ActivitySource.collected_at.desc()))).all()
    st.subheader(f"Загружено источников: {len(sources)}")
    if sources:
        st.dataframe([{"Тип": SOURCE_TYPE_LABELS.get(s.type, s.type), "Название": s.title, "Независимый голос": s.origin_key,
                       "Дата": f"{s.published_at:%d.%m.%Y}" if s.published_at else "—",
                       "Утверждений": sum((s.meta or {}).get("claims", {}).values()),
                       "Отброшено": (s.meta or {}).get("rejected", 0),
                       "Разбор": {"done": "готов", "partial": "частично", "failed": "сбой"}.get((s.meta or {}).get("extraction"), "—")}
                      for s in sources], hide_index=True, width="stretch")
        failed = [s for s in sources if (s.meta or {}).get("extraction") in ("failed", "partial")]
        if failed and st.button(f"Повторить извлечение ({len(failed)})", icon=":material/replay:"):
            llm = extractor(factory)
            for source in failed:
                await store.retry_extraction(factory, source.id, profession.name, llm)
            await store.link_claims(factory, profession.id, actor.id)
            await store.build_version(factory, profession.id)
            st.rerun()


async def screen_experts(actor, factory, users):
    rows, _, _ = await professions_list(factory)
    profession = choose_profession(rows)
    if profession is None:
        return
    demo_banner(profession.is_demo)
    st.header("Эксперты")
    config = load_config()["experts"]
    async with factory() as db:
        experts = await store.expert_rows(db, profession.id)
        version = await store.latest_version(db, profession.id)
        votes = (await db.scalars(select(ExpertVote).where(ExpertVote.expert_id.in_([e.id for e in experts])))).all() \
            if experts else []
    round_number = (profession.meta or {}).get("delphi_round", 1)
    first = sum(1 in (e.submitted_rounds or []) for e in experts)
    second = sum(2 in (e.submitted_rounds or []) for e in experts)
    content = version.content if version else None
    kpis([(len(experts), "приглашено"), (first, "прошли раунд 1"), (second, "прошли раунд 2"),
          (content["experts"]["reliable"] if content else 0, "надёжных"),
          (len(content["experts"]["excluded"]) if content else 0, "исключено"), (f"{round_number}", "текущий раунд Дельфи")])
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.subheader("Пригласить экспертов")
        directory = [user for user in await users.list_experts()]
        invited = {e.user_id for e in experts}
        options = [user for user in directory if user.id not in invited]
        chosen = st.multiselect("Эксперты из справочника", options, format_func=lambda user: user.display_name,
                                placeholder="Выберите экспертов")
        if st.button("Пригласить", type="primary", disabled=not chosen):
            for user in chosen:
                await store.invite_expert(factory, actor, profession.id, user.id)
            st.toast(f"Приглашено: {len(chosen)}")
            st.rerun()
        st.caption(f"Эксперт увидит анкету в кабинете эксперта. Нужно не меньше {config['n_min']} надёжных голосов на задачу; "
                   "разные сегменты (размер компании, роль) снижают риск расслоения.")
        st.subheader("Метод Дельфи")
        share = changed_share([Vote(expert_id=v.expert_id, item_id=v.item_id, verdict=v.verdict, round=v.round) for v in votes])
        if round_number == 1:
            st.write("Идёт раунд 1: статусы и ответы коллег эксперты не видят.")
            if st.button("Запустить раунд 2", disabled=first < config["n_min"] or version is None,
                         help=f"Нужно не меньше {config['n_min']} завершённых анкет раунда 1"):
                await store.build_version(factory, profession.id)
                await store.start_round_two(factory, actor, profession.id)
                st.rerun()
            st.caption("Раунд 2 включает только спорные задачи и задачи, предложенные экспертами. Эксперты увидят "
                       "распределение ответов коллег и обезличенные комментарии.")
        else:
            st.write("Идёт раунд 2. В расчёте берётся последний ответ каждого эксперта.")
            if share is not None:
                stable = share < config["delphi_stable_share"]
                st.write(f"Изменили ответ во втором раунде: {percent(share)} "
                         + ("— мнения устойчивы, опрос можно останавливать." if stable else "— мнения ещё сдвигаются."))
    with right:
        with st.container(border=True):
            st.markdown("**Почему 10 или 50 экспертов**")
            needed_hint()
    st.subheader("Контрольные карточки")
    if content:
        controls = [task for task in content["tasks"].values() if task["is_control"]]
        st.caption("Около 10 % карточек анкеты. Эксперт не знает, какие из них контрольные. «Чужая» задача — правильный "
                   "ответ «Нет»; дубль — ответ должен совпасть с ответом на оригинал.")
        for task in controls:
            kind = "чужая задача" if task["control_kind"] == "foreign" else \
                f"дубль задачи «{content['tasks'].get(task['duplicate_of'], {}).get('name', '—')}»"
            experts_metrics = task["metrics"]["experts"]
            st.write(f"«{task['name']}» — {kind}; ответили «да» {experts_metrics['k']} из {experts_metrics['n']} надёжных.")
        with st.form("control"):
            text = st.text_input("Новая контрольная карточка (формулировка)")
            originals = {key: task["name"] for key, task in content["tasks"].items() if not task["is_control"]}
            duplicate = st.selectbox("Дубль какой задачи (пусто — «чужая» задача)", [None, *sorted(originals, key=originals.get)],
                                     format_func=lambda key: "— «чужая» задача" if key is None else originals[key])
            if st.form_submit_button("Добавить карточку"):
                await store.add_control(factory, actor, profession.id, text, duplicate)
                st.rerun()
        st.subheader("Голоса по задачам")

        def still_needed(task: dict) -> str:
            n = task["metrics"]["experts"]["n"]
            if n < config["n_min"]:
                return f"{config['n_min'] - n} до минимума"
            if task["dispute"]:
                return "спорно по существу" if task["dispute"]["essential"] else f"≈ {task['dispute']['more']}"
            return "достаточно"

        st.dataframe([{"Задача": task["name"], "Статус": task["status_label"],
                       "Голосов (надёжных)": task["metrics"]["experts"]["n"], "Нужно ещё": still_needed(task)}
                      for task in sorted(content["tasks"].values(), key=lambda t: t["metrics"]["experts"]["n"])
                      if not task["is_control"]], hide_index=True, width="stretch")


async def head_programs(actor, factory):
    async with factory() as db:
        return (await db.scalars(select(Program).where(Program.head_user_id == actor.id).order_by(Program.created_at))).all()


async def program_form(actor, factory, professions, program=None):
    names = {row.id: row.name + (" · ДЕМО" if row.is_demo else "") for row in professions}
    current = {item["profession_id"]: item["weight"] for item in (program.target_professions if program else [])}
    with st.form(f"program-{program.id if program else 'new'}"):
        name = st.text_input("Название программы", value=program.name if program else "")
        left, middle, right = st.columns(3)
        institution = left.text_input("Вуз", value=program.institution if program else "")
        year = middle.text_input("Учебный год", value=program.academic_year if program else "", placeholder="2026/2027")
        total = right.number_input("Объём программы, ЗЕТ", 1, 600, program.total_zet if program else 240)
        level = left.selectbox("Уровень", ["bachelor", "specialist", "master"], format_func={
            "bachelor": "бакалавриат", "specialist": "специалитет", "master": "магистратура"}.get,
            index=["bachelor", "specialist", "master"].index(program.level) if program else 0)
        targets = st.multiselect("Кого готовит программа (целевые профессии)", list(names), default=list(current),
                                 format_func=names.get)
        weights = {pid: st.slider(f"Вес «{names[pid]}»", 0.05, 1.0, float(current.get(pid, 1.0)), 0.05, key=f"w-{pid}")
                   for pid in targets}
        if st.form_submit_button("Сохранить программу", type="primary"):
            total_weight = sum(weights.values()) or 1
            target_list = [{"profession_id": pid, "weight": round(weight / total_weight, 3)} for pid, weight in weights.items()]
            if program:
                await store.update_program(factory, actor, program.id, name=name, institution=institution,
                                           academic_year=year, total_zet=int(total), level=level,
                                           target_professions=target_list)
            else:
                st.session_state["program_id"] = await store.create_program(
                    factory, actor, name, level=level, total_zet=int(total), institution=institution,
                    academic_year=year, targets=target_list)
            st.rerun()


def matrix_html(analysis: dict, disciplines: list) -> str:
    head = "".join(f"<th><span>{escape(d.name)}</span></th>" for d in disciplines)
    rows = []
    for row in analysis["matrix"]:
        if not row["eligible"]:
            continue
        gap = row["eligible"] - row["covered"]
        cells = []
        for d in disciplines:
            cell = row["cells"][d.id]
            level = 0 if not cell["covered"] else min(3, 1 + cell["covered"] * 3 // max(row["eligible"], 1))
            cells.append(f'<td><span class="am-cell am-cell--{level}" title="полностью: {cell["full"]}">'
                         f'{cell["covered"] or "·"}</span></td>')
        status = f'<span class="am-gap">не покрыто {gap}</span>' if gap else "покрыто"
        rows.append(f"<tr><td>{escape(row['function'])}<br><span class='am-muted'>{row['eligible']} задач · {status}</span></td>"
                    + "".join(cells) + "</tr>")
    return (f'<div class="am-matrix-wrap"><table class="am-matrix"><thead><tr><th>Функция профессии</th>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


async def recommendation_card(actor, factory, rec: dict, row: Recommendation | None):
    decision = row.decision if row else None
    confidence = CONFIDENCE_LABELS[rec["confidence"]]
    status = f" · решение: {DECISION_LABELS[decision]}" if decision else ""
    st.html(f'<div class="am-rec"><span class="am-rule">{rec["rule"]} · {escape(RULE_LABELS[rec["rule"]]).upper()}</span>'
            f'<b>{escape(rec["action_text"])}</b><div class="am-muted">Уверенность: {confidence} · освобождается '
            f'{rec["zet"]["frees"]:g} ЗЕТ · нужно {rec["zet"]["needs"]:g} ЗЕТ{escape(status)}</div></div>')
    with st.expander("Доводы, цифры и решение"):
        for line in rec["lines"]:
            st.write(f"— {line}")
        st.caption(f"Если не делать: {rec['consequence']}")
        if rec["tasks"]:
            st.dataframe([{"Задача": t["name"], "Статус": t["status_label"], "Тренд": TREND_LABELS[t["trend"]],
                           "Приоритет": t["priority"], "Рынок": t["market"], "Эксперты": t["experts"]}
                          for t in rec["tasks"]], hide_index=True, width="stretch")
        if row is None:
            return
        comment = st.text_input("Комментарий к решению", value=row.decision_comment or "", key=f"comment-{row.id}")
        columns = st.columns(3)
        for column, (value, label) in zip(columns, (("accepted", "Принять"), ("rejected", "Отклонить"), ("postponed", "Отложить"))):
            if column.button(label, key=f"{value}-{row.id}", width="stretch",
                             type="primary" if value == "accepted" and decision != "accepted" else "secondary"):
                await store.decide(factory, actor, row.id, value, comment)
                st.rerun()
        if row.decided_at:
            st.caption(f"Решение записано {row.decided_at:%d.%m.%Y %H:%M}.")


async def screen_program(actor, factory, users):
    st.header("Моя программа")
    professions, _, _ = await professions_list(factory)
    programs = await head_programs(actor, factory)
    if not programs:
        st.info("Создайте программу: укажите, кого она готовит, и добавьте дисциплины.")
        await program_form(actor, factory, professions)
        return
    ids = [program.id for program in programs]
    if st.session_state.get("program_id") not in ids:
        st.session_state["program_id"] = ids[0]
    program = next(p for p in programs if p.id == st.selectbox("Программа", ids, key="program_id",
                                                              format_func={p.id: p.name for p in programs}.get))
    demo_banner(program.is_demo)
    async with factory() as db:
        disciplines = (await db.scalars(select(ProgramDiscipline).where(ProgramDiscipline.program_id == program.id)
                                        .order_by(ProgramDiscipline.position, ProgramDiscipline.name))).all()
    tab_recs, tab_matrix, tab_disciplines, tab_content, tab_settings = st.tabs(
        ["Рекомендации", "Матрица покрытия", "Дисциплины", "Содержание дисциплин", "Параметры программы"])
    with tab_settings:
        await program_form(actor, factory, professions, program)
    with tab_disciplines:
        teachers = await users.list_teachers()
        teacher_names = {user.id: user.display_name for user in teachers}
        for d in disciplines:
            if d.teacher_user_id and d.teacher_user_id not in teacher_names:
                teacher_names[d.teacher_user_id] = d.teacher_name or "преподаватель вне справочника"
        by_name = {name: key for key, name in teacher_names.items()}
        st.caption("Часть «обязательная по ФГОС» система не предлагает сокращать. Преподаватель увидит закреплённую "
                   "дисциплину в своём кабинете и запустит генерацию РПД по модели деятельности.")
        edited = st.data_editor(
            [{"id": d.id, "name": d.name, "zet": d.zet, "hours": d.hours, "semester": d.semester, "part": d.part,
              "teacher": teacher_names.get(d.teacher_user_id, "")} for d in disciplines],
            column_config={"id": None, "name": st.column_config.TextColumn("Дисциплина", required=True, width="large"),
                           "zet": st.column_config.NumberColumn("ЗЕТ", min_value=0, max_value=60, step=0.5),
                           "hours": st.column_config.NumberColumn("Часов", min_value=0, max_value=2000),
                           "semester": st.column_config.NumberColumn("Семестр", min_value=1, max_value=12),
                           "part": st.column_config.SelectboxColumn("Часть", options=list(PART_LABELS), required=True),
                           "teacher": st.column_config.SelectboxColumn("Преподаватель", options=["", *by_name])},
            num_rows="dynamic", hide_index=True, width="stretch", key=f"editor-{program.id}")
        st.caption("Часть: " + "; ".join(f"{key} — {label}" for key, label in PART_LABELS.items()))
        if st.button("Сохранить дисциплины", type="primary"):
            await store.save_disciplines(factory, actor, program.id, edited)
            async with factory() as db:
                fresh = {d.name: d for d in (await db.scalars(select(ProgramDiscipline).where(
                    ProgramDiscipline.program_id == program.id))).all()}
            for row in edited:
                discipline = fresh.get(" ".join(str(row.get("name") or "").split()))
                if discipline is not None:
                    teacher = by_name.get(row.get("teacher") or "")
                    await store.assign_teacher(factory, actor, discipline.id, teacher, row.get("teacher") or "")
            await store.recompute(factory, actor, program.id)
            st.rerun()
    with tab_content:
        st.caption("Темы, результаты обучения и задания — по одному в строке. По ним считается покрытие задач модели: "
                   "full — пункт есть в темах и в результатах или заданиях, partial — только в одном из них.")
        for d in disciplines:
            with st.expander(d.name):
                with st.form(f"content-{d.id}"):
                    topics = st.text_area("Темы", "\n".join((d.content or {}).get("topics", [])), key=f"t-{d.id}")
                    outcomes = st.text_area("Результаты обучения", "\n".join((d.content or {}).get("outcomes", [])), key=f"o-{d.id}")
                    tasks = st.text_area("Задания", "\n".join((d.content or {}).get("assignments", [])), key=f"a-{d.id}")
                    if st.form_submit_button("Сохранить содержание"):
                        await store.save_content(factory, actor, d.id, topics.splitlines(), outcomes.splitlines(), tasks.splitlines())
                        await store.recompute(factory, actor, program.id)
                        st.rerun()
    if not program.target_professions:
        with tab_recs:
            st.info("Укажите целевые профессии в параметрах программы.")
        return
    if not disciplines:
        with tab_recs:
            st.info("Добавьте дисциплины программы.")
        return
    analysis = await store.recompute(factory, actor, program.id)
    async with factory() as db:
        saved = {row.key: row for row in (await db.scalars(select(Recommendation).where(
            Recommendation.program_id == program.id))).all()}
    with tab_recs:
        balance = analysis["balance"]
        kpis([(len(analysis["recommendations"]), "рекомендаций"),
              (f"{balance['disciplines_zet']:g} / {balance['program_zet']}", "ЗЕТ дисциплин / программы"),
              (f"−{balance['frees']:g}", "ЗЕТ освобождается"), (f"+{balance['needs']:g}", "ЗЕТ нужно новым"),
              (f"{balance['after']:g}", "ЗЕТ после всех рекомендаций"),
              (sum(1 for row in saved.values() if row.current and row.decision), "решений принято")])
        st.caption("Рекомендации считает код по правилам R1–R7 и порогам config/activity_model.yaml. Учитываются задачи "
                   "со статусами «Подтверждено» и «Предварительно». Обязательные по ФГОС дисциплины не трогаются. "
                   "Окончательно решает руководитель.")
        recommendations = analysis["recommendations"]
        rule_filter = st.pills("Правила", list(RULE_LABELS), selection_mode="multi", format_func=lambda r: f"{r} · {RULE_LABELS[r]}")
        # Изменения программы — сверху по Σ приоритета; «сохранить как фундамент» (R5) — отдельным блоком ниже.
        changes = [rec for rec in recommendations if rec["rule"] != "R5" and (not rule_filter or rec["rule"] in rule_filter)]
        keep = [rec for rec in recommendations if rec["rule"] == "R5" and (not rule_filter or "R5" in rule_filter)]
        top = changes if rule_filter or st.session_state.get(f"all-{program.id}") else changes[:7]
        if top:
            st.subheader("Главные рекомендации" if len(top) < len(changes) else "Рекомендации")
        for rec in top:
            await recommendation_card(actor, factory, rec, saved.get(rec["key"]))
        if len(changes) > len(top) and st.button(f"Показать все изменения ({len(changes)})"):
            st.session_state[f"all-{program.id}"] = True
            st.rerun()
        if keep:
            st.subheader("Фундамент: сохранить")
            st.caption("Дисциплины покрывают фундаментальные области, на которые опираются подтверждённые задачи. "
                       "Они защищены от правила «сократить или убрать».")
            for rec in keep:
                await recommendation_card(actor, factory, rec, saved.get(rec["key"]))
    with tab_matrix:
        st.caption("Сколько подтверждённых и предварительных задач функции покрывает дисциплина. Наведите на число, "
                   "чтобы увидеть, сколько из них покрыто полностью.")
        st.html(matrix_html(analysis, disciplines))
        st.subheader("Покрытие по дисциплинам")
        models = {pid: content for pid, content in (await latest_models(factory)).items()}
        for d in disciplines:
            links = analysis["links"].get(d.id, {})
            with st.expander(f"{d.name} · {len(links)} {plural(len(links), 'задача', 'задачи', 'задач')} · "
                             f"без связи {analysis['unallocated_hours'].get(d.id, 0):.0f} ч"):
                for key, link in links.items():
                    profession_id, task_id = key.split(":", 1)
                    task = models[profession_id]["tasks"][task_id]
                    left, right = st.columns([5, 1])
                    left.write(f"**{task['name']}** — {link['coverage']}, {task['status_label'].lower()}. {link['reason']}")
                    if right.button("Убрать", key=f"rm-{d.id}-{task_id}", help="Ручная правка важнее автоматики"):
                        await store.set_coverage(factory, actor, d.id, task_id, "none", "убрано вручную")
                        st.rerun()
                options = {f"{pid}:{tid}": t["name"] for pid in models if pid in {t["profession_id"] for t in program.target_professions}
                           for tid, t in models[pid]["tasks"].items() if not t["is_control"] and f"{pid}:{tid}" not in links}
                add = st.selectbox("Отметить задачу как покрытую", [None, *sorted(options, key=options.get)],
                                   format_func=lambda key: "—" if key is None else options[key], key=f"add-{d.id}")
                if add and st.button("Добавить покрытие", key=f"add-btn-{d.id}"):
                    await store.set_coverage(factory, actor, d.id, add.split(":", 1)[1], "partial", "добавлено вручную")
                    st.rerun()


async def latest_models(factory) -> dict[str, dict]:
    async with factory() as db:
        return await store.latest_contents(db)


async def screen_signal(actor, factory):
    rows, _, _ = await professions_list(factory)
    profession = choose_profession(rows)
    if profession is None:
        return
    demo_banner(profession.is_demo)
    models = await latest_models(factory)
    content = models.get(profession.id)
    st.header("Сигнал по профилю")
    if content is None:
        st.info("Модель ещё не собрана.")
        return
    signal = content["profile_signal"]
    css = {"risk": "risk", "transforming": "transforming", "stable": "stable"}[signal["label"]]
    lines = signal_lines(signal, content)
    st.html(f'<div class="am-signal am-signal--{css}"><h3>{escape(SIGNAL_LABELS[signal["label"]])}</h3>'
            f'<div class="am-muted">{escape(DISCLAIMER)}</div></div>')
    config = load_config()["profile_signal"]
    columns = st.columns(3)
    market, tasks, experts = signal["market"], signal["tasks"], signal["experts"]
    with columns[0].container(border=True):
        st.markdown("**A. Рынок** · запаздывает")
        st.metric("Изменение числа работодателей", f"{market['value'] * 100:+.0f} %" if market["value"] is not None else "мало данных")
        st.caption(lines[0] + f" Порог риска: {config['a_risk'] * 100:+.0f} %.")
    with columns[1].container(border=True):
        st.markdown("**B. Задачи** · текущий сигнал")
        st.metric("Доля приоритета снижающихся задач", percent(tasks["value"]))
        names = [content["tasks"][key]["name"] for key in tasks["falling"]]
        st.caption(f"Пороги: «трансформируется» от {percent(config['b_transform'])}, «под риском» от {percent(config['b_risk'])}. "
                   + ("Задачи: " + "; ".join(names) if names else "Таких задач нет."))
    with columns[2].container(border=True):
        st.markdown("**C. Эксперты** · опережающий сигнал")
        st.metric("«Вырастет» минус «снизится»", f"{experts['value']:+.2f}" if experts["value"] is not None else "мало ответов")
        st.caption(f"Ответов {experts['answered']}: " + "; ".join(f"{DEMAND_LABELS[key]} — {value}"
                                                               for key, value in experts["demand"].items())
                   + f". Порог риска: {config['c_risk']:+.2f}.")
    st.subheader("Во что превратится роль")
    if not signal["successors"]:
        st.info("Эксперты ещё не отвечали на этот вопрос.")
    for successor in signal["successors"]:
        model = find_profession(models, successor["name"], profession.id, load_config())
        with st.container(border=True):
            st.markdown(f"**{successor['name']}** — назвали {successor['count']} "
                        f"{plural(successor['count'], 'эксперт', 'эксперта', 'экспертов')}: {percent(successor['share'])} "
                        f"надёжных (на вопрос ответили {signal['successors_answered']})")
            if model is None:
                st.caption("Модели деятельности для этой роли нет. Постройте её, чтобы оценить, сколько программы сохранится.")
                continue
            kept = retention(content, model, load_config())
            growth = model["profile_signal"]["market"]["value"]
            st.write(f"При переходе на «{model['profession']['name']}» сохраняется ≈ {percent(kept)} содержания "
                     f"(по приоритету задач)" + (f"; рост рынка {growth * 100:+.0f} %." if growth is not None else "."))
            st.caption("Задача считается сохранённой, если у смежной роли есть похожая задача или все её области знаний. "
                       "Что добавить в программу — в рекомендации R7 на экране «Моя программа».")


async def screen_map(factory):
    st.header("Карта знаний между профессиями")
    st.caption("Межпрофессиональные области знаний: вес области = Σ приоритета задач, которые её используют / Σ приоритета "
               "подтверждённых задач профессии.")
    knowledge_map(await latest_models(factory))


def knowledge_map(models: dict[str, dict]) -> None:
    """Матрица «профессии × области знаний» и сценарий преподавателя: где нужна моя область и какие задачи — идеи проектов."""
    if not models:
        st.info("Моделей деятельности пока нет.")
        return
    areas = sorted({area for content in models.values() for area in content["areas"]})
    rows = []
    for content in sorted(models.values(), key=lambda c: c["profession"]["name"]):
        row = {"Профессия": content["profession"]["name"] + (" (ДЕМО)" if content["profession"].get("is_demo") else "")}
        for area in areas:
            weight = content["areas"].get(area, {}).get("weight")
            row[area] = round(weight * 100) if weight else None
        rows.append(row)
    st.dataframe(rows, hide_index=True, width="stretch",
                 column_config={area: st.column_config.ProgressColumn(area, format="%d %%", min_value=0, max_value=100)
                                for area in areas})
    chosen = st.selectbox("Моя область знаний", areas, index=areas.index("Теория графов") if "Теория графов" in areas else 0)
    st.subheader(f"Где нужна «{chosen}»")
    found = False
    for content in sorted(models.values(), key=lambda c: c["profession"]["name"]):
        tasks = [content["tasks"][key] for key in content["areas"].get(chosen, {}).get("tasks", [])]
        if not tasks:
            continue
        found = True
        with st.container(border=True):
            st.markdown(f"**{content['profession']['name']}** · вес области {percent(content['areas'][chosen]['weight'])}")
            for task in tasks:
                tools = sorted({content["knowledge"][link["id"]]["name"] for link in task["related"]
                                if content["knowledge"][link["id"]]["level"] == "tool"})
                source = task["quotes"][0]["source_title"] if task["quotes"] else "—"
                st.markdown(f"Проект: **{task['name']}** в контексте «{content['profession']['name']}»; инструменты: "
                            f"{', '.join(tools) or '—'}; источник: {source}. Статус: {task['status_label'].lower()}.")
                if task["quotes"]:
                    st.html(quotes_html(task["quotes"], 1))
    if not found:
        st.info("Подтверждённых задач с этой областью пока нет.")
    st.caption("Система не придумывает проекты: это реальные задачи из моделей деятельности, с цитатами источников.")


async def screen_queue(actor, factory):
    rows, _, _ = await professions_list(factory)
    profession = choose_profession(rows)
    if profession is None:
        return
    demo_banner(profession.is_demo)
    st.header("Очередь методиста")
    st.caption("Слияния формулировок, разметка функций и правки формулировок подтверждает человек. Всё пишется в журнал "
               "и обратимо.")
    async with factory() as db:
        version = await store.latest_version(db, profession.id)
        queue = await store.review_queue(db, profession.id, version.content if version else None)
        log = (await db.scalars(select(ActivityMergeLog).where(ActivityMergeLog.profession_id == profession.id)
                                .order_by(ActivityMergeLog.created_at.desc()).limit(30))).all()
    functions = queue["functions"]
    tab_duplicates, tab_functions, tab_wording, tab_log = st.tabs([
        f"Похожие формулировки ({len(queue['duplicates'])})", f"Без функции ({len(queue['unassigned'])})",
        f"Предложения экспертов ({len(queue['wordings'])})", "Журнал"])
    with tab_duplicates:
        if not queue["duplicates"]:
            st.info("Кандидатов в дубли нет.")
        for pair in queue["duplicates"]:
            with st.container(border=True):
                st.write(f"«{pair['left'].name}»  ↔  «{pair['right'].name}» · сходство понятий {pair['similarity']}")
                a, b, c = st.columns(3)
                if a.button("Оставить первую", key=f"m1-{pair['key']}", width="stretch"):
                    await store.merge_items(factory, actor, pair["right"].id, pair["left"].id)
                    st.rerun()
                if b.button("Оставить вторую", key=f"m2-{pair['key']}", width="stretch"):
                    await store.merge_items(factory, actor, pair["left"].id, pair["right"].id)
                    st.rerun()
                if c.button("Это разные задачи", key=f"d-{pair['key']}", width="stretch"):
                    await store.dismiss(factory, actor, profession.id, pair["key"])
                    st.rerun()
        if queue["merged"]:
            st.subheader("Объединённые пункты")
            for item in queue["merged"]:
                left, right = st.columns([4, 1])
                left.write(f"«{item.name}» объединён с другим пунктом.")
                if right.button("Разъединить", key=f"un-{item.id}"):
                    await store.unmerge_item(factory, actor, item.id)
                    st.rerun()
    with tab_functions:
        names = {function.id: function.name + (f" · {function.standard_code}" if function.standard_code else "")
                 for function in functions}
        for entry in queue["unassigned"]:
            task = entry["task"]
            with st.container(border=True):
                hint = f" Подсказка: «{entry['suggestion'].name}» (сходство {entry['similarity']})." if entry["suggestion"] else ""
                st.write(f"«{task.name}».{hint}")
                options = list(names)
                default = options.index(entry["suggestion"].id) if entry["suggestion"] else 0
                left, right = st.columns([4, 1], vertical_alignment="bottom")
                target = left.selectbox("Функция", options, index=default, format_func=names.get, key=f"f-{task.id}")
                if right.button("Отнести", key=f"fa-{task.id}", width="stretch"):
                    await store.set_function(factory, actor, task.id, target)
                    st.rerun()
        if not queue["unassigned"]:
            st.info("Все задачи отнесены к функциям.")
        with st.form("new-function"):
            name = st.text_input("Новая функция (если её нет в профстандарте)")
            if st.form_submit_button("Создать функцию"):
                await store.create_function(factory, actor, profession.id, name)
                st.rerun()
    with tab_wording:
        if not queue["wordings"]:
            st.info("Предложений формулировок нет.")
        for entry in queue["wordings"]:
            with st.container(border=True):
                st.write(f"Сейчас: «{entry['name']}»")
                st.write(f"Эксперт предлагает: «{entry['text']}»")
                st.caption("Новая формулировка — новая версия пункта: голоса за прежнюю формулировку остаются при ней.")
                a, b = st.columns(2)
                if a.button("Принять формулировку", key=f"wa-{entry['key']}", width="stretch"):
                    await store.reword_item(factory, actor, entry["task_id"], entry["text"])
                    await store.dismiss(factory, actor, profession.id, entry["key"], "принято")
                    st.rerun()
                if b.button("Оставить как есть", key=f"wr-{entry['key']}", width="stretch"):
                    await store.dismiss(factory, actor, profession.id, entry["key"], "отклонено")
                    st.rerun()
    with tab_log:
        labels = {"merge": "объединение", "unmerge": "разъединение", "reparent": "отнесение к функции", "reword": "новая формулировка",
                  "dismiss": "решение по очереди", "function": "новая функция", "control": "контрольная карточка",
                  "delphi": "раунд Дельфи", "item_link": "связь задачи со знанием"}
        st.dataframe([{"Когда": f"{row.created_at:%d.%m.%Y %H:%M}", "Действие": labels.get(row.action, row.action),
                       "Подробности": "; ".join(f"{key}: {value}" for key, value in row.payload.items()
                                                if key in ("name", "into_name", "from", "to", "text", "note", "round"))}
                      for row in log], hide_index=True, width="stretch")


async def screen_export(actor, factory):
    st.header("Экспорт")
    models = await latest_models(factory)
    rows, _, _ = await professions_list(factory)
    profession = choose_profession(rows)
    if profession is not None and profession.id in models:
        content = models[profession.id]
        demo_banner(profession.is_demo)
        st.subheader(f"Модель деятельности «{profession.name}»")
        st.caption("Паспорт, функции и задачи со статусами, трендами, «почему здесь» и цитатами; отклонённое, гипотезы, "
                   "отброшенное, исключённые эксперты, карта областей знаний, изменения с прошлой версии.")
        document = model_document(content)
        columns = st.columns(3)
        stem = f"activity-model-v{content.get('version')}"
        columns[0].download_button("HTML", document.html(), f"{stem}.html", "text/html", width="stretch")
        columns[1].download_button("Markdown", document.markdown(), f"{stem}.md", "text/markdown", width="stretch")
        columns[2].download_button("JSON", model_json(content), f"{stem}.json", "application/json", width="stretch")
    programs = await head_programs(actor, factory)
    for program in programs:
        async with factory() as db:
            disciplines = (await db.scalars(select(ProgramDiscipline).where(ProgramDiscipline.program_id == program.id)
                                            .order_by(ProgramDiscipline.position))).all()
        if not disciplines or not program.target_professions:
            continue
        analysis = await store.recompute(factory, actor, program.id)
        async with factory() as db:
            saved = (await db.scalars(select(Recommendation).where(Recommendation.program_id == program.id,
                                                                  Recommendation.current.is_(True)))).all()
        decisions = [{"key": row.key, "rule": row.rule, "action_text": row.action_text, "decision": row.decision,
                      "comment": row.decision_comment, "decided_at": f"{row.decided_at:%d.%m.%Y}" if row.decided_at else ""}
                     for row in saved]
        data = {"id": program.id, "name": program.name, "institution": program.institution,
                "academic_year": program.academic_year, "total_zet": program.total_zet, "is_demo": program.is_demo,
                "disciplines": [{"id": d.id, "name": d.name, "zet": d.zet, "part": d.part, "teacher_name": d.teacher_name}
                                for d in disciplines]}
        targets = {item["profession_id"]: models[item["profession_id"]] for item in program.target_professions
                   if item["profession_id"] in models}
        st.subheader(f"Программа «{program.name}»")
        demo_banner(program.is_demo)
        recommendations = program_document(data, analysis, decisions, targets)
        justification = justification_document(data, analysis, decisions, targets)
        columns = st.columns(5)
        columns[0].download_button("Рекомендации · HTML", recommendations.html(), "recommendations.html", "text/html",
                                   width="stretch", key=f"rh-{program.id}")
        columns[1].download_button("Рекомендации · MD", recommendations.markdown(), "recommendations.md", "text/markdown",
                                   width="stretch", key=f"rm-{program.id}")
        columns[2].download_button("Рекомендации · JSON", program_json(data, analysis, decisions), "recommendations.json",
                                   "application/json", width="stretch", key=f"rj-{program.id}")
        columns[3].download_button("Обоснование ОП · HTML", justification.html(), "program-justification.html", "text/html",
                                   width="stretch", key=f"jh-{program.id}")
        columns[4].download_button("Обоснование ОП · MD", justification.markdown(), "program-justification.md",
                                   "text/markdown", width="stretch", key=f"jm-{program.id}")


async def render(actor, settings, factory, users):
    if requested := st.query_params.get("profession"):
        st.session_state.update(profession_id=requested, head_screen="model")
        del st.query_params["profession"]
    st.sidebar.text(actor.display_name)
    screen = st.session_state.get("head_screen", "professions")
    for key, (label, icon) in SCREENS.items():
        if st.sidebar.button(label, icon=icon, key=f"nav-{key}", width="stretch",
                             type="secondary" if key == screen else "tertiary"):
            st.session_state["head_screen"] = key
            st.rerun()
    if screen == "professions":
        await screen_professions(actor, factory)
    elif screen == "model":
        await screen_model(actor, factory)
    elif screen == "sources":
        await screen_sources(actor, factory)
    elif screen == "experts":
        await screen_experts(actor, factory, users)
    elif screen == "program":
        await screen_program(actor, factory, users)
    elif screen == "signal":
        await screen_signal(actor, factory)
    elif screen == "map":
        await screen_map(factory)
    elif screen == "queue":
        await screen_queue(actor, factory)
    else:
        await screen_export(actor, factory)


def main():
    portal("Кабинет руководителя программы", {"program_head", "admin"}, render)


if __name__ == "__main__":
    main()
