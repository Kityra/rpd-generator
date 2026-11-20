"""Streamlit interface for the RPD workbench."""

from io import BytesIO
import json
import re
from zipfile import ZIP_DEFLATED, ZipFile

import streamlit as st

from run_course import Discipline, discover_disciplines, readable_fields, render_course_report, render_rpd, unique_disciplines
from run_rpd import ROOT, open_yandex, yandex_settings
from web_preview import Batch, PreviewSession, Task, read_attachment, request_error


STATUSES = {
    "queued": "В очереди", "ready": "РПД готова", "error": "Ошибка обработки",
}


@st.cache_data
def document_schema() -> dict:
    return json.loads((ROOT / "final_schema.json").read_text(encoding="utf-8-sig"))


def run_queue(preview: PreviewSession, batch: Batch) -> None:
    queued = [task for task in batch.tasks if task.status == "queued"]
    if not queued:
        return
    progress = st.progress(0, text="Запускаю подготовку")
    for index, task in enumerate(queued, 1):
        progress.progress((index - 1) / len(queued), text=f"{index} из {len(queued)} · {task.request.discipline}")
        preview.advance(task)
    progress.empty()


def new_request(preview: PreviewSession) -> None:
    st.subheader("Новая заявка")
    mode = st.radio("Вид заявки", ("Одна дисциплина", "Комплект на курс"), horizontal=True)
    course = mode == "Комплект на курс"
    with st.form("new_request"):
        title = st.text_input("Название курса или программы" if course else "Название дисциплины", max_chars=300)
        columns = st.columns([2, 1])
        institution = columns[0].text_input("Вуз", max_chars=300, placeholder="Южный федеральный университет")
        year = columns[1].text_input("Учебный год", max_chars=100, placeholder="2026/2027")
        if course:
            study_year = st.number_input("Курс обучения", min_value=1, step=1, value=None, placeholder="Необязательно")
            names = st.text_area("Дисциплины", help="По одному названию на строку. Пустой список можно подобрать по названию курса.",
                                 max_chars=12000, key="course_names")
            buttons = st.columns(2)
            suggest = buttons[0].form_submit_button("Подобрать дисциплины")
            start = buttons[1].form_submit_button("Подготовить комплект", type="primary")
            facts = ""
        else:
            facts = st.text_area("Известные сведения", max_chars=16000,
                                 placeholder="Учебный план, часы, компетенции, темы, форма аттестации")
            start = st.form_submit_button("Подготовить РПД", type="primary")
            suggest = False

    if not start and not suggest:
        return
    title, institution, year = title.strip(), institution.strip(), year.strip()
    if not all((title, institution, year)):
        st.warning("Укажите название, вуз и учебный год.")
        return
    scope = (title, institution, year, study_year) if course else None
    try:
        if suggest:
            with st.spinner("Ищу дисциплины и темы…"):
                settings = yandex_settings()
                with open_yandex(settings, timeout=45) as client:
                    items = discover_disciplines(client, settings, *scope)
            if not items:
                st.info("Введите перечень дисциплин в форму.")
                return
            st.session_state["proposed_course"] = (scope, items)
            st.session_state["populate_course_names"] = "\n".join(item.name for item in items)
            st.rerun()
        selected = [Discipline(title)]
        if course:
            proposal_scope, proposed = st.session_state.get("proposed_course", (None, []))
            suggestions = {item.name.casefold(): item for item in proposed} if proposal_scope == scope else {}
            selected = unique_disciplines([
                suggestions.get(name.casefold(), Discipline(name))
                for line in names.splitlines() if (name := " ".join(line.split()))
            ])
        batch_id = preview.create_batch(title, institution, year, selected, facts,
                                        kind="course" if course else "single", study_year=study_year if course else None)
        st.session_state["active_batch"] = batch_id
        st.session_state["navigate_to"] = "Черновики"
        st.session_state["start_batch"] = batch_id
        st.rerun()
    except Exception as error:
        st.error(request_error(error))


def download_package(batch: Batch) -> bytes:
    if not batch.tasks or any(task.document is None for task in batch.tasks):
        raise ValueError("Архив доступен, когда подготовлены РПД всех дисциплин.")
    if any(task.official_rpd_error for task in batch.tasks):
        raise ValueError("Экспорт РПД приостановлен до подтверждения официальных материалов по всем дисциплинам.")
    heading = [batch.title, f"Вуз — {batch.institution}", f"Учебный год — {batch.year}"]
    if batch.approved:
        heading.append("Курс утверждён в текущей сессии.")
    rows = []
    output = BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for index, task in enumerate(batch.tasks, 1):
            lines = [f"{STATUSES[task.status]}: {task.request.discipline}"]
            if doc := task.document:
                lines[0] = f"Готово: {task.request.discipline}"
                if task.approved:
                    lines.append("Дисциплина утверждена в текущей сессии.")
                name = discipline_filename(task, index, "txt")
                archive.writestr(name, readable_document(task))
                lines.append(f"РПД — {name}.")
                load = doc.get("workload", {})
                if {"totalHours", "totalCredits"} <= load.keys():
                    lines.append(f"Объём — {load['totalHours']} ч, {load['totalCredits']} ЗЕТ.")
                for semester in doc.get("curriculum", []):
                    topics = ", ".join(topic["title"] for topic in semester["topics"])
                    lines.append(f"Семестр {semester['semester']} — {topics}.")
                goals = doc.get("purposeAndTasks", {}).get("goals") or []
                if goals:
                    lines.append("Цели — " + " / ".join(goals))
            rows.append(lines)
        archive.writestr("course.txt", render_course_report(heading, rows))
    return output.getvalue()


def discipline_filename(task: Task, index: int, extension: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", task.request.discipline)[:110].strip(" .")
    return f"{index:02} - {name or 'РПД'}.{extension}"


def missing_official_fields(document: dict) -> list[str]:
    missing = []
    if not any(item.get("programCode") for item in document.get("programs", [])):
        missing.append("шифр направления")
    if not document.get("competencies"):
        missing.append("утверждённые компетенции")
    if not {"totalHours", "totalCredits"} <= document.get("workload", {}).keys():
        missing.append("часы и ЗЕТ")
    if not document.get("curriculum"):
        missing.append("семестры")
    if not document.get("assessment", {}).get("interimForms") and not any(
        semester.get("assessmentForm") for semester in document.get("workload", {}).get("bySemester", [])
    ):
        missing.append("форма аттестации")
    return missing


def readable_document(task: Task) -> str:
    text = render_rpd(task.document, document_schema())
    notes = missing_official_fields(task.document)
    if notes:
        text += "\nОфициальные сведения для сверки — " + ", ".join(notes) + ".\n"
    if task.approved:
        text += "\nДисциплина утверждена в текущей сессии приложения.\n"
    return text


def show_document(task: Task) -> None:
    document = task.document
    schema = document_schema()
    st.markdown("#### Рабочая программа дисциплины")
    st.text(document["title"], width="stretch")
    fields = schema["properties"]
    for key in dict.fromkeys([*fields, *document]):
        if key in ("title", "documentType") or key not in document:
            continue
        lines = readable_fields({key: document[key]}, schema, schema)
        if not lines:
            continue
        if len(lines) > 1:
            st.write(lines[0])
            lines = lines[1:]
        st.text("\n".join(lines), width="stretch")
    if notes := missing_official_fields(document):
        st.caption("Официальные сведения для сверки — " + ", ".join(notes) + ".")


def preview_requests(preview: PreviewSession) -> None:
    st.subheader("Черновики")
    if not preview.batches:
        st.info("Создайте заявку в разделе «Новая РПД».")
        return
    selected = st.session_state.pop("start_batch", None)
    if selected in preview.batches:
        run_queue(preview, preview.batches[selected])
    active = st.session_state.get("active_batch")
    keys = list(reversed(preview.batches))
    batch_id = st.selectbox("Заявка", keys, index=keys.index(active) if active in keys else 0,
                            format_func=lambda value: f"{preview.batches[value].title} · {preview.batches[value].year}")
    st.session_state["active_batch"] = batch_id
    batch = preview.batches[batch_id]
    count = sum(task.document is not None for task in batch.tasks)
    st.caption(f"Готово {count} из {len(batch.tasks)}")
    if any(task.status == "queued" for task in batch.tasks) and st.button("Запустить оставшиеся дисциплины"):
        run_queue(preview, batch)
        st.rerun()
    if batch.kind == "course":
        download, approve = st.columns(2)
        complete = count == len(batch.tasks)
        can_export = complete and all(not task.official_rpd_error for task in batch.tasks)
        download.download_button("Скачать отчёт и готовые РПД", download_package(batch) if can_export else b"",
                                 file_name="course-rpd.zip", mime="application/zip", disabled=not can_export,
                                 help="В архив входят отчёт и отдельная РПД для каждой дисциплины.")
        if approve.button("Утвердить курс", disabled=batch.approved or not complete,
                          help="Утверждает текущие версии РПД всех дисциплин."):
            batch.approve()
            st.rerun()
        if batch.approved:
            st.success("Курс утверждён")
    for index, task in enumerate(batch.tasks, 1):
        status = "Утверждена" if task.approved else STATUSES[task.status]
        label = f"{task.request.discipline} · {status}"
        with st.expander(label, expanded=len(batch.tasks) == 1):
            if doc := task.document:
                show_document(task)
                outputs = st.columns(3)
                if reason := task.official_rpd_error:
                    st.warning(f"Экспорт РПД приостановлен. {reason}")
                    if st.button("Перепроверить материалы РПД", key=f"official_{task.id}"):
                        try:
                            preview.recheck_official(task)
                            st.rerun()
                        except Exception as error:
                            st.error(request_error(error))
                else:
                    outputs[0].download_button("Скачать JSON", json.dumps(doc, ensure_ascii=False, indent=2),
                                               discipline_filename(task, index, "json"), "application/json", key=f"json_{task.id}")
                    outputs[1].download_button("Скачать TXT", readable_document(task),
                                               discipline_filename(task, index, "txt"), "text/plain", key=f"txt_{task.id}")
                if outputs[2].button("Утвердить дисциплину", key=f"approve_{task.id}", disabled=task.approved):
                    task.approve()
                    st.rerun()
                show_revisions(preview, task)
            if task.error:
                st.error(task.error)
            if task.error and st.button("Повторить обработку", key=f"retry_{task.id}"):
                with st.spinner("Обрабатываю заявку…"):
                    preview.advance(task)
                st.rerun()


def show_revisions(preview: PreviewSession, task: Task) -> None:
    st.markdown("#### Правки к РПД")
    for revision in task.revisions:
        with st.chat_message("user"):
            if revision.text:
                st.text(revision.text)
            for file in revision.attachments:
                st.text(f"Вложение: {file.name}")
        if revision.applied:
            with st.chat_message("assistant"):
                st.text("Правка обработана. Текущая версия РПД выше.")
    edit = st.chat_input(
        "Что изменить или уточнить",
        key=f"revision_{task.id}_{len(task.revisions)}",
        max_chars=6000,
        accept_file="multiple",
        file_type=["txt", "docx", "pdf"],
        max_upload_size=10,
        disabled=bool(task.error),
    )
    st.caption("Файлы можно прикрепить скрепкой: до трёх TXT, DOCX или PDF по 10 МБ.")
    if edit is not None:
        try:
            if len(edit.files) > 3:
                raise ValueError("Можно приложить до трёх файлов.")
            attachments = tuple(read_attachment(file.name, file.getvalue()) for file in edit.files)
            with st.spinner("Продолжаю подготовку РПД…"):
                preview.advance(task, edit.text, attachments)
            st.rerun()
        except ValueError as error:
            st.error(str(error))


def main() -> None:
    st.set_page_config(page_title="РПД · Рабочее пространство", layout="wide")
    st.title("Генератор РПД")
    if st.get_option("server.address") not in ("127.0.0.1", "::1", "localhost"):
        st.info("Веб-доступ откроется после подключения облачного входа.")
        return
    if st.session_state.get("preview_version") != 2:
        preview = PreviewSession.from_previous(st.session_state.get("preview"))
        st.session_state["preview"] = preview
        st.session_state["preview_version"] = 2
        active = st.session_state.get("active_batch")
        if active in preview.batches:
            st.session_state["start_batch"] = active
    preview = st.session_state["preview"]
    with st.sidebar:
        st.caption("Локальная демонстрация. Черновики доступны в текущей вкладке. Скачайте нужные файлы перед её закрытием.")
        if target := st.session_state.pop("navigate_to", None):
            st.session_state["navigation"] = target
        if st.session_state.get("navigation") == "Science":
            st.session_state["navigation"] = "Новая РПД"
        page = st.radio("Раздел", ("Новая РПД", "Черновики"), key="navigation")
    if page == "Новая РПД":
        prefill = st.session_state.pop("populate_course_names", None)
        if prefill:
            st.session_state["course_names"] = prefill
        if proposal := st.session_state.get("proposed_course"):
            with st.expander("Материалы для проверки предложенного состава"):
                for item in proposal[1]:
                    st.text(f"{item.name} · {', '.join(item.sources) or 'предложение модели'}")
                    if item.topics:
                        st.text(", ".join(item.topics))
        new_request(preview)
    else:
        preview_requests(preview)


if __name__ == "__main__":
    main()
