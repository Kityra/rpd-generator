"""Local preview data for a single browser session."""

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

from langgraph.checkpoint.memory import InMemorySaver
from openai import APIConnectionError, APIStatusError, APITimeoutError
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from rpd_graph import RpdRequest
from run_course import Discipline
from run_rpd import create_rpd_graph, open_yandex, yandex_settings


UPLOAD_BYTES = 10 * 1024 * 1024


@dataclass(frozen=True)
class Attachment:
    name: str
    text: str


def read_attachment(name: str, data: bytes) -> Attachment:
    name = Path(name.replace("\\", "/")).name
    suffix = Path(name).suffix.lower()
    if not data or len(data) > UPLOAD_BYTES:
        raise ValueError(f"{name}: допустим непустой файл размером до 10 МБ.")
    try:
        if suffix == ".txt":
            text = data.decode("utf-8-sig")
        elif suffix == ".docx":
            with ZipFile(BytesIO(data)) as archive:
                info = archive.getinfo("word/document.xml")
                if info.file_size > 4 * 1024**2:
                    raise ValueError("слишком большой текст документа")
                root = ElementTree.fromstring(archive.read(info))
            word = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
            text = "\n".join("".join(node.itertext()) for node in root.iter(f"{word}t"))
        elif suffix == ".pdf":
            reader = PdfReader(BytesIO(data))
            if reader.is_encrypted or len(reader.pages) > 50:
                raise ValueError("используйте PDF без пароля и размером до 50 страниц")
            chunks = []
            for page in reader.pages:
                chunks.append((page.extract_text() or "")[:3000])
                if sum(map(len, chunks)) >= 3000:
                    break
            text = "\n".join(chunks)
        else:
            raise ValueError("доступные форматы — TXT, DOCX и PDF")
    except (BadZipFile, KeyError, ElementTree.ParseError, PdfReadError, ValueError, OSError) as error:
        raise ValueError(f"{name}: не удалось прочитать файл. {error}") from error
    return Attachment(name[:180], text.replace("\x00", "").strip()[:3000])


def request_error(error: Exception) -> str:
    if isinstance(error, APITimeoutError):
        return "Yandex AI Studio не ответила вовремя. Запрос можно повторить."
    if isinstance(error, APIConnectionError):
        return "Нет связи с Yandex AI Studio. Запрос можно повторить."
    if isinstance(error, APIStatusError):
        return f"Yandex AI Studio вернула HTTP {error.status_code}."
    if isinstance(error, (ValueError, RuntimeError)) and type(error).__module__ in ("builtins", "run_rpd"):
        return str(error)
    return "Обработка прервалась. Запрос можно повторить."


@dataclass
class Revision:
    id: str
    text: str
    attachments: tuple[Attachment, ...]
    model_input: str
    applied: bool = False


@dataclass
class Task:
    request: RpdRequest
    id: str = field(default_factory=lambda: uuid4().hex)
    status: str = "queued"
    revisions: list[Revision] = field(default_factory=list)
    document: dict[str, Any] | None = None
    error: str | None = None
    version: int = 0
    approved: bool = False

    def approve(self) -> None:
        if self.document is None:
            raise ValueError("Сначала подготовьте РПД.")
        self.approved = True


@dataclass
class Batch:
    title: str
    institution: str
    year: str
    kind: str
    tasks: list[Task]
    approved_versions: tuple[int, ...] = ()

    @property
    def approved(self) -> bool:
        return bool(self.approved_versions) and self.approved_versions == tuple(task.version for task in self.tasks)

    def approve(self) -> None:
        if not self.tasks or any(task.document is None for task in self.tasks):
            raise ValueError("Сначала подготовьте РПД всех дисциплин.")
        for task in self.tasks:
            task.approve()
        self.approved_versions = tuple(task.version for task in self.tasks)


@dataclass
class PreviewSession:
    batches: dict[str, Batch] = field(default_factory=dict)
    checkpoint: InMemorySaver = field(default_factory=InMemorySaver)

    @classmethod
    def from_previous(cls, previous) -> "PreviewSession":
        session = cls()
        if previous is None:
            return session
        for batch_id, batch in previous.batches.items():
            tasks = []
            for old in batch.tasks:
                request = RpdRequest.model_validate(old.request.model_dump())
                additions = "\n".join(reply.model_input for reply in getattr(old, "replies", []))
                if additions:
                    room = 16_000 - len(request.facts) - 1
                    if room > 0:
                        request.facts = request.facts + "\n" + additions[:room]
                task = Task(request, document=old.document)
                if task.document is not None:
                    task.status = "ready"
                    task.version = 1
                tasks.append(task)
            session.batches[batch_id] = Batch(batch.title, batch.institution, batch.year, batch.kind, tasks)
        return session

    def create_batch(self, title: str, institution: str, year: str, subjects: list[Discipline],
                     facts: str, *, kind: str, study_year: int | None = None) -> str:
        if not subjects:
            raise ValueError("Укажите хотя бы одну дисциплину.")
        tasks = []
        for item in subjects:
            notes = facts
            if kind == "course":
                notes += f"\nКонтекст комплекта — курс «{title}»."
            if study_year:
                notes += f"\nГод обучения — {study_year}. Семестр уточняется отдельно."
            if item.topics:
                notes += "\nТемы для обсуждения: " + ", ".join(item.topics)
            tasks.append(Task(RpdRequest(discipline=item.name, institution=institution, academic_year=year,
                                        program=title if kind == "course" else "", facts=notes)))
        batch_id = uuid4().hex
        self.batches[batch_id] = Batch(title, institution, year, kind, tasks)
        return batch_id

    def advance(self, task: Task, edit: str | None = None, attachments: tuple[Attachment, ...] = ()) -> None:
        if edit is not None:
            if task.document is None or task.error:
                raise ValueError("Правки доступны после подготовки РПД и завершения предыдущей обработки.")
            edit = edit.strip()
            excerpts = [f"Файл «{file.name}»:\n{file.text}" for file in attachments if file.text]
            model_input = "\n\n".join(part for part in (edit, *excerpts) if part)
            if len(edit) > 6000 or not model_input or len(attachments) > 3:
                raise ValueError("Введите правку до 6 000 символов или приложите до трёх файлов с текстом.")
            task.revisions.append(Revision(uuid4().hex, edit, attachments, model_input))
        elif task.status != "queued" and not task.error:
            raise ValueError("РПД уже подготовлена.")

        try:
            settings = yandex_settings()
            config = {"configurable": {"thread_id": task.id}}
            revision = task.revisions[-1] if task.revisions and not task.revisions[-1].applied else None
            edit_data = {"id": revision.id, "text": revision.model_input} if revision else None
            with open_yandex(settings) as client:
                graph = create_rpd_graph(client, settings, checkpointer=self.checkpoint, draft_first=True)
                previous = graph.get_state(config)
                if not previous.values:
                    result = graph.invoke({"request": task.request.model_dump(), "document": task.document,
                                           "revision": edit_data}, config=config)
                elif previous.next:
                    result = graph.invoke(None, config=config)
                elif revision and previous.values.get("applied_revision_id") != revision.id:
                    result = graph.invoke({"revision": edit_data}, config=config)
                else:
                    result = previous.values
            document = result["document"]
            if document != task.document:
                task.document = document
                task.version += 1
                task.approved = False
            if revision:
                revision.applied = True
            task.status = "ready"
            task.error = None
        except Exception as error:
            if task.document is None:
                task.status = "error"
            task.error = request_error(error)
