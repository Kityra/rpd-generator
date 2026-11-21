from io import BytesIO
import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from zipfile import ZipFile

from streamlit.testing.v1 import AppTest

from preview_app import download_package, show_revisions
from rpd_graph import build_rpd_graph
from run_course import Discipline
from test_pipeline import ROOT, SCHEMA, proposal
from web_preview import PreviewSession, read_attachment


def web_proposal(*, edited=False, only_pedagogy=False):
    payload = json.loads(proposal(("Какая форма аттестации?",)))
    document = payload["document"]
    document["purposeAndTasks"]["tasks"] = ["Составить архитектурную диаграмму" if edited else "Спроектировать модель"]
    if only_pedagogy:
        for key in ("workload", "programs", "competencies", "curriculum"):
            document.pop(key)
        document["appendices"] = [{"title": "Предлагаемый тематический план", "content": "Проектирование и SQL"}]
    return json.dumps(payload)


class WebPreviewTests(unittest.TestCase):
    def setUp(self):
        self.preview = PreviewSession()
        self.research = MagicMock(return_value={"fragments": [], "findings": []})
        self.generate = MagicMock()
        for target, options in (
            ("web_preview.yandex_settings", {"return_value": {}}),
            ("web_preview.open_yandex", {}),
            ("web_preview.create_rpd_graph", {"side_effect": lambda client, settings, checkpointer, draft_first:
                build_rpd_graph(SCHEMA, self.research, self.generate, checkpointer, draft_first=draft_first)}),
        ):
            patcher = patch(target, **options)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_revision_composer_accepts_attachment_without_text_and_limits_files(self):
        batch_id = self.preview.create_batch("Базы данных", "Вуз", "2026/2027",
                                             [Discipline("Базы данных")], "", kind="single")
        task = self.preview.batches[batch_id].tasks[0]
        file = SimpleNamespace(name="plan.txt", getvalue=lambda: "Добавить SQL".encode())
        preview = MagicMock()
        with patch("preview_app.st") as ui:
            ui.chat_input.return_value = SimpleNamespace(text="", files=[file])
            show_revisions(preview, task)
            attachments = preview.advance.call_args.args[2]
            self.assertEqual(preview.advance.call_args.args[1], "")
            self.assertEqual(attachments[0].text, "Добавить SQL")
            ui.rerun.assert_called_once()

            preview.reset_mock()
            ui.chat_input.return_value = SimpleNamespace(text="Правка", files=[file] * 4)
            show_revisions(preview, task)
            preview.advance.assert_not_called()
            ui.error.assert_called_with("Можно приложить до трёх файлов.")

    def test_failed_revision_preserves_document_and_retry_invalidates_approval(self):
        self.generate.side_effect = [
            web_proposal(), RuntimeError("Временная ошибка"), web_proposal(edited=True),
        ]
        batch_id = self.preview.create_batch("Базы данных", "Вуз", "2026/2027",
                                               [Discipline("Базы данных")], "", kind="single")
        batch = self.preview.batches[batch_id]
        task = batch.tasks[0]
        self.preview.advance(task)
        self.assertEqual(task.status, "ready")
        self.assertEqual(PreviewSession().batches, {})
        batch.approve()
        previous = task.document
        file = read_attachment("plan.txt", "Утверждённая форма: зачёт.".encode())
        self.preview.advance(task, "", (file,))
        self.assertEqual(task.error, "Временная ошибка")
        self.assertEqual(task.document, previous)
        self.assertTrue(batch.approved)
        self.preview.advance(task)
        self.assertEqual(task.status, "ready")
        self.assertEqual(self.research.call_count, 3)
        self.assertEqual(len(task.revisions), 1)
        self.assertFalse(batch.approved)
        self.assertFalse(task.approved)
        self.assertEqual(task.version, 2)
        payload = json.loads(self.generate.call_args.args[1])
        self.assertEqual(payload["previousDraft"], previous)
        self.assertIn("Утверждённая форма: зачёт.", payload["revision"]["text"])

    def test_course_can_be_approved_with_readable_file_per_discipline(self):
        self.generate.side_effect = [web_proposal(), web_proposal(only_pedagogy=True)]
        batch_id = self.preview.create_batch("Информатика", "Вуз", "2026/2027",
                                               [Discipline("Базы данных"), Discipline("Алгоритмы")], "", kind="course")
        batch = self.preview.batches[batch_id]
        with self.assertRaises(ValueError):
            download_package(batch)
        with self.assertRaises(ValueError):
            batch.approve()
        for task in batch.tasks:
            self.preview.advance(task)
        batch.approve()
        with ZipFile(BytesIO(download_package(batch))) as archive:
            self.assertCountEqual(archive.namelist(), ["01 - Базы данных.txt", "02 - Алгоритмы.txt", "course.txt"])
            report = archive.read("course.txt").decode()
            self.assertIn("Готовых РПД: 2 из 2", report)
            self.assertIn("Курс утверждён", report)
            plan = archive.read("02 - Алгоритмы.txt").decode()
            self.assertIn("Тематический план".casefold(), plan.casefold())
            self.assertIn("утверждённые компетенции", plan)
            self.assertNotIn("часы и ЗЕТ", plan)
        self.assertTrue(all(task.approved for task in batch.tasks))

    def test_network_binding_leaves_public_generation_unavailable(self):
        with patch("streamlit.get_option", return_value="0.0.0.0"):
            app = AppTest.from_file(str(ROOT / "preview_app.py"), default_timeout=20).run()
        self.assertFalse(app.exception)
        self.assertIn("после подключения облачного входа", app.info[0].value)
        self.assertFalse(app.button)
        self.assertFalse(app.text_input)
        self.assertEqual(self.generate.call_count, 0)

    def test_streamlit_course_suggestions_can_be_edited_before_submission(self):
        self.generate.return_value = web_proposal()
        app = AppTest.from_file(str(ROOT / "preview_app.py"), default_timeout=20).run()
        next(item for item in app.radio if item.label == "Вид заявки").set_value("Комплект на курс").run()
        for field, value in zip(app.text_input, ("Информатика", "Вуз", "2026/2027"), strict=True):
            field.set_value(value)
        with patch("run_rpd.yandex_settings", return_value={}), patch("run_rpd.open_yandex"), \
             patch("run_course.discover_disciplines", return_value=[
                 Discipline("Базы данных", ("SQL",), ("plan.pdf",)),
                 Discipline("Алгоритмы"),
             ]):
            next(item for item in app.button if item.label == "Подобрать дисциплины").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.text_area(key="course_names").value, "Базы данных\nАлгоритмы")
        self.assertEqual(app.session_state["preview"].batches, {})
        app.text_area(key="course_names").set_value("Базы данных\nСети")
        next(item for item in app.button if item.label == "Подготовить комплект").click().run()
        self.assertFalse(app.exception)
        self.assertEqual([item.label for item in app.expander], ["Базы данных · РПД готова", "Сети · РПД готова"])
        button = next(item for item in app.button if item.label == "Утвердить курс")
        self.assertFalse(button.disabled)
        button.click().run()
        self.assertFalse(app.exception)
        self.assertTrue(all("Утверждена" in item.label for item in app.expander))

    def test_streamlit_displays_document_before_optional_revision(self):
        self.generate.side_effect = [web_proposal(only_pedagogy=True), web_proposal(edited=True)]

        def button(app, label):
            return next(item for item in app.button if item.label == label)

        app = AppTest.from_file(str(ROOT / "preview_app.py"), default_timeout=20).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.radio(key="navigation").options, ["Новая РПД", "Черновики"])
        self.assertFalse(any(item.label == "Войти" for item in app.button))
        for field, value in zip(app.text_input, ("Базы данных", "Вуз", "2026/2027"), strict=True):
            field.set_value(value)
        button(app, "Подготовить РПД").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.radio(key="navigation").value, "Черновики")
        self.assertIn("РПД готова", app.expander[0].label)
        self.assertFalse(app.get("chat_message"))
        self.assertTrue(app.get("download_button"))
        self.assertEqual(self.generate.call_args.args[2]["required"], ["document"])
        button(app, "Утвердить дисциплину").click().run()
        self.assertIn("Утверждена", app.expander[0].label)
        app.chat_input[0].set_value("Добавь архитектурную диаграмму").run()
        self.assertFalse(app.exception)
        self.assertIn("РПД готова", app.expander[0].label)
        self.assertTrue(any("Составить архитектурную диаграмму" in item.value for item in app.text))


if __name__ == "__main__":
    unittest.main()
