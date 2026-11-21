import os
import hashlib
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from argon2 import PasswordHasher
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from datetime import timedelta
from streamlit.testing.v1 import AppTest

from apps.presentation import progress_html
from storage.models import Base, Course, ExpertInvitation, Interview, now


class PortalUITests(unittest.TestCase):
    def test_progress_distinguishes_waiting_approval_and_failure(self):
        waiting = progress_html("waiting_for_expert")
        self.assertIn("Ждёт интервью эксперта", waiting)
        self.assertIn('aria-current="step"><span class="portal-step-line" aria-hidden="true"></span><span>Модель деятельности', waiting)
        self.assertEqual(progress_html("approved").count("portal-step--complete"), 6)
        failed = progress_html("failed")
        self.assertNotIn('aria-current="step"', failed)
        self.assertNotIn("portal-step--complete", failed)

    def test_sign_in_create_course_and_role_boundaries(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "portal.db"
            engine = create_engine("sqlite:///" + path.as_posix())
            Base.metadata.create_all(engine)
            with Session(engine) as db:
                db.add(Course(id="other", thread_id="other", owner_user_id="other-teacher",
                              status="waiting_for_expert", course_meta={"title": "Интервью о данных"}))
                db.add(Course(id="own", thread_id="own", owner_user_id="users:1",
                              status="waiting_for_expert", course_meta={"title": "Аналитика продаж"}))
                db.flush()
                db.add(Interview(id="i", course_id="other", interrupt_id="ask", gap_report={"prompt": "Как сверяете данные?"}))
                db.add(Interview(id="i-own", course_id="own", interrupt_id="ask-own",
                                 gap_report={"prompt": "Какие метрики используете?"}))
                db.flush()
                db.add(ExpertInvitation(id="invite", interview_id="i", expert_user_id="users:2",
                       token_hash=hashlib.sha256(b"sample").hexdigest(), expires_at=now()+timedelta(days=1), created_by="other-teacher"))
                db.commit()
            engine.dispose()
            with closing(sqlite3.connect(path)) as conn, conn:
                conn.execute("create table users(id integer, email text, display_name text, role text, password_hash text)")
                conn.execute("insert into users values (1, 'tester@example.org', 'Преподаватель', 'teacher', ?)",
                             (PasswordHasher().hash("fixture"),))
                conn.execute("insert into users values (2, 'expert@example.org', 'Эксперт', 'expert', ?)",
                             (PasswordHasher().hash("fixture"),))
            url = "sqlite+aiosqlite:///" + path.as_posix()
            env = dict(APP_DATABASE_URL=url, USER_DATABASE_URL=url, TEACHERS_INDEX_ID="", EXPERTS_INDEX_ID="",
                       SESSION_SIGNING_KEY="t" * 48)
            with patch.dict(os.environ, env):
                app = AppTest.from_string("from apps.teacher_app import main\nmain()", default_timeout=30).run()
                self.assertFalse(app.exception)
                self.assertIn("Кабинет методиста", [title.value for title in app.title])
                app.text_input[0].set_value("tester@example.org")
                app.text_input[1].set_value("fixture")
                app.button[0].click().run()
                self.assertFalse(app.exception)
                self.assertIn("Создать курс", [b.label for b in app.button],
                              ([e.value for e in app.error], [b.label for b in app.button]))
                next(b for b in app.button if b.label == "Создать курс").click().run()
                self.assertFalse(app.number_input)
                for field, value in zip(app.text_input, ("Аналитика", "Вуз", "2026/2027", "Аналитик данных"), strict=True):
                    field.set_value(value)
                next(b for b in app.button if b.label == "Добавить предметы автоматически").click().run()
                self.assertFalse(app.exception)
                self.assertIn("Аналитика", [x.value for x in app.header])
                self.assertTrue(any("В очереди" in item.proto.body and "Статус обновляется автоматически" in item.proto.body
                                    for item in app.get("html")))
                self.assertNotIn("Обновить статус", [b.label for b in app.button])
                app.button(key="nav-own").click().run()
                next(b for b in app.button if b.label == "Сформировать черновик по имеющимся данным").click().run()
                self.assertFalse(app.exception)
                self.assertTrue(any("В очереди" in item.proto.body for item in app.get("html")))
                expert = AppTest.from_string("from apps.expert_app import main\nmain()", default_timeout=30)
                expert.session_state["session_token"] = app.session_state["session_token"]
                expert.run()
                self.assertTrue(expert.error)
                self.assertFalse(expert.chat_input)
                next(b for b in expert.button if b.label == "Выйти").click().run()
                self.assertTrue(expert.text_input)
                expert.query_params["token"] = "sample"
                expert.text_input[0].set_value("expert@example.org")
                expert.text_input[1].set_value("fixture")
                expert.button[0].click().run()
                self.assertFalse(expert.exception)
                expert.chat_input[0].set_value("Сверяю результаты запросов.").run()
                next(b for b in expert.button if b.label.startswith("Завершить интервью")).click().run()
                self.assertFalse(expert.exception)
                self.assertTrue(expert.success)
            with closing(sqlite3.connect(path)) as conn:
                self.assertEqual(conn.execute("select count(*) from workflow_commands").fetchone()[0], 3)
                self.assertEqual(conn.execute("select status from interviews where id = 'i-own'").fetchone(), ("closed",))

    def test_course_page_with_legacy_draft_and_approved_version_renders(self):
        from test_course_architecture import program, records
        from storage.models import CourseVersion, SourceRow
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "portal.db"
            engine = create_engine("sqlite:///" + path.as_posix())
            Base.metadata.create_all(engine)
            legacy = program()  # версия до новых полей: нет generated_by, rationale_check, workload_proposal
            content = {"program": legacy, "activity": {}, "evidence_gaps": {"gap_types": ["missing_tasks"]}}
            with Session(engine) as db:
                db.add(Course(id="own", thread_id="own", owner_user_id="users:1", status="waiting_for_review",
                              course_meta={"title": "Аналитика", "institution": "Вуз", "year": "2026"},
                              draft_version_id="draft", active_version_id="active"))
                db.flush()
                db.add(CourseVersion(id="active", course_id="own", revision=1, status="approved",
                                     content={**content, "workloads": {}}))
                db.add(CourseVersion(id="draft", course_id="own", revision=2, status="draft", content=content))
                for key, record in records().items():
                    db.add(SourceRow(course_id="own", source_id=key, record=record.model_dump(mode="json")))
                db.commit()
            engine.dispose()
            with closing(sqlite3.connect(path)) as conn, conn:
                conn.execute("create table users(id integer, email text, display_name text, role text, password_hash text)")
                conn.execute("insert into users values (1, 'tester@example.org', 'Преподаватель', 'teacher', ?)",
                             (PasswordHasher().hash("fixture"),))
            url = "sqlite+aiosqlite:///" + path.as_posix()
            env = dict(APP_DATABASE_URL=url, USER_DATABASE_URL=url, TEACHERS_INDEX_ID="", EXPERTS_INDEX_ID="",
                       SESSION_SIGNING_KEY="t" * 48)
            with patch.dict(os.environ, env):
                app = AppTest.from_string("from apps.teacher_app import main\nmain()", default_timeout=30).run()
                app.text_input[0].set_value("tester@example.org")
                app.text_input[1].set_value("fixture")
                app.button[0].click().run()
                next(b for b in app.button if b.label == "Открыть курс").click().run()
                self.assertFalse(app.exception, app.exception)
                self.assertEqual([tab.label for tab in app.tabs], ["Программа", "Отчёт по курсу"])
                html = " ".join(item.proto.body for item in app.get("html"))
                self.assertIn("Базы данных", html)
                self.assertTrue(any("Рыночные данные не загружены" in item.value for item in app.info))
                self.assertTrue(any(b.label.startswith("Скачать утверждённую версию") for b in app.get("download_button")))
