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

from storage.models import Base, Course, ExpertInvitation, Interview, now


class PortalUITests(unittest.TestCase):
    def test_sign_in_create_course_and_role_boundaries(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "portal.db"
            engine = create_engine("sqlite:///" + path.as_posix())
            Base.metadata.create_all(engine)
            with Session(engine) as db:
                db.add(Course(id="other", thread_id="other", owner_user_id="other-teacher",
                              status="waiting_for_expert", course_meta={"title": "Интервью о данных"}))
                db.flush()
                db.add(Interview(id="i", course_id="other", interrupt_id="ask", gap_report={"prompt": "Как сверяете данные?"}))
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
                app.text_input[0].set_value("tester@example.org")
                app.text_input[1].set_value("fixture")
                app.button[0].click().run()
                self.assertFalse(app.exception)
                self.assertIn("Создать курс", [b.label for b in app.button],
                              ([e.value for e in app.error], [b.label for b in app.button]))
                next(b for b in app.button if b.label == "Создать курс").click().run()
                for field, value in zip(app.text_input, ("Аналитика", "Вуз", "2026/2027"), strict=True):
                    field.set_value(value)
                next(b for b in app.button if b.label.startswith("✨")).click().run()
                self.assertFalse(app.exception)
                self.assertIn("Аналитика", [x.value for x in app.header])
                self.assertIn("В очереди", [x.value for x in app.markdown])
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
                self.assertEqual(conn.execute("select count(*) from workflow_commands").fetchone()[0], 2)
