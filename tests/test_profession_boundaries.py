import hashlib
import sqlite3
import tempfile
import unittest
from datetime import timedelta
from contextlib import closing
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from zipfile import ZipFile

from alembic import command
from alembic.config import Config
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from auth.settings import Settings
from domain.export import export_course
from orchestration.interview_skills import BOUNDARY_SLOTS, SKILL_FILES, load_interview_skills, select_interview_method
from orchestration.profession_boundaries import (boundary_conflicts, check_subject_boundary_violations,
                                                 compute_adjacent_profession_candidates, with_boundary_warnings)
from storage.activity_models import ModelVersionRow, Profession
from storage.database import MIGRATIONS, ensure_schema, make_engine
from storage.models import Base, Course, CourseVersion, ProfessionBoundary, now
from storage.portal import create_course
from storage.profession_boundaries import (create_boundary_interview_invite, save_boundary_review,
                                            submit_boundary_turn, sync_boundary_candidates, validate_boundary_invite)
from test_course_architecture import program, records
from test_portals import user


class MigrationTests(unittest.TestCase):
    def test_upgrade_from_previous_revision_preserves_course(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.sqlite"
            url = "sqlite+aiosqlite:///" + path.as_posix()
            config = Config()
            config.set_main_option("script_location", str(MIGRATIONS))
            config.attributes["database_url"] = url
            command.upgrade(config, "0005_judge_subject_id")
            with closing(sqlite3.connect(path)) as db:
                db.execute("INSERT INTO courses (id, owner_user_id, thread_id, course_meta, status, created_at, updated_at)"
                           " VALUES ('legacy','owner','legacy','{}','approved',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)")
                db.commit()
            ensure_schema(url)
            with closing(sqlite3.connect(path)) as db:
                self.assertEqual(db.execute("SELECT id, status FROM courses").fetchall(), [("legacy", "approved")])
                columns = {row[1] for row in db.execute("PRAGMA table_info(profession_boundaries)")}
                self.assertTrue({"id", "course_id", "primary_profession", "adjacent_profession", "skill_jaccard_overlap",
                    "overlap_tasks_json", "exclusive_to_primary_json", "exclusive_to_adjacent_json", "shared_core_json",
                    "boundary_conflicts_json", "invite_token_hash", "invite_expires_at", "source_interview_id",
                    "verified_by_methodologist", "created_at"} <= columns)
                db.execute("INSERT INTO profession_boundaries (course_id, primary_profession, adjacent_profession,"
                           " skill_jaccard_overlap) VALUES ('legacy','A','B',0.45)")
                self.assertEqual(db.execute("SELECT overlap_tasks_json, verified_by_methodologist, invite_token_hash "
                                            "FROM profession_boundaries").fetchone(), ("[]", 0, None))


class CorrelationTests(unittest.TestCase):
    def test_candidate_thresholds_and_exact_intersection(self):
        primary = [f"skill{index:02}" for index in range(18)]
        extras = ["extralib01", "extralib02"]
        catalog = {"Инженер данных": primary, "Аналитик данных": primary[:9] + extras,
                   "Дубликат": primary + extras, "Далёкая роль": primary[:2] + extras}
        result, = compute_adjacent_profession_candidates("Инженер данных", primary, catalog)
        self.assertEqual((result.adjacent_profession, result.skill_jaccard_overlap), ("Аналитик данных", 0.45))
        self.assertEqual(result.overlapping_items, primary[:9])

    def test_contrastive_skill_has_fixed_slots_and_wins_router(self):
        directory = Path(__file__).resolve().parents[1] / "skills" / "interview_methods"
        skills = load_interview_skills(directory)
        self.assertEqual(set(skills), set(SKILL_FILES))
        method = select_interview_method(sample_size=0, rag_count=0, has_unverified=True, missing_lab_tasks=True,
                                         missing_slots=[], interview_mode="boundary_disambiguation", overlapping_items=["SQL"])
        self.assertEqual(method, "contrastive_boundary_interview")
        self.assertEqual(skills[method].output_slots, list(BOUNDARY_SLOTS))
        self.assertEqual(skills[method].max_turns, 3)

    def test_cross_expert_conflict_matches_inflection_and_context(self):
        items = ["Проектирование витрин данных", "Подготовка интервью"]
        primary = ["Проектирование витрин данных для BI"]
        self.assertEqual(boundary_conflicts(items, primary), [items[0]])

    def test_post_validator_keeps_subject_and_zip_includes_exact_warning(self):
        value = program()
        value["subjects"][0]["topics"][0]["title"] = "Проектирование витрин данных"
        row = {"primary_profession": "Инженер данных", "adjacent_profession": "BI-аналитик",
               "skill_jaccard_overlap": 0.45, "exclusive_to_adjacent": ["Проектирование витрин данных"],
               "boundary_conflicts": [], "verified_by_methodologist": True, "source_interview_id": "interview-123"}
        with patch("orchestration.graph.make_outline") as outline:
            warning, = check_subject_boundary_violations(value, [row])
            decorated = with_boundary_warnings(value, [row], threshold=0.5)
        outline.assert_not_called()
        self.assertEqual(warning.source_interview_id, "interview-123")
        self.assertEqual(warning.matched_topic, value["subjects"][0]["topics"][0]["title"])
        subject = decorated["subjects"][0]
        self.assertTrue(subject["needs_review"])
        self.assertEqual(subject["id"], value["subjects"][0]["id"])
        self.assertEqual(subject["topics"], value["subjects"][0]["topics"])
        self.assertFalse(check_subject_boundary_violations(value, [{**row, "verified_by_methodologist": False}]))
        version = SimpleNamespace(status="approved", revision=1,
                                  content={"program": decorated, "profession_boundaries": [row]})
        corpus = {key: source.model_dump(mode="json") for key, source in records().items()}
        with ZipFile(BytesIO(export_course({"institution": "Вуз", "year": "2026"}, version, corpus))) as archive:
            for path in ("course-report.html", "rpd/01/rpd.html"):
                report = archive.read(path).decode()
                self.assertIn("[Пересечение со смежной профессией «BI-аналитик» — требует решения методиста]", report)
                self.assertIn("interview-123", report)


class BoundaryStorageTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "boundaries.sqlite"
        engine = make_engine("sqlite+aiosqlite:///" + self.path.as_posix())
        self.addAsyncCleanup(engine.dispose)
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        self.factory = async_sessionmaker(engine, expire_on_commit=False)
        self.actor = user("teacher")
        await create_course(self.factory, self.actor, "course", {"title": "Инженер данных"}, [])
        async with self.factory.begin() as db:
            row = ProfessionBoundary(course_id="course", primary_profession="Инженер данных",
                adjacent_profession="BI-аналитик", skill_jaccard_overlap=0.45, primary_claims_json=[
                    "Проектирование витрин данных"],
                overlap_tasks_json=["Проектирование витрин данных", "Архитектура БД", "SQL", "Подготовка отчёта"])
            db.add(row)
            await db.flush()
            self.boundary_id = row.id

    async def invite(self, **options):
        return await create_boundary_interview_invite(self.factory, self.actor, "course", self.boundary_id, **options)

    async def test_invite_hash_expiration_rotation_and_course_scope(self):
        expired = await self.invite(ttl=timedelta(seconds=-1))
        self.assertIsNone(await validate_boundary_invite(self.factory, expired))
        token = await self.invite()
        session = await validate_boundary_invite(self.factory, token)
        self.assertEqual(session["id"], self.boundary_id)
        self.assertNotIn("invite_token_hash", session)
        tampered = token[:-1] + ("x" if token[-1] != "x" else "y")
        self.assertIsNone(await validate_boundary_invite(self.factory, tampered))
        with self.assertRaises(PermissionError):
            await create_boundary_interview_invite(self.factory, user("teacher", "other"), "course", self.boundary_id)
        with self.assertRaises(PermissionError):
            await create_boundary_interview_invite(self.factory, self.actor, "other-course", self.boundary_id)
        async with self.factory() as db:
            row = await db.get(ProfessionBoundary, self.boundary_id)
            self.assertEqual(row.invite_token_hash, hashlib.sha256(token.encode()).hexdigest())
        self.assertNotIn(token.encode(), self.path.read_bytes())
        await self.invite()
        self.assertIsNone(await validate_boundary_invite(self.factory, token))

    async def test_gibberish_whitelist_and_conflict_persist_across_turns(self):
        token = await self.invite()
        settings = Settings(_env_file=None, session_signing_key="x" * 40)
        fake = AsyncMock()
        first, note = await submit_boundary_turn(self.factory, token, "бла-бла-бла", fake, settings, expected_turn=0)
        fake.generate.assert_not_awaited()
        self.assertEqual(first["rejected_noise_turns"], [1])
        self.assertTrue(note)
        self.assertFalse(any(first[slot] for slot in BOUNDARY_SLOTS))

        async def generate(output, prompt, payload, context, **options):
            self.assertEqual(payload["requested_slots"], list(BOUNDARY_SLOTS))
            self.assertEqual(payload["overlapping_items"], first["overlapping_items"])
            self.assertNotIn(token, payload["answer"])
            if output.__name__ == "InterviewerTurnQuality":
                return output(classification="valid_answer")
            return {"exclusive_to_primary": ["Архитектура БД"], "exclusive_to_adjacent": [
                        "Проектирование витрин данных", "Подготовка отчёта", "Настройка Kubernetes"],
                    "shared_core": ["SQL"], "handoff_artifacts": ["схема обмена", "выдуманный договор"],
                    "extra_file": "result.docx", "subjects": ["Тест"]}

        fake.generate.side_effect = generate
        message = ("Архитектура БД — целевая. Проектирование витрин данных и Подготовка отчёта — смежная. "
                   "SQL — общий пункт, артефакт — схема обмена. Настройка Kubernetes — лишнее." + token)
        second, note = await submit_boundary_turn(self.factory, token, message, fake, settings, expected_turn=1, finish=True)
        self.assertIsNone(note)
        self.assertEqual(second["exclusive_to_primary"], ["Архитектура БД"])
        self.assertEqual(second["exclusive_to_adjacent"], ["Подготовка отчёта"])
        self.assertEqual(second["shared_core"], ["SQL"])
        self.assertEqual(second["handoff_artifacts"], ["схема обмена"])
        self.assertEqual(second["boundary_conflicts"], ["Проектирование витрин данных"])
        self.assertTrue(second["interview_completed"])
        self.assertNotIn("extra_file", second)
        self.assertNotIn("subjects", second)
        self.assertEqual(second["rejected_noise_turns"], [1])
        with self.assertRaises(PermissionError):
            await submit_boundary_turn(self.factory, token, message, fake, settings, expected_turn=1)

    async def test_sync_uses_confirmed_catalog_and_approval_updates_existing_version(self):
        async with self.factory.begin() as db:
            role = Profession(id="neighbor", name="Инженер платформы")
            db.add(role)
            db.add(ModelVersionRow(profession_id=role.id, number=1, as_of=now(), config_hash="test", input_hash="test",
                content={"tasks": {"a": {"name": "SQL", "status": "confirmed"},
                                   "b": {"name": "Оркестрация", "status": "confirmed"},
                                   "c": {"name": "Непроверенная задача", "status": "preliminary"}}}))
            version = CourseVersion(id="draft", course_id="course", revision=1, content={"program": program()})
            db.add(version)
            (await db.get(Course, "course")).draft_version_id = version.id
        rows = await sync_boundary_candidates(self.factory, "course", {"profession_name": "Инженер данных", "claims": [
            {"category": "tool_stack", "statement": name, "verification_status": "verified"}
            for name in ("SQL", "Kafka")]})
        candidate = next(row for row in rows if row["adjacent_profession"] == "Инженер платформы")
        self.assertEqual(candidate["overlapping_items"], ["SQL"])
        self.assertAlmostEqual(candidate["skill_jaccard_overlap"], 1 / 3)
        await save_boundary_review(self.factory, self.actor, "course", self.boundary_id,
                                   {"exclusive_to_adjacent": ["SQL"]}, True, threshold=0.5)
        async with self.factory() as db:
            version = await db.get(CourseVersion, "draft")
            boundary = await db.scalar(select(ProfessionBoundary).where(ProfessionBoundary.id == self.boundary_id))
        self.assertTrue(boundary.verified_by_methodologist)
        self.assertTrue(version.content["program"]["subjects"][0]["needs_review"])
