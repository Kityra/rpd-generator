import json
import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from zipfile import ZipFile

from apps.teacher_app import market_report
from domain.export import export_course, market_html
from domain.schemas import MarketDataset, MarketVacancy
from domain.skill_mapper import FALLBACK_COMPETENCE, ML_COMPETENCE, TOOL_COLUMN_NOTE, SkillMappingBatch, skill_mapper
from orchestration.clients import YandexModelClient
from orchestration.market_analysis import course_market
from test_course_architecture import program, records


DATABASE = "Проектирование и работа с базами данных"
OOP = "Объектно-ориентированное программирование"


class MappingTests(unittest.IsolatedAsyncioTestCase):
    def test_combines_distinct_tools_and_counts_unique_vacancies(self):
        dataset = MarketDataset(source="HH", vacancies=[
            MarketVacancy(id="a", title="Разработчик", skills=["c sharp", "C++", "Git", "SQL"]),
            MarketVacancy(id="a", title="Разработчик", skills=["C#"]),
            MarketVacancy(id="b", title="Разработчик", skills=["C++", "PostgreSQL"]),
            MarketVacancy(id="c", title="Аналитик", skills=["бизнес-анализ"]),
        ])
        meta = {"profession": "ML-инженер", "market": {"dataset": dataset.model_dump(mode="json")}}
        report = course_market(meta, program(), {}, [], None)
        self.assertEqual(report.sample_size, 3)
        rows = {row.competence_name: row for row in report.competences}
        self.assertCountEqual(rows[OOP].tools, ["C#", "C++"])
        self.assertCountEqual(rows[OOP].raw_skills, ["c sharp", "C++", "C#"])
        self.assertEqual((rows[OOP].vacancies, rows[OOP].share, rows[OOP].weight, rows[OOP].critical),
                         (2, 0.667, 0.667, True))
        self.assertEqual(rows[DATABASE].vacancies, 2)
        self.assertCountEqual(rows[DATABASE].tools, ["SQL", "PostgreSQL"])
        self.assertIn("Базы данных", rows[DATABASE].subjects)
        self.assertEqual(rows["Контроль версий и командная разработка"].tools, ["Git"])
        self.assertEqual(skill_mapper().resolve("PYTHON", "ML-инженер").competence_name, ML_COMPETENCE)
        self.assertNotEqual(skill_mapper().resolve("Python", "Бэкенд-разработчик").competence_name, ML_COMPETENCE)

    async def test_one_batch_and_sanitizer_ignore_invalid_model_items(self):
        fake = AsyncMock()
        fake.generate.return_value = SkillMappingBatch(items=[
            {"raw_skill": "NeoTool", "competence_name": "Сервисная разработка на C++ и NeoTool", "tool_name": "NeoTool"},
            {"raw_skill": "Инженерный эксперимент", "competence_name": "Проверка инженерных гипотез", "tool_name": None},
            {"raw_skill": "Выдуманный тег", "competence_name": "Проектирование", "tool_name": None},
        ])
        mapper = skill_mapper()
        mappings = await mapper.normalize_unknown(["SQL", "NeoTool", "Инженерный эксперимент", "neotool"], "Инженер", fake)
        fake.generate.assert_awaited_once()
        self.assertEqual(fake.generate.call_args.args[2]["raw_skills"], ["NeoTool", "Инженерный эксперимент"])
        self.assertEqual(mappings["neotool"], {"competence_name": FALLBACK_COMPETENCE, "tool_name": "NeoTool"})
        self.assertEqual(mapper.resolve("NeoTool", learned=mappings).competence_name, FALLBACK_COMPETENCE)
        self.assertEqual(mapper.resolve("Инженерный эксперимент", learned=mappings).tool_name, None)
        self.assertNotIn("выдуманный тег", mappings)
        fake.reset_mock()
        self.assertEqual(await mapper.normalize_unknown(["NeoTool"], "Инженер", fake, previous=mappings),
                         {"neotool": mappings["neotool"]})
        fake.generate.assert_not_awaited()

    def test_data_scientist_merges_python_into_ml_for_new_and_saved_reports(self):
        dataset = MarketDataset(source="HH", vacancies=[
            MarketVacancy(title="Data Scientist", skills=["Python", "машинное обучение"]),
            MarketVacancy(title="Data Scientist", skills=["машинное обучение"]),
            MarketVacancy(title="Data Scientist", skills=["Python"]),
        ])
        saved = course_market({"market": {"dataset": dataset.model_dump(mode="json")}}, program(), {}, [], None)
        meta = {"profession": "Data Scientist"}
        new = course_market({**meta, "market": {"dataset": dataset.model_dump(mode="json")}}, program(), {}, [], None)
        restored = course_market(meta, program(), {}, [], None, saved=saved.model_dump(mode="json"))
        for report in (new, restored):
            self.assertEqual(len(report.competences), 1)
            row = report.competences[0]
            self.assertEqual((row.competence_name, row.tools, row.vacancies), (ML_COMPETENCE, ["Python"], 3))

    def test_methodology_has_no_tool_and_ui_keeps_column_order(self):
        dataset = MarketDataset(source="HH", vacancies=[MarketVacancy(title="Аналитик", skills=["бизнес-анализ", "Git"])])
        report = course_market({"market": {"dataset": dataset.model_dump(mode="json")}}, program(), {}, [], None)
        with patch("apps.teacher_app.st") as ui:
            market_report(report)
            rows = ui.dataframe.call_args.args[0]
            method = next(row for row in rows if row["Компетенция"] == "Бизнес-анализ и работа с требованиями")
            self.assertEqual(list(method), ["Компетенция", "Инструменты", "Доля вакансий", "Вакансий",
                                           "Вес", "Критичный", "Покрытие", "Дисциплина"])
            self.assertEqual(method["Инструменты"], "—")
            self.assertIsInstance(method["Вес"], float)
            self.assertIs(method["Критичный"], True)
            self.assertIn("Контроль версий и командная разработка (Git)", str(ui.write.call_args_list))
            ui.caption.assert_any_call(TOOL_COLUMN_NOTE)
        self.assertIn(TOOL_COLUMN_NOTE, market_html(report.model_dump(mode="json")))

    def test_legacy_course_exports_clean_competence_and_subject_tools(self):
        legacy = {"status": "ok", "sample_size": 10, "skills": [
            {"skill": skill, "vacancies": count, "share": count / 10, "weight": None, "critical": True,
             "level": "full" if linked else "none", "subjects": ["Базы данных"] if linked else [],
             "topics": [], "tasks": [], "priority": 0.0}
            for skill, count, linked in [("SQL", 6, True), ("PostgreSQL", 4, False)]],
        }
        version = SimpleNamespace(status="approved", revision=1, content={"program": program(), "market_analysis": legacy})
        sources = {key: record.model_dump(mode="json") for key, record in records().items()}
        package = export_course({"title": "Курс", "institution": "Вуз", "year": "2026"}, version, sources)
        with ZipFile(BytesIO(package)) as archive:
            course = json.loads(archive.read("course.json"))
            report = archive.read("course-report.html").decode()
            rpd = json.loads(archive.read("rpd/01/rpd.json"))
            html = archive.read("rpd/01/rpd.html").decode()
        row = course["market_analysis"]["competences"][0]
        self.assertEqual((row["competence_name"], row["vacancies"], row["weight"]), (DATABASE, 6, 0.6))
        self.assertIn("нижняя граница", report)
        self.assertEqual(rpd["competenceLearningOutcomes"], [
            {"competence_name": DATABASE, "learningOutcomes": {"skillsAndTools": ["SQL"]}},
        ])
        self.assertIn(f"<td>{DATABASE}</td>", html)
        self.assertIn("Навыки и инструменты</strong><br>SQL", html)
        self.assertNotIn("<td>SQL</td>", html)
        self.assertNotIn("competences", version.content["market_analysis"])

    def test_business_model_has_priority_for_skill_mapping(self):
        config = {"YANDEX_FOLDER_ID": "folder", "YANDEX_MODEL_URI_BUSINESS": "gpt://folder/business"}
        self.assertEqual(YandexModelClient(config).model_for(SkillMappingBatch, True), "gpt://folder/business")
        self.assertEqual(YandexModelClient({"YANDEX_FOLDER_ID": "folder"}).model_for(SkillMappingBatch, True),
                         "gpt://folder/aliceai-llm-flash")
