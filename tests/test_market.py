"""Этап 6: анализ рынка и навыков кодом."""
import json
import unittest
from pathlib import Path

from domain.export import market_html
from orchestration.evidence_cache import source_record
from orchestration.market_analysis import analyse_market, course_market, dataset_from_registry, parse_market_file
from test_course_architecture import program


ROOT = Path(__file__).resolve().parents[1]


def course():
    value = program()
    base = value["subjects"][0]
    def subject(key, title, topic, assignment=""):
        topic_block = {**base["topics"][0], "id": f"{key}-t", "title": topic, "activity_task_ids": [],
                       "assignments": [{**base["topics"][0]["assignments"][0], "id": f"{key}-a",
                                        "instructions": assignment}] if assignment else []}
        return {**base, "id": key, "title": title, "topics": [topic_block]}
    value["subjects"] = [subject("s1", "Базы данных", "Запросы SQL"),
                         subject("s2", "Анализ данных на Python", "Библиотека pandas",
                                 "Постройте дашборд в Power BI по результатам анализа"),
                         subject("s3", "История философии", "Античная философия")]
    return value


def vacancies(rows):
    return json.dumps({"source": "Тестовая выборка", "region": "Ростов-на-Дону", "period": "2026-10",
                       "vacancies": [{"title": title, "skills": skills, "text": text} for title, skills, text in rows]},
                      ensure_ascii=False).encode()


SAMPLE = [("Аналитик данных", ["SQL", "Python"], "Опыт работы с Excel"),
          ("Аналитик", ["SQL"], "Power BI, визуализация данных"),
          ("Младший аналитик", [], "Знание SQL и Python, английский язык"),
          ("BI-аналитик", ["Power BI", "SQL"], ""),
          ("Data analyst", ["Python"], "A/B тесты")]


class FrequencyTests(unittest.TestCase):
    def test_skill_frequencies_and_thresholds_are_counted_by_code(self):
        analysis = analyse_market(course(), parse_market_file("sample.json", vacancies(SAMPLE)))
        rows = {row.skill: row for row in analysis.skills}
        self.assertEqual(analysis.sample_size, 5)
        self.assertEqual((rows["SQL"].vacancies, rows["SQL"].share), (4, 0.8))
        self.assertEqual((rows["Python"].vacancies, rows["Python"].share), (3, 0.6))
        self.assertEqual(rows["Power BI"].vacancies, 2)
        self.assertEqual(rows["Excel"].vacancies, 1)
        # Критичные — не ниже 30 % выборки (config/market_skills.yaml), остальные — дополнительные.
        self.assertTrue(rows["SQL"].critical and rows["Python"].critical and rows["Power BI"].critical)
        self.assertFalse(rows["Excel"].critical or rows["английский язык"].critical)
        self.assertEqual(analysis.warnings, [])

    def test_coverage_levels_gaps_and_formula(self):
        analysis = analyse_market(course(), parse_market_file("sample.json", vacancies(SAMPLE)))
        rows = {row.skill: row for row in analysis.skills}
        self.assertEqual(rows["SQL"].level, "full")
        self.assertEqual(rows["SQL"].subjects, ["Базы данных"])
        self.assertEqual(rows["Python"].level, "full")
        self.assertEqual(rows["Power BI"].level, "partial")
        self.assertEqual(rows["Excel"].level, "none")
        self.assertIn("Excel", analysis.uncovered)
        self.assertEqual(analysis.weak_subjects, ["История философии"])
        # Критичные: SQL (1), Python (1), Power BI (0.5) → 2.5 / 3.
        self.assertAlmostEqual(analysis.market_fit, 0.833, places=3)
        self.assertEqual(analysis.skills[0].skill, "Power BI")  # наибольший дефицит среди критичных
        self.assertEqual(rows["SQL"].priority, 0.0)

    def test_manual_override_changes_coverage(self):
        dataset = parse_market_file("sample.json", vacancies(SAMPLE))
        analysis = analyse_market(course(), dataset, overrides={"Excel": ["Анализ данных на Python"]})
        excel = next(row for row in analysis.skills if row.skill == "Excel")
        self.assertEqual((excel.level, excel.manual, excel.subjects), ("full", True, ["Анализ данных на Python"]))

    def test_small_sample_is_flagged(self):
        analysis = analyse_market(course(), parse_market_file("sample.json", vacancies(SAMPLE[:2])))
        self.assertTrue(analysis.warnings and "ненадёжны" in analysis.warnings[0])


class InputTests(unittest.TestCase):
    def test_no_data_does_not_invent_numbers(self):
        analysis = course_market({"title": "Курс"}, course(), {}, [], None)
        self.assertEqual(analysis.status, "no_data")
        self.assertIsNone(analysis.market_fit)
        self.assertEqual(analysis.skills, [])
        html = market_html(analysis.model_dump(mode="json"))
        self.assertIn("Рыночные данные не загружены", html)
        self.assertNotIn("%", html)

    def test_csv_skill_weights_and_vacancy_rows(self):
        skills = parse_market_file("skills.csv", "skill;weight;aliases\n1С;10;1C Бухгалтерия\nМСФО;4;\n".encode())
        analysis = analyse_market(course(), skills)
        rows = {row.skill: row for row in analysis.skills}
        self.assertTrue(rows["1С"].critical)
        self.assertFalse(rows["МСФО"].critical)
        self.assertIsNone(rows["1С"].share)
        rows_csv = parse_market_file("v.csv", "title,skills\nБухгалтер,1С; МСФО\nКассир,1С\n".encode())
        self.assertEqual(len(rows_csv.vacancies), 2)

    def test_templates_are_empty_and_valid_structure(self):
        template = json.loads((ROOT / "templates" / "market_data.json").read_text(encoding="utf-8"))
        self.assertEqual((template["skills"], template["vacancies"]), ([], []))
        self.assertTrue((ROOT / "templates" / "market_skills.csv").read_text(encoding="utf-8").startswith("skill;"))

    def test_found_vacancy_cards_are_a_sample(self):
        cards = [source_record("Rostrud", f"https://trudvsem.ru/vacancy/{i}", f"Аналитик {i}",
                               f"Работодатель: А\nОпубликована: 2026-10-0{i}\nТребования: SQL, Excel")
                 for i in range(1, 6)]
        registry = {card.source_id: card.model_dump(mode="json") for card in cards}
        dataset = dataset_from_registry(registry, list(registry))
        self.assertEqual(dataset.period, "2026-10-01 — 2026-10-05")
        analysis = analyse_market(course(), dataset)
        sql = next(row for row in analysis.skills if row.skill == "SQL")
        self.assertEqual((sql.vacancies, sql.share), (5, 1.0))


if __name__ == "__main__":
    unittest.main()
