"""Этап 5: проверка смысла обоснований, состава курса и качества покрытия."""
import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock

from pydantic import ValidationError

from domain.export import rationale_report_html
from domain.schemas import SubjectRationale
from literature_catalog import course_concepts, references_for_discipline
from orchestration.evidence_cache import source_record
from orchestration.semantic_check import (composition_report, concepts, duplicate_subjects, excerpt_supports_title,
                                          limit_fundamental, rationale_problem, task_support, weighted_coverage)
from orchestration.subject_rationale import explain_subjects
from test_course_architecture import program


WORKLOAD_QUOTE = "трудоемкость самостоятельной работы обучающегося по дисциплине (час) 302, контактная работа (час) 22"
REGRESSION = ("Линейная регрессия оценивает зависимость отклика от признаков методом наименьших квадратов; "
              "качество регрессионной модели проверяют по остаткам.")
REGRESSION_SIDE = ("Регрессионное тестирование программного обеспечения повторяет проверки после изменений "
                   "кода и не связано с анализом данных.")
POLYMERS = "Физика упругости каучука объясняет высокоэластическое состояние полимеров и энтропию."


def subject(key, title, topics=("Основы",)):
    value = program()["subjects"][0]
    return {**value, "id": key, "title": title, "is_fundamental": False, "locked": False,
            "topics": [{**value["topics"][0], "id": f"{key}-{i}", "title": name} for i, name in enumerate(topics)]}


class CodeCheckTests(unittest.TestCase):
    def test_synonyms_and_inflections_are_one_concept(self):
        self.assertEqual(concepts("Теория слабого и сильного ИИ"),
                         concepts("theory of weak (narrow) and strong AI"))
        self.assertIn("c:база данных", concepts("системы баз данных"))

    def test_workload_quote_is_rejected(self):
        self.assertFalse(excerpt_supports_title("Регрессионный анализ", WORKLOAD_QUOTE))
        problem = rationale_problem("Регрессионный анализ", "Дисциплина требует самостоятельной работы.", [WORKLOAD_QUOTE])
        self.assertIn("служебного", problem)
        record = source_record("RPD", "https://example.org/rpd", "РПД", WORKLOAD_QUOTE)
        with self.assertRaises(ValidationError):
            SubjectRationale.model_validate({
                "explanation": "Дисциплина требует самостоятельной работы обучающегося.",
                "source_ids": [record.source_id], "evidence": [{"source_id": record.source_id, "excerpt": WORKLOAD_QUOTE}],
            }, context={"sources": {record.source_id: record}, "rationale_subject": "Регрессионный анализ"})

    def test_supporting_quote_passes_and_unrelated_fails(self):
        self.assertIsNone(rationale_problem("Регрессия", "Регрессия оценивает зависимость отклика от признаков.",
                                            [REGRESSION]))
        self.assertIsNotNone(rationale_problem("Регрессия", "Курс опирается на физику полимеров.", [POLYMERS]))
        # Совпадение по слову есть, но объяснение не опирается на цитату.
        self.assertIsNotNone(rationale_problem("Регрессия", "Нужна для прогнозирования продаж компании.",
                                               [REGRESSION_SIDE]))


class JudgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_supports_weak_and_unrelated_verdicts(self):
        sources = [source_record("Books", f"https://example.org/{key}", key, text)
                   for key, text in (("reg", REGRESSION),
                                     ("ts", "Регрессия временных рядов: модели с лагами, где отклик зависит от признаков "
                                            "прошлых периодов."),
                                     ("glm", "Обобщённые линейные модели и логистическая регрессия: оценка параметров."))]
        evidence = AsyncMock()
        evidence.search.return_value = sources
        subjects = [subject("s1", "Регрессия"), subject("s2", "Регрессия временных рядов"), subject("s3", "Логистическая регрессия")]

        async def generate(output, prompt, payload, context, flash=False, **options):
            if output.__name__ == "SubjectRationaleBatch":
                return output.model_validate({"items": [
                    {"subject_id": row["subject_id"], "excerpt_ids": [row["fragments"][0]["excerpt_id"]],
                     "explanation": "Регрессия оценивает зависимость отклика от признаков модели."}
                    for row in payload["disciplines"]]}, context=context)
            status = {"с лагами": "weak", "логистическая": "unrelated"}
            return output.model_validate({"items": [
                {"pair_id": row["pair_id"], "verdict": next(
                    (verdict for marker, verdict in status.items() if marker in row["evidence"]), "supports")}
                for row in payload["pairs"]]}, context=context)

        llm = AsyncMock()
        llm.generate.side_effect = generate
        result, _ = await explain_subjects({"title": "Аналитик", "subjects": subjects}, {}, evidence, llm, "teacher")
        by_title = {item["title"]: item for item in result["subjects"]}
        self.assertIsNotNone(by_title["Регрессия"]["inclusion_rationale"])
        self.assertEqual(by_title["Регрессия"]["rationale_check"]["status"], "supports")
        for title, status in (("Регрессия временных рядов", "weak"), ("Логистическая регрессия", "unrelated")):
            self.assertIsNotNone(by_title[title]["inclusion_rationale"])
            self.assertEqual(by_title[title]["rationale_check"]["status"], status)
            self.assertEqual(by_title[title]["rationale_check"]["verification_status"], "needs_review")
        html = rationale_report_html(result, {source.source_id: source.model_dump(mode="json") for source in sources})
        self.assertEqual(html.count("Требует проверки"), 2)
        judged = [call for call in llm.generate.call_args_list if call.args[0].__name__ == "RationaleVerdicts"]
        self.assertEqual(len(judged), 3)
        for call in judged:
            self.assertEqual(set(call.args[2]), {"pairs"})
            self.assertTrue(all(set(pair) == {"pair_id", "claim", "source_type", "evidence"}
                                for pair in call.args[2]["pairs"]))
            self.assertEqual(call.kwargs["temperature"], 0.0)


class CompositionTests(unittest.TestCase):
    def test_duplicates_and_near_duplicates(self):
        subjects = [subject("a", "Машинное обучение"), subject("b", "Основы машинного обучения"),
                    subject("c", "Базы данных"), subject("d", "Базы данных и SQL"),
                    subject("e", "Визуализация данных")]
        pairs = [tuple(item["subjects"]) for item in duplicate_subjects(subjects)]
        self.assertIn(("Машинное обучение", "Основы машинного обучения"), pairs)
        self.assertIn(("Базы данных", "Базы данных и SQL"), pairs)
        self.assertFalse(any("Визуализация данных" in pair for pair in pairs))

    def test_disciplines_without_link_to_profession_are_flagged(self):
        subjects = [subject("a", "Промпт-инжиниринг", ("Составление промптов",)),
                    subject("b", "Физика полимеров", ("Упругость каучука",))]
        for item in subjects:
            for topic in item["topics"]:
                topic["activity_task_ids"] = []
        report = composition_report({"subjects": subjects}, {"title": "Промпт-инженер"},
                                    {"tasks": [{"title": "Составлять промпты для LLM", "stack": ["LLM"]}]})
        self.assertEqual(report["unlinked"], ["Физика полимеров"])

    def test_fundamental_flag_needs_rpd_or_books_and_limited_share(self):
        sources = {"Books": source_record("Books", "https://example.org/b", "b", "текст"),
                   "HH": source_record("HH", "https://example.org/h", "h", "текст")}
        sources = {"Books": sources["Books"], "HH": sources["HH"]}
        suggestions = [{"name": f"Д{i}", "is_fundamental": True, "conflict_with": None,
                        "importance": "core" if i == 5 else "standard", "source_ids": ["Books"]} for i in range(10)]
        suggestions.append({"name": "Без учебника", "is_fundamental": True, "conflict_with": None, "source_ids": ["HH"]})
        result, notes = limit_fundamental(suggestions, sources)
        flagged = [item["name"] for item in result if item["is_fundamental"]]
        self.assertEqual(len(flagged), 4)
        self.assertIn("Д5", flagged)
        self.assertNotIn("Без учебника", flagged)
        self.assertEqual(len(notes), 2)


class CoverageQualityTests(unittest.TestCase):
    def test_coverage_counts_type_freshness_and_independent_sources(self):
        now = datetime(2026, 11, 21, tzinfo=timezone.utc)
        hh = source_record("HH", "https://ai.api.cloud.yandex.net/v1/files/a", "hh.csv", "Вакансия 2026 SQL")
        hh_same = source_record("HH", "https://ai.api.cloud.yandex.net/v1/files/b", "hh.csv", "Вакансия 2026 Python")
        priklad = source_record("Priklad", "https://ai.api.cloud.yandex.net/v1/files/c", "practice.md", "Практика 2025")
        old = source_record("Crossref", "https://doi.org/10.1/x", "Статья", "Опубликовано 2012")
        fresh = source_record("Crossref", "https://doi.org/10.1/y", "Статья", "Опубликовано 2025")
        sources = {item.source_id: item for item in (hh, hh_same, priklad, old, fresh)}
        business = {"HH", "Priklad"}
        self.assertAlmostEqual(task_support([hh.source_id], sources, business, now), 0.8)
        # Два фрагмента одного файла — один источник.
        self.assertAlmostEqual(task_support([hh.source_id, hh_same.source_id], sources, business, now), 0.8)
        self.assertAlmostEqual(task_support([hh.source_id, priklad.source_id], sources, business, now), 1.0)
        self.assertLess(task_support([old.source_id], sources, {"Crossref"}, now),
                        task_support([fresh.source_id], sources, {"Crossref"}, now))

        class Task:
            def __init__(self, ids):
                self.source_ids = ids

        self.assertAlmostEqual(weighted_coverage([Task([hh.source_id]), Task([])], sources, business, now), 0.4)


class CatalogContextTests(unittest.TestCase):
    def test_catalog_uses_course_context_instead_of_topic_patch(self):
        rl_course = course_concepts(["Обучение с подкреплением", "Методы Монте-Карло"])
        self.assertTrue(references_for_discipline("Методы Монте-Карло", course=rl_course))
        self.assertEqual(references_for_discipline("Методы Монте-Карло", course=course_concepts(["Статистика"])), ())


if __name__ == "__main__":
    unittest.main()
