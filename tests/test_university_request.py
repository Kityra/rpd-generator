import json
import unittest

from pydantic import ValidationError

from domain.schemas import SubjectPlan
from orchestration.clients import grounded_schema
from orchestration.evidence_cache import source_record
from orchestration.university_request import build_university_request


class UniversityRequestTests(unittest.TestCase):
    def test_compact_academic_request_keeps_relevant_excerpts_and_full_records_for_validation(self):
        topic = "Оптимизация SQL и индексация в базах данных"
        academic = source_record("Books", "https://example.org/book", "Книга", "Введение " * 330 + topic)
        rpd = source_record("RPD", "https://example.org/plan", "Учебный план", "Базы данных. SQL и транзакции.")
        science = source_record("Crossref", "https://example.org/paper", "Статья", "Транзакции.")
        method = source_record("Interview", "https://example.org/method", "Методика", "Техника интервью.")
        jobs = [source_record("HH", f"https://example.org/jobs/{i}", f"Вакансия {i}", "Работа " * 800)
                for i in range(30)]
        registry = {source.source_id: source for source in [academic, rpd, science, method, *jobs]}
        state = {
            "course_meta": {"title": "Базы данных", "year": "2026/2027", "expand_subjects": False},
            "manual_subjects": ["Базы данных"],
            "university_report": {"rpd_source_ids": [rpd.source_id],
                                  "rpd_lookup": {"status": "snapshot", "reason": ""}},
            "books_report": {"findings": [{"summary": topic, "source_ids": [academic.source_id]}]},
            "science_report": {"findings": [{"summary": "Транзакции", "source_ids": [science.source_id]}]},
            "applied_activity": {"tasks": [{"id": "task", "title": "Индексация",
                                           "source_ids": [jobs[0].source_id]}]},
            "baseline": {"subjects": [{"title": "Основы БД", "is_fundamental": True, "locked": True,
                                       "topics": [{"title": "Помеченный длинный текст " * 300}]}]},
            "interview_answers": [{"content": "Отдельный текст интервью"}],
        }

        request, selected = build_university_request(state, registry)

        self.assertEqual(request["manual_subjects"], state["manual_subjects"])
        self.assertFalse(request["course_meta"]["expand_subjects"])
        self.assertEqual(request["baseline_subjects"], [{"title": "Основы БД", "is_fundamental": True, "locked": True}])
        self.assertEqual(request["rpd_lookup"]["status"], "snapshot")
        self.assertEqual(set(selected), {academic.source_id, rpd.source_id, science.source_id,
                                         jobs[0].source_id, jobs[-1].source_id})
        self.assertIs(selected[academic.source_id], academic)
        excerpt = next(source["content"] for source in request["sources"] if source["source_id"] == academic.source_id)
        self.assertIn(topic, excerpt)
        self.assertLess(len(excerpt), len(academic.content))
        self.assertLess(len(json.dumps(request, ensure_ascii=False)), 4500)
        self.assertNotIn("interview_answers", request)

        context = {"sources": selected}
        schema = grounded_schema(SubjectPlan, context)
        ids = schema["$defs"]["SubjectSuggestion"]["properties"]["source_ids"]["items"]["enum"]
        self.assertEqual(set(ids), set(selected))
        subject = {"name": "Базы данных", "is_fundamental": True, "conflict_with": None,
                   "source_ids": [academic.source_id]}
        SubjectPlan.model_validate({"subjects": [subject]}, context=context)
        subject["source_ids"] = [jobs[1].source_id]
        with self.assertRaises(ValidationError):
            SubjectPlan.model_validate({"subjects": [subject]}, context=context)
