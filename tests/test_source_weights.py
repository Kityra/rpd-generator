import unittest
from unittest.mock import AsyncMock

from pydantic import ValidationError

from domain.source_weights import SourceWeights, course_source_weights
from orchestration.graph import context, subject_request
from orchestration.source_conflicts import ConflictingPair, SourceConflicts, resolve_source_conflicts
from orchestration.clients import source_record


class SourceWeightsTests(unittest.IsolatedAsyncioTestCase):
    async def test_conflicting_database_choices_follow_weights_and_require_real_quotes(self):
        interview = source_record("expert", "https://example.org/interview", "Интервью", "Для хранилища используется PostgreSQL.")
        books = source_record("Books", "https://example.org/books", "Базы данных", "Для хранилища используется SQLite.")
        pair = ConflictingPair(left_source_id=books.source_id, left_quote=books.content,
                               right_source_id=interview.source_id, right_quote=interview.content)
        invalid = pair.model_copy(update={"left_quote": "Хранилище работает на другой базе."})
        llm = AsyncMock()
        llm.generate.return_value = SourceConflicts(pairs=[pair, invalid])
        rows = [item.model_dump(mode="json") for item in (books, interview)]
        for weights, winner in ((SourceWeights(books=0.2), interview), (SourceWeights(interview=0.2), books),
                                (SourceWeights(books=0.2, interview=0.2), None)):
            with self.subTest(weights=weights):
                results = await resolve_source_conflicts(rows, weights, llm)
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0]["preferred_source_id"], winner.source_id if winner else None)
                self.assertEqual(results[0]["preferred_quote"], winner.content if winner else None)

        weights = SourceWeights(books=0.0)
        conflicts = await resolve_source_conflicts(rows, weights, llm)
        state = {"course_meta": {"title": "Базы данных", "source_weights": weights.model_dump()},
                 "sources": rows, "source_conflicts": conflicts}
        request, selected = subject_request(state, ["Базы данных"], [], context(state))
        self.assertEqual(selected[0], interview.source_id)
        self.assertEqual(request["source_conflicts"][0]["preferred_quote"], interview.content)
        self.assertEqual(request["sources"][0]["source_weight"], 1.0)

    def test_old_courses_have_neutral_defaults_and_range_is_validated(self):
        self.assertEqual(set(course_source_weights({}).model_dump().values()), {1.0})
        for value in (True, "0.5", -0.01, 1.01, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                SourceWeights(books=value)
