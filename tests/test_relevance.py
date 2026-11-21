"""Этап 4: поиск источников по смыслу — запрос, оценка близости, служебные фрагменты."""
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from orchestration import yandex_calls
from orchestration.clients import YandexEvidenceClient
from orchestration.evidence_cache import source_record
from orchestration.graph import merge_records
from orchestration.rabota_business import _course_topic
from orchestration.relevance_filter import filter_fragments, search_query, service_reason
from orchestration.resilience import Resilience


def fragment(index, title, text, score):
    return source_record(index, f"https://ai.api.cloud.yandex.net/v1/files/{abs(hash(text))}", title, text,
                         relevance_score=score)


RUBBER = "Физика упругости каучука. Высокоэластическое состояние полимеров и энтропийная природа упругости."
PROMPTS = ("Промпт-инженер составляет инструкции для языковых моделей, проверяет ответы на тестовых наборах "
           "и фиксирует критерии качества генерации.")
BIBLIOGRAPHY = "\n".join([
    "1. Иванов И. И. Основы программирования. – М.: Наука, 2019. – 320 с.",
    "2. Петров П. П. Нейронные сети. – СПб.: Питер, 2020. – 256 с. ISBN 978-5-4461-0000-0",
    "3. Сидоров С. С. Машинное обучение. – М.: Издательство МГТУ, 2021. – 410 с.",
    "4. Электронно-библиотечная система Лань. URL: https://e.lanbook.com"])
WORKLOAD = ("Вид учебной работы Трудоемкость (час) Всего 324 Контактная работа 22 Лекции 8 Практические 14 "
            "трудоемкость самостоятельной работы обучающегося (час) 302 Зачёт 4")


class QueryTests(unittest.TestCase):
    def test_query_is_built_from_course_profession_and_tasks_not_json(self):
        query = search_query({"title": "Промпт-инженер", "profession": "Специалист по LLM", "institution": "ЮФУ",
                              "year": "2026"}, ["Оценка ответов модели", "Составление инструкций"])
        self.assertFalse(query.lstrip().startswith("{"))
        self.assertNotIn("ЮФУ", query)
        self.assertNotIn("2026", query)
        self.assertEqual(query.splitlines()[0], "Промпт-инженер")
        self.assertIn("Специалист по LLM", query)
        self.assertIn("Оценка ответов модели", query)
        self.assertEqual(_course_topic(query), "Промпт-инженер")

    def test_service_fragments_are_recognised(self):
        self.assertEqual(service_reason(BIBLIOGRAPHY), "список литературы")
        self.assertEqual(service_reason(WORKLOAD), "таблица трудоёмкости")
        self.assertIsNone(service_reason(PROMPTS))

    def test_low_score_and_service_fragments_are_dropped(self):
        kept, dropped = filter_fragments([fragment("Books", "rubber.pdf", RUBBER, 0.12),
                                          fragment("Books", "llm.pdf", PROMPTS, 0.71),
                                          fragment("Books", "list.pdf", BIBLIOGRAPHY, 0.9),
                                          fragment("Books", "old.pdf", PROMPTS + " Старая запись.", None)], 0.3)
        self.assertEqual([item.title for item in kept], ["llm.pdf", "old.pdf"])
        self.assertEqual(dropped, {"близость": 1, "список литературы": 1})

    def test_same_fragment_with_different_scores_merges(self):
        first = fragment("Books", "llm.pdf", PROMPTS, 0.71).model_dump(mode="json")
        second = {**first, "relevance_score": 0.5}
        self.assertEqual(len(merge_records([first], [second])), 1)


class EvidenceClientTests(unittest.TestCase):
    def setUp(self):
        self.settings = {"YANDEX_FOLDER_ID": "folder", "YANDEX_API_KEY": "fixture", "VECTOR_MIN_SCORE": "0.3",
                         **{f"YANDEX_VECTOR_STORE_ID_{name}": name for name in ("PRIKLAD", "HH", "RPD", "BOOKS")}}

    @patch("orchestration.clients.open_yandex")
    @patch("orchestration.clients.search_fragments")
    def test_irrelevant_low_score_fragment_does_not_reach_registry(self, search, open_client):
        search.side_effect = [((fragment("Priklad", "rubber.pdf", RUBBER, 0.12),
                                fragment("Priklad", "prompts.pdf", PROMPTS, 0.68)), None),
                              ((fragment("HH", "hh.csv", WORKLOAD, 0.8),), None)]
        with self.assertLogs("orchestration.clients", level="WARNING"):
            found = YandexEvidenceClient(self.settings)._search("business", search_query({"title": "Промпт-инженер"}),
                                                                "teacher")
        self.assertEqual([item.title for item in found], ["prompts.pdf"])
        self.assertNotIn("Физика упругости каучука", " ".join(item.content for item in found))
        self.assertEqual(found[0].relevance_score, 0.68)
        self.assertTrue(any("ниже порога близости — 1" in note for note in found.notices))
        self.assertTrue(any("таблица трудоёмкости — 1" in note for note in found.notices))

    @patch("orchestration.clients.open_yandex")
    @patch("orchestration.clients.search_fragments")
    def test_required_rpd_keeps_content_fragments_with_warning(self, search, open_client):
        search.side_effect = [((fragment("RPD", "rpd.pdf", PROMPTS, 0.1),), None), ((), None)]
        with self.assertLogs("orchestration.clients", level="WARNING"):
            found = YandexEvidenceClient(self.settings)._search("university", "Промпт-инженер", "teacher")
        self.assertEqual([item.title for item in found], ["rpd.pdf"])
        self.assertTrue(any("ниже порога" in note for note in found.notices))


class VectorScoreTests(unittest.TestCase):
    def test_score_from_vector_search_is_saved_in_source(self):
        hit = SimpleNamespace(file_id="f1", filename="methods.md", score=0.83,
                              content=[SimpleNamespace(type="text", text="Метод критических инцидентов")])
        client = MagicMock()
        client.vector_stores.search.return_value = SimpleNamespace(data=[hit])
        with patch.object(yandex_calls, "EXTERNAL", Resilience(wait=lambda delay: None)):
            fragments, taken_at = yandex_calls.search_fragments(client, {"YANDEX_FOLDER_ID": "f"}, "Interview",
                                                                "store", "запрос", 2)
        self.assertIsNone(taken_at)
        self.assertEqual(fragments[0].relevance_score, 0.83)


if __name__ == "__main__":
    unittest.main()
