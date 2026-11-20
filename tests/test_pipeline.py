import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import httpx

from jsonschema import Draft202012Validator
from langgraph.types import Command

import run_course
import run_rpd
import science_literature as science
from rpd_graph import build_rpd_graph, response_schema_for
from rpd_specialists import PROFILES


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "final_schema.json"
REQUEST = {
    "discipline": "Базы данных", "academic_year": "2026/2027",
    "institution": "Учебный полигон",
}
ENVIRONMENT = {
    "YANDEX_API_KEY": "test", "YANDEX_FOLDER_ID": "test", "YANDEX_MODEL_URI": "test",
    "YANDEX_VECTOR_STORE_ID_RPD": "test", "YANDEX_BASE_URL": run_rpd.YANDEX_BASE_URL,
}
RPD_ARGS = [
    "run_rpd.py", "--discipline", REQUEST["discipline"], "--year", REQUEST["academic_year"],
    "--institution", REQUEST["institution"],
]


def proposal(questions=()):
    return json.dumps({
        "document": {
            "metadata": {"sourceFiles": None},
            "programs": [{"programCode": "01.03.02"}],
            "purposeAndTasks": {"goals": ["Проектировать базы данных"]},
            "competencies": [{"code": "ПК-1", "description": "Проектирует базы данных"}],
            "workload": {"totalHours": 108, "totalCredits": 3},
            "curriculum": [{"semester": 3, "topics": [{"title": "SQL"}]}],
        },
        "questions": list(questions),
    })


class RpdPipelineTests(unittest.TestCase):
    def setUp(self):
        self.schema = json.loads(SCHEMA.read_text(encoding="utf-8-sig"))
        self.config = {"configurable": {"thread_id": self.id()}}
        self.research = MagicMock(return_value={"fragments": [], "findings": []})

    def test_schema_and_source_roles(self):
        Draft202012Validator.check_schema(self.schema)
        projected = response_schema_for(self.schema)
        Draft202012Validator.check_schema(projected)
        self.assertEqual(
            {role: [search.index for search in profile.searches] for role, profile in PROFILES.items()},
            {"university": ["RPD", "Books"], "science": ["Books"], "business": ["Priklad", "HH"]},
        )
        draft_schema = projected["properties"]["document"]
        self.assertEqual(draft_schema["required"], list(draft_schema["properties"]))
        self.assertFalse(draft_schema["additionalProperties"])

    def test_repairs_schema_and_resumes_without_searching_again(self):
        invalid = json.loads(proposal())
        invalid["document"]["workload"]["unknownField"] = 30
        generate = MagicMock(side_effect=[
            json.dumps(invalid), proposal(("Какую форму аттестации указать?", "|", ".")), proposal((".",)),
        ])
        graph = build_rpd_graph(SCHEMA, self.research, generate)
        paused = graph.invoke({"request": REQUEST}, config=self.config)
        self.assertEqual(paused["__interrupt__"][0].value["questions"], ["Какую форму аттестации указать?"])
        retry_payload = json.loads(generate.call_args_list[1].args[1])
        self.assertTrue(any("$.workload" in error for error in retry_payload["schemaErrors"]))
        answer = "Зачёт."
        result = graph.invoke(Command(resume=answer), config=self.config)
        resume_payload = json.loads(generate.call_args_list[2].args[1])
        self.assertEqual(resume_payload["interviewAnswers"][0]["answer"], answer)
        self.assertCountEqual([call.args[0] for call in self.research.call_args_list], PROFILES)
        self.assertTrue(result["ready"])
        self.assertNotIn("sourceFiles", result["document"]["metadata"])
        Draft202012Validator(self.schema).validate(result["document"])

    def test_stops_after_three_unresolved_answers(self):
        generate = MagicMock(return_value='{"document": {}, "questions": []}')
        graph = build_rpd_graph(SCHEMA, self.research, generate)
        result = graph.invoke({"request": REQUEST}, config=self.config)
        for _ in range(3):
            self.assertTrue(result["__interrupt__"])
            result = graph.invoke(Command(resume="Уточняю сведения."), config=self.config)
        self.assertFalse(result.get("__interrupt__"))
        self.assertFalse(result["ready"])
        self.assertEqual(generate.call_count, 4)
        self.assertEqual(self.research.call_count, 3)

    def test_cli_saves_one_valid_rpd_without_overwriting(self):
        client = MagicMock()
        client.vector_stores.search.return_value = SimpleNamespace(data=[])
        client.responses.create.return_value = SimpleNamespace(
            incomplete_details=None, output=[], status="completed", output_text=proposal(),
        )
        with TemporaryDirectory() as directory:
            output = Path(directory) / "rpd.json"
            with patch.dict("os.environ", ENVIRONMENT, clear=True), \
                 patch("sys.argv", [*RPD_ARGS, "--output", str(output)]), \
                 patch.object(run_rpd, "load_dotenv"), patch.object(run_rpd, "OpenAI") as sdk, \
                 patch.object(run_rpd, "collect_recent_articles", return_value=[]), \
                 patch("sys.stdout"), patch("sys.stderr"):
                sdk.return_value.__enter__.return_value = client
                self.assertEqual(run_rpd.main(), 0)
                original = output.read_text(encoding="utf-8")
                with self.assertRaises(SystemExit) as error:
                    run_rpd.main()
                self.assertEqual(error.exception.code, 2)
                self.assertEqual(output.read_text(encoding="utf-8"), original)
                self.assertEqual(client.responses.create.call_count, 1)
        Draft202012Validator(self.schema).validate(json.loads(original))

    def test_university_does_not_forward_fragments_after_moderation(self):
        for incomplete_details, output in (
            (SimpleNamespace(reason="content_filter"), []),
            (None, [SimpleNamespace(type="message", content=[SimpleNamespace(type="refusal")])]),
        ):
            with self.subTest(incomplete_details=incomplete_details):
                client = MagicMock()
                client.vector_stores.search.return_value = SimpleNamespace(data=[
                    SimpleNamespace(filename="source.pdf", content=[SimpleNamespace(type="text", text="Материал")])
                ])
                client.responses.create.return_value = SimpleNamespace(
                    incomplete_details=incomplete_details, output=output, status="incomplete", output_text="",
                )
                graph = MagicMock()

                def build(schema_path, research, generate, checkpointer=None, draft_first=False):
                    report = research("university", run_rpd.RpdRequest(**REQUEST))
                    self.assertEqual(report, {"fragments": [], "findings": []})
                    return graph

                graph.invoke.return_value = {"ready": False, "questions": []}
                with patch.dict("os.environ", ENVIRONMENT, clear=True), patch("sys.argv", RPD_ARGS), \
                     patch.object(run_rpd, "load_dotenv"), patch.object(run_rpd, "OpenAI") as sdk, \
                     patch.object(run_rpd, "build_rpd_graph", side_effect=build), \
                     patch("sys.stdout"), patch("sys.stderr"):
                    sdk.return_value.__enter__.return_value = client
                    self.assertEqual(run_rpd.main(), 1)


class SciencePipelineTests(unittest.TestCase):
    def test_course_retries_a_short_crossref_rate_limit(self):
        response = httpx.Response(429, headers={"Retry-After": "1"}, request=httpx.Request("GET", science.CROSSREF))
        error = httpx.HTTPStatusError("Rate limited", request=response.request, response=response)
        with patch.object(science, "has_retraction", side_effect=[error, False]) as check, \
             patch.object(science, "sleep") as sleep:
            self.assertFalse(science.check_course_retraction(MagicMock(), "10.1234/good"))
        sleep.assert_called_once_with(1)
        self.assertEqual(check.call_count, 2)

    def test_course_search_rejects_tangential_papers_and_retracted_versions(self):
        good = science.Article("Relational database query optimization", 2026, "https://doi.org/10.1234/good",
                               "published_candidate", doi="10.1234/good", citations=15)
        retracted = science.Article("Database query planning", 2026, "https://doi.org/10.1234/retracted",
                                    "published_candidate", doi="10.1234/retracted")
        unrelated = science.Article("Navigation in ocean currents", 2026, "https://doi.org/10.1234/other",
                                    "published_candidate", doi="10.1234/other", citations=200)
        preprints = [
            science.Article(retracted.title, 2026, "https://arxiv.org/abs/2611.00001", "arxiv_candidate",
                            doi=retracted.doi),
            science.Article("Database query workloads", 2026, "https://arxiv.org/abs/2611.00002", "arxiv_candidate"),
        ]
        with patch.object(science, "crossref_articles", return_value=[unrelated, good, retracted]) as crossref, \
             patch.object(science, "arxiv_articles", return_value=preprints), \
             patch.object(science, "has_retraction", side_effect=lambda client, doi: doi == retracted.doi) as check:
            cards, unavailable = science.find_course_articles("Relational database query", verified_dois={good.doi})
        self.assertEqual(crossref.call_args.kwargs["query"], "relational database query")
        self.assertCountEqual([call.args[1] for call in check.call_args_list], [good.doi, retracted.doi])
        self.assertEqual(unavailable, [])
        self.assertEqual([(card.url, card.status) for card in cards],
                         [(good.url, "verified"), (preprints[1].url, "arxiv_candidate")])

    def test_only_science_sees_article_cards_and_proposals_reach_rpd(self):
        settings = {**ENVIRONMENT, "YANDEX_VECTOR_STORE_ID_BOOKS": "books"}
        client = MagicMock()
        client.vector_stores.search.return_value = SimpleNamespace(data=[])
        article = science.Article("Relational database query optimization", 2026, "https://doi.org/10.1234/good",
                                  "published_candidate", doi="10.1234/good", authors=("Example Author",), citations=15,
                                  abstract="The evaluation compares physical query plans.")
        proposal_data = json.loads(proposal())
        proposal_data["document"]["purposeAndTasks"]["tasks"] = ["Сравнить планы запросов"]
        responses = {
            "science_course_query": {"query": "database query optimization"},
            "rpd_science": {"findings": [{"fragment": 1, "summary": "Сравнить планы SQL-запросов."}]},
            "rpd_draft": proposal_data,
        }
        calls = {}

        def model(**request):
            name = request["text"]["format"]["name"]
            calls[name] = request
            return SimpleNamespace(incomplete_details=None, output=[], status="completed",
                                   output_text=json.dumps(responses[name]))

        client.responses.create.side_effect = model
        graph = run_rpd.create_rpd_graph(client, settings, draft_first=True)
        with patch.object(run_rpd, "find_course_articles", return_value=([article], [])):
            result = graph.invoke({"request": {**REQUEST, "facts": "Внутренние данные вуза"}},
                                  config={"configurable": {"thread_id": "science-in-rpd"}})
        self.assertNotIn("Внутренние данные вуза", calls["science_course_query"]["input"])
        self.assertIn("Example Author", calls["rpd_science"]["input"])
        self.assertIn(article.abstract, calls["rpd_science"]["input"])
        payload = json.loads(calls["rpd_draft"]["input"])
        self.assertNotIn("Example Author", calls["rpd_draft"]["input"])
        self.assertNotIn(article.abstract, calls["rpd_draft"]["input"])
        self.assertEqual(payload["recentScientificSuggestions"][0]["url"], article.url)
        addition = result["document"]["appendices"][0]["content"]
        self.assertIn("Сравнить планы SQL-запросов.", addition)
        self.assertIn(article.url, addition)

    def test_checks_approved_dois_and_keeps_citation_counts_out_of_model_input(self):
        accepted = science.Article("Approved", 2026, "https://doi.org/10.1234/ok", "published_candidate",
                                   doi="10.1234/ok", area="data")
        retracted = science.Article("Retracted", 2026, "https://doi.org/10.1234/retracted", "published_candidate",
                                    doi="10.1234/retracted", area="data")
        preprint = science.Article("Preprint", 2026, "https://arxiv.org/abs/2611.00001", "arxiv_candidate",
                                   doi=retracted.doi, area="data")
        with patch.object(science, "crossref_articles", return_value=[accepted, retracted]), \
             patch.object(science, "arxiv_articles", return_value=[preprint]), \
             patch.object(science, "has_retraction", side_effect=[False, True]) as check, \
             patch.object(science, "semantic_citations", return_value={accepted.doi: 12}):
            cards, unavailable = science.search_frontier(
                "data", verified_dois={accepted.doi, retracted.doi}, semantic_key="test",
            )
        self.assertEqual(unavailable, [])
        self.assertEqual(check.call_count, 2)
        self.assertEqual([(card.doi, card.status) for card in cards], [(accepted.doi, "verified")])
        self.assertEqual(cards[0].citations, 12)
        self.assertNotIn("Semantic Scholar", cards[0].as_science_source()["text"])


class CoursePipelineTests(unittest.TestCase):
    def test_searches_rpd_and_books_but_only_rpd_can_support_a_discipline(self):
        def hits(filename):
            return SimpleNamespace(data=[
                SimpleNamespace(filename=filename, content=[SimpleNamespace(type="text", text="Учебный материал")]),
            ])

        client = MagicMock()
        client.vector_stores.search.side_effect = [hits("plan.pdf"), hits("book.pdf")]
        plan = {"disciplines": [
            {"name": "Алгоритмы", "topics": ["Графы"], "fragments": [1, 2, 9]},
            {"name": " Базы данных ", "topics": ["SQL"], "fragments": [2]},
            {"name": "базы   данных", "topics": [], "fragments": []},
        ]}
        client.responses.create.return_value = SimpleNamespace(
            incomplete_details=None, output=[], status="completed", output_text=json.dumps(plan),
        )
        with patch.dict("os.environ", {"YANDEX_VECTOR_STORE_ID_BOOKS": "books"}, clear=True):
            result = run_course.discover_disciplines(client, ENVIRONMENT, "Информатика", "Вуз", "2026/2027", 2)
        self.assertEqual(result, [
            run_course.Discipline("Алгоритмы", ("Графы",), ("plan.pdf",)),
            run_course.Discipline("Базы данных", ("SQL",)),
        ])
        self.assertIn("2 курс обучения", client.vector_stores.search.call_args_list[0].kwargs["query"])
        payload = json.loads(client.responses.create.call_args.kwargs["input"])
        self.assertEqual([part["index"] for part in payload["fragments"]], ["RPD", "Books"])

    def test_batch_saves_completed_documents_and_lists_unresolved_disciplines(self):
        client = MagicMock()
        client.vector_stores.search.return_value = SimpleNamespace(data=[])
        document = SimpleNamespace(incomplete_details=None, output=[], status="completed", output_text=proposal())
        incomplete = SimpleNamespace(incomplete_details=SimpleNamespace(reason="max_output_tokens"),
                                     output=[], status="incomplete", output_text="")
        client.responses.create.side_effect = [document, incomplete, document]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            disciplines = root / "disciplines.txt"
            disciplines.write_text("Базы данных\nТеория алгоритмов\nСети\nбазы данных\n", encoding="utf-8")
            output = root / "course"
            args = ["--course", "Информатика", "--institution", "Вуз", "--year", "2026/2027",
                    "--disciplines-file", str(disciplines), "--output-dir", str(output)]
            with patch.dict("os.environ", ENVIRONMENT, clear=True), \
                 patch.object(run_course, "load_dotenv"), patch.object(run_rpd, "load_dotenv"), \
                 patch.object(run_rpd, "collect_recent_articles", return_value=[]), \
                 patch.object(run_rpd, "OpenAI") as sdk, patch("sys.stdout"), patch("sys.stderr"):
                sdk.return_value.__enter__.return_value = client
                self.assertEqual(run_course.main(args), 1)
            schema = json.loads(SCHEMA.read_text(encoding="utf-8-sig"))
            validator = Draft202012Validator(schema)
            for number, title in [(1, "Базы данных"), (3, "Сети")]:
                rpd = json.loads((output / f"{number:02}-rpd.json").read_text(encoding="utf-8"))
                validator.validate(rpd)
                self.assertEqual(rpd["title"], title)
                self.assertIn("Тематический план", (output / f"{number:02}-rpd.txt").read_text(encoding="utf-8"))
            self.assertFalse((output / "02-rpd.json").exists())
            summary = (output / "course.txt").read_text(encoding="utf-8")
            self.assertIn("Готовых РПД: 2 из 3", summary)
            self.assertIn("Черновик не сохранён: Теория алгоритмов", summary)
            self.assertIn("Семестр 3 — SQL", summary)
            for call in client.responses.create.call_args_list:
                request = json.loads(call.kwargs["input"])["request"]
                self.assertIn("курс «Информатика»", request["facts"])


if __name__ == "__main__":
    unittest.main()
