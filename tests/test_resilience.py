from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import date
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import httpx
from openai import AuthenticationError, InternalServerError

from orchestration import yandex_calls
from orchestration.clients import YandexEvidenceClient, YandexModelClient
from orchestration.graph import AgentReport
from orchestration.resilience import ExternalUnavailable, Resilience
import run_rpd
import science_literature as science


def sdk_error(status):
    response = httpx.Response(status, request=httpx.Request("POST", "https://example.org"))
    return (AuthenticationError if status == 401 else InternalServerError)("error", response=response, body=None)


def reply(text="", *, moderated=False):
    return SimpleNamespace(incomplete_details=SimpleNamespace(reason="content_filter") if moderated else None,
                           output=[], status="completed", output_text=text)


class ResilienceTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.delays = []
        self.policy = Resilience(clock=lambda: self.now, wait=self.advance, jitter=lambda low, high: low,
                                 invoke=lambda request, remaining: request())
        self.service = ("Crossref", "search")

    def advance(self, seconds):
        self.delays.append(seconds)
        self.now += seconds

    def call(self, request, budget=8):
        return self.policy.call(self.service, request, budget=budget, read=3)

    def test_503_recovers_and_retry_after_cannot_exceed_total_budget(self):
        request = MagicMock(side_effect=[sdk_error(503), "ok"])
        self.assertEqual(self.call(request), "ok")
        self.assertEqual(self.delays, [0.25])
        self.assertEqual(request.call_args.args[0].connect, 3)
        self.assertEqual(request.call_args.args[0].read, 3)

        response = httpx.Response(429, headers={"Retry-After": "10"},
                                  request=httpx.Request("GET", "https://example.org"))
        request = MagicMock(side_effect=lambda timeout: response.raise_for_status())
        with self.assertRaises(ExternalUnavailable):
            self.call(request, budget=2)
        request.assert_called_once()
        self.assertEqual(request.call_args.args[0].connect, 2)
        self.assertEqual(self.delays, [0.25])

    def test_deadline_stops_waiting_for_a_running_request(self):
        release = Event()
        request = MagicMock(side_effect=lambda timeout: release.wait(3))
        try:
            policy = Resilience()
            with self.assertRaises(ExternalUnavailable):
                policy.call(self.service, request, budget=0.05, read=2)
            request.assert_called_once()
            self.assertFalse(release.is_set())
        finally:
            release.set()

    def test_circuit_isolates_provider_and_allows_only_one_recovery_probe(self):
        failing = MagicMock(side_effect=sdk_error(503))
        for _ in range(3):
            with self.assertRaises(ExternalUnavailable):
                self.call(failing)
        with self.assertRaises(ExternalUnavailable):
            self.call(failing)
        self.assertEqual(failing.call_count, 9)
        self.assertTrue(self.policy.call(("arXiv", "search"), lambda timeout: True, budget=5, read=2))
        self.now += 30

        started, release = Event(), Event()

        def probe(timeout):
            started.set()
            return release.wait(3)

        with ThreadPoolExecutor(1) as pool:
            recovered = pool.submit(self.call, probe)
            try:
                self.assertTrue(started.wait(2))
                with self.assertRaises(ExternalUnavailable):
                    self.call(failing)
                self.assertEqual(failing.call_count, 9)
            finally:
                release.set()
            self.assertTrue(recovered.result())
        self.assertEqual(self.call(lambda timeout: "closed"), "closed")

    def test_snapshot_expiry_empty_success_and_auth_failure(self):
        key = (self.service, "query")
        fetch = MagicMock(return_value=("record",))
        self.assertEqual(self.policy.search(key, fetch, max_age=5), (("record",), None))
        self.now = 3
        fetch.side_effect = ExternalUnavailable()
        records, timestamp = self.policy.search(key, fetch, max_age=5)
        self.assertEqual(records, ("record",))
        self.assertIsNotNone(timestamp)
        self.now = 6
        with self.assertRaises(ExternalUnavailable):
            self.policy.search(key, fetch, max_age=5)

        fetch.side_effect = None
        self.policy.search(key, fetch, max_age=5)
        fetch.return_value = ()
        self.assertEqual(self.policy.search(key, fetch, max_age=5), ((), None))
        fetch.side_effect = ExternalUnavailable()
        self.assertEqual(self.policy.search(key, fetch, max_age=5)[0], ())
        fetch.side_effect = sdk_error(401)
        with self.assertRaises(AuthenticationError):
            self.policy.search(key, lambda: self.call(fetch), max_age=5)
        fetch.side_effect = ExternalUnavailable()
        with self.assertRaises(ExternalUnavailable):
            self.policy.search(key, fetch, max_age=5)
        self.assertEqual(fetch.call_count, 8)


class YandexResilienceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.settings = {"YANDEX_FOLDER_ID": "folder", "YANDEX_API_KEY": "fixture", "YANDEX_MODEL_URI": "model",
                         "YANDEX_VECTOR_STORE_ID_RPD": "rpd", "YANDEX_VECTOR_STORE_ID_BOOKS": "books"}
        self.client = MagicMock()
        patcher = patch.object(yandex_calls, "EXTERNAL", Resilience(wait=lambda delay: None))
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_model_refusal_and_schema_repair_are_distinct(self):
        model = YandexModelClient(self.settings)
        with patch("orchestration.clients.open_yandex", return_value=nullcontext(self.client)):
            self.client.responses.create.return_value = reply(moderated=True)
            with self.assertRaises(run_rpd.YandexModeratedResponse):
                await model.generate(AgentReport, "instructions", {}, {})
            self.client.responses.create.assert_called_once()
            self.client.responses.create.reset_mock()
            self.client.responses.create.side_effect = [reply("{}"), reply('{"findings": []}')]
            self.assertEqual((await model.generate(AgentReport, "instructions", {}, {})).findings, [])
        self.assertEqual(self.client.responses.create.call_count, 2)
        self.assertIn('"repair"', self.client.responses.create.call_args.kwargs["input"])

    async def test_vector_snapshot_and_optional_failure_preserve_evidence(self):
        hit = SimpleNamespace(file_id="f1", filename="plan.pdf",
                              content=[SimpleNamespace(type="text", text="Подтверждённый учебный план")])
        self.client.vector_stores.search.side_effect = lambda index, **kwargs: (
            SimpleNamespace(data=[hit]) if index == "rpd" else self.raise_unavailable())
        with patch("orchestration.clients.open_yandex", return_value=nullcontext(self.client)):
            evidence = YandexEvidenceClient(self.settings)
            live = await evidence.search("university", "exact-course")
            self.assertEqual([source.source_type for source in live], ["RPD"])
            self.assertIn("Books", live.notices[0])
            self.client.vector_stores.search.side_effect = sdk_error(503)
            cached = await evidence.search("university", "exact-course")
            self.assertEqual(cached[0].source_id, live[0].source_id)
            self.assertTrue(any("снимок" in note for note in cached.notices))
            with self.assertRaises(ExternalUnavailable):
                await evidence.search("university", "other-course")
            other_scope = YandexEvidenceClient({**self.settings, "YANDEX_API_KEY": "other-fixture"})
            with self.assertRaises(ExternalUnavailable):
                await other_scope.search("university", "exact-course")

    @staticmethod
    def raise_unavailable():
        raise sdk_error(503)

    def test_required_source_outage_blocks_document_generation(self):
        request_data = {"discipline": "Database testing", "institution": "Test University", "academic_year": "2026"}
        # An earlier successful empty response does not supply evidence during an outage.
        profile_search = run_rpd.PROFILES["university"].searches[0]
        self.client.vector_stores.search.return_value = SimpleNamespace(data=[])
        yandex_calls.search_fragments(self.client, self.settings, "RPD", "rpd",
                                      profile_search.query.format_map(request_data), profile_search.result_count)
        self.client.vector_stores.search.side_effect = sdk_error(503)
        self.client.responses.create.return_value = reply('{"query":""}')
        graph = run_rpd.create_rpd_graph(self.client, self.settings)
        with self.assertRaises(ExternalUnavailable):
            graph.invoke({"request": request_data}, config={"configurable": {"thread_id": "outage"}})
        self.assertNotIn("rpd_draft", [call.kwargs["text"]["format"]["name"]
                                      for call in self.client.responses.create.call_args_list])


class ScienceResilienceTests(unittest.TestCase):
    def test_cached_publication_is_not_verified_with_stale_retraction_check(self):
        doi = "10.1234/test"
        published = {"message": {"items": [{"title": ["Database query optimization"], "DOI": doi,
                                             "issued": {"date-parts": [[date.today().year]]}}]}}
        failing = False

        def serve(request):
            if request.url.host == "export.arxiv.org":
                return httpx.Response(200, text='<feed xmlns="http://www.w3.org/2005/Atom"/>')
            if failing:
                return httpx.Response(503)
            return httpx.Response(200, json={"message": {"total-results": 0}}
                                  if request.url.params.get("rows") == "0" else published)

        client_class = httpx.Client
        with patch.object(science, "EXTERNAL", Resilience(wait=lambda delay: None)), \
             patch.object(science.httpx, "Client", side_effect=lambda **kw: client_class(
                 transport=httpx.MockTransport(serve), **kw)):
            cards, notices = science.find_course_articles("database query optimization", verified_dois={doi})
            self.assertEqual(cards[0].status, "verified")
            self.assertEqual(notices, [])
            failing = True
            cards, notices = science.find_course_articles("database query optimization", verified_dois={doi})
        self.assertEqual(cards[0].status, "published_candidate")
        self.assertIsNotNone(cards[0].cached_at)
        self.assertTrue(any("снимок" in note for note in notices))
