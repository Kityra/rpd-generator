from datetime import timedelta
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import httpx

from rabota_it import BusinessAPI, VacancyStore, import_export, sync
from rabota_it.__main__ import main
from rabota_it.service import now


VACANCY = {
    "id": 42, "title": "Разработчик Python", "url": "/vacancy/42?track=test",
    "company": {"name": "Команда"}, "region": {"name": "Москва"},
    "requirements": "<p>Опыт с Docker</p><script>skip me</script>",
    "published_at": "2026-11-20T10:00:00+03:00", "skills": ["SQL", "sql"],
    "salary": {"from": "100 000", "to": 200000, "currency": "руб", "period": "month"},
}


class RabotaITTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.store = VacancyStore(self.root / "rabota.sqlite")
        self.api = BusinessAPI(self.store)

    def test_export_normalisation_search_and_summary(self):
        export = self.root / "source.ndjson"
        records = [VACANCY, {**VACANCY, "id": 43}, {"id": 1, "title": "Кассир"}]
        export.write_text("\n".join(map(json.dumps, records)), encoding="utf-8")
        result = import_export(self.store, export)
        self.assertEqual((result["vacancies"], result["duplicates"], result["non_it"]), (1, 1, 1))
        result = self.api.handle(query="РАЗРАБОТЧИК", region="москва", technologies=["docker"])
        card, = result["vacancies"]
        self.assertEqual(card["published_at"], "2026-11-20T07:00:00+00:00")
        self.assertEqual(card["skills"], ["SQL"])
        self.assertEqual(card["url"], "https://www.rabota.ru/vacancy/42")
        self.assertEqual(card["requirements"], "Опыт с Docker")
        self.assertIsNone(card["updated_at"])
        self.assertEqual(self.api.handle(operation="get", source_id="42")["vacancy"], card)
        self.assertEqual(self.api.handle(published_after="2026-11-21")["matched_count"], 0)
        group, = self.api.handle(operation="summary")["salary_groups"]
        self.assertEqual(group["median_of_range_midpoints_or_single_bounds"], 150000)

    def test_invalid_snapshot_preserves_previous_data(self):
        self.store.replace([VACANCY])
        with self.assertRaises(ValueError):
            self.store.replace([{"id": "broken"}])
        self.assertEqual(self.api.handle()["matched_count"], 1)
        self.store.replace([])
        self.assertEqual(self.api.handle()["matched_count"], 0)

    def test_http_cache_rate_limit_and_outage(self):
        requests = []
        responses = iter([httpx.Response(200, json={"vacancies": [VACANCY]}, headers={"ETag": "v1"}),
                          httpx.Response(304), httpx.Response(429, headers={"Retry-After": "180"})])
        def serve(request):
            requests.append(request)
            return next(responses)
        start = now()
        endpoint = "https://partner.example/feed"
        with httpx.Client(transport=httpx.MockTransport(serve)) as client:
            with self.assertRaises(ValueError):
                sync(self.store, endpoint, client=client)
            def refresh(at):
                with patch("rabota_it.service.now", return_value=start + timedelta(seconds=at)):
                    return sync(self.store, endpoint, authorised=True, client=client, interval=60)["status"]
            self.assertEqual([refresh(at) for at in (0, 10, 60, 120, 181)],
                             ["ok", "skipped", "not_modified", "failed", "skipped"])
        self.assertEqual(len(requests), 3)
        self.assertEqual(requests[1].headers["If-None-Match"], "v1")
        result = self.api.handle()
        self.assertEqual(result["matched_count"], 1)
        self.assertIn("429", result["last_error"])

    def test_cli_search_without_query(self):
        with patch("builtins.print") as output:
            self.assertEqual(main(["--database", str(self.store.path), "search"]), 0)
        self.assertEqual(json.loads(output.call_args.args[0])["vacancies"], [])

    def test_inflight_request_cannot_overwrite_newer_import(self):
        def serve(request):
            self.store.replace([{**VACANCY, "title": "Разработчик Java"}])
            return httpx.Response(200, json=[VACANCY])

        with httpx.Client(transport=httpx.MockTransport(serve)) as client:
            result = sync(self.store, "https://partner.example/feed", authorised=True, client=client)
        self.assertEqual(result["status"], "superseded")
        self.assertEqual(self.api.handle(query="java")["matched_count"], 1)

    def test_incomplete_feed_preserves_snapshot(self):
        self.store.replace([VACANCY])
        def serve(request):
            return httpx.Response(200, json={"vacancies": [], "next": "page-2"})

        with httpx.Client(transport=httpx.MockTransport(serve)) as client:
            result = sync(self.store, "https://partner.example/feed", authorised=True, client=client)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(self.api.handle()["matched_count"], 1)
