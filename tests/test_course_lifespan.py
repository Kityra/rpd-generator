from datetime import date, datetime, timedelta, timezone
import unittest

import httpx

from orchestration.course_lifespan import (DOTNET_RELEASES, NODE_SCHEDULE, calculate_lifespan, collect_market_sample,
                                          compare_market_samples, fetch_support_deadlines)
from orchestration.evidence_cache import source_record


def program(*titles):
    return {"subjects": [{"title": "Практикум", "topics": [{"title": title} for title in titles]}]}


class LifespanRulesTests(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_versions_use_official_deadlines_and_the_earliest_one(self):
        responses = {
            NODE_SCHEDULE: {"v22": {"end": "2027-04-30"}},
            DOTNET_RELEASES: {"releases-index": [{"channel-version": "8.0", "eol-date": "2026-11-10"}]},
        }
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json=responses[str(request.url)]))
        deadlines, failed = await fetch_support_deadlines(program("Node.js v22.3.1", ".NET 8"),
                                                          transport=transport)
        self.assertFalse(failed)
        self.assertEqual({item.technology for item in deadlines}, {"Node.js 22", ".NET 8.0"})
        today = date(2026, 1, 1)
        assessment = calculate_lifespan(today, deadlines, None, {})
        self.assertEqual(assessment.review_by, date(2026, 11, 10))
        self.assertEqual(assessment.estimated_lifespan_years, round((assessment.review_by - today).days / 365.2425, 2))
        self.assertEqual(assessment, calculate_lifespan(today, list(reversed(deadlines)), None, {}))
        past = calculate_lifespan(date(2029, 1, 1), deadlines, None, {})
        self.assertEqual(past.estimated_lifespan_years, 0)
        self.assertEqual({item.source.source_id for item in deadlines}, set(assessment.source_ids))

    async def test_unversioned_tools_do_not_trigger_lookup_and_one_provider_can_fail(self):
        paths = []

        def respond(request):
            paths.append(str(request.url))
            if str(request.url) == NODE_SCHEDULE:
                return httpx.Response(503)
            return httpx.Response(200, json={"releases-index": [{"channel-version": "8.0", "eol-date": "2026-11-10"}]})

        transport = httpx.MockTransport(respond)
        self.assertEqual(await fetch_support_deadlines(program("Node.js", ".NET", "Node.js 22+", "Node.js 20–22"),
                                                       transport=transport), ([], False))
        self.assertFalse(paths)
        deadlines, failed = await fetch_support_deadlines(program("Node.js 22.x", ".NET Core 8.0"), transport=transport)
        self.assertTrue(failed)
        self.assertEqual([item.technology for item in deadlines], [".NET 8.0"])

    def test_market_signal_requires_separate_windows_minimum_sample_and_a_taught_skill(self):
        today = date(2026, 11, 22)

        def sample(day, postgres_count, size=20, topic="Инженер данных", identity=""):
            retrieved = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
            records = [source_record("Rostrud", f"https://example.org/{identity or day.isoformat()}/{i}", "Инженер",
                                     f"Опубликована: {day}\n{'SQL' if i < postgres_count else 'Командная работа'}",
                                     retrieved_at=retrieved) for i in range(size)]
            return collect_market_sample(topic, records)

        current = sample(today, 7)
        previous = sample(today - timedelta(days=45), 12)
        self.assertIsNone(sample(today, 10, size=19))
        self.assertIsNone(compare_market_samples(program("SQL"), current, [], today))
        self.assertIsNone(compare_market_samples(program("SQL"), current, [sample(today - timedelta(days=44), 20)], today))
        self.assertIsNone(compare_market_samples(program("SQL"), current, [sample(today - timedelta(days=60), 11)], today))
        self.assertIsNone(compare_market_samples(program("SQL"), current,
                          [sample(today - timedelta(days=60), 20, topic="Аналитик")], today))
        self.assertIsNone(compare_market_samples(program("SQL"), current,
                          [sample(today - timedelta(days=60), 20, identity=today.isoformat())], today))
        self.assertIsNone(compare_market_samples(program("Документирование"), current, [previous], today))
        self.assertEqual(compare_market_samples(program("SQL"), current, [previous], today).skill, "SQL")
