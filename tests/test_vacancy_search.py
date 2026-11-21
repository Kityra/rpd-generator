from datetime import datetime, timedelta, timezone
import json
import unittest
from unittest.mock import AsyncMock

import httpx

from domain.schemas import EvidenceResults
from orchestration.evidence_cache import source_record
from orchestration.rabota_business import BusinessEvidenceClient
from orchestration.vacancy_search import PublicVacancySearch


class PublicVacancySearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_fresh_relevant_records_from_both_sources_and_no_duplicate_or_contacts(self):
        now = datetime.now(timezone.utc).date()
        date = now.isoformat()
        vacancy = {
            "job-name": "ML-инженер", "creation-date": date,
            "vac_url": "https://trudvsem.ru/vacancy/card/1/job",
            "company": {"name": "ООО «Пример»", "email": "hidden@example.org"},
            "duty": "<p>Обучать и развёртывать модели.</p>",
            "contact_person": "Скрытый контакт",
        }
        rows = [
            vacancy,
            {**vacancy, "vac_url": "https://trudvsem.ru/vacancy/card/1/copy",
             "creation-date": (now - timedelta(days=2)).isoformat()},
            {**vacancy, "job-name": "Машинист с обучением"},
            {**vacancy, "job-name": "ML Lead", "creation-date": (now - timedelta(days=75)).isoformat()},
        ]
        job_posting = {"@type": "JobPosting", "title": "ML-разработчик",
                       "description": "<p>Оценивать качество моделей.</p>", "datePosted": date}

        def respond(request):
            if request.url.host == "opendata.trudvsem.ru":
                self.assertIn(request.url.params["text"], {"ML", "LLM", "Data Scientist"})
                self.assertNotIn("Вуз", str(request.url))
                return httpx.Response(200, json={"status": "200", "results": {"vacancies": [
                    {"vacancy": item} for item in rows
                ]}})
            if request.url.path == "/jobs/vacancies":
                return httpx.Response(200, text=(
                    '<a href="/jobs/vacancies/ml-developer-100">ML-разработчик</a>'
                    '<a href="/jobs/vacancies/product-manager-101">Product Manager ML</a>'
                ))
            self.assertEqual(request.url.path, "/jobs/vacancies/ml-developer-100")
            return httpx.Response(200, text='<script type="application/ld+json">' + json.dumps(job_posting) + "</script>")

        base = AsyncMock()
        hh = source_record("HH", "https://hh.ru/vacancy/1", "HH", "Историческая вакансия")
        base.search.return_value = EvidenceResults([hh], ["Существующее примечание"])
        client = BusinessEvidenceClient(base, None, source_record,
                                        public_search=PublicVacancySearch(transport=httpx.MockTransport(respond)))
        records = await client.search("business", json.dumps({"title": "ML-инженер", "institution": "Вуз"}))
        self.assertEqual([record.source_type for record in records], ["HH", "Rostrud", "EmployerCareer"])
        self.assertEqual(records[1].url, vacancy["vac_url"])
        self.assertIn("Обучать и развёртывать модели.", records[1].content)
        self.assertIn(date, records[2].content)
        self.assertNotIn("hidden@example.org", records[1].content)
        self.assertIn("Существующее примечание", records.notices)
        self.assertIn("работодателей: 2", records.notices[-1])

    async def test_unavailable_provider_leaves_existing_business_evidence(self):
        base = AsyncMock()
        base.search.return_value = EvidenceResults(notices=["Индекс HH недоступен."])
        transport = httpx.MockTransport(lambda request: httpx.Response(503))
        client = BusinessEvidenceClient(base, None, source_record,
                                        public_search=PublicVacancySearch(employer_careers=False, transport=transport))
        records = await client.search("business", "DevOps")
        self.assertFalse(records)
        self.assertEqual(records.notices[0], "Индекс HH недоступен.")
        self.assertIn("Работа в России", records.notices[1])
