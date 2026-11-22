import json
import unittest
from io import BytesIO
from types import SimpleNamespace
from zipfile import ZipFile

from domain.export import export_course, gap_coverage_html
from orchestration.evidence_cache import source_record
from orchestration.gap_coverage import GapParameters, coverage_formula, freshness
from orchestration.profession_profile import EvidenceRef, ProfessionClaim
from test_course_architecture import program, records


class GapCoverageTests(unittest.TestCase):
    def test_gap_analysis_formula_and_independence_dedup(self):
        claim_text = "Проектирует схемы PostgreSQL"
        params = GapParameters()

        def score(parents):
            sources = [source_record("Priklad", f"https://example.org/file/{index}", f"Фрагмент {index}",
                                     f"{claim_text}. Фрагмент {index}", parent_document_id=parent,
                                     publication_year=2025) for index, parent in enumerate(parents)]
            claim = ProfessionClaim(claim_id="postgres", category="work_task", statement=claim_text,
                verification_status="verified", verification_reason="Проверено", evidence=[
                    EvidenceRef(source_type="rag_chunk", source_id=source.source_id,
                                quote_or_metric=claim_text, judge_verdict="supports") for source in sources])
            return coverage_formula(claim, {source.source_id: source for source in sources}, {}, params, now_year=2026)

        single = score(["doc-a", "doc-a", "doc-a"])
        self.assertEqual(single, {"best_quality_freshness": 0.9, "independent_sources_count": 1,
                                  "independence_factor": 0.5, "final_score": 0.45, "status": "partial_gap"})
        distinct = score(["doc-a", "doc-b", "doc-c"])
        self.assertEqual(distinct, {"best_quality_freshness": 0.9, "independent_sources_count": 3,
                                    "independence_factor": 1.0, "final_score": 0.9, "status": "covered"})
        gaps = {"coverage_parameters": {**vars(params), "as_of_year": 2026},
                "formula_breakdown": [{"claim_id": "postgres", "statement": claim_text, **single}]}
        version = SimpleNamespace(status="approved", revision=1, content={"program": program(), "evidence_gaps": gaps})
        sources = {key: source.model_dump(mode="json") for key, source in records().items()}
        with ZipFile(BytesIO(export_course({"institution": "Вуз", "year": "2026"}, version, sources))) as archive:
            report = archive.read("course-report.html").decode()
            saved = json.loads(archive.read("course.json"))
        self.assertIn(gap_coverage_html(gaps), report)
        self.assertIn("0.900 × 0.500 = 0.450", report)
        self.assertEqual(saved["evidence_gaps"]["formula_breakdown"][0]["independent_sources_count"], 1)
        portal_report = gap_coverage_html(gaps, include_methodology=False)
        self.assertIn("<details><summary>Расчёт по утверждениям</summary>", portal_report)
        self.assertIn("0.900 × 0.500 = 0.450", portal_report)
        self.assertNotIn("Коэффициент независимости", portal_report)
        self.assertNotIn("Порог покрытия", portal_report)

    def test_weak_unknown_year_and_rejected_evidence(self):
        source = source_record("Priklad", "https://example.org/work", "Рабочий процесс",
                               "Проектирует схемы PostgreSQL")
        claim = ProfessionClaim(claim_id="postgres", category="work_task", statement=source.content,
            verification_status="needs_review", verification_reason="Слабое подтверждение", evidence=[
                EvidenceRef(source_type="rag_chunk", source_id=source.source_id,
                            quote_or_metric=source.content, judge_verdict="weak"),
                EvidenceRef(source_type="expert_interview", source_id="turn:2", quote_or_metric=source.content,
                            expert_status="rejected_by_expert")])
        settings = GapParameters()
        score = coverage_formula(claim, {source.source_id: source}, {}, settings, now_year=2026)
        self.assertAlmostEqual(score["final_score"], 0.4 * 0.75 * 0.5)
        rejected = claim.model_copy(update={"verification_status": "rejected"})
        self.assertEqual(coverage_formula(rejected, {}, {}, settings, now_year=2026)["status"], "critical_gap")
        self.assertEqual(freshness(1999, 2026, 5), 0.5)
        self.assertEqual(freshness(2024, 2026, 10), 0.9)
