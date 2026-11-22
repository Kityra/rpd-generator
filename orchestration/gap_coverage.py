import re
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from math import log1p
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit

from domain.schemas import SourceRecord
from orchestration.profession_profile import EvidenceRef, ProfessionClaim, ProfessionProfile


YEAR_LABEL = re.compile(r"(?im)^\s*(?:год(?: публикации)?\s*:|опубликован[ао]?\s*:?|published\s*:)\s*((?:19|20)\d{2})\b")
EXPERT_CONFIRMED = {"confirmed_by_expert", "edited_by_expert"}


@dataclass(frozen=True)
class GapParameters:
    weak_weight: float = 0.4
    freshness_half_life_years: float = 5
    target_independent_sources: int = 3
    covered_threshold: float = 0.7

    @classmethod
    def from_settings(cls, settings):
        defaults = cls()
        return cls(**{entry.name: getattr(settings, f"gap_{entry.name}", getattr(defaults, entry.name))
                      for entry in fields(cls)})


def document_origin(source: SourceRecord) -> str:
    if source.parent_document_id:
        return source.parent_document_id
    url = urlsplit(source.url)
    if url.hostname == "ai.api.cloud.yandex.net" and url.path.startswith("/v1/files/"):
        return "yandex:" + unquote(url.path.split("/")[-1])
    if url.hostname in {"doi.org", "dx.doi.org"}:
        return "doi:" + unquote(url.path.strip("/")).lower()
    query = urlencode(sorted((key, value) for key, value in parse_qsl(url.query)
                             if not key.casefold().startswith("utm_") and key not in {"gclid", "fbclid"}))
    return f"{url.netloc}{url.path.rstrip('/')}?{query}" if query else f"{url.netloc}{url.path.rstrip('/')}"


def freshness(year: int | None, now_year: int, half_life: float) -> float:
    if year is None:
        return 0.75
    return max(0.5, 1.0 - 0.5 * max(0, now_year - year) / half_life)


def reference_weight(ref: EvidenceRef, sources: dict[str, SourceRecord], market: dict,
                     params: GapParameters, now_year: int, source_types: set[str] | None, expert_origins: dict[str, str]):
    if ref.expert_status == "rejected_by_expert":
        return None
    if ref.source_type == "expert_interview":
        if source_types is None and ref.expert_status in EXPERT_CONFIRMED:
            # Реплики одного эксперта — один источник. В старой записи автор может отсутствовать.
            return "expert:" + expert_origins.get(ref.source_id, "interview"), 1.0
        return None
    weight = {"supports": 1.0, "weak": params.weak_weight}.get(ref.judge_verdict, 0.0)
    if not weight:
        return None
    if ref.source_type == "market_cluster":
        measurement = market.get(ref.source_id)
        if measurement and (source_types is None or "market_cluster" in source_types):
            factor = (1.0 if measurement.get("current_sample", True) and not measurement.get("publication_year") else
                      freshness(measurement.get("publication_year"), now_year, params.freshness_half_life_years))
            return "market:" + (measurement.get("sample_id") or measurement.get("provider", "выборка")), weight * factor
    if ref.source_type == "rag_chunk":
        source = sources.get(ref.source_id)
        if source is not None and (source_types is None or source.source_type in source_types):
            explicit = YEAR_LABEL.search(source.content[:1500])
            year = source.publication_year or (int(explicit.group(1)) if explicit else None)
            return document_origin(source), weight * freshness(year, now_year, params.freshness_half_life_years)
    return None


def coverage_formula(claim: ProfessionClaim, sources: dict[str, SourceRecord], market: dict, params: GapParameters,
                     *, now_year: int, source_types: set[str] | None = None,
                     expert_origins: dict[str, str] | None = None) -> dict:
    expert_origins = expert_origins or {}
    candidates = [] if claim.verification_status == "rejected" else [
        entry for ref in claim.evidence
        if (entry := reference_weight(ref, sources, market, params, now_year, source_types, expert_origins))
    ]
    best = max((score for _, score in candidates), default=0.0)
    independent = len({origin for origin, _ in candidates})
    factor = min(1.0, log1p(independent) / log1p(params.target_independent_sources))
    score = best * factor
    return {"best_quality_freshness": best, "independent_sources_count": independent,
            "independence_factor": factor, "final_score": score,
            "status": ("covered" if score >= params.covered_threshold else "partial_gap" if score else "critical_gap")}


def profile_coverage(profile: ProfessionProfile, sources: dict[str, SourceRecord], market: dict,
                     params: GapParameters, business_types: set[str], *, expert_origins: dict[str, str] | None = None) -> dict:
    now_year = datetime.now(timezone.utc).year
    rows = [{"claim_id": claim.claim_id, "statement": claim.statement, "category": claim.category,
             **coverage_formula(claim, sources, market, params, now_year=now_year, expert_origins=expert_origins)}
            for claim in profile.claims if claim.verification_status != "rejected"]
    tasks = [claim for claim in profile.claims if claim.category == "work_task" and claim.verification_status != "rejected"]

    def average(source_types):
        return sum(coverage_formula(claim, sources, market, params, now_year=now_year, source_types=source_types)["final_score"]
                   for claim in tasks) / len(tasks) if tasks else 0.0

    return {"formula_breakdown": rows, "coverage_parameters": {**vars(params), "as_of_year": now_year},
            "business_coverage": average(business_types | {"market_cluster"}),
            "science_coverage": average({"Crossref", "arXiv"})}
