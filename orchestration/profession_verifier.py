import logging
import re
from collections.abc import Mapping

from orchestration.llm_judge import JudgePair, JudgeVerdicts, can_judge, judge_batch
from orchestration.profession_profile import EvidenceRef, ProfessionClaim, ProfessionProfile
from orchestration.relevance_filter import service_reason


LOG = logging.getLogger(__name__)
class ProfessionEvidenceVerdicts(JudgeVerdicts):
    pass


def normalized(text: str) -> str:
    return " ".join(text.split()).casefold()


def _trusted_ref(claim: ProfessionClaim, ref: EvidenceRef, rag, market, turns) -> EvidenceRef | None:
    source_id, quote = ref.source_id, " ".join(ref.quote_or_metric.split())
    unreviewed = {"expert_status": "unreviewed", "judge_verdict": "pending"}
    if ref.source_type == "rag_chunk":
        chunk = rag.get(source_id)
        if chunk is None or quote not in " ".join(chunk.split()):
            return None
        return ref.model_copy(update=unreviewed)
    if ref.source_type == "market_cluster":
        measurement = market.get(source_id)
        if not measurement or ref.quote_or_metric != measurement["metric"]:
            return None
        return ref.model_copy(update=unreviewed)
    if ref.source_type == "expert_interview":
        entry = turns.get(source_id, {}).get(claim.claim_id)
        if (not re.fullmatch(r"turn:[1-9][0-9]*", source_id) or not entry
                or quote != " ".join(entry["quote"].split()) or claim.statement != entry["statement"]):
            return None
        return ref.model_copy(update={**unreviewed, "expert_status": entry["status"]})
    return None  # Ввод методиста включается отдельным явным действием в кабинете.


async def verify_profession_profile(profile: ProfessionProfile, rag_index_map: Mapping[str, str],
                                    market_map: Mapping[str, dict], interview_turns: Mapping[str, dict],
                                    judge_client, *, rejected_noise_turns: list[int] | None = None,
                                    reuse_recorded_verdicts: bool = False) -> ProfessionProfile:
    rejected_ids = {f"turn:{index}" for index in [*profile.rejected_noise_turns, *(rejected_noise_turns or [])]}
    pending: list[tuple[str, int, JudgePair]] = []
    trusted = {}
    for claim in profile.claims:
        refs = []
        seen = set()
        for ref in claim.evidence:
            checked = (None if ref.source_id in rejected_ids else
                       _trusted_ref(claim, ref, rag_index_map, market_map, interview_turns))
            if checked is None:
                LOG.warning("Отброшена ссылка профиля профессии (%s)", ref.source_type,
                            extra={"event": "profession_ref_stripped", "source_type": ref.source_type,
                                   "claim_id": claim.claim_id, "source_id": ref.source_id})
                continue
            key = (checked.source_type, checked.source_id, normalized(checked.quote_or_metric))
            if key not in seen:
                if checked.source_type in {"rag_chunk", "market_cluster"}:
                    evidence = (market_map[checked.source_id]["skill"] if checked.source_type == "market_cluster"
                                else checked.quote_or_metric)
                    if not can_judge(claim.statement, evidence) or (
                        checked.source_type == "rag_chunk" and service_reason(rag_index_map[checked.source_id])
                    ):
                        checked = checked.model_copy(update={"judge_verdict": "unrelated"})
                    elif reuse_recorded_verdicts:
                        checked = checked.model_copy(update={"judge_verdict": ref.judge_verdict})
                refs.append(checked)
                seen.add(key)
        trusted[claim.claim_id] = refs
        if reuse_recorded_verdicts or any(ref.expert_status != "unreviewed" for ref in refs):
            continue
        for index, ref in enumerate(refs):
            if ref.judge_verdict == "pending" and ref.source_type in {"rag_chunk", "market_cluster"}:
                snippet = ref.quote_or_metric
                if ref.source_type == "market_cluster":
                    snippet = f"{market_map[ref.source_id]['skill']} — {snippet}"
                pending.append((claim.claim_id, index, {"pair_id": str(len(pending) + 1), "claim": claim.statement,
                    "source_type": ref.source_type, "evidence": snippet}))

    verdicts = await judge_batch(judge_client, ProfessionEvidenceVerdicts, [pair for _, _, pair in pending])
    judge_failed = verdicts is None or (reuse_recorded_verdicts and profile.judge_unavailable)
    if verdicts is not None:
        for claim_id, index, pair in pending:
            ref = trusted[claim_id][index]
            trusted[claim_id][index] = ref.model_copy(update={"judge_verdict": verdicts[pair["pair_id"]]})

    claims = []
    for claim in profile.claims:
        refs = trusted[claim.claim_id]
        statuses = {ref.expert_status for ref in refs}
        if "rejected_by_expert" in statuses:
            status, reason = "rejected", "Эксперт отклонил формулировку."
        elif statuses & {"confirmed_by_expert", "edited_by_expert"}:
            status, reason = "verified", "Эксперт подтвердил или уточнил формулировку."
        elif any(ref.judge_verdict == "supports" for ref in refs):
            status, reason = "verified", "Содержание утверждения подтверждено процитированным источником."
        elif judge_failed and any(ref.source_type in {"rag_chunk", "market_cluster"} and ref.judge_verdict == "pending"
                                  for ref in refs):
            status, reason = "needs_review", "Автоматическая проверка источников недоступна."
        else:
            status, reason = "needs_review", "Прямого подтверждения в проверенных источниках нет."
        claims.append(claim.model_copy(update={"evidence": refs, "verification_status": status,
                                               "verification_reason": reason}))
    return profile.model_copy(update={"claims": claims, "judge_unavailable": judge_failed, "rejected_noise_turns": sorted(
        set(profile.rejected_noise_turns) | set(rejected_noise_turns or []))})
