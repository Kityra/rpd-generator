from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


SourceKind = Literal["rag_chunk", "market_cluster", "expert_interview", "user_input"]
ExpertStatus = Literal["unreviewed", "confirmed_by_expert", "edited_by_expert", "rejected_by_expert"]
Verdict = Literal["supports", "weak", "unrelated"]
Category = Literal["duty_area", "work_task", "work_artifact", "tool_stack", "domain_constraint"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class EvidenceRef(Contract):
    source_type: SourceKind
    source_id: str = Field(min_length=1)
    quote_or_metric: str = Field(min_length=1)
    expert_status: ExpertStatus = "unreviewed"
    judge_verdict: Literal["pending", "supports", "weak", "unrelated"] = "pending"


class ProfessionClaim(Contract):
    claim_id: str = Field(min_length=1)
    category: Category
    statement: str = Field(min_length=1, max_length=2000)
    evidence: list[EvidenceRef]
    verification_status: Literal["verified", "needs_review", "rejected"]
    verification_reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_evidence(self):
        if self.verification_status != "needs_review" and not self.evidence:
            raise ValueError("Утверждению требуется источник.")
        return self


class CandidateClaim(ProfessionClaim):
    evidence: list[EvidenceRef] = Field(min_length=1)


class ProfessionProfile(Contract):
    profession_name: str = Field(min_length=1)
    claims: list[ProfessionClaim]
    sample_adequacy: Literal["sufficient", "low_sample", "empty_external_data"]
    missing_slots: list[str]
    interview_method_used: str | None = None

    @model_validator(mode="after")
    def unique_claim_ids(self):
        if len({claim.claim_id for claim in self.claims}) != len(self.claims):
            raise ValueError("Идентификаторы утверждений должны быть уникальны.")
        return self


class ExpertDecision(Contract):
    claim_id: str = Field(min_length=1)
    action: Literal["confirm", "edit", "reject", "add"]
    statement: str | None = Field(default=None, min_length=1, max_length=2000)
    category: Category = "work_task"

    @model_validator(mode="after")
    def require_statement(self):
        if (self.action in {"edit", "add"}) != bool(self.statement):
            raise ValueError("Для новой или исправленной формулировки требуется текст.")
        return self


def pending_claim(claim_id: str, category: Category, statement: str, evidence: list[EvidenceRef]) -> CandidateClaim:
    return CandidateClaim(claim_id=claim_id, category=category, statement=statement, evidence=evidence,
                          verification_status="needs_review", verification_reason="Ожидает проверки.")
