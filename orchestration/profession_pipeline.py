import re
from decimal import Decimal
from urllib.parse import urlsplit

from domain.schemas import SourceRecord
from orchestration.evidence_cache import source_record
from orchestration.interview_skills import MIN_MARKET_SAMPLE, MIN_RAG_CHUNKS
from orchestration.llm_judge import terms
from orchestration.market_analysis import course_market
from orchestration.profession_profile import EvidenceRef, ExpertDecision, ProfessionProfile, pending_claim
from orchestration.profession_verifier import normalized


ACTIVITY_SOURCE_TYPES = {"Priklad", "HH", "Crossref", "arXiv", "documentation", "Rostrud",
                         "EmployerCareer", "RabotaRu", "ActivityModel"}
EXPERT_ACTIONS = {"confirm": "confirmed_by_expert", "edit": "edited_by_expert",
                  "add": "confirmed_by_expert", "reject": "rejected_by_expert"}
FIELDS = {"stack": "tool_stack", "artifacts": "work_artifact",
          "quality_criteria": "domain_constraint", "typical_errors": "domain_constraint"}
SLOT_LABELS = {"duty_area": "рабочие обязанности", "work_task": "рабочие задачи",
               "work_artifact": "проверяемые результаты работы", "tool_stack": "инструменты",
               "domain_constraint": "условия и критерии качества"}


def market_evidence(meta: dict, sources: dict[str, SourceRecord], vacancy_ids: list[str]) -> tuple[int, dict]:
    # Фрагмент CSV из индекса HH не равен одной вакансии. Учитываем только реальные карточки.
    hh_cards = [key for key, record in sources.items() if record.source_type == "HH"
                and (urlsplit(record.url).hostname or "").split(".")[-2:] == ["hh", "ru"]
                and re.fullmatch(r"/vacancy/\d+/?", urlsplit(record.url).path)]
    candidates = list(dict.fromkeys([*vacancy_ids, *hh_cards,
                                    *(key for key, record in sources.items() if record.source_type == "RabotaRu")]))
    by_vacancy = {}
    for key in candidates:
        if key in sources:
            url = urlsplit(sources[key].url)
            by_vacancy.setdefault((url.netloc.casefold(), url.path.rstrip("/")), key)
    ids = list(by_vacancy.values())
    records = {key: sources[key].model_dump(mode="json") for key in ids if key in sources}
    market = course_market(meta, {}, records, ids, None)
    size = market.sample_size or 0
    uploaded_dataset = (meta.get("market") or {}).get("dataset")
    provider = "выборка методиста" if uploaded_dataset else (
        "HH" if records and all(record["source_type"] == "HH" for record in records.values()) else "выборка вакансий")
    dates = [str((uploaded_dataset or {}).get("collected_at") or ""), market.period or ""]
    years = re.findall(r"\b(?:19|20)\d{2}\b", dates[0] or dates[1])
    year = max(map(int, years), default=None)
    measurements = {}
    for row in market.skills:
        if size and row.vacancies is not None and 0 < row.vacancies <= size:
            share = format((Decimal(row.vacancies) * 100 / size).quantize(Decimal("0.1")), "f").removesuffix(".0")
            measurements[f"skill:{row.skill}"] = {"skill": row.skill, "provider": provider,
                "metric": f"{row.vacancies}/{size} ({share}%)", "publication_year": year,
                "current_sample": not uploaded_dataset and year is None}
    return size, measurements


def _excerpt(text: str, statement: str) -> str:
    words = terms(statement)
    # Короткое дословное окно. Учитывает фрагменты индекса, пришедшие одной длинной строкой.
    tokens = text.split()
    chunks = (" ".join(tokens[start:start + 74]) for start in range(0, len(tokens), 56))
    return max(chunks, key=lambda block: len(words & terms(block)), default="")


def initial_profile(name: str, activity: dict, rag: dict[str, str], market_sample: int, market: dict) -> ProfessionProfile:
    claims = []

    def add(claim_id, category, statement, source_ids):
        if not statement.strip():
            return
        refs = [EvidenceRef(source_type="rag_chunk", source_id=key, quote_or_metric=_excerpt(rag[key], statement))
                for key in source_ids if rag.get(key, "").strip()][:3]
        if refs:
            claims.append(pending_claim(claim_id, category, statement[:2000], refs))

    for task in activity.get("tasks", []):
        add(f"work_task:{task['id']}", "work_task", task["title"], task["source_ids"])
        for field, category in FIELDS.items():
            for index, value in enumerate(task.get(field, [])):
                add(f"{task['id']}:{field}:{index}", category, value, task["source_ids"])
    for group in ("processes", "aspects"):
        for item in activity.get(group, []):
            statement = item.get("title") or item.get("description", "")
            add(f"{group}:{item['id']}", "duty_area", statement, item["source_ids"])
    for key, row in list(market.items())[:20]:
        claims.append(pending_claim(key, "tool_stack", f"В выборке вакансий указан навык {row['skill']}.",
                                    [EvidenceRef(source_type="market_cluster", source_id=key,
                                                 quote_or_metric=row["metric"])]))
    adequacy = "empty_external_data" if not (rag or market_sample) else (
        "sufficient" if len(rag) >= MIN_RAG_CHUNKS or market_sample >= MIN_MARKET_SAMPLE else "low_sample")
    return ProfessionProfile(profession_name=name, sample_adequacy=adequacy, claims=claims, missing_slots=[])


def with_missing_slots(profile: ProfessionProfile, meta: dict) -> ProfessionProfile:
    present = {claim.category for claim in profile.claims if claim.verification_status == "verified"}
    missing = [label for category, label in SLOT_LABELS.items() if category not in present]
    missing.extend(label for key, label in (("institution", "образовательная организация"),
                                            ("year", "учебный год")) if not meta.get(key))
    return profile.model_copy(update={"missing_slots": missing})


def apply_expert_turn(profile: ProfessionProfile, decisions: list[dict], turn_id: str, *,
                      verbatim_confirmations: set[str] | None = None) -> tuple[ProfessionProfile, dict]:
    claims = {claim.claim_id: claim for claim in profile.claims}
    turns = {}
    for raw in decisions:
        decision = ExpertDecision.model_validate(raw)
        original = claims.get(decision.claim_id)
        if decision.action == "add":
            claim_id, category = f"expert:{turn_id}:{decision.claim_id}", decision.category
        elif original is None or original.verification_status == "rejected":
            continue
        else:
            claim_id, category = original.claim_id, original.category
        statement = decision.statement or original.statement
        quote = {"confirm": f"Подтверждаю: {statement}", "reject": f"Отклоняю: {statement}"}.get(
            decision.action, statement)
        if decision.action == "confirm" and claim_id in (verbatim_confirmations or ()):
            quote = statement
        status = EXPERT_ACTIONS[decision.action]
        ref = EvidenceRef(source_type="expert_interview", source_id=turn_id, quote_or_metric=quote,
                          expert_status=status)
        old_refs = original.evidence if original and decision.action == "confirm" else []
        claims[claim_id] = pending_claim(claim_id, category, statement, [*old_refs, ref])
        turns[claim_id] = {"statement": statement, "quote": quote, "status": status}
    return profile.model_copy(update={"claims": list(claims.values())}), turns


def activity_for_design(profile: ProfessionProfile, raw_activity: dict, acknowledged: list[str], url: str,
                        sources: dict[str, SourceRecord]) -> tuple[dict, list[SourceRecord]]:
    selected = {claim.claim_id: claim for claim in profile.claims if claim.verification_status == "verified" or
                (claim.verification_status == "needs_review" and claim.claim_id in acknowledged)}
    records = {}
    references = {}
    for claim_id, claim in selected.items():
        refs = [ref for ref in claim.evidence if ref.judge_verdict == "supports" or
                ref.expert_status in {"confirmed_by_expert", "edited_by_expert"}]
        ids = [ref.source_id for ref in refs if ref.source_type == "rag_chunk" and ref.source_id in sources]
        for ref in refs:
            if ref.source_type == "expert_interview":
                number = ref.source_id.split(":")[-1]
                record = source_record("expert", f"{url}&turn={number}", f"Эксперт · реплика #{number}",
                                       ref.quote_or_metric)
                records[record.source_id] = record
                ids.append(record.source_id)
        if claim.verification_status == "needs_review":
            record = source_record("teacher", url + "&input=acknowledged", "Включено методистом для проверки", claim.statement)
            records[record.source_id] = record
            ids = [record.source_id]
        references[claim_id] = list(dict.fromkeys(ids))

    tasks = []
    for claim_id, claim in selected.items():
        if claim.category != "work_task" or not references[claim_id]:
            continue
        task_id = claim_id.removeprefix("work_task:")
        task = {"id": task_id, "title": claim.statement, "source_ids": list(references[claim_id]),
                **{field: [] for field in FIELDS}}
        for field in FIELDS:
            for key, detail in selected.items():
                if key.startswith(f"{task_id}:{field}:") and references[key]:
                    task[field].append(detail.statement)
                    task["source_ids"].extend(references[key])
        task["source_ids"] = list(dict.fromkeys(task["source_ids"]))
        tasks.append(task)
    groups = {}
    present = {task["id"] for task in tasks}
    for group in ("processes", "aspects"):
        groups[group] = []
        for item in raw_activity.get(group, []):
            claim_id = f"{group}:{item['id']}"
            task_ids = [key for key in item["task_ids"] if key in present]
            if task_ids and references.get(claim_id):
                field = "title" if group == "processes" else "description"
                groups[group].append({**item, field: selected[claim_id].statement,
                                      "task_ids": task_ids, "source_ids": references[claim_id]})
    ids = list(dict.fromkeys(key for item in [*tasks, *groups["processes"], *groups["aspects"]]
                             for key in item["source_ids"]))
    return ({"tasks": tasks, **groups, "source_ids": ids} if ids else {}), list(records.values())


def manual_profile(profile: ProfessionProfile, tasks: list[str], record: SourceRecord) -> ProfessionProfile:
    claims = list(profile.claims)
    seen = {normalized(claim.statement) for claim in claims}
    for i, task in enumerate(tasks, 1):
        if normalized(task) not in seen:
            claim = pending_claim(f"manual:{i}", "work_task", task, [
                EvidenceRef(source_type="methodologist_input", source_id=record.source_id, quote_or_metric=task)])
            claims.append(claim.model_copy(update={"verification_reason": "Рабочую задачу указал методист."}))
            seen.add(normalized(task))
    return profile.model_copy(update={"claims": claims})
