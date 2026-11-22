"""Хранилище модели деятельности: источники, нормализация формулировок, эксперты, версии модели, программы.

Все изменения пишутся в журнал (activity_merge_log) и обратимы. Расчёт — activity/engine.py и activity/program.py.
"""

import hashlib
from collections import Counter
from datetime import date, datetime, timezone

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from orchestration.semantic_check import concepts
from storage.database import write_transaction
from storage.activity_models import (ActivityClaim, ActivityExpert, ActivityItem, ActivityMergeLog, ActivitySource,
                                     ClaimLink, DisciplineLink, ExpertVote, ExtractionCache, ItemLink, ItemMetrics,
                                     KnowledgeArea, ModelVersionRow, Profession, Program, ProgramDiscipline,
                                     Recommendation)

from activity import engine
from activity.config import load_areas, load_config
from activity.experts import Expert, Vote, assign_cards
from activity.extract import LlmExtractor, deduplicate, parse_lines, parse_standard, verify
from activity.program import Discipline, ProgramInput, analyse
from activity.text import AreaDictionary, canonical_key, jaccard, normalize, object_concepts


SOURCE_TYPES = ("standard", "vacancy", "rpd", "science", "employer_doc", "forecast", "expert")
Factory = async_sessionmaker[AsyncSession]


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def as_date(value: datetime | None) -> date | None:
    return value.date() if value else None


def aware(value: datetime | date | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def require_head(actor) -> None:
    if actor.role not in {"program_head", "admin"}:
        raise PermissionError("Действие доступно руководителю образовательной программы.")


class SqlExtractionCache:
    """Кэш ответов LLM-экстрактора по хэшу (текст + версия промта + модель)."""

    def __init__(self, factory: Factory):
        self.factory = factory

    async def get(self, key: str) -> dict | None:
        async with self.factory() as db:
            row = await db.get(ExtractionCache, key)
            return row.response if row else None

    async def put(self, key: str, prompt_version: str, model: str, response: dict) -> None:
        async with write_transaction(self.factory) as db:
            if await db.get(ExtractionCache, key) is None:
                db.add(ExtractionCache(key=key, prompt_version=prompt_version, model=model, response=response))


async def log(db: AsyncSession, profession_id: str, action: str, payload: dict, actor: str) -> None:
    db.add(ActivityMergeLog(profession_id=profession_id, action=action, payload=payload, actor=actor))


# --- Профессии и источники. ---

async def create_profession(factory: Factory, actor, name: str, region: str = "", horizon_years: int = 3,
                            standard_codes: list[str] | None = None, *, is_demo: bool = False) -> str:
    require_head(actor)
    if not name.strip():
        raise ValueError("Введите название профессии.")
    async with write_transaction(factory) as db:
        row = Profession(name=" ".join(name.split()), region=region.strip(), horizon_years=horizon_years,
                         standard_codes=[code.strip() for code in standard_codes or [] if code.strip()],
                         is_demo=is_demo, created_by=actor.id, meta={"delphi_round": 1})
        db.add(row)
        await db.flush()
        return row.id


async def add_source(factory: Factory, actor, profession_id: str, kind: str, title: str, content: str, *,
                     origin_key: str = "", url: str = "", published_at: date | datetime | None = None,
                     collected_at: date | datetime | None = None, standard_code: str | None = None,
                     standard_year: int | None = None, meta: dict | None = None) -> tuple[str, bool]:
    """Сохранить источник. Возвращает (id, создан ли новый). Повтор того же текста не создаёт дубль."""
    if kind not in SOURCE_TYPES:
        raise ValueError(f"Неизвестный тип источника {kind}.")
    text = content.strip()
    if not text:
        raise ValueError("В источнике нет текста.")
    async with write_transaction(factory) as db:
        existing = await db.scalar(select(ActivitySource.id).where(
            ActivitySource.profession_id == profession_id, ActivitySource.content_hash == digest(text),
            ActivitySource.type == kind, ActivitySource.origin_key == (origin_key or title).strip().casefold()))
        if existing:
            return existing, False
        row = ActivitySource(profession_id=profession_id, type=kind, title=" ".join(title.split())[:500] or "Без названия",
                             url=url.strip(), origin_key=" ".join((origin_key or title).split()).casefold()[:300],
                             published_at=aware(published_at), collected_at=aware(collected_at) or datetime.now(timezone.utc),
                             content_hash=digest(text), content=text, standard_code=standard_code,
                             standard_year=standard_year, meta={**(meta or {}), "added_by": getattr(actor, "id", "")})
        db.add(row)
        await db.flush()
        return row.id, True


async def extract_source(factory: Factory, source_id: str, profession_name: str,
                         extractor: LlmExtractor | None = None) -> dict:
    """Извлечь утверждения: сначала правила, для свободного текста — LLM. Каждая цитата проверяется кодом."""
    async with factory() as db:
        source = await db.get(ActivitySource, source_id)
        already = await db.scalar(select(func.count()).select_from(ActivityClaim).where(ActivityClaim.source_id == source_id))
    if already:
        return {"claims": already, "note": "утверждения уже извлечены"}
    if source.type == "standard":
        result = parse_standard(source.content, source.standard_code)
    else:
        result = parse_lines(source.content, source.type)
    drafts = list(result.claims)
    status = "done"
    if result.needs_llm or (source.type != "standard" and not any(d.kind == "task" for d in drafts)):
        if extractor is None:
            if not drafts or result.needs_llm:
                result.notes.append("Свободный текст не разобран: LLM-экстрактор не настроен (заполните .env).")
                status = "partial"
        else:
            try:
                drafts += await extractor.extract(profession_name, source.type, source.title,
                                                  source.published_at.date().isoformat() if source.published_at else None,
                                                  source.content)
            except Exception as error:  # сбой модели не должен терять правила и сам источник
                result.notes.append(f"LLM-экстрактор не ответил: {type(error).__name__}. Повторите извлечение позже.")
                status = "failed"
    checked = verify(deduplicate(drafts), source.content)
    async with write_transaction(factory) as db:
        for draft in checked:
            db.add(ActivityClaim(source_id=source_id, kind=draft.kind, text=draft.text, verb=draft.verb,
                                 object=draft.object, tool=draft.tool, result=draft.result, quote=draft.quote,
                                 quote_ok=draft.quote_ok, foreign_role=draft.foreign_role,
                                 rejection_reason=draft.rejection_reason, function_hint=draft.function_hint,
                                 extractor=draft.extractor))
        row = await db.get(ActivitySource, source_id)
        row.meta = {**(row.meta or {}), "extraction": status, "notes": result.notes,
                    "claims": dict(Counter(d.kind for d in checked if d.quote_ok)),
                    "rejected": sum(not d.quote_ok for d in checked)}
    return {"claims": len(checked), "verified": sum(d.quote_ok for d in checked), "notes": result.notes, "status": status}


async def retry_extraction(factory: Factory, source_id: str, profession_name: str, extractor: LlmExtractor | None) -> dict:
    """Повторное извлечение: прежние утверждения источника и их связи удаляются (источник остаётся)."""
    async with write_transaction(factory) as db:
        ids = (await db.scalars(select(ActivityClaim.id).where(ActivityClaim.source_id == source_id))).all()
        if ids:
            await db.execute(delete(ClaimLink).where(ClaimLink.claim_id.in_(ids)))
            await db.execute(delete(ActivityClaim).where(ActivityClaim.id.in_(ids)))
    return await extract_source(factory, source_id, profession_name, extractor)


# --- Нормализация формулировок (раздел 7.3). ---

def function_similarity(task_name: str, function: ActivityItem, members: list[ActivityItem]) -> float:
    terms = concepts(task_name)
    return max([jaccard(terms, concepts(function.name)), *(jaccard(terms, concepts(item.name)) for item in members)],
               default=0.0)


def nearest_function(task_name: str, functions: list[ActivityItem], tasks: list[ActivityItem]) -> tuple[ActivityItem | None, float]:
    best, score = None, 0.0
    for function in sorted(functions, key=lambda item: item.id):
        value = function_similarity(task_name, function, [t for t in tasks if t.parent_id == function.id])
        if value > score:
            best, score = function, value
    return best, score


async def link_claims(factory: Factory, profession_id: str, actor: str = "system") -> dict:
    """Привязать проверенные утверждения к пунктам модели: точное совпадение ключа или новый пункт."""
    config = load_config()
    areas = AreaDictionary()
    area_ids = await sync_areas(factory)
    counts = Counter()
    async with write_transaction(factory) as db:
        items = (await db.scalars(select(ActivityItem).where(ActivityItem.profession_id == profession_id))).all()
        linked = set((await db.scalars(select(ClaimLink.claim_id).join(ActivityClaim).join(ActivitySource)
                                       .where(ActivitySource.profession_id == profession_id))).all())
        claims = (await db.scalars(select(ActivityClaim).join(ActivitySource).where(
            ActivitySource.profession_id == profession_id, ActivityClaim.quote_ok.is_(True),
            ActivityClaim.foreign_role.is_(False)).order_by(ActivitySource.collected_at, ActivityClaim.created_at,
                                                             ActivityClaim.id))).all()
        sources = {s.id: s for s in (await db.scalars(select(ActivitySource).where(
            ActivitySource.profession_id == profession_id))).all()}
        by_key = {(item.level, item.canonical_key): item for item in items if item.active and not item.is_control}
        functions = {item.standard_code: item for item in items if item.level == "function" and item.standard_code}
        for claim in claims:
            if claim.id in linked:
                continue
            level = claim.kind
            key = canonical_key(claim.text, level, config)
            item = by_key.get((level, key))
            hint = claim.function_hint or {}
            if item is None:
                parent = None
                if hint.get("code"):
                    parent = functions.get(hint["code"])
                    if parent is None:
                        parent = ActivityItem(profession_id=profession_id, level="function",
                                              name=hint.get("name") or hint["code"], canonical_key=f"f:{hint['code']}",
                                              standard_code=hint["code"], origin="sources")
                        db.add(parent)
                        await db.flush()
                        functions[hint["code"]] = parent
                        items = [*items, parent]
                        counts["functions"] += 1
                elif level == "task":
                    candidates = [i for i in items if i.level == "function" and i.active]
                    best, score = nearest_function(claim.text, candidates, [i for i in items if i.level == "task"])
                    if best is not None and score >= config["grouping"]["auto"] - 1e-9:
                        parent = best
                origin = "expert_proposal" if sources[claim.source_id].type == "expert" and level == "task" else "sources"
                found = areas.detect(claim.text)
                item = ActivityItem(profession_id=profession_id, level=level, name=claim.text, canonical_key=key,
                                    parent_id=parent.id if parent else None, origin=origin,
                                    first_round=2 if origin == "expert_proposal" else 1,
                                    standard_code=hint.get("code"),
                                    knowledge_area_id=area_ids.get(found[0]) if found and level != "task" else None)
                db.add(item)
                await db.flush()
                items = [*items, item]
                by_key[(level, key)] = item
                counts[f"new_{level}"] += 1
                method = "exact"
            else:
                method = "exact" if normalize(item.name) == normalize(claim.text) else "synonym"
                counts[f"linked_{method}"] += 1
            db.add(ClaimLink(claim_id=claim.id, item_id=item.id, method=method, similarity=1.0))
    return dict(counts)


async def sync_areas(factory: Factory) -> dict[str, str]:
    """Справочник областей знаний из config/knowledge_areas.yaml → таблица knowledge_areas."""
    async with write_transaction(factory) as db:
        rows = {row.name: row for row in (await db.scalars(select(KnowledgeArea))).all()}
        for area in load_areas():
            row = rows.get(area["name"])
            if row is None:
                row = KnowledgeArea(name=area["name"], kind=area["kind"], synonyms=list(area.get("synonyms", [])))
                db.add(row)
                rows[area["name"]] = row
            else:
                row.kind, row.synonyms = area["kind"], list(area.get("synonyms", []))
        await db.flush()
        return {name: row.id for name, row in rows.items()}


async def ensure_controls(factory: Factory, profession_id: str) -> int:
    """Добавить «чужие» контрольные карточки из конфига, не пересекающиеся по понятиям с задачами профессии."""
    config = load_config()
    async with write_transaction(factory) as db:
        items = (await db.scalars(select(ActivityItem).where(ActivityItem.profession_id == profession_id))).all()
        # Сравниваются понятия объекта без глагола: «Проектирует» есть у многих профессий.
        words = set().union(*(object_concepts(item.name) for item in items if item.level == "task" and not item.is_control))
        existing = {normalize(item.name) for item in items if item.is_control}
        added = 0
        for text in config["controls"]["foreign_tasks"]:
            if normalize(text) in existing or object_concepts(text) & words:
                continue
            db.add(ActivityItem(profession_id=profession_id, level="task", name=text, canonical_key=canonical_key(text),
                                is_control=True, control_kind="foreign", origin="control"))
            added += 1
        return added


async def add_control(factory: Factory, actor, profession_id: str, text: str, duplicate_of: str | None = None) -> None:
    require_head(actor)
    if not text.strip():
        raise ValueError("Введите формулировку карточки.")
    async with write_transaction(factory) as db:
        db.add(ActivityItem(profession_id=profession_id, level="task", name=" ".join(text.split()),
                            canonical_key=canonical_key(text), is_control=True,
                            control_kind="duplicate" if duplicate_of else "foreign", duplicate_of=duplicate_of,
                            origin="control"))
        await log(db, profession_id, "control", {"text": text, "duplicate_of": duplicate_of}, actor.id)


# --- Очередь методиста: слияния, разметка функций, формулировки. ---

async def merge_items(factory: Factory, actor, source_id: str, target_id: str) -> None:
    require_head(actor)
    async with write_transaction(factory) as db:
        source, target = await db.get(ActivityItem, source_id), await db.get(ActivityItem, target_id)
        if source is None or target is None or source.id == target.id or source.level != target.level:
            raise ValueError("Объединять можно два разных пункта одного уровня.")
        source.active, source.merged_into = False, target.id
        await log(db, source.profession_id, "merge", {"item": source.id, "into": target.id, "name": source.name,
                                                      "into_name": target.name}, actor.id)


async def unmerge_item(factory: Factory, actor, item_id: str) -> None:
    require_head(actor)
    async with write_transaction(factory) as db:
        item = await db.get(ActivityItem, item_id)
        if item is None or not item.merged_into:
            raise ValueError("Пункт не был объединён.")
        target = item.merged_into
        item.active, item.merged_into = True, None
        await log(db, item.profession_id, "unmerge", {"item": item.id, "from": target, "name": item.name}, actor.id)


async def set_function(factory: Factory, actor, task_id: str, function_id: str | None) -> None:
    require_head(actor)
    async with write_transaction(factory) as db:
        task = await db.get(ActivityItem, task_id)
        old = task.parent_id
        task.parent_id = function_id
        await log(db, task.profession_id, "reparent", {"item": task.id, "from": old, "to": function_id,
                                                       "name": task.name}, actor.id)


async def create_function(factory: Factory, actor, profession_id: str, name: str) -> str:
    require_head(actor)
    if not name.strip():
        raise ValueError("Введите название функции.")
    async with write_transaction(factory) as db:
        row = ActivityItem(profession_id=profession_id, level="function", name=" ".join(name.split()),
                           canonical_key="f:" + " ".join(sorted(concepts(name))), origin="manual")
        db.add(row)
        await db.flush()
        await log(db, profession_id, "function", {"item": row.id, "name": row.name}, actor.id)
        return row.id


async def reword_item(factory: Factory, actor, item_id: str, name: str) -> None:
    """Новая формулировка — новая версия пункта: голоса за прежнюю формулировку остаются при ней."""
    require_head(actor)
    if not name.strip():
        raise ValueError("Введите формулировку.")
    async with write_transaction(factory) as db:
        item = await db.get(ActivityItem, item_id)
        item.history = [*(item.history or []), {"version": item.version, "name": item.name, "changed_by": actor.id,
                                                "at": datetime.now(timezone.utc).isoformat()}]
        old = item.name
        item.name, item.version = " ".join(name.split()), item.version + 1
        item.canonical_key = canonical_key(item.name, item.level)
        await log(db, item.profession_id, "reword", {"item": item.id, "from": old, "to": item.name,
                                                     "version": item.version}, actor.id)


async def dismiss(factory: Factory, actor, profession_id: str, key: str, note: str = "") -> None:
    require_head(actor)
    async with write_transaction(factory) as db:
        await log(db, profession_id, "dismiss", {"key": key, "note": note}, actor.id)


async def review_queue(db: AsyncSession, profession_id: str, content: dict | None) -> dict:
    """Кандидаты в дубли, задачи без функции с подсказкой, предложенные экспертами формулировки."""
    config = load_config()["grouping"]
    items = (await db.scalars(select(ActivityItem).where(ActivityItem.profession_id == profession_id))).all()
    dismissed = {row.payload["key"] for row in (await db.scalars(select(ActivityMergeLog).where(
        ActivityMergeLog.profession_id == profession_id, ActivityMergeLog.action == "dismiss"))).all()}
    tasks = sorted((i for i in items if i.level == "task" and i.active and not i.is_control), key=lambda i: (i.name, i.id))
    functions = [i for i in items if i.level == "function" and i.active]
    duplicates = []
    for index, left in enumerate(tasks):
        for right in tasks[index + 1:]:
            score = jaccard(concepts(left.name), concepts(right.name))
            key = "dup:" + ":".join(sorted([left.id, right.id]))
            if score >= config["duplicate_review"] - 1e-9 and key not in dismissed:
                duplicates.append({"key": key, "left": left, "right": right, "similarity": round(score, 2)})
    unassigned = []
    for task in tasks:
        if task.parent_id:
            continue
        best, score = nearest_function(task.name, functions, tasks)
        unassigned.append({"task": task, "suggestion": best if score >= config["review"] - 1e-9 else None,
                           "similarity": round(score, 2)})
    wordings = []
    for task_id, task in sorted(((content or {}).get("tasks") or {}).items()):
        for text in task["metrics"]["experts"]["wording_suggestions"]:
            key = "word:" + task_id + ":" + digest(text)[:12]
            if key not in dismissed and normalize(text) != normalize(task["name"]):
                wordings.append({"key": key, "task_id": task_id, "name": task["name"], "text": text})
    merged = [item for item in items if item.merged_into]
    return {"duplicates": duplicates, "unassigned": unassigned, "wordings": wordings, "merged": merged,
            "functions": sorted(functions, key=lambda f: (f.standard_code or "~", f.name))}


# --- Эксперты и анкета (раздел 7.5). ---

async def invite_expert(factory: Factory, actor, profession_id: str, user_id: str, *, demo: bool = False) -> str:
    if not demo:
        require_head(actor)
    async with write_transaction(factory) as db:
        existing = await db.scalar(select(ActivityExpert).where(ActivityExpert.user_id == user_id,
                                                                ActivityExpert.profession_id == profession_id))
        if existing:
            return existing.id
        row = ActivityExpert(user_id=user_id, profession_id=profession_id, invited_by=getattr(actor, "id", "system"))
        db.add(row)
        await db.flush()
        return row.id


async def expert_cards(factory: Factory, expert_id: str, round_number: int) -> list[str]:
    """Карточки эксперта на раунд: назначаются один раз и сохраняются (порядок не меняется между визитами)."""
    config = load_config()
    async with write_transaction(factory) as db:
        expert = await db.get(ActivityExpert, expert_id)
        cards = dict(expert.cards or {})
        if str(round_number) in cards:
            return cards[str(round_number)]
        items = (await db.scalars(select(ActivityItem).where(
            ActivityItem.profession_id == expert.profession_id, ActivityItem.level == "task",
            ActivityItem.active.is_(True)))).all()
        latest = await latest_version(db, expert.profession_id)
        statuses = {key: task["status"] for key, task in (latest.content["tasks"].items() if latest else [])}
        controls = [{"id": i.id, "kind": i.control_kind, "duplicate_of": i.duplicate_of} for i in items if i.is_control]
        if round_number == 1:
            tasks = [i.id for i in items if not i.is_control and i.first_round == 1]
            disputed = [key for key in tasks if statuses.get(key) == "disputed"]
        else:
            tasks = [i.id for i in items if not i.is_control and (statuses.get(i.id) == "disputed"
                                                                   or i.origin == "expert_proposal")]
            disputed, controls = tasks, []
        others = (await db.scalars(select(ActivityExpert).where(ActivityExpert.profession_id == expert.profession_id,
                                                                ActivityExpert.id != expert.id))).all()
        load = Counter(item for other in others for item in (other.cards or {}).get(str(round_number), []))
        chosen = assign_cards(expert.id, sorted(tasks), sorted(disputed), sorted(controls, key=lambda c: c["id"]),
                              load, config) if tasks else []
        cards[str(round_number)] = chosen
        expert.cards = cards
        return chosen


async def save_vote(factory: Factory, expert_id: str, item_id: str, round_number: int, answer: dict) -> None:
    verdict = answer["verdict"]
    if verdict not in ("regular", "sometimes", "other_role", "outdated", "dont_know"):
        raise ValueError("Выберите ответ.")
    yes = verdict in ("regular", "sometimes")
    async with write_transaction(factory) as db:
        item = await db.get(ActivityItem, item_id)
        row = await db.scalar(select(ExpertVote).where(ExpertVote.expert_id == expert_id, ExpertVote.item_id == item_id,
                                                       ExpertVote.item_version == item.version,
                                                       ExpertVote.round == round_number))
        if row is None:
            row = ExpertVote(expert_id=expert_id, item_id=item_id, item_version=item.version, round=round_number,
                             verdict=verdict)
            db.add(row)
        row.verdict = verdict
        row.importance = answer.get("importance") if yes else None
        row.forecast = answer.get("forecast") if verdict != "dont_know" else None
        row.wording_ok = answer.get("wording_ok")
        row.wording_suggestion = (answer.get("wording_suggestion") or "").strip() or None if row.wording_ok is False else None
        row.comment = (answer.get("comment") or "").strip() or None
        row.elapsed_ms = answer.get("elapsed_ms")


async def finish_round(factory: Factory, actor, expert_id: str, round_number: int, profile: dict,
                       extractor: LlmExtractor | None = None) -> dict:
    """Завершить раунд: профиль эксперта, прогноз роли и «чего не хватает» (идёт в экстрактор как источник expert)."""
    async with write_transaction(factory) as db:
        expert = await db.get(ActivityExpert, expert_id)
        profession = await db.get(Profession, expert.profession_id)
        for field in ("years", "role_kind", "company_size", "company_kind", "region", "role_forecast_text",
                      "role_demand", "missing_text"):
            if field in profile:
                value = profile[field]
                setattr(expert, field, value.strip() if isinstance(value, str) else value)
        expert.submitted_rounds = sorted({*(expert.submitted_rounds or []), round_number})
        missing = (expert.missing_text or "").strip()
        name = profession.name
    result = {"missing": None}
    if missing and round_number == 1:
        source_id, created = await add_source(factory, actor, expert.profession_id, "expert",
                                              "Ответ эксперта: чего не хватает в списке", missing,
                                              origin_key=f"expert:{expert_id}")
        if created:
            result["missing"] = await extract_source(factory, source_id, name, extractor)
            await link_claims(factory, expert.profession_id)
    return result


async def start_round_two(factory: Factory, actor, profession_id: str) -> None:
    require_head(actor)
    async with write_transaction(factory) as db:
        profession = await db.get(Profession, profession_id)
        profession.meta = {**(profession.meta or {}), "delphi_round": 2}
        await log(db, profession_id, "delphi", {"round": 2}, actor.id)


async def expert_rows(db: AsyncSession, profession_id: str) -> list[ActivityExpert]:
    return (await db.scalars(select(ActivityExpert).where(ActivityExpert.profession_id == profession_id)
                             .order_by(ActivityExpert.created_at, ActivityExpert.id))).all()


# --- Сборка версии модели. ---

async def load_data(db: AsyncSession, profession_id: str) -> engine.ModelData:
    profession = await db.get(Profession, profession_id)
    sources = (await db.scalars(select(ActivitySource).where(ActivitySource.profession_id == profession_id))).all()
    items = (await db.scalars(select(ActivityItem).where(ActivityItem.profession_id == profession_id))).all()
    by_id = {item.id: item for item in items}

    def resolve(item_id: str | None) -> str | None:
        seen = set()
        while item_id and item_id in by_id and not by_id[item_id].active and by_id[item_id].merged_into and item_id not in seen:
            seen.add(item_id)
            item_id = by_id[item_id].merged_into
        return item_id if item_id in by_id and by_id[item_id].active else None

    source_ids = [source.id for source in sources]
    claims = (await db.scalars(select(ActivityClaim).where(ActivityClaim.source_id.in_(source_ids)))).all() if source_ids else []
    links = {row.claim_id: row for row in (await db.scalars(select(ClaimLink).where(
        ClaimLink.claim_id.in_([claim.id for claim in claims])))).all()} if claims else {}
    experts = await expert_rows(db, profession_id)
    votes = (await db.scalars(select(ExpertVote).where(ExpertVote.expert_id.in_([e.id for e in experts])))).all() if experts else []
    manual = (await db.scalars(select(ItemLink).where(ItemLink.task_id.in_(list(by_id))))).all() if by_id else []
    active = {item.id for item in items if item.active}
    return engine.ModelData(
        profession={"id": profession.id, "name": profession.name, "region": profession.region,
                    "horizon_years": profession.horizon_years, "standard_codes": profession.standard_codes,
                    "is_demo": profession.is_demo, "meta": profession.meta or {}},
        sources=[engine.Source(id=s.id, type=s.type, title=s.title, origin_key=s.origin_key, url=s.url or "",
                               collected_at=s.collected_at.date(), published_at=as_date(s.published_at),
                               standard_code=s.standard_code, standard_year=s.standard_year) for s in sources],
        claims=[engine.Claim(id=c.id, source_id=c.source_id, kind=c.kind, text=c.text, quote=c.quote, quote_ok=c.quote_ok,
                             extractor=c.extractor, item_id=resolve(links[c.id].item_id) if c.id in links else None,
                             link_method=links[c.id].method if c.id in links else None, foreign_role=c.foreign_role,
                             rejection_reason=c.rejection_reason, tool=c.tool, result=c.result,
                             function_hint=c.function_hint) for c in claims],
        items=[engine.Item(id=i.id, level=i.level, name=i.name, canonical_key=i.canonical_key, parent_id=i.parent_id,
                           version=i.version, is_control=i.is_control, control_kind=i.control_kind,
                           duplicate_of=i.duplicate_of, standard_code=i.standard_code, origin=i.origin,
                           first_round=i.first_round) for i in items if i.active],
        experts=[Expert(id=e.id, label=f"Эксперт {index:02d}", years=e.years, role_kind=e.role_kind,
                        company_size=e.company_size, company_kind=e.company_kind, region=e.region,
                        role_forecast_text=e.role_forecast_text, role_demand=e.role_demand)
                 for index, e in enumerate(experts, 1)],
        votes=[Vote(expert_id=v.expert_id, item_id=v.item_id, verdict=v.verdict, item_version=v.item_version,
                    round=v.round, importance=v.importance, forecast=v.forecast, wording_ok=v.wording_ok,
                    wording_suggestion=v.wording_suggestion, comment=v.comment, elapsed_ms=v.elapsed_ms)
               for v in votes if v.item_id in active],
        manual_links=[(link.task_id, link.item_id, link.reason) for link in manual])


async def latest_version(db: AsyncSession, profession_id: str) -> ModelVersionRow | None:
    return await db.scalar(select(ModelVersionRow).where(ModelVersionRow.profession_id == profession_id)
                           .order_by(ModelVersionRow.number.desc()).limit(1))


async def build_version(factory: Factory, profession_id: str, as_of: date | None = None) -> ModelVersionRow:
    """Пересчитать модель. Если входные данные и конфиг не изменились, новая версия не создаётся."""
    as_of = as_of or datetime.now(timezone.utc).date()
    await ensure_controls(factory, profession_id)
    async with factory() as db:
        data = await load_data(db, profession_id)
        previous = await latest_version(db, profession_id)
    content = engine.build(data, as_of, previous=previous.content if previous else None)
    if previous and previous.input_hash == content["input_hash"]:
        return previous
    async with write_transaction(factory) as db:
        number = (previous.number if previous else 0) + 1
        statuses = Counter(task["status"] for task in content["tasks"].values())
        row = ModelVersionRow(profession_id=profession_id, number=number, as_of=aware(as_of),
                              config_hash=content["config_hash"], input_hash=content["input_hash"],
                              counts={"tasks": len(content["tasks"]), "statuses": dict(statuses),
                                      "experts": content["experts"]["reliable"]},
                              diff_from_previous=content["diff"], content={**content, "version": number})
        db.add(row)
        await db.flush()
        for task in content["tasks"].values():
            db.add(ItemMetrics(item_id=task["id"], model_version=row.id, metrics=task["metrics"], status=task["status"],
                               trend=task["trend"], lag_flag="; ".join(task["lag_flags"]) or None,
                               reason_text="\n".join([task["reason"]["title"], *task["reason"]["lines"]])))
        excluded = {item["label"]: "; ".join(item["reasons"]) for item in content["experts"]["excluded"]}
        for index, expert in enumerate(await expert_rows(db, profession_id), 1):
            label = f"Эксперт {index:02d}"
            expert.excluded, expert.exclusion_reason = label in excluded, excluded.get(label)
        profession = await db.get(Profession, profession_id)
        profession.status = "built"
        return row


async def latest_contents(db: AsyncSession) -> dict[str, dict]:
    """Последние версии моделей всех профессий: для карты знаний и смежных ролей."""
    numbers = select(ModelVersionRow.profession_id, func.max(ModelVersionRow.number).label("number")).group_by(
        ModelVersionRow.profession_id).subquery()
    rows = (await db.scalars(select(ModelVersionRow).join(numbers, (ModelVersionRow.profession_id == numbers.c.profession_id)
                                                          & (ModelVersionRow.number == numbers.c.number)))).all()
    return {row.profession_id: row.content for row in rows}


# --- Программа и рекомендации (разделы 7.10–7.11). ---

async def create_program(factory: Factory, actor, name: str, *, level: str = "bachelor", total_zet: int = 240,
                         institution: str = "", academic_year: str = "", targets: list[dict] | None = None,
                         is_demo: bool = False) -> str:
    require_head(actor)
    if not name.strip():
        raise ValueError("Введите название программы.")
    async with write_transaction(factory) as db:
        row = Program(head_user_id=actor.id, name=" ".join(name.split()), level=level, total_zet=total_zet,
                      institution=institution.strip(), academic_year=academic_year.strip(),
                      target_professions=targets or [], is_demo=is_demo)
        db.add(row)
        await db.flush()
        return row.id


async def update_program(factory: Factory, actor, program_id: str, **fields) -> None:
    require_head(actor)
    async with write_transaction(factory) as db:
        program = await owned_program(db, actor, program_id)
        for key in ("name", "level", "total_zet", "institution", "academic_year", "target_professions"):
            if key in fields:
                setattr(program, key, fields[key])


async def owned_program(db: AsyncSession, actor, program_id: str) -> Program:
    program = await db.get(Program, program_id)
    if program is None or (program.head_user_id != actor.id and actor.role != "admin"):
        raise PermissionError("Программа недоступна.")
    return program


async def save_disciplines(factory: Factory, actor, program_id: str, rows: list[dict]) -> None:
    """Сохранить список дисциплин из таблицы редактора (новые — без id, удалённые — исчезают из списка)."""
    require_head(actor)
    async with write_transaction(factory) as db:
        await owned_program(db, actor, program_id)
        current = {d.id: d for d in (await db.scalars(select(ProgramDiscipline).where(
            ProgramDiscipline.program_id == program_id))).all()}
        keep = set()
        for position, row in enumerate(rows):
            name = " ".join(str(row.get("name") or "").split())
            if not name:
                continue
            discipline = current.get(row.get("id") or "")
            if discipline is None:
                discipline = ProgramDiscipline(program_id=program_id, name=name, content={})
                db.add(discipline)
            part = row.get("part") or "mandatory"
            if part not in ("mandatory_fgos", "mandatory", "elective"):
                raise ValueError(f"Неизвестная часть программы: {part}.")
            discipline.name, discipline.part, discipline.position = name, part, position
            discipline.zet = float(row.get("zet") or 0)
            if "hours" in row:
                discipline.hours = int(row["hours"] or 0)
            discipline.semester = int(row["semester"]) if row.get("semester") not in (None, "") else None
            await db.flush()
            keep.add(discipline.id)
        for discipline_id, discipline in current.items():
            if discipline_id not in keep:
                await db.execute(delete(DisciplineLink).where(DisciplineLink.discipline_id == discipline_id))
                await db.delete(discipline)


async def save_content(factory: Factory, actor, discipline_id: str, topics: list[str], outcomes: list[str],
                       assignments: list[str]) -> None:
    require_head(actor)
    async with write_transaction(factory) as db:
        discipline = await db.get(ProgramDiscipline, discipline_id)
        await owned_program(db, actor, discipline.program_id)
        discipline.content = {"topics": [t for t in topics if t.strip()], "outcomes": [o for o in outcomes if o.strip()],
                              "assignments": [a for a in assignments if a.strip()]}


async def assign_teacher(factory: Factory, actor, discipline_id: str, teacher_id: str | None, teacher_name: str) -> None:
    require_head(actor)
    async with write_transaction(factory) as db:
        discipline = await db.get(ProgramDiscipline, discipline_id)
        await owned_program(db, actor, discipline.program_id)
        discipline.teacher_user_id, discipline.teacher_name = teacher_id, teacher_name if teacher_id else ""


async def set_coverage(factory: Factory, actor, discipline_id: str, item_id: str, coverage: str | None, reason: str = "") -> None:
    """Ручная правка покрытия важнее автоматики. coverage=None — вернуть автоматический расчёт."""
    if actor.role not in {"program_head", "admin", "teacher"}:
        raise PermissionError("Правка покрытия недоступна.")
    async with write_transaction(factory) as db:
        discipline = await db.get(ProgramDiscipline, discipline_id)
        if actor.role == "teacher" and discipline.teacher_user_id != actor.id:
            raise PermissionError("Дисциплина закреплена за другим методистом.")
        await db.execute(delete(DisciplineLink).where(DisciplineLink.discipline_id == discipline_id,
                                                      DisciplineLink.item_id == item_id))
        if coverage:
            db.add(DisciplineLink(discipline_id=discipline_id, item_id=item_id, coverage=coverage, method="manual",
                                  reason=reason, created_by=actor.id))


async def program_input(db: AsyncSession, program: Program) -> tuple[ProgramInput, dict[str, dict]]:
    disciplines = (await db.scalars(select(ProgramDiscipline).where(ProgramDiscipline.program_id == program.id)
                                    .order_by(ProgramDiscipline.position, ProgramDiscipline.name))).all()
    links = (await db.scalars(select(DisciplineLink).where(DisciplineLink.discipline_id.in_([d.id for d in disciplines])))).all() \
        if disciplines else []
    contents = await latest_contents(db)
    targets = [(item["profession_id"], float(item.get("weight") or 0)) for item in program.target_professions or []]
    data = ProgramInput(
        id=program.id, name=program.name, total_zet=program.total_zet,
        targets=targets, models={pid: contents[pid] for pid, _ in targets if pid in contents},
        disciplines=[Discipline(id=d.id, name=d.name, zet=d.zet, hours=d.hours, part=d.part, semester=d.semester,
                                topics=tuple((d.content or {}).get("topics", [])),
                                outcomes=tuple((d.content or {}).get("outcomes", [])),
                                assignments=tuple((d.content or {}).get("assignments", [])))
                     for d in disciplines],
        overrides={(link.discipline_id, link.item_id): link.coverage for link in links})
    return data, contents


async def recompute(factory: Factory, actor, program_id: str) -> dict:
    """Пересчитать рекомендации. Решение руководителя сохраняется, пока рекомендация не изменилась (ключ)."""
    async with factory() as db:
        program = await owned_program(db, actor, program_id) if actor else await db.get(Program, program_id)
        data, contents = await program_input(db, program)
    result = analyse(data, contents)
    versions = ", ".join(f"{contents[pid]['profession']['name']} v{contents[pid].get('version')}" for pid in data.models)
    async with write_transaction(factory) as db:
        rows = {row.key: row for row in (await db.scalars(select(Recommendation).where(
            Recommendation.program_id == program_id))).all()}
        keys = set()
        for rec in result["recommendations"]:
            keys.add(rec["key"])
            row = rows.get(rec["key"])
            if row is None:
                row = Recommendation(program_id=program_id, key=rec["key"], rule=rec["rule"], target=rec["target"],
                                     action_text=rec["action_text"], evidence={}, confidence=rec["confidence"])
                db.add(row)
            row.rule, row.target, row.action_text, row.confidence = rec["rule"], rec["target"], rec["action_text"], rec["confidence"]
            row.evidence = {key: rec[key] for key in ("lines", "tasks", "consequence", "zet", "weight", "rule_label")}
            row.model_version, row.current = versions, True
        for key, row in rows.items():
            if key not in keys:
                row.current = False
    return result


async def decide(factory: Factory, actor, recommendation_id: str, decision: str, comment: str) -> None:
    require_head(actor)
    if decision not in ("accepted", "rejected", "postponed"):
        raise ValueError("Неизвестное решение.")
    if decision == "rejected" and not comment.strip():
        raise ValueError("Укажите причину отклонения: она попадёт в журнал решений.")
    async with write_transaction(factory) as db:
        row = await db.get(Recommendation, recommendation_id)
        await owned_program(db, actor, row.program_id)
        row.decision, row.decision_comment = decision, comment.strip() or None
        row.decided_by, row.decided_at = actor.id, datetime.now(timezone.utc)


async def teacher_disciplines(db: AsyncSession, teacher_id: str) -> list[tuple[ProgramDiscipline, Program]]:
    return (await db.execute(select(ProgramDiscipline, Program).join(Program, Program.id == ProgramDiscipline.program_id)
                             .where(ProgramDiscipline.teacher_user_id == teacher_id)
                             .order_by(Program.name, ProgramDiscipline.position))).all()


async def link_course(factory: Factory, discipline_id: str, course_id: str) -> None:
    async with write_transaction(factory) as db:
        discipline = await db.get(ProgramDiscipline, discipline_id)
        discipline.course_id = course_id
