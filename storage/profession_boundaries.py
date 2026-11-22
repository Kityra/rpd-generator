import hashlib
import re
import secrets
from datetime import timedelta
from uuid import uuid4

from sqlalchemy import func, select

from auth.session import aware
from orchestration.interview_skills import BOUNDARY_METHOD, BOUNDARY_SLOTS, load_interview_skills
from orchestration.profession_boundaries import (boundary_conflicts, compute_adjacent_profession_candidates,
                                                 item_key, process_boundary_turn, role_key, with_boundary_warnings)
from storage.activity_models import ModelVersionRow, Profession
from storage.database import write_transaction
from storage.models import Course, CourseVersion, ProfessionBoundary, now


CONFIRMED = {"confirmed", "confirmed_early"}
PARTITIONS = (*BOUNDARY_SLOTS[:3], "boundary_conflicts")
EDITABLE = (*BOUNDARY_SLOTS, "boundary_conflicts")
TOKEN = re.compile(r"[A-Za-z0-9_-]{43}")
TOKEN_IN_ANSWER = re.compile(r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{43}(?![A-Za-z0-9_-])")
MARKET_SKILL_PREFIX = "В выборке вакансий указан навык "


def public_boundary(row: ProfessionBoundary) -> dict:
    return {"id": row.id, "primary_profession": row.primary_profession, "adjacent_profession": row.adjacent_profession,
            "skill_jaccard_overlap": row.skill_jaccard_overlap, "overlapping_items": row.overlap_tasks_json,
            "source_interview_id": row.source_interview_id, "verified_by_methodologist": row.verified_by_methodologist,
            "turn_count": row.turn_count, "interview_completed": row.interview_completed,
            "rejected_noise_turns": row.rejected_noise_turns_json,
            **{field: getattr(row, field + "_json") for field in EDITABLE}}


async def course_boundaries(db, course_id: str, *, lock: bool = False) -> list[ProfessionBoundary]:
    query = select(ProfessionBoundary).where(ProfessionBoundary.course_id == course_id).order_by(
        ProfessionBoundary.skill_jaccard_overlap.desc(), ProfessionBoundary.adjacent_profession)
    return list((await db.scalars(query.with_for_update() if lock else query)).all())


async def role_skill_catalog(db) -> dict[str, list[str]]:
    latest = select(ModelVersionRow.profession_id, func.max(ModelVersionRow.number).label("number")).group_by(
        ModelVersionRow.profession_id).subquery()
    rows = await db.execute(select(Profession.name, ModelVersionRow.content).join(
        ModelVersionRow, ModelVersionRow.profession_id == Profession.id).join(
        latest, (latest.c.profession_id == ModelVersionRow.profession_id) &
        (latest.c.number == ModelVersionRow.number)).where(Profession.is_demo.is_(False)).order_by(Profession.name))
    catalog: dict[str, list[str]] = {}
    for name, version in rows:
        knowledge = version.get("knowledge", {})
        for task in version.get("tasks", {}).values():
            if task.get("status") not in CONFIRMED:
                continue
            items = catalog.setdefault(name, [])
            items.append(task["name"])
            for link in task.get("related", []):
                item = knowledge.get(link["id"])
                if item and item.get("level") in {"skill", "tool"}:
                    items.append(item["name"])
    return catalog


async def sync_boundary_candidates(factory, course_id: str, profile: dict) -> list[dict]:
    primary = profile["profession_name"]
    verified = []
    for claim in profile["claims"]:
        if claim["verification_status"] != "verified" or claim["category"] not in {"work_task", "tool_stack"}:
            continue
        value = claim["statement"]
        if value.startswith(MARKET_SKILL_PREFIX) and any(
                ref["source_type"] == "market_cluster" for ref in claim.get("evidence", [])):
            value = value.removeprefix(MARKET_SKILL_PREFIX).removesuffix(".")
        verified.append(value)
    async with factory() as db:
        catalog = await role_skill_catalog(db)
    primary_items = [*verified, *(item for role, items in catalog.items()
                                  if role_key(role) == role_key(primary) for item in items)]
    candidates = compute_adjacent_profession_candidates(primary, primary_items, catalog)
    async with write_transaction(factory) as db:
        course = await db.scalar(select(Course).where(Course.id == course_id).with_for_update())
        if course is None:
            return []
        rows = {role_key(row.adjacent_profession): row for row in await course_boundaries(db, course_id, lock=True)
                if role_key(row.primary_profession) == role_key(primary)}
        for candidate in candidates:
            key = role_key(candidate.adjacent_profession)
            row = rows.get(key)
            if row is None:
                row = rows[key] = ProfessionBoundary(course_id=course_id, primary_profession=primary,
                    adjacent_profession=candidate.adjacent_profession, skill_jaccard_overlap=candidate.skill_jaccard_overlap,
                    overlap_tasks_json=candidate.overlapping_items, primary_claims_json=verified)
                db.add(row)
            elif not row.verified_by_methodologist:
                row.primary_claims_json = verified
                if not row.source_interview_id:
                    row.overlap_tasks_json = candidate.overlapping_items
                    row.skill_jaccard_overlap = candidate.skill_jaccard_overlap
        await db.flush()
        return [public_boundary(row) for row in sorted(rows.values(), key=lambda row: (-row.skill_jaccard_overlap,
                                                                                      row.adjacent_profession))]


async def _owned(db, actor, course_id: str, boundary_id: int):
    if actor.role not in {"teacher", "admin", "program_head"}:
        raise PermissionError("Доступно методисту курса.")
    course = await db.scalar(select(Course).where(Course.id == course_id, Course.owner_user_id == actor.id,
                             Course.status != "cancelled").with_for_update())
    if course is None:
        raise PermissionError("Курс недоступен.")
    row = await db.scalar(select(ProfessionBoundary).where(ProfessionBoundary.id == boundary_id,
                          ProfessionBoundary.course_id == course_id).with_for_update())
    if row is None:
        raise PermissionError("Граница профессий недоступна.")
    return course, row


async def create_boundary_interview_invite(factory, actor, course_id: str, boundary_id: int, *,
                                           ttl: timedelta = timedelta(days=7)) -> str:
    async with write_transaction(factory) as db:
        _, row = await _owned(db, actor, course_id, boundary_id)
        if row.verified_by_methodologist or row.interview_completed or not row.overlap_tasks_json:
            raise ValueError("Интервью по этой границе завершено или недоступно.")
        token = secrets.token_urlsafe(32)
        row.invite_token_hash = hashlib.sha256(token.encode()).hexdigest()
        row.invite_expires_at = now() + ttl
        if not row.source_interview_id:
            row.source_interview_id = str(uuid4())
    return token


async def _find_invite(db, token: str, *, lock: bool = False):
    if not isinstance(token, str) or not TOKEN.fullmatch(token):
        return None
    query = select(ProfessionBoundary).where(
        ProfessionBoundary.invite_token_hash == hashlib.sha256(token.encode()).hexdigest())
    row = await db.scalar(query.with_for_update() if lock else query)
    if row is None or row.invite_expires_at is None or aware(row.invite_expires_at) <= now():
        return None
    return row


async def validate_boundary_invite(factory, token: str) -> dict | None:
    async with factory() as db:
        row = await _find_invite(db, token)
        return public_boundary(row) if row is not None else None


def _merge_guest_result(row, fields: dict, conflict_threshold: float) -> None:
    partitions = {field: {item_key(item): item for item in getattr(row, field + "_json")} for field in PARTITIONS}
    new = {}
    for field in BOUNDARY_SLOTS[:3]:
        for item in fields.get(field, []):
            key = item_key(item)
            destination = "boundary_conflicts" if key in new else field
            new[key] = (destination, item)
    for key, (field, item) in new.items():
        for group in partitions.values():
            group.pop(key, None)
        if field == "exclusive_to_adjacent" and boundary_conflicts([item], row.primary_claims_json,
                                                                  threshold=conflict_threshold):
            field = "boundary_conflicts"
        partitions[field][key] = item
    for field, items in partitions.items():
        setattr(row, field + "_json", list(items.values()))
    row.handoff_artifacts_json = list(dict.fromkeys([*row.handoff_artifacts_json, *fields.get("handoff_artifacts", [])]))


async def submit_boundary_turn(factory, token: str, answer: str, llm, settings, *,
                               expected_turn: int, finish: bool = False) -> tuple[dict, str | None]:
    async with factory() as db:
        row = await _find_invite(db, token)
        if row is None or row.interview_completed or row.verified_by_methodologist or row.turn_count != expected_turn:
            raise PermissionError("Реплика устарела или приглашение недоступно.")
        overlap = list(row.overlap_tasks_json)
        earlier = tuple(getattr(row, field + "_json") for field in EDITABLE)
    skill = load_interview_skills(settings.interview_skill_directory).get(BOUNDARY_METHOD)
    limit = min(3, skill.max_turns) if skill else 3
    if expected_turn >= limit:
        raise ValueError("Лимит реплик исчерпан.")
    if len(answer) > 16000:
        raise ValueError("Ответ должен быть не длиннее 16 000 символов.")
    result = await process_boundary_turn(overlap, TOKEN_IN_ANSWER.sub("", answer.replace(token, "")),
                                         llm, settings.interview_skill_directory)
    async with write_transaction(factory) as db:
        row = await _find_invite(db, token, lock=True)
        if (row is None or row.turn_count != expected_turn or row.interview_completed or row.verified_by_methodologist
                or tuple(getattr(row, field + "_json") for field in EDITABLE) != earlier):
            raise PermissionError("Реплика устарела или приглашение недоступно.")
        row.turn_count += 1
        if result.classification == "nonsense_or_offtopic":
            row.rejected_noise_turns_json = [*row.rejected_noise_turns_json, row.turn_count]
        else:
            _merge_guest_result(row, result.fields, settings.boundary_conflict_threshold)
        row.interview_completed = row.turn_count >= limit or (finish and bool(result.fields))
        return public_boundary(row), result.clarification


async def save_boundary_review(factory, actor, course_id: str, boundary_id: int, values: dict, approved: bool, *,
                               threshold: float) -> None:
    lists = {field: list({item_key(item): item.strip() for item in values.get(field, []) if item.strip()}.values())
             for field in EDITABLE}
    if any(len(items) > 100 or any(len(item) > 2000 for item in items) for items in lists.values()):
        raise ValueError("Для каждого поля допускается до 100 пунктов, до 2000 символов каждый.")
    keys = [{item_key(item) for item in lists[field]} for field in PARTITIONS]
    if len(set().union(*keys)) != sum(map(len, keys)):
        raise ValueError("Один пункт может находиться только в одной зоне или в списке конфликтов.")
    async with write_transaction(factory) as db:
        course, row = await _owned(db, actor, course_id, boundary_id)
        for field, items in lists.items():
            setattr(row, field + "_json", items)
        row.verified_by_methodologist = approved
        if approved:
            row.interview_completed = True
            row.invite_token_hash = None
            row.invite_expires_at = None
        boundaries = [public_boundary(item) for item in await course_boundaries(db, course_id)]
        for version_id in dict.fromkeys(filter(None, (course.draft_version_id, course.active_version_id))):
            version = await db.get(CourseVersion, version_id)
            if version and version.course_id == course_id:
                primary = (version.content.get("profession_profile") or {}).get("profession_name")
                relevant = [item for item in boundaries if not primary or role_key(item["primary_profession"]) == role_key(primary)]
                program = with_boundary_warnings(version.content["program"], relevant, threshold=threshold)
                version.content = {**version.content, "program": program, "profession_boundaries": relevant}
