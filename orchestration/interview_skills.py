import logging
from pathlib import Path

import yaml
from pydantic import Field

from orchestration.market_analysis import load_market_config
from orchestration.profession_profile import Contract


LOG = logging.getLogger(__name__)
MIN_MARKET_SAMPLE = load_market_config()["thresholds"]["min_sample"]
MIN_RAG_CHUNKS = 3
SKILL_FILES = {
    "dacum_task_analysis": "01_dacum_task_analysis.md",
    "member_checking_validation": "02_member_checking_validation.md",
    "critical_incident_technique": "03_critical_incident_technique.md",
    "gap_focused_slot_filling": "04_gap_focused_slot_filling.md",
}


class InterviewSkill(Contract):
    id: str
    max_turns: int = Field(ge=1, le=8, strict=True)
    title: str = Field(min_length=1)
    instructions: str = Field(min_length=1)


FALLBACK = InterviewSkill(id="gap_focused_slot_filling", max_turns=1, title="Уточнение пробела",
                          instructions="Уточни один недостающий факт. Попроси назвать действие и проверяемый результат.")


def load_interview_skills(directory: str | Path) -> dict[str, InterviewSkill]:
    path = Path(directory)
    skills = {}
    for skill_id, filename in SKILL_FILES.items():
        try:
            raw = (path / filename).read_text(encoding="utf-8-sig")
            first, header, body = raw.split("---", 2)
            if first.strip():
                raise ValueError("Отсутствует YAML-заголовок.")
            skill = InterviewSkill.model_validate({**yaml.safe_load(header), "instructions": body})
            if skill.id != skill_id:
                raise ValueError("Идентификатор методики отличается от ожидаемого.")
            skills[skill_id] = skill
        except FileNotFoundError:
            continue
        except (OSError, TypeError, ValueError, yaml.YAMLError):
            LOG.warning("Некорректная методика интервью: %s", filename,
                        extra={"event": "interview_skill_invalid", "skill_id": skill_id})
    if not skills:
        LOG.warning("Локальные методики интервью недоступны. Используется уточнение пробелов.",
                    extra={"event": "interview_skills_unavailable"})
    return skills


def select_interview_method(*, sample_size: int, rag_count: int, has_unverified: bool,
                            missing_lab_tasks: bool, missing_slots: list[str]) -> str:
    if sample_size < MIN_MARKET_SAMPLE and rag_count < MIN_RAG_CHUNKS:
        return "dacum_task_analysis"
    if has_unverified:
        return "member_checking_validation"
    if missing_lab_tasks:
        return "critical_incident_technique"
    return "gap_focused_slot_filling" if missing_slots else FALLBACK.id
