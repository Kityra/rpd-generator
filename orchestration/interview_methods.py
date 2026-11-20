from pathlib import Path

import yaml
from pydantic import Field

from domain.schemas import GapType, Model


class InterviewMethod(Model):
    title: str
    applicable_gap_types: list[GapType] = Field(min_length=1)
    body: str = Field(min_length=1)


def load_interview_methods(directory: str, gap_types: list[str]) -> str:
    selected = []
    for path in sorted(Path(directory).glob("*.md")):
        text = path.read_text(encoding="utf-8-sig")
        if not text.startswith("---\n"):
            raise ValueError(f"У методики {path.name} отсутствуют YAML-метаданные.")
        parts = text.split("---", 2)
        if len(parts) != 3:
            raise ValueError(f"Повреждены метаданные {path.name}.")
        method = InterviewMethod.model_validate({**yaml.safe_load(parts[1]), "body": parts[2].strip()})
        if set(method.applicable_gap_types) & set(gap_types):
            selected.append(f"{method.title}\n{method.body}")
    return "\n\n".join(selected)
