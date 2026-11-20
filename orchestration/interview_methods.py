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
    folder = Path(directory)
    paths = sorted(path for ext in ("*.md", "*.yaml", "*.yml") for path in folder.glob(ext))
    for path in paths:
        text = path.read_text(encoding="utf-8-sig")
        if path.suffix == ".md":
            if not text.startswith("---\n"):
                raise ValueError(f"У методики {path.name} отсутствуют YAML-метаданные.")
            parts = text.split("---", 2)
            if len(parts) != 3:
                raise ValueError(f"Повреждены метаданные {path.name}.")
            data = {**yaml.safe_load(parts[1]), "body": parts[2].strip()}
        else:
            data = yaml.safe_load(text)
        method = InterviewMethod.model_validate(data)
        if set(method.applicable_gap_types) & set(gap_types):
            selected.append(f"{method.title}\n{method.body}")
    return "\n\n".join(selected)
