from collections.abc import Mapping, Sequence
from typing import Annotated

from pydantic import Field

from domain.schemas import Model, SourceRecord


Weight = Annotated[float, Field(strict=True, ge=0, le=1)]
SOURCE_GROUPS = {
    "expert": "interview",
    "Books": "books",
    "HH": "business", "Priklad": "business", "RabotaRu": "business",
    "Rostrud": "business", "EmployerCareer": "business",
    "Crossref": "science", "arXiv": "science", "documentation": "science",
    "RPD": "university",
}


def source_group(source_type: str, content: str = "") -> str | None:
    if source_type == "Literature":
        return "science" if "Тип: научная статья" in content.splitlines() else "books"
    return SOURCE_GROUPS.get(source_type)


class SourceWeights(Model):
    interview: Weight = 1.0
    books: Weight = 1.0
    business: Weight = 1.0
    science: Weight = 1.0
    university: Weight = 1.0

    @property
    def customized(self) -> bool:
        return self != SourceWeights()

    def for_type(self, source_type: str, content: str = "") -> float:
        group = source_group(source_type, content)
        return getattr(self, group) if group else 1.0

    def for_source(self, source: SourceRecord) -> float:
        return self.for_type(source.source_type, source.content)

    def for_references(self, source_ids: Sequence[str], sources: Mapping[str, SourceRecord]) -> float:
        return max((self.for_source(sources[key]) for key in source_ids if key in sources), default=1.0)


def course_source_weights(meta: dict) -> SourceWeights:
    return SourceWeights.model_validate(meta.get("source_weights", {}))
