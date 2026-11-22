"""Grounded edits to an existing discipline when the methodologist changes source priorities."""

from pydantic import Field, ValidationInfo, model_validator

from domain.schemas import ChangeJustification, CourseSubject, Model, all_blocks, block_value


class BlockEdit(Model):
    block_id: str
    title: str | None = Field(default=None, min_length=1)
    instructions: str | None = Field(default=None, min_length=1)
    expected_artifacts: list[str] | None = Field(default=None, min_length=1)
    quality_criteria: list[str] | None = Field(default=None, min_length=2)
    justification: ChangeJustification


class SubjectWeightRevision(Model):
    edits: list[BlockEdit]

    @model_validator(mode="after")
    def check_targets(self, info: ValidationInfo):
        ctx = info.context
        blocks = ctx["revision_blocks"]
        seen = set()
        for edit in self.edits:
            block = blocks.get(edit.block_id)
            if block is None or edit.block_id in seen:
                raise ValueError("Укажите каждый изменяемый блок один раз по его ID из existing_blocks.")
            seen.add(edit.block_id)
            fields = edit.model_dump(exclude_none=True, exclude={"block_id", "justification"})
            if fields.keys() - type(block).model_fields.keys():
                raise ValueError("У блока отсутствует указанное поле.")
            if (isinstance(block, CourseSubject) or block.locked) and fields.get("title", block.title) != block.title:
                raise ValueError("Сохраните название дисциплины и названия защищённых блоков.")
            if not set(edit.justification.source_ids) <= set(ctx["allowed_source_ids"]):
                raise ValueError("В обосновании укажите источники из запроса этой дисциплины.")
        return self


def apply_subject_revision(subject: CourseSubject, edits: list[BlockEdit]) -> dict:
    old = all_blocks([subject])
    changes = {edit.block_id: edit for edit in edits}
    data = subject.model_dump(mode="json")
    for block in [data, *data["topics"], *(item for topic in data["topics"] for item in topic["assignments"])]:
        block["change_status"] = "fundamental_protected" if block["is_fundamental"] else "unchanged"
        block["previous_value"] = None
        if edit := changes.get(block["id"]):
            values = edit.model_dump(exclude_none=True, exclude={"block_id", "justification"})
            if any(block[key] != value for key, value in values.items()):
                block.update(values, change_status="modified", previous_value=block_value(old[block["id"]]))
            block["justification"] = edit.justification.model_dump(mode="json")
            block["source_ids"] = list(dict.fromkeys([*edit.justification.source_ids, *block["source_ids"]]))
    return data
