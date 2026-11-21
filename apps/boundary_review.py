import streamlit as st

from domain.export import boundary_table_html
from storage.profession_boundaries import EDITABLE, create_boundary_interview_invite, save_boundary_review


LABELS = {"exclusive_to_primary": "Ядро целевой профессии", "shared_core": "Общая база",
          "exclusive_to_adjacent": "Зона смежной профессии (исключить)", "boundary_conflicts": "Конфликты",
          "handoff_artifacts": "Артефакты передачи"}


async def boundary_review(actor, settings, factory, course_id: str, boundaries: list[dict]) -> None:
    if not boundaries:
        return
    st.subheader("Границы смежных профессий")
    for row in boundaries:
        identifier = row["id"]
        label = f"{row['adjacent_profession']} · пересечение {row['skill_jaccard_overlap']:.0%}"
        with st.expander(label):
            st.caption("Подтверждено методистом" if row["verified_by_methodologist"] else "Требуется решение методиста")
            st.html(boundary_table_html(row))
            if row["overlapping_items"]:
                st.caption("Предмет интервью — " + ", ".join(row["overlapping_items"]))
            if not (row["verified_by_methodologist"] or row["interview_completed"]):
                if st.button("Создать гостевое приглашение", key=f"boundary-invite-{identifier}"):
                    token = await create_boundary_interview_invite(factory, actor, course_id, identifier)
                    st.code(settings.public_origin.rstrip("/") + "/boundary#invite=" + token, language=None)
                    st.caption("Скопируйте приглашение для эксперта. При создании нового прежняя ссылка перестанет действовать.")
            with st.form(f"boundary-review-{identifier}-{row['turn_count']}"):
                values = {field: st.text_area(LABELS[field], "\n".join(row[field]),
                                              key=f"boundary-{identifier}-{row['turn_count']}-{field}")
                          for field in EDITABLE}
                approve = st.checkbox("Граница подтверждена методистом", value=row["verified_by_methodologist"])
                saved = st.form_submit_button("Сохранить решение по границе")
            if saved:
                try:
                    await save_boundary_review(factory, actor, course_id, identifier,
                                               {field: value.splitlines() for field, value in values.items()}, approve,
                                               threshold=settings.boundary_violation_threshold)
                except (PermissionError, ValueError) as error:
                    st.warning(str(error))
                else:
                    st.rerun()
