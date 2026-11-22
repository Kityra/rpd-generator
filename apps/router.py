import streamlit as st

from apps import boundary_guest, expert_app, teacher_app
from apps.presentation import portal_brand


CABINETS = (
    ("teacher", "Кабинет методиста",
     "Курсы и дисциплины, модели деятельности профессий, управление программами и карта знаний."),
    ("expert", "Кабинет эксперта", "Анкета по задачам профессии и интервью по курсам."),
)


def main():
    st.set_page_config(page_title="Проектирование программ", layout="wide", initial_sidebar_state=200)
    pages = {"teacher": st.Page(teacher_app.main, title="Кабинет методиста", url_path="teacher"),
             "expert": st.Page(expert_app.main, title="Кабинет эксперта", url_path="expert")}

    def legacy_program_link():
        st.session_state["screen"] = "management"
        st.switch_page(pages["teacher"], query_params=st.query_params.to_dict())

    def landing():
        portal_brand()
        st.title("Выберите кабинет")
        st.caption("Модель деятельности профессии, проверенная экспертами, — основа программы и РПД.")
        for key, title, description in CABINETS:
            with st.container(border=True, key=f"portal-card-cabinet-{key}"):
                st.page_link(pages[key], label=title, icon=":material/arrow_forward:")
                st.caption(description)

    home = st.Page(landing, title="Проектирование программ", default=True)
    boundary = st.Page(boundary_guest.main, title="Интервью по границам профессий", url_path="boundary")
    legacy = st.Page(legacy_program_link, title="Управление программами", url_path="head")
    st.navigation([home, *pages.values(), boundary, legacy], position="hidden").run()
