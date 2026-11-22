import streamlit as st

from apps import boundary_guest, expert_app, head_app, teacher_app
from apps.presentation import portal_brand


CABINETS = (
    ("head", "Кабинет руководителя программы",
     "Модели деятельности профессий, экспертная валидация, состав программы и рекомендации."),
    ("teacher", "Кабинет методиста",
     "Закреплённые дисциплины, проверенные задачи профессии, РПД и карта знаний."),
    ("expert", "Кабинет эксперта", "Анкета по задачам профессии и интервью по курсам."),
)


def main():
    st.set_page_config(page_title="Проектирование программ", layout="wide", initial_sidebar_state=200)
    pages = {"head": st.Page(head_app.main, title="Кабинет руководителя программы", url_path="head"),
             "teacher": st.Page(teacher_app.main, title="Кабинет методиста", url_path="teacher"),
             "expert": st.Page(expert_app.main, title="Кабинет эксперта", url_path="expert")}

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
    st.navigation([home, *pages.values(), boundary], position="hidden").run()
