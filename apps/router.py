import streamlit as st

from apps import teacher_app, expert_app
from apps.presentation import portal_brand


def main():
    st.set_page_config(page_title="Проектирование курсов", layout="wide")
    teacher = st.Page(teacher_app.main, title="Кабинет преподавателя", url_path="teacher")
    expert = st.Page(expert_app.main, title="Кабинет эксперта", url_path="expert")

    def landing():
        portal_brand()
        st.title("Выберите кабинет")
        for page in (teacher, expert):
            with st.container(border=True):
                st.page_link(page, label=page.title)

    home = st.Page(landing, title="Проектирование курсов", default=True)
    st.navigation([home, teacher, expert], position="hidden").run()
