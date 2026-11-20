import streamlit as st

from apps import teacher_app, expert_app


def main():
    st.set_page_config(page_title="Проектирование курсов", layout="wide")
    teacher = st.Page(teacher_app.main, title="Кабинет преподавателя", url_path="teacher")
    expert = st.Page(expert_app.main, title="Кабинет эксперта", url_path="expert")

    def landing():
        st.title("Проектирование курсов")
        st.page_link(teacher, label="Кабинет преподавателя")
        st.page_link(expert, label="Кабинет эксперта")

    home = st.Page(landing, title="Проектирование курсов", default=True)
    st.navigation([home, teacher, expert], position="hidden").run()
