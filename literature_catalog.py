"""Bibliographic starting points for common ML topics from authors and publishers."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class AcademicReference:
    title: str
    authors: tuple[str, ...]
    year: int
    publisher: str
    url: str
    kind: Literal["textbook", "article"] = "textbook"
    doi: str | None = None
    pdf: str | None = None


def _rl(chapter: int, title: str) -> AcademicReference:
    return AcademicReference(
        f"{title} (глава {chapter})", ("Richard S. Sutton", "Andrew G. Barto"), 2018,
        "Reinforcement Learning: An Introduction, 2-е изд. · MIT Press",
        "https://mitpress.mit.edu/9780262039246/reinforcement-learning/",
        pdf="https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf",
    )


STAT_LEARNING = AcademicReference(
    "An Introduction to Statistical Learning with Applications in Python",
    ("Gareth James", "Daniela Witten", "Trevor Hastie", "Robert Tibshirani", "Jonathan Taylor"), 2023,
    "Springer", "https://www.statlearning.com/", doi="10.1007/978-3-031-38747-0",
)
DEEP_FORWARD = AcademicReference(
    "Deep Feedforward Networks (глава 6)", ("Ian Goodfellow", "Yoshua Bengio", "Aaron Courville"), 2016,
    "Deep Learning · MIT Press", "https://www.deeplearningbook.org/contents/mlp.html",
)
CONV_NETS = AcademicReference(
    "Convolutional Networks (глава 9)", ("Ian Goodfellow", "Yoshua Bengio", "Aaron Courville"), 2016,
    "Deep Learning · MIT Press", "https://www.deeplearningbook.org/contents/convnets.html",
)
TD_PAPER = AcademicReference(
    "Learning to predict by the methods of temporal differences", ("Richard S. Sutton",), 1988,
    "Machine Learning", "https://link.springer.com/article/10.1007/BF00115009", kind="article", doi="10.1007/BF00115009",
)
NUMERICAL_ACCURACY = AcademicReference(
    "Problems and conditioning (раздел 1.2)", ("Tobin A. Driscoll", "Richard J. Braun"), 2022,
    "Fundamentals of Numerical Computation · SIAM", "https://fncbook.com/conditioning/",
)
STRONG_AI = AcademicReference(
    "Minds, brains, and programs", ("John R. Searle",), 1980, "Behavioral and Brain Sciences",
    "https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/minds-brains-and-programs/DC644B47A4299C637C89772FACC2706A",
    kind="article", doi="10.1017/S0140525X00005756",
    pdf=("https://openlearninglibrary.mit.edu/assets/courseware/v1/894920e796501e08c6628331d21e651b/"
         "asset-v1%3AMITx%2B24.09x%2B3T2019%2Btype%40asset%2Bblock/2_searle_minds_brains_and_programs.pdf"),
)


def course_concepts(subject_titles) -> set[str]:
    """Понятия всех дисциплин курса — контекст для каталога (одна тема в разных курсах значит разное)."""
    from orchestration.semantic_check import concepts
    return set().union(*(concepts(title) for title in subject_titles)) if subject_titles else set()


def references_for_discipline(discipline: str, *, course: set[str] = frozenset()) -> tuple[AcademicReference, ...]:
    title = " ".join(discipline.casefold().replace("ё", "е").replace("-", " ").split())
    reinforcement_learning = bool({"c:обучение с подкреплением", "подкрепл"} & course)
    if "слабого и сильного ии" in title or "сильного и слабого ии" in title:
        return (STRONG_AI,)
    if "теория погрешности" in title or ("абсолютная" in title and "относительная" in title):
        return (NUMERICAL_ACCURACY,)
    if "глубокое обучение" in title:
        return (DEEP_FORWARD,)
    if "архитектур" in title and "нейронных сет" in title:
        return (CONV_NETS, DEEP_FORWARD)
    if "с учителем" in title:
        basics = (STAT_LEARNING,)
        return (*basics, _rl(3, "Finite Markov Decision Processes")) if "с подкреплением" in title else basics
    if "временных различ" in title or "temporal difference" in title:
        return (TD_PAPER, _rl(6, "Temporal-Difference Learning"))
    if "обучение с подкреплением" in title or "reinforcement learning" in title:
        return (_rl(3, "Finite Markov Decision Processes"),)
    if reinforcement_learning:
        if "динамического программирования" in title:
            return (_rl(4, "Dynamic Programming"),)
        if "монте карло" in title:
            return (_rl(5, "Monte Carlo Methods"),)
        if "аппроксимац" in title and "функци" in title and "ценност" in title:
            return (_rl(9, "On-policy Prediction with Approximation"),)
        if "приближенное управление" in title and "аппроксимац" in title:
            return (_rl(10, "On-policy Control with Approximation"),)
    return ()
