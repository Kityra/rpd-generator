"""Демоданные модуля «Модель деятельности профессии». ДЕМО: синтетические данные, не реальный опрос.

Создаёт профессию «Backend-разработчик» (фрагмент ПС 06.001 из фикстуры, синтетические вакансии двух периодов,
45 синтетических экспертов с фиксированным seed, из них 6 недобросовестных, ловушку «грядки», два раунда
Дельфи), три смежные профессии для карты знаний и смежных ролей и демо-программу из 12 дисциплин.

    .venv\\Scripts\\python.exe -m scripts.seed_activity_demo
    .venv\\Scripts\\python.exe -m scripts.seed_activity_demo --reset --head-login head@demo.local

Все извлечения идут через те же правила и ту же проверку цитат, что и реальные источники.
"""

import argparse
import hashlib
import random
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from auth.settings import Settings
from storage.activity_models import (ActivityClaim, ActivityExpert, ActivityItem, ActivityMergeLog, ActivitySource,
                                     ClaimLink, DisciplineLink, ExpertVote, ItemLink, ItemMetrics, ModelVersionRow,
                                     Profession, Program, ProgramDiscipline, Recommendation)
from storage.database import ensure_schema, make_engine, run_async, write_transaction

from activity import store
from activity.extract import ClaimDraft


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "activity_demo"
SEED = 20261121
DEMO = "ДЕМО: синтетические данные"


def user_id(role: str, login: str) -> str:
    """Тот же ID, что выдаёт справочник учётных записей (auth/yandex_users.py)."""
    return f"{role}:{hashlib.sha256(login.strip().casefold().encode()).hexdigest()}"


# --- Backend-разработчик: обязанности в вакансиях (инфинитив, как в реальных вакансиях) и их доли. ---
# (строка обязанности, доля работодателей сейчас, год назад)
BACKEND_DUTIES = [
    ("Разрабатывать REST API для веб- и мобильных клиентов", 0.85, 0.85),
    ("Проектировать схемы баз данных под нагрузку", 0.55, 0.5),
    ("Оптимизировать SQL-запросы и индексы", 0.5, 0.45),
    ("Писать модульные и интеграционные тесты", 0.6, 0.5),
    ("Проводить код-ревью", 0.45, 0.4),
    ("Настраивать CI/CD для сервисов", 0.4, 0.3),
    ("Упаковывать сервисы в Docker-контейнеры", 0.6, 0.45),
    ("Настраивать мониторинг и логирование сервисов", 0.35, 0.3),
    ("Разбирать инциденты в продакшене", 0.25, 0.2),
    ("Интегрировать сервисы через очереди сообщений Kafka", 0.5, 0.1),
    ("Проектировать граф зависимостей микросервисов", 0.15, 0.1),
    ("Встраивать LLM в продуктовые сервисы", 0.25, 0.0),
    ("Поддерживать SOAP-сервисы", 0.03, 0.35),
    ("Писать хранимые процедуры в базе данных", 0.0, 0.25),
    ("Документировать API в формате OpenAPI", 0.3, 0.25),
    ("Обеспечивать безопасность API", 0.4, 0.35),
    ("Кэшировать данные в Redis", 0.45, 0.04),
    ("Писать шаблонный CRUD-код", 0.2, 0.25),
    ("Настраивать Kubernetes-кластеры", 0.35, 0.25),
    ("Участвовать в планировании задач команды", 0.2, 0.15),
]
REQUIREMENTS = ["Знание Python или Go", "Опыт работы с PostgreSQL", "Понимание принципов REST", "Опыт работы с Git",
                "Знание основ Linux", "Опыт работы с Docker", "Понимание алгоритмов и структур данных"]
TITLES = ["Backend-разработчик (Python)", "Go-разработчик", "Backend-разработчик", "Python-разработчик (backend)",
          "Разработчик серверной части", "Java-разработчик (backend)"]

# Ответы экспертов по задачам: (да из 39 надёжных, «не знаю», медиана важности, прогноз: more, same, less, automated).
VOTES = {
    "Разрабатывает REST API для веб- и мобильных клиентов": (38, 0, 5, (0.3, 0.6, 0.1, 0.0)),
    "Проектирует схемы баз данных под нагрузку": (36, 0, 5, (0.3, 0.6, 0.1, 0.0)),
    "Оптимизирует SQL-запросы и индексы": (35, 1, 4, (0.2, 0.6, 0.1, 0.1)),
    "Пишет модульные и интеграционные тесты": (37, 0, 4, (0.3, 0.5, 0.0, 0.2)),
    "Проводит код-ревью": (36, 0, 4, (0.4, 0.5, 0.1, 0.0)),
    "Настраивает CI/CD для сервисов": (31, 1, 3, (0.3, 0.5, 0.1, 0.1)),
    "Упаковывает сервисы в Docker-контейнеры": (33, 0, 4, (0.2, 0.7, 0.1, 0.0)),
    "Настраивает мониторинг и логирование сервисов": (32, 1, 4, (0.4, 0.5, 0.1, 0.0)),
    "Разбирает инциденты в продакшене": (30, 1, 4, (0.4, 0.5, 0.1, 0.0)),
    "Интегрирует сервисы через очереди сообщений Kafka": (30, 2, 4, (0.5, 0.4, 0.1, 0.0)),
    "Проектирует граф зависимостей микросервисов": (27, 2, 3, (0.4, 0.5, 0.1, 0.0)),
    "Встраивает LLM в продуктовые сервисы": (29, 3, 4, (0.8, 0.2, 0.0, 0.0)),
    "Поддерживает SOAP-сервисы": (8, 2, 2, (0.0, 0.2, 0.8, 0.0)),
    "Пишет хранимые процедуры в базе данных": (28, 1, 3, (0.0, 0.4, 0.5, 0.1)),
    "Документирует API в формате OpenAPI": (30, 1, 3, (0.1, 0.5, 0.0, 0.4)),
    "Обеспечивает безопасность API": (33, 0, 5, (0.5, 0.5, 0.0, 0.0)),
    "Кэширует данные в Redis": (31, 1, 3, (0.3, 0.6, 0.1, 0.0)),
    "Пишет шаблонный CRUD-код": (33, 0, 2, (0.0, 0.2, 0.2, 0.6)),
    "Составление формализованных описаний решений поставленных задач": (30, 2, 3, (0.1, 0.7, 0.2, 0.0)),
    "Разработка алгоритмов решения поставленных задач": (34, 0, 4, (0.2, 0.7, 0.1, 0.0)),
    "Оценка и согласование сроков выполнения поставленных задач": (10, 2, 2, (0.0, 0.6, 0.4, 0.0)),
    "Создание программного кода в соответствии с техническим заданием": (38, 0, 5, (0.2, 0.6, 0.1, 0.1)),
    "Оптимизация программного кода с использованием специализированных программных средств": (27, 2, 3, (0.2, 0.6, 0.1, 0.1)),
    "Анализ возможностей реализации требований к программному обеспечению": (31, 1, 3, (0.1, 0.8, 0.1, 0.0)),
    "Оценка времени и трудоемкости реализации требований к программному обеспечению": (29, 1, 3, (0.1, 0.7, 0.2, 0.0)),
    "Согласование требований к программному обеспечению с заинтересованными сторонами": (31, 1, 3, (0.1, 0.8, 0.1, 0.0)),
    "Разработка, изменение и согласование архитектуры программного обеспечения с системным аналитиком и архитектором программного обеспечения": (26, 2, 4, (0.3, 0.6, 0.1, 0.0)),
    "Проектирование структур данных": (33, 0, 4, (0.2, 0.7, 0.1, 0.0)),
    "Проектирование баз данных": (35, 0, 4, (0.2, 0.7, 0.1, 0.0)),
    "Проектирование программных интерфейсов": (34, 0, 4, (0.3, 0.6, 0.1, 0.0)),
}
# Спорная по существу: в малых компаниях это делает бэкендер, в крупных — DevOps (p ≈ 0.54).
K8S = "Настраивает Kubernetes-кластеры"
PROPOSAL = "Проверять код, сгенерированный ИИ-ассистентами"
DUPLICATES = {"Создаёт программные интерфейсы REST для веб- и мобильных приложений": "Разрабатывает REST API для веб- и мобильных клиентов",
              "Пишет автотесты: модульные и интеграционные": "Пишет модульные и интеграционные тесты"}
# Разметка методиста (ДЕМО): функции, которых нет в ПС, и отнесение задач рынка к функциям.
FUNCTIONS = {
    "D/03.6": ["Проектирует схемы баз данных под нагрузку", "Проектирует граф зависимостей микросервисов"],
    "A/02.3": ["Разрабатывает REST API для веб- и мобильных клиентов", "Оптимизирует SQL-запросы и индексы",
               "Пишет хранимые процедуры в базе данных", "Пишет шаблонный CRUD-код", "Кэширует данные в Redis",
               "Документирует API в формате OpenAPI", "Обеспечивает безопасность API"],
    "Качество кода и тестирование (нет в ПС)": ["Пишет модульные и интеграционные тесты", "Проводит код-ревью"],
    "Интеграция сервисов (нет в ПС)": ["Интегрирует сервисы через очереди сообщений Kafka", "Поддерживает SOAP-сервисы"],
    "Эксплуатация и наблюдаемость (нет в ПС)": ["Настраивает CI/CD для сервисов", "Упаковывает сервисы в Docker-контейнеры",
                                                "Настраивает мониторинг и логирование сервисов",
                                                "Разбирает инциденты в продакшене", K8S],
    "Работа с ИИ-инструментами (нет в ПС)": ["Встраивает LLM в продуктовые сервисы", "Проверяет код, сгенерированный ИИ-ассистентами"],
}
SUCCESSORS = ["Платформенный инженер"] * 14 + ["AI-инженер (разработчик LLM-сервисов)"] * 9 + \
             ["Бэкенд-разработчик, который работает с ИИ-ассистентами"] * 6
FORECAST_TEXT = """ДЕМО. Обзор тенденций рынка серверной разработки, 2026.
Компании всё чаще встраивают большие языковые модели в продуктовые сервисы и ждут от бэкенд-разработчиков умения работать с ними.
Ручное написание шаблонного кода уходит к ИИ-ассистентам, растёт роль проверки сгенерированного кода.
Поддержка SOAP-интеграций сохраняется только в унаследованных системах банков и госсектора.
В отчёте также описаны требования к бухгалтерам при переходе на новые формы отчётности."""


class DemoExtractor:
    """Заранее размеченные утверждения для демо-документов (вместо вызова модели). Проверка цитат — та же."""

    label = "demo:fixture"

    def __init__(self, drafts: dict[str, list[ClaimDraft]]):
        self.drafts = drafts

    async def extract(self, profession, source_type, title, published_at, content):
        return self.drafts.get(title, [])


def forecast_drafts() -> list[ClaimDraft]:
    label = "demo:fixture (ДЕМО: заранее размеченные утверждения)"
    return [
        ClaimDraft(kind="task", text="Встраивает LLM в продуктовые сервисы", extractor=label, verb="Встраивает",
                   object="LLM в продуктовые сервисы", quote="встраивают большие языковые модели в продуктовые сервисы"),
        # Перефразированная цитата: код не найдёт её в тексте и отбросит.
        ClaimDraft(kind="task", text="Пишет шаблонный CRUD-код", extractor=label, verb="Пишет", object="шаблонный код",
                   quote="шаблонный код больше никто не пишет вручную"),
        # Задача другой роли: экстрактор отметил foreign_role.
        ClaimDraft(kind="task", text="Готовит бухгалтерскую отчётность по новым формам", extractor=label, verb="Готовит",
                   object="бухгалтерскую отчётность", foreign_role=True,
                   quote="требования к бухгалтерам при переходе на новые формы отчётности"),
    ]


ADJACENT = {
    "Платформенный инженер": {
        "duties": [("Настраивать Kubernetes-кластеры", 0.8), ("Строить внутреннюю платформу разработчика", 0.6),
                   ("Настраивать CI/CD для сервисов", 0.8), ("Настраивать мониторинг и логирование сервисов", 0.7),
                   ("Описывать инфраструктуру кодом в Terraform", 0.6), ("Упаковывать сервисы в Docker-контейнеры", 0.6),
                   ("Разбирать инциденты в продакшене", 0.5), ("Проектировать граф зависимостей микросервисов", 0.3)],
        "employers": (14, 10), "titles": ["Платформенный инженер", "DevOps-инженер", "SRE-инженер"],
        "requirements": ["Опыт работы с Kubernetes", "Знание основ Linux", "Опыт работы с Terraform"]},
    "Аналитик данных": {
        "duties": [("Анализировать графы связей клиентов", 0.3), ("Писать SQL-запросы к хранилищу данных", 0.9),
                   ("Строить дашборды в Power BI", 0.6), ("Проводить A/B-тесты продуктовых изменений", 0.5),
                   ("Строить прогнозные модели спроса", 0.4)],
        "employers": (12, 11), "titles": ["Аналитик данных", "Продуктовый аналитик"],
        "requirements": ["Знание SQL", "Понимание статистики и теории вероятностей"]},
    "Специалист по логистике": {
        "duties": [("Строить маршруты доставки с учётом кратчайшего пути", 0.6), ("Планировать загрузку складов", 0.7),
                   ("Вести учёт запасов в 1С", 0.6), ("Согласовывать графики поставок с перевозчиками", 0.5)],
        "employers": (11, 10), "titles": ["Логист", "Специалист по логистике"],
        "requirements": ["Опыт работы с 1С", "Знание основ складской логистики"]},
}

# Демо-программа: 12 дисциплин. (название, ЗЕТ, семестр, часть, темы, результаты/задания)
PROGRAM = [
    ("История России", 4, 1, "mandatory_fgos", ["История России с древнейших времён до наших дней"], []),
    ("Философия", 3, 3, "mandatory_fgos", ["Основные направления философской мысли"], []),
    ("Физическая культура и спорт", 2, 1, "mandatory_fgos", ["Общая физическая подготовка"], []),
    ("Дискретная математика и теория графов", 5, 2, "mandatory",
     ["Множества, отношения и комбинаторика", "Графы и деревья, кратчайший путь, топологическая сортировка",
      "Граф зависимостей микросервисов как ориентированный граф"],
     ["Строит граф зависимостей и находит циклы"]),
    ("Алгоритмы и структуры данных", 5, 2, "mandatory",
     ["Сложность алгоритмов", "Разработка алгоритмов решения поставленных задач", "Структуры данных: хеш-таблицы, деревья, индексы"],
     ["Разрабатывает алгоритмы решения задач и оценивает их сложность", "Проектирование структур данных под задачу"]),
    ("Программирование на Python", 6, 1, "mandatory",
     ["Создание программного кода в соответствии с техническим заданием", "Шаблонный CRUD-код на Python",
      "Модульные тесты на pytest"],
     ["Пишет программный код по техническому заданию", "Пишет шаблонный CRUD-код для веб-приложения"]),
    ("Базы данных", 5, 3, "mandatory",
     ["Проектирование схем баз данных под нагрузку", "Оптимизация SQL-запросов и индексов",
      "Хранимые процедуры в базе данных"],
     ["Проектирует схему базы данных под нагрузку", "Оптимизирует SQL-запросы и индексы",
      "Пишет хранимые процедуры в базе данных"]),
    ("Веб-разработка на PHP и SOAP", 4, 4, "elective",
     ["Поддержка SOAP-сервисов", "Вёрстка страниц на jQuery", "Шаблонизатор Smarty"],
     ["Разрабатывает сайт на PHP с вёрсткой на jQuery"]),
    ("Технологии Java EE и CORBA", 4, 5, "elective",
     ["Объектные брокеры запросов CORBA", "Компоненты EJB 2.x", "Апплеты Java"], []),
    ("Тестирование программного обеспечения", 3, 5, "mandatory",
     ["Модульные и интеграционные тесты", "Код-ревью и статический анализ"],
     ["Пишет модульные и интеграционные тесты", "Проводит код-ревью"]),
    ("Контроль качества программного кода", 3, 6, "elective",
     ["Модульные и интеграционные тесты", "Проведение код-ревью"],
     ["Пишет модульные и интеграционные тесты", "Проводит код-ревью"]),
    ("Проектирование программных систем", 5, 6, "mandatory",
     ["Проектирование программных интерфейсов REST API", "Разработка REST API для веб- и мобильных клиентов",
      "Документирование API в формате OpenAPI", "Безопасность API: аутентификация и авторизация",
      "Граф зависимостей микросервисов", "Архитектура программного обеспечения и её согласование"],
     ["Разрабатывает REST API для веб- и мобильных клиентов", "Документирует API в формате OpenAPI",
      "Обеспечивает безопасность API", "Проектирует граф зависимостей микросервисов"]),
]


def actor_for(head: str) -> SimpleNamespace:
    return SimpleNamespace(id=head, role="program_head")


async def reset(factory) -> None:
    """Удалить только прежние демоданные (флаг is_demo)."""
    async with write_transaction(factory) as db:
        professions = (await db.scalars(select(Profession.id).where(Profession.is_demo.is_(True)))).all()
        programs = (await db.scalars(select(Program.id).where(Program.is_demo.is_(True)))).all()
        disciplines = (await db.scalars(select(ProgramDiscipline.id).where(ProgramDiscipline.program_id.in_(programs)))).all()
        await db.execute(delete(DisciplineLink).where(DisciplineLink.discipline_id.in_(disciplines)))
        await db.execute(delete(Recommendation).where(Recommendation.program_id.in_(programs)))
        await db.execute(delete(ProgramDiscipline).where(ProgramDiscipline.program_id.in_(programs)))
        await db.execute(delete(Program).where(Program.id.in_(programs)))
        items = (await db.scalars(select(ActivityItem.id).where(ActivityItem.profession_id.in_(professions)))).all()
        sources = (await db.scalars(select(ActivitySource.id).where(ActivitySource.profession_id.in_(professions)))).all()
        claims = (await db.scalars(select(ActivityClaim.id).where(ActivityClaim.source_id.in_(sources)))).all()
        experts = (await db.scalars(select(ActivityExpert.id).where(ActivityExpert.profession_id.in_(professions)))).all()
        versions = (await db.scalars(select(ModelVersionRow.id).where(ModelVersionRow.profession_id.in_(professions)))).all()
        await db.execute(delete(ItemMetrics).where(ItemMetrics.model_version.in_(versions)))
        await db.execute(delete(ModelVersionRow).where(ModelVersionRow.id.in_(versions)))
        await db.execute(delete(ExpertVote).where(ExpertVote.expert_id.in_(experts)))
        await db.execute(delete(ActivityExpert).where(ActivityExpert.id.in_(experts)))
        await db.execute(delete(ClaimLink).where(ClaimLink.claim_id.in_(claims)))
        await db.execute(delete(ItemLink).where(ItemLink.task_id.in_(items)))
        await db.execute(delete(ActivityClaim).where(ActivityClaim.id.in_(claims)))
        await db.execute(delete(ActivitySource).where(ActivitySource.id.in_(sources)))
        await db.execute(delete(ActivityItem).where(ActivityItem.id.in_(items)))
        await db.execute(delete(ActivityMergeLog).where(ActivityMergeLog.profession_id.in_(professions)))
        await db.execute(delete(Profession).where(Profession.id.in_(professions)))


def sample(rng: random.Random, population: list, share: float) -> set:
    return set(rng.sample(population, round(share * len(population))))


def vacancy_text(title: str, duties: list[str], requirements: list[str]) -> str:
    duty_block = ["Обязанности:", *(f"- {line}" for line in duties)] if duties else []
    return "\n".join([title, *duty_block, "Требования:",
                      *(f"- {line}" for line in requirements), "Условия:", "- Удалённая работа или гибрид",
                      "- Дружный коллектив и печеньки в офисе"])


async def add_vacancies(factory, actor, profession_id: str, name: str, duties: list[tuple], employers: tuple[int, int],
                        titles: list[str], requirements: list[str], today: date, rng: random.Random) -> None:
    """Вакансии текущего года и прошлого года; одна строка обязанности на работодателя-участника."""
    for period, count in (("now", employers[0]), ("prev", employers[1])):
        keys = [f"Демо-работодатель {name[:3].upper()}-{period}-{index:02d}" if period == "prev" and index > count // 2
                else f"Демо-работодатель {name[:3].upper()}-{index:02d}" for index in range(1, count + 1)]
        chosen = {line: sample(rng, keys, now if period == "now" else prev) for line, now, prev in duties}
        for employer in keys:
            lines = [line for line, members in chosen.items() if employer in members]
            for _ in range(1 + (rng.random() < 0.4)):
                offset = rng.randint(5, 355) + (365 if period == "prev" else 0)
                title = rng.choice(titles)
                text = vacancy_text(title, lines, rng.sample(requirements, min(3, len(requirements))))
                source_id, created = await store.add_source(
                    factory, actor, profession_id, "vacancy", f"{title} — {employer}", text, origin_key=employer,
                    url="", published_at=today - timedelta(days=offset), collected_at=today,
                    meta={"demo": True, "platform": "ДЕМО-выгрузка"})
                if created:
                    await store.extract_source(factory, source_id, name)


async def seed_backend(factory, actor, today: date) -> str:
    rng = random.Random(SEED)
    profession_id = await store.create_profession(factory, actor, "Backend-разработчик", "Россия", 3, ["06.001"],
                                                  is_demo=True)
    async with write_transaction(factory) as db:
        row = await db.get(Profession, profession_id)
        row.meta = {"delphi_round": 1, "market_totals": {"now": 1200, "prev": 1000,
                                                          "source": "ДЕМО: всего ИТ-работодателей в выгрузке"}}
    standard = (FIXTURES / "standard_06_001.txt").read_text(encoding="utf-8")
    source_id, _ = await store.add_source(factory, actor, profession_id, "standard", "ПС 06.001 «Программист» (ДЕМО-фрагмент)",
                                          standard, origin_key="Минтруд России: ПС 06.001", standard_code="06.001",
                                          standard_year=2022, published_at=date(2022, 7, 20), collected_at=today)
    await store.extract_source(factory, source_id, "Backend-разработчик")
    await add_vacancies(factory, actor, profession_id, "Backend-разработчик",
                        [(line, now, prev) for line, now, prev in BACKEND_DUTIES], (32, 28), TITLES, REQUIREMENTS, today, rng)
    forecast_id, _ = await store.add_source(factory, actor, profession_id, "forecast", "ДЕМО: обзор тенденций серверной разработки",
                                            FORECAST_TEXT, origin_key="ДЕМО-аналитическое агентство",
                                            published_at=today - timedelta(days=40), collected_at=today)
    await store.extract_source(factory, forecast_id, "Backend-разработчик",
                               DemoExtractor({"ДЕМО: обзор тенденций серверной разработки": forecast_drafts()}))
    await store.link_claims(factory, profession_id)
    await mark_up(factory, actor, profession_id)
    await store.ensure_controls(factory, profession_id)
    async with factory() as db:
        tasks = {item.name: item.id for item in (await db.scalars(select(ActivityItem).where(
            ActivityItem.profession_id == profession_id, ActivityItem.level == "task"))).all()}
    for text, original in DUPLICATES.items():
        await store.add_control(factory, actor, profession_id, text, duplicate_of=tasks[original])
    await store.build_version(factory, profession_id, today - timedelta(days=30))
    await seed_experts(factory, actor, profession_id, today)
    return profession_id


async def mark_up(factory, actor, profession_id: str) -> None:
    """Разметка методиста (ДЕМО): новые функции и отнесение задач рынка к функциям; всё пишется в журнал."""
    async with factory() as db:
        items = (await db.scalars(select(ActivityItem).where(ActivityItem.profession_id == profession_id))).all()
    functions = {item.standard_code: item.id for item in items if item.level == "function" and item.standard_code}
    tasks = {item.name: item.id for item in items if item.level == "task"}
    for function, names in FUNCTIONS.items():
        function_id = functions.get(function) or await store.create_function(factory, actor, profession_id, function)
        for name in names:
            if name in tasks:
                await store.set_function(factory, actor, tasks[name], function_id)


def stratified(experts: list[str], strata: dict[str, tuple], yes: int, unknown: int,
               rng: random.Random) -> tuple[set[str], set[str]]:
    """«Да» распределяется по сегментам пропорционально их размеру: в синтетике нет случайных расслоений."""
    groups: dict[tuple, list[str]] = {}
    for expert in experts:
        groups.setdefault(strata[expert], []).append(expert)
    for members in groups.values():
        rng.shuffle(members)

    def split(total: int) -> dict[tuple, int]:
        exact = {key: total * len(members) / len(experts) for key, members in groups.items()}
        counts = {key: int(value) for key, value in exact.items()}
        for key in sorted(exact, key=lambda key: (counts[key] - exact[key], str(key)))[:total - sum(counts.values())]:
            counts[key] += 1
        return counts

    yes_counts, unknown_counts = split(yes), split(unknown)
    chosen, unsure = set(), set()
    for key, members in sorted(groups.items(), key=lambda item: str(item[0])):
        chosen |= set(members[:yes_counts[key]])
        unsure |= set(members[yes_counts[key]:yes_counts[key] + unknown_counts[key]])
    return chosen, unsure


def demo_votes(experts: list[str], cheaters: dict[str, str], items: dict[str, str], controls: dict[str, dict],
               small: set[str], strata: dict[str, tuple], rng: random.Random) -> list[dict]:
    """Ответы раунда 1. Надёжные эксперты голосуют по таблице VOTES; недобросовестные — по своему шаблону."""
    honest = [expert for expert in experts if expert not in cheaters]
    names_of = {control_id: control["name"] for control_id, control in controls.items()}
    rows = []

    def answer(expert, item_id, verdict, importance=None, forecast=None, seconds=None):
        rows.append({"expert_id": expert, "item_id": item_id, "verdict": verdict,
                     "importance": importance if verdict in ("regular", "sometimes") else None,
                     "forecast": forecast if verdict != "dont_know" else None,
                     "elapsed_ms": int((seconds if seconds is not None else rng.uniform(5, 16)) * 1000),
                     "wording_ok": True})

    plans = {}
    for name, (yes, dont_know, importance, forecast) in VOTES.items():
        plans[name] = (*stratified(honest, strata, yes, dont_know, rng), importance, forecast)
    small_list, large_list = sorted(small & set(honest)), sorted(set(honest) - small)
    k8s_yes = set(rng.sample(small_list, 17)) | set(rng.sample(large_list, 4))  # 21 из 39 ≈ 0.54
    plans[K8S] = (k8s_yes, set(), 3, (0.2, 0.6, 0.1, 0.1))
    for expert in honest:
        verdicts = {}
        for name, (yes, unknown, importance, forecast) in plans.items():
            verdict = ("dont_know" if expert in unknown else
                       rng.choice(["regular", "regular", "sometimes"]) if expert in yes else
                       rng.choice(["other_role", "other_role", "outdated"]))
            importance_value = max(1, min(5, importance + rng.choice([-1, 0, 0, 0, 1])))
            forecast_value = rng.choices(["more", "same", "less", "automated"], weights=forecast)[0]
            verdicts[name] = verdict
            answer(expert, items[name], verdict, importance_value, forecast_value)
        for control_id, control in controls.items():
            if control["kind"] == "foreign":
                answer(expert, control_id, "other_role")
            else:
                original = next(name for name, item_id in items.items() if item_id == control["duplicate_of"])
                answer(expert, control_id, verdicts[original], 4, "same")
    for expert, pattern in cheaters.items():
        for name, item_id in [*items.items(), *((f"control:{key}", key) for key in controls)]:
            if pattern == "fast_yes":
                answer(expert, item_id, "regular", 5, "more", seconds=rng.uniform(0.6, 1.6))
            elif pattern == "yes_to_foreign":
                # «Да» на грядки и ещё одну «чужую» задачу: 2 из 5 ловушек.
                foreign = item_id in controls and controls[item_id]["kind"] == "foreign"
                trap = foreign and any(word in names_of[item_id] for word in ("грядок", "паркетной"))
                verdict = "regular" if trap else "other_role" if foreign else rng.choice(["regular", "sometimes", "other_role"])
                answer(expert, item_id, verdict, rng.randint(2, 5), rng.choice(["more", "same"]))
            elif pattern == "inconsistent":
                if item_id in controls:
                    continue
                answer(expert, item_id, "regular" if rng.random() < 0.8 else "sometimes", 4, "same")
            else:  # same_pattern: одна и та же кнопка на всех карточках при обычной скорости
                answer(expert, item_id, "sometimes", 3, "same")
        if pattern == "inconsistent":
            for control_id, control in controls.items():
                if control["kind"] == "duplicate":
                    original = next(row for row in rows if row["expert_id"] == expert and row["item_id"] == control["duplicate_of"])
                    answer(expert, control_id, "other_role" if original["verdict"] in ("regular", "sometimes") else "regular", 3, "same")
                else:
                    answer(expert, control_id, "other_role")
    return rows


async def seed_experts(factory, actor, profession_id: str, today: date) -> None:
    rng = random.Random(SEED + 1)
    expert_ids = []
    roles = ["practitioner"] * 6 + ["lead"] * 3 + ["hr", "teacher"]
    async with write_transaction(factory) as db:
        for index in range(1, 46):
            row = ActivityExpert(user_id=f"demo:expert-{index:02d}", profession_id=profession_id,
                                 years=rng.randint(2, 15), role_kind=roles[index % len(roles)],
                                 company_size="small" if index % 2 else "large", region=rng.choice(["Москва", "Санкт-Петербург", "Казань", "Новосибирск", "Ростов-на-Дону"]),
                                 company_kind=rng.choice(["продуктовая", "аутсорс", "банк", "госсектор"]),
                                 invited_by=actor.id, submitted_rounds=[1], created_at=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=index))
            db.add(row)
            await db.flush()
            expert_ids.append(row.id)
    cheaters = {expert_ids[4]: "fast_yes", expert_ids[11]: "yes_to_foreign", expert_ids[17]: "yes_to_foreign",
                expert_ids[23]: "fast_yes", expert_ids[30]: "inconsistent", expert_ids[38]: "same_pattern"}
    honest = [expert for expert in expert_ids if expert not in cheaters]
    async with factory() as db:
        items = (await db.scalars(select(ActivityItem).where(ActivityItem.profession_id == profession_id,
                                                             ActivityItem.level == "task"))).all()
    names = {item.name: item.id for item in items if not item.is_control}
    missing = [name for name in [*VOTES, K8S] if name not in names]
    if missing:
        raise RuntimeError(f"В демо-модели нет задач: {missing}")
    controls = {item.id: {"kind": item.control_kind, "duplicate_of": item.duplicate_of, "name": item.name}
                for item in items if item.is_control}
    honest_sorted = sorted(honest)
    # Компании надёжных экспертов: половина малые, половина крупные (для сегментов).
    small = set(honest_sorted[::2])
    async with write_transaction(factory) as db:
        for expert in honest_sorted:
            row = await db.get(ActivityExpert, expert)
            row.company_size = "small" if expert in small else "large"
    async with factory() as db:
        roles_of = {row.id: row.role_kind for row in (await db.scalars(select(ActivityExpert).where(
            ActivityExpert.profession_id == profession_id))).all()}
    strata = {expert: (expert in small, roles_of[expert] == "practitioner") for expert in honest_sorted}
    rows = demo_votes(sorted(expert_ids), cheaters, {name: names[name] for name in [*VOTES, K8S]}, controls, small,
                      strata, rng)
    async with write_transaction(factory) as db:
        for row in rows:
            db.add(ExpertVote(round=1, item_version=1, **row))
        demand = ["more"] * 10 + ["same"] * 17 + ["less"] * 12
        for index, expert in enumerate(honest_sorted):
            item = await db.get(ActivityExpert, expert)
            item.role_demand = demand[index % len(demand)]
            item.role_forecast_text = SUCCESSORS[index] if index < len(SUCCESSORS) else None
            item.cards = {"1": sorted({row["item_id"] for row in rows if row["expert_id"] == expert})}
    # «Чего не хватает в списке?» — свободный ответ проходит через экстрактор как источник expert.
    for expert in honest_sorted[:5]:
        await store.finish_round(factory, actor, expert, 1, {"missing_text": PROPOSAL})
    await store.link_claims(factory, profession_id)
    async with factory() as db:
        proposal = await db.scalar(select(ActivityItem).where(ActivityItem.profession_id == profession_id,
                                                              ActivityItem.origin == "expert_proposal"))
        function = await db.scalar(select(ActivityItem).where(ActivityItem.profession_id == profession_id,
                                                              ActivityItem.name == "Работа с ИИ-инструментами (нет в ПС)"))
    await store.set_function(factory, actor, proposal.id, function.id)
    await store.build_version(factory, profession_id, today - timedelta(days=7))
    # Раунд 2 Дельфи: спорные задачи и предложение экспертов. 13 экспертов, 9 «за» (нужно ещё ≈ 10).
    await store.start_round_two(factory, actor, profession_id)
    voters = honest_sorted[:13]
    async with write_transaction(factory) as db:
        for index, expert in enumerate(voters):
            verdict = "regular" if index < 6 else "sometimes" if index < 9 else "other_role"
            db.add(ExpertVote(expert_id=expert, item_id=proposal.id, item_version=1, round=2, verdict=verdict,
                              importance=4 if index < 9 else None, forecast="more", wording_ok=index != 3,
                              wording_suggestion="Проверяет и исправляет код, предложенный ИИ-ассистентом" if index == 3 else None,
                              comment="В нашей команде это уже ежедневная работа" if index == 0 else
                              "У нас ассистенты запрещены политикой безопасности" if index == 10 else None,
                              elapsed_ms=int(rng.uniform(6, 14) * 1000)))
        # Повторный голос по Kubernetes во втором раунде: мнения почти не меняются (консенсуса нет).
        k8s = names[K8S]
        first = {row["expert_id"]: row for row in rows if row["item_id"] == k8s}
        for expert in honest_sorted[13:26]:
            vote = first[expert]
            db.add(ExpertVote(expert_id=expert, item_id=k8s, item_version=1, round=2, verdict=vote["verdict"],
                              importance=vote["importance"], forecast=vote["forecast"], wording_ok=True,
                              elapsed_ms=int(rng.uniform(6, 14) * 1000),
                              comment="В крупных компаниях кластеры настраивает платформенная команда" if expert == honest_sorted[13] else None))
        for expert in [*voters, *honest_sorted[13:26]]:
            row = await db.get(ActivityExpert, expert)
            row.submitted_rounds = [1, 2]
            row.cards = {**(row.cards or {}), "2": [proposal.id] if expert in voters else [k8s]}


async def seed_adjacent(factory, actor, today: date) -> dict[str, str]:
    result = {}
    for offset, (name, spec) in enumerate(ADJACENT.items()):
        rng = random.Random(SEED + 10 + offset)
        profession_id = await store.create_profession(factory, actor, name, "Россия", 3, [], is_demo=True)
        result[name] = profession_id
        await add_vacancies(factory, actor, profession_id, name, [(line, share, share * 0.8) for line, share in spec["duties"]],
                            spec["employers"], spec["titles"], spec["requirements"], today, rng)
        document = "\n".join([f"ДЕМО. Описание должности «{name}» от работодателя.", "Обязанности:",
                              *(f"- {line}" for line, _ in spec["duties"])])
        source_id, _ = await store.add_source(factory, actor, profession_id, "employer_doc",
                                              f"ДЕМО: должностная инструкция «{name}»", document,
                                              origin_key=f"ДЕМО-работодатель-документ {name}",
                                              published_at=today - timedelta(days=90), collected_at=today)
        await store.extract_source(factory, source_id, name)
        await store.link_claims(factory, profession_id)
        await store.build_version(factory, profession_id, today)
    return result


async def seed_program(factory, actor, profession_id: str, teacher: str, today: date) -> str:
    program_id = await store.create_program(
        factory, actor, "Программная инженерия: бэкенд-разработка (ДЕМО)", level="bachelor", total_zet=240,
        institution="ДЕМО-университет", academic_year="2026/2027",
        targets=[{"profession_id": profession_id, "weight": 1.0}], is_demo=True)
    rows = [{"name": name, "zet": zet, "semester": semester, "part": part}
            for name, zet, semester, part, _, _ in PROGRAM]
    await store.save_disciplines(factory, actor, program_id, rows)
    async with factory() as db:
        disciplines = {d.name: d.id for d in (await db.scalars(select(ProgramDiscipline).where(
            ProgramDiscipline.program_id == program_id))).all()}
    for name, _, _, _, topics, outcomes in PROGRAM:
        await store.save_content(factory, actor, disciplines[name], topics, outcomes, [])
    await store.assign_teacher(factory, actor, disciplines["Дискретная математика и теория графов"], teacher,
                               "Методист по теории графов (демо)")
    await store.assign_teacher(factory, actor, disciplines["Базы данных"], teacher, "Методист по теории графов (демо)")
    await store.recompute(factory, actor, program_id)
    return program_id


async def run(head_login: str, teacher_login: str, expert_login: str, do_reset: bool) -> None:
    settings = Settings()
    engine = make_engine(settings.app_database_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with factory() as db:
            exists = await db.scalar(select(Profession.id).where(Profession.is_demo.is_(True)).limit(1))
        if exists and not do_reset:
            raise SystemExit("Демоданные уже есть. Запустите с --reset, чтобы пересоздать их (удаляются только демоданные).")
        if exists:
            await reset(factory)
        today = datetime.now(timezone.utc).date()
        actor = actor_for(user_id("program_head", head_login))
        backend = await seed_backend(factory, actor, today)
        await seed_adjacent(factory, actor, today)
        await store.build_version(factory, backend, today)
        await store.invite_expert(factory, actor, backend, user_id("expert", expert_login))
        program_id = await seed_program(factory, actor, backend, user_id("teacher", teacher_login), today)
        async with factory() as db:
            version = await store.latest_version(db, backend)
        statuses = version.counts["statuses"]
        print(f"{DEMO}. Модель «Backend-разработчик» v{version.number}: {statuses}.")
        print(f"Программа {program_id}: руководитель {head_login}, методист {teacher_login}, эксперт {expert_login}.")
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reset", action="store_true", help="пересоздать демоданные (удаляются только записи is_demo)")
    parser.add_argument("--head-login", default="head@demo.local", help="логин руководителя программы")
    parser.add_argument("--teacher-login", default="teacher@demo.local", help="логин методиста")
    parser.add_argument("--expert-login", default="expert@demo.local", help="логин эксперта для прохождения анкеты")
    arguments = parser.parse_args()
    ensure_schema(Settings().app_database_url)
    run_async(run(arguments.head_login, arguments.teacher_login, arguments.expert_login, arguments.reset))


if __name__ == "__main__":
    main()
