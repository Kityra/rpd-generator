"""Run the RPD graph from a terminal using Yandex AI Studio."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from dotenv import load_dotenv
from jsonschema import SchemaError
from openai import APIConnectionError, APIResponseValidationError, APIStatusError, APITimeoutError, OpenAI
from pydantic import BaseModel, ConfigDict, ValidationError

from check_yandex_indexes import content_indexes
from orchestration.rabota_business import RabotaBusiness, SOURCE_GUIDANCE
from orchestration.resilience import ExternalUnavailable, snapshot_notice
from orchestration.yandex_calls import InvalidIndexResponse, lookup_rpd, model_response, search_fragments
from rpd_graph import RpdRequest, build_rpd_graph
from rpd_specialists import PROFILE_SCHEMA, PROFILES, ProfileName, SourceFragment, SpecialistReport, analyse_fragments
from science_literature import Article, find_course_articles, normalise_doi


ROOT = Path(__file__).resolve().parent
YANDEX_BASE_URL = "https://ai.api.cloud.yandex.net/v1"


class YandexModeratedResponse(RuntimeError):
    """The provider filtered the response or the model refused the request."""


class ScientificSearch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str


def collect_recent_articles(discipline: str, plan: Callable[[str, str], str],
                            notify: Callable[[str], None]) -> list[Article]:
    instructions = (
        "Преобразуй название дисциплины в тематический поисковый запрос для свежих ИТ-статей. "
        "Верни в query от трёх до восьми английских технических слов. "
        "Добавь ключевые понятия дисциплины, убери общие учебные слова и названия вузов."
    )
    try:
        response = ScientificSearch.model_validate_json(plan(instructions, discipline))
    except (APIStatusError, APIConnectionError, RuntimeError, ValidationError):
        notify("Science: не удалось сформировать запрос по свежим статьям.")
        return []
    verified = {doi for item in os.environ.get("SCIENCE_VERIFIED_DOIS", "").split(",") if (doi := normalise_doi(item))}
    articles, unavailable = find_course_articles(response.query, verified_dois=verified)
    if unavailable:
        notify("Science: часть поиска недоступна (" + ", ".join(unavailable) + ").")
    return articles


def require_response_text(response: Any) -> str:
    reason = getattr(response.incomplete_details, "reason", None)
    refused = any(
        part.type == "refusal"
        for item in response.output if item.type == "message"
        for part in item.content
    )
    if reason == "content_filter" or refused:
        raise YandexModeratedResponse("Yandex AI Studio ограничила ответ модели.")
    if response.status != "completed" or not response.output_text.strip():
        details = f"{response.status}, {reason}" if reason else response.status
        raise RuntimeError(f"Модель не вернула полный ответ (статус {details}).")
    return response.output_text


def read_answer(prompt: str) -> str:
    while True:
        answer = input(prompt).strip()
        if answer:
            return answer
        print("Введите ответ или завершите работу сочетанием Ctrl+C.")


def yandex_settings() -> dict[str, str]:
    load_dotenv(ROOT / ".env", encoding="utf-8-sig", interpolate=False)
    indexes = content_indexes()
    required = ("YANDEX_API_KEY", "YANDEX_FOLDER_ID", "YANDEX_MODEL_URI")
    settings = {name: os.environ.get(name, "").strip() for name in required}
    settings.update(indexes)
    missing = [name for name in required if not settings[name]]
    if missing:
        raise ValueError("Заполните .env: " + ", ".join(missing))
    base_url = os.environ.get("YANDEX_BASE_URL", YANDEX_BASE_URL).strip().rstrip("/")
    if base_url != YANDEX_BASE_URL:
        raise ValueError(f"Укажите в YANDEX_BASE_URL адрес {YANDEX_BASE_URL}")
    for key in ("YANDEX_VECTOR_STORE_ID_BOOKS", "YANDEX_VECTOR_STORE_ID_HH",
                "YANDEX_VECTOR_STORE_ID_PRIKLAD", "YANDEX_VECTOR_STORE_ID_INTERVIEW",
                *(f"YANDEX_MODEL_URI_{role.upper()}" for role in PROFILES)):
        if key not in indexes:
            settings[key] = os.environ.get(key, "").strip()
    return settings


def open_yandex(settings: dict[str, str], timeout: float = 120):
    return OpenAI(api_key=settings["YANDEX_API_KEY"], project=settings["YANDEX_FOLDER_ID"],
                  base_url=YANDEX_BASE_URL, timeout=timeout, max_retries=0)


def request_rpd_materials(client, settings, request: RpdRequest) -> SpecialistReport:
    search = PROFILES["university"].searches[0]
    query = search.query.format_map(request.model_dump())
    if request.program:
        query += f" Образовательная программа: {request.program}."
    scope = ("local", request.institution, request.program, request.discipline, request.academic_year)
    results, taken_at, lookup = lookup_rpd(client, settings, query, search.result_count, scope=scope, budget=20)
    fragments = [SourceFragment(file=hit.title, text=hit.content, index="RPD",
                                url=hit.url, retrieved_at=hit.retrieved_at.isoformat()) for hit in results]
    notices = [lookup["reason"]] if lookup["reason"] else []
    if taken_at is not None:
        notices.append(snapshot_notice("RPD", taken_at))
    return {"fragments": fragments, "findings": [], "source_notices": notices, "rpd_lookup": lookup}


def create_rpd_graph(client: OpenAI, settings: dict[str, str], *, notify=lambda message: None, checkpointer=None,
                     draft_first: bool = False):
    def request_json(
        model: str, instructions: str, payload: str, schema: dict[str, Any],
        schema_name: str, max_tokens: int, timeout: float,
    ) -> str:
        response = model_response(client, settings,
            model=model,
            instructions=instructions,
            input=payload,
            text={"format": {
                "type": "json_schema",
                "name": schema_name,
                "schema": schema,
                "strict": True,
            }},
            max_output_tokens=max_tokens,
            store=False,
            budget=timeout,
        )
        return require_response_text(response)

    flash_uri = f"gpt://{settings['YANDEX_FOLDER_ID']}/aliceai-llm-flash"
    rabota = RabotaBusiness.from_environment()

    def research(name: ProfileName, request: RpdRequest) -> SpecialistReport:
        profile = PROFILES[name]
        fragments: list[SourceFragment] = []
        notices: list[str] = []
        rpd: SpecialistReport = {"fragments": [], "findings": []}

        def note(message: str) -> None:
            notices.append(message)
            notify(message)

        fields = request.model_dump()
        for search in profile.searches:
            if search.index == "RPD":
                rpd = request_rpd_materials(client, settings, request)
                fragments.extend(rpd["fragments"])
                for message in rpd.get("source_notices", []):
                    note(message)
                continue
            index_id = settings.get(f"YANDEX_VECTOR_STORE_ID_{search.index.upper()}", "")
            if not index_id:
                note(f"{profile.label}: индекс {search.index} пока не указан.")
                continue
            query = search.query.format_map(fields)
            try:
                results, taken_at = search_fragments(client, settings, search.index, index_id,
                                                    query, search.result_count, budget=20)
            except (APIStatusError, APIResponseValidationError, ExternalUnavailable, InvalidIndexResponse):
                note(f"{profile.label}: индекс {search.index} недоступен.")
                continue
            if taken_at is not None:
                note(snapshot_notice(search.index, taken_at))
            for hit in results:
                fragments.append(SourceFragment(file=hit.title, text=hit.content[:2_000], index=search.index,
                                                url=hit.url, retrieved_at=hit.retrieved_at.isoformat()))

        vacancy_fragments = rabota.fragments(request.discipline) if rabota and name == "business" else []
        fragments.extend(vacancy_fragments)
        model = settings.get(f"YANDEX_MODEL_URI_{name.upper()}", "") or flash_uri
        book_count = len(fragments)
        articles: list[Article] = []
        if name == "science":
            def plan(instructions: str, subject: str) -> str:
                return request_json(model, instructions, subject, ScientificSearch.model_json_schema(),
                                    "science_course_query", 384, 20)

            articles = collect_recent_articles(request.discipline, plan, note)
            fragments.extend(article.as_science_source(include_abstract=True) for article in articles)

        def summarize(instructions: str, payload: str) -> str:
            return request_json(model, instructions, payload, PROFILE_SCHEMA, f"rpd_{name}", 896, 35)

        try:
            if vacancy_fragments:
                report = analyse_fragments(profile, request.discipline, fragments, summarize,
                                           scope=f"{profile.scope} {SOURCE_GUIDANCE}")
            else:
                report = analyse_fragments(profile, request.discipline, fragments, summarize)
        except YandexModeratedResponse:
            note(f"{profile.label}: Yandex AI Studio ограничила ответ, материалы узла пропущены.")
            return {"fragments": [], "findings": [], "source_notices": notices}
        except (APIStatusError, APIConnectionError, RuntimeError, ValidationError):
            note(f"{profile.label}: модель не сформировала краткий отчёт.")
            return {**rpd, "fragments": fragments if name == "university" else [], "findings": [],
                    "source_notices": notices}

        notify(f"{profile.label}: найдено фрагментов — {len(fragments)}, "
               f"наблюдений — {len(report['findings'])}.")
        if name != "university" and not report["findings"]:
            return {"fragments": [], "findings": [], "source_notices": notices}
        if articles:
            used = {}
            for finding in report["findings"]:
                if finding["fragment"] > book_count:
                    article = articles[finding["fragment"] - book_count - 1]
                    used.setdefault(article.url, {"title": article.title, "url": article.url,
                                                   "status": article.status, "idea": finding["summary"]})
            if used:
                report["articles"] = list(used.values())
            report["fragments"] = fragments[:book_count]
            report["findings"] = [finding for finding in report["findings"] if finding["fragment"] <= book_count]
        report["source_notices"] = notices
        if "rpd_lookup" in rpd:
            report["rpd_lookup"] = rpd["rpd_lookup"]
        return report

    def generate(instructions: str, payload: str, response_schema: dict[str, Any]) -> str:
        notify("Собираю черновик РПД…")
        return request_json(settings["YANDEX_MODEL_URI"], instructions, payload,
                            response_schema, "rpd_draft", 16_384, 120)

    return build_rpd_graph(ROOT / "final_schema.json", research, generate, checkpointer=checkpointer,
                           draft_first=draft_first)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Подготовить черновик РПД и сохранить его в JSON.")
    parser.add_argument("--discipline", help="Название дисциплины")
    parser.add_argument("--year", help="Целевой учебный год")
    parser.add_argument("--institution", help="Название вуза")
    parser.add_argument("--program", default="", help="Название образовательной программы")
    parser.add_argument("--facts-file", type=Path, help="Файл UTF-8 с известными сведениями для РПД")
    parser.add_argument("--output", type=Path, default=ROOT / "rpd_draft.json", help="Куда сохранить черновик")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Выходной файл уже существует. Укажите другое имя через --output.")

    try:
        settings = yandex_settings()
    except ValueError as error:
        parser.error(str(error))

    try:
        request = RpdRequest(
            discipline=args.discipline or read_answer("Название дисциплины: "),
            academic_year=args.year or read_answer("Учебный год: "),
            institution=args.institution or read_answer("Название вуза: "),
            program=args.program.strip(),
            facts=args.facts_file.read_text(encoding="utf-8-sig") if args.facts_file else "",
        )

        with open_yandex(settings) as client:
            graph = create_rpd_graph(client, settings, notify=lambda message: print(message, flush=True))
            config = {"configurable": {"thread_id": str(uuid4())}}
            result = graph.invoke({"request": request.model_dump()}, config=config)

        if result.get("official_rpd_error"):
            print(f"Экспорт РПД приостановлен. {result['official_rpd_error']}", file=sys.stderr)
            return 1
        if result.get("__interrupt__"):
            print("Для интервью создайте курс в кабинете преподавателя.", file=sys.stderr)
            return 1
        if not result["ready"]:
            print("РПД требует уточнения. JSON не сохранён.", file=sys.stderr)
            for question in result["questions"]:
                print(f"• {question}", file=sys.stderr)
            return 1

        content = json.dumps(result["document"], ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        with args.output.open("x", encoding="utf-8", newline="\n") as output:
            output.write(content)
        print(f"Черновик прошёл проверку по схеме. Файл: {args.output.resolve()}")
        return 0
    except APIStatusError as error:
        print(f"Yandex AI Studio вернула HTTP {error.status_code}.", file=sys.stderr)
    except APITimeoutError:
        print("Yandex AI Studio не ответила за отведённое время. Запусти сценарий ещё раз.", file=sys.stderr)
    except APIConnectionError:
        print("Не удалось подключиться к Yandex AI Studio.", file=sys.stderr)
    except (OSError, ValueError, ValidationError, SchemaError, RuntimeError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
    except (KeyboardInterrupt, EOFError):
        print("\nРабота прервана. JSON не сохранён.", file=sys.stderr)
        return 130
    return 1


if __name__ == "__main__":
    sys.exit(main())
