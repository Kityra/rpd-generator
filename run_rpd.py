"""Run the RPD graph from a terminal using Yandex AI Studio."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from jsonschema import SchemaError
from langgraph.types import Command
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from pydantic import ValidationError

from rpd_graph import RpdRequest, build_rpd_graph
from rpd_specialists import PROFILE_SCHEMA, PROFILES, ProfileName, SourceFragment, SpecialistReport, analyse_fragments


ROOT = Path(__file__).resolve().parent
YANDEX_BASE_URL = "https://ai.api.cloud.yandex.net/v1"


def read_answer(prompt: str) -> str:
    while True:
        answer = input(prompt).strip()
        if answer:
            return answer
        print("Введите ответ или завершите работу сочетанием Ctrl+C.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Подготовить черновик РПД и сохранить его в JSON.")
    parser.add_argument("--discipline", help="Название дисциплины")
    parser.add_argument("--year", help="Целевой учебный год")
    parser.add_argument("--institution", help="Название вуза")
    parser.add_argument("--facts-file", type=Path, help="Файл UTF-8 с известными сведениями для РПД")
    parser.add_argument("--output", type=Path, default=ROOT / "rpd_draft.json", help="Куда сохранить черновик")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Выходной файл уже существует. Укажите другое имя через --output.")

    load_dotenv(ROOT / ".env", encoding="utf-8-sig", interpolate=False)
    required = ("YANDEX_API_KEY", "YANDEX_FOLDER_ID", "YANDEX_MODEL_URI", "YANDEX_VECTOR_STORE_ID_RPD")
    settings = {name: os.environ.get(name, "").strip() for name in required}
    missing = [name for name, value in settings.items() if not value]
    if missing:
        parser.error("Заполните .env: " + ", ".join(missing))
    base_url = os.environ.get("YANDEX_BASE_URL", YANDEX_BASE_URL).strip().rstrip("/")
    if base_url != YANDEX_BASE_URL:
        parser.error(f"Укажите в YANDEX_BASE_URL адрес {YANDEX_BASE_URL}")

    try:
        request = RpdRequest(
            discipline=args.discipline or read_answer("Название дисциплины: "),
            academic_year=args.year or read_answer("Учебный год: "),
            institution=args.institution or read_answer("Название вуза: "),
            facts=args.facts_file.read_text(encoding="utf-8-sig") if args.facts_file else "",
        )

        with OpenAI(
            api_key=settings["YANDEX_API_KEY"],
            project=settings["YANDEX_FOLDER_ID"],
            base_url=base_url,
            timeout=120,
            max_retries=0,
        ) as client:

            def request_json(
                model: str, instructions: str, payload: str, schema: dict[str, Any],
                schema_name: str, max_tokens: int, timeout: float,
            ) -> str:
                response = client.responses.create(
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
                    timeout=timeout,
                )
                if response.status != "completed" or not response.output_text.strip():
                    raise RuntimeError(f"Модель не вернула полный ответ (статус {response.status}).")
                return response.output_text

            flash_uri = f"gpt://{settings['YANDEX_FOLDER_ID']}/aliceai-llm-flash"

            def research(name: ProfileName, request: RpdRequest) -> SpecialistReport:
                profile = PROFILES[name]
                fragments: list[SourceFragment] = []
                fields = request.model_dump()
                for search in profile.searches:
                    index_id = os.environ.get(f"YANDEX_VECTOR_STORE_ID_{search.index.upper()}", "").strip()
                    if not index_id:
                        print(f"{profile.label}: индекс {search.index} пока не указан.", flush=True)
                        continue
                    try:
                        results = client.vector_stores.search(
                            index_id, query=search.query.format_map(fields), max_num_results=search.result_count, timeout=20
                        )
                    except (APIStatusError, APIConnectionError) as error:
                        if search.required:
                            raise
                        detail = f"HTTP {error.status_code}" if isinstance(error, APIStatusError) else "нет связи"
                        print(f"{profile.label}: индекс {search.index} недоступен ({detail}).",
                              file=sys.stderr, flush=True)
                        continue

                    for hit in results.data:
                        text = "\n".join(item.text for item in hit.content if item.type == "text" and item.text)
                        if text.strip():
                            fragments.append(SourceFragment(file=hit.filename, text=text[:2_000], index=search.index))

                model = os.environ.get(f"YANDEX_MODEL_URI_{name.upper()}", "").strip() or flash_uri

                def summarize(instructions: str, payload: str) -> str:
                    return request_json(model, instructions, payload, PROFILE_SCHEMA, f"rpd_{name}", 896, 35)

                try:
                    report = analyse_fragments(profile, request.discipline, fragments, summarize)
                except (APIStatusError, APIConnectionError, RuntimeError, ValidationError):
                    print(f"{profile.label}: модель не сформировала краткий отчёт.", file=sys.stderr, flush=True)
                    return {"fragments": fragments if name == "university" else [], "findings": []}

                print(f"{profile.label}: найдено фрагментов — {len(fragments)}, "
                      f"наблюдений — {len(report['findings'])}.", flush=True)
                if name != "university" and not report["findings"]:
                    return {"fragments": [], "findings": []}
                return report

            def generate(instructions: str, payload: str, response_schema: dict[str, Any]) -> str:
                print("Собираю черновик РПД…", flush=True)
                return request_json(settings["YANDEX_MODEL_URI"], instructions, payload,
                                    response_schema, "rpd_draft", 8_192, 120)

            graph = build_rpd_graph(ROOT / "final_schema.json", research, generate)
            config = {"configurable": {"thread_id": str(uuid4())}}
            result = graph.invoke({"request": request.model_dump()}, config=config)
            while result.get("__interrupt__"):
                print("\nУточнения для РПД")
                for question in result["__interrupt__"][0].value["questions"]:
                    print(f"• {question}")
                answer = read_answer("Ответь на вопросы одним сообщением: ")
                result = graph.invoke(Command(resume=answer), config=config)

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
    return 1


if __name__ == "__main__":
    sys.exit(main())
