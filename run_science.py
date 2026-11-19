"""Prepare an independent Science briefing on recent IT research."""

import argparse
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAI
from pydantic import ValidationError

from rpd_specialists import PROFILE_SCHEMA, PROFILES, analyse_fragments
from science_literature import FRONTIER_WINDOW_DAYS, IT_AREAS, normalise_doi, search_frontier


ROOT = Path(__file__).resolve().parent
ARTICLE_STATUSES = {
    "verified": "подтверждён экспертом",
    "published_candidate": "публикация найдена, рецензирование требует подтверждения",
    "arxiv_candidate": "arXiv, опубликованная версия не проверена",
}
REVIEW_SCOPE = (
    "Подготовь краткий обзор свежих исследований по ИТ по библиографическим карточкам. "
    "Выдели направления, которые имеет смысл изучить, и объясни, какие карточки на них указывают. "
    "Не описывай методы, новизну или результаты, которые нельзя установить по предоставленным данным. "
    "Ответ предназначен для научного обзора."
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Подобрать свежие научные работы по ИТ для Science.")
    parser.add_argument("--area", choices=IT_AREAS, help="Подобласть ИТ. По умолчанию ищем по всем.")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env", encoding="utf-8-sig", interpolate=False)

    verified_dois = {
        doi for entry in os.environ.get("SCIENCE_VERIFIED_DOIS", "").split(",")
        if (doi := normalise_doi(entry))
    }
    semantic_key = ""
    if os.environ.get("SCIENCE_SEMANTIC_SCHOLAR_ENABLED", "0").strip() == "1":
        semantic_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()
        if not semantic_key:
            print("Semantic Scholar: укажите SEMANTIC_SCHOLAR_API_KEY.", file=sys.stderr)

    print(f"Science ищет публикации по ИТ за последние {FRONTIER_WINDOW_DAYS} дней…", flush=True)
    cards, unavailable = search_frontier(args.area, verified_dois=verified_dois, semantic_key=semantic_key)
    if unavailable:
        print("Недоступные источники: " + ", ".join(unavailable), file=sys.stderr)
    if not cards:
        print("Подходящие карточки не найдены.", file=sys.stderr)
        return 1

    for card in cards:
        count = f", цитирования {card.citations} ({card.citation_source})" if card.citations is not None else ""
        print(f"  {card.title} ({card.year or 'год не указан'}) [{card.area}]\n"
              f"  {ARTICLE_STATUSES[card.status]}{count}. {card.url}")

    fragments = [card.as_science_source() for card in cards if card.status == "verified"]
    if not fragments:
        print("Подборка готова для проверки. Обзор Science появится, когда в ней будет подтверждённый DOI.")
        return 0

    key = os.environ.get("YANDEX_API_KEY", "").strip()
    folder = os.environ.get("YANDEX_FOLDER_ID", "").strip()
    if not key or not folder:
        print("Для обзора Science заполните YANDEX_API_KEY и YANDEX_FOLDER_ID.", file=sys.stderr)
        return 1
    model = os.environ.get("YANDEX_MODEL_URI_SCIENCE", "").strip() or f"gpt://{folder}/aliceai-llm-flash"
    try:
        with OpenAI(api_key=key, project=folder, base_url="https://ai.api.cloud.yandex.net/v1",
                    timeout=35, max_retries=0) as client:
            def generate(instructions: str, payload: str) -> str:
                response = client.responses.create(
                    model=model, instructions=instructions, input=payload,
                    text={"format": {"type": "json_schema", "name": "science_frontier",
                                     "schema": PROFILE_SCHEMA, "strict": True}},
                    max_output_tokens=1_024, store=False,
                )
                if response.status != "completed" or not response.output_text.strip():
                    raise RuntimeError(f"Модель вернула статус {response.status}.")
                return response.output_text

            report = analyse_fragments(PROFILES["science"], "Свежие исследования по ИТ",
                                       fragments, generate, scope=REVIEW_SCOPE)
        print("\nОбзор Science")
        for item in report["findings"]:
            print(f"• {item['summary']} ({fragments[item['fragment'] - 1]['file']})")
        return 0
    except APIStatusError as error:
        print(f"Yandex AI Studio вернула HTTP {error.status_code}.", file=sys.stderr)
    except APIConnectionError:
        print("Science не получила ответ от Yandex AI Studio.", file=sys.stderr)
    except (RuntimeError, ValidationError):
        print("Science не удалось сформировать краткий обзор.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
