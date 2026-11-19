"""Check Yandex AI Studio index metadata, search, and model access."""

import argparse
import os
import sys
from pathlib import Path


INDEX_SETTINGS = {
    "rpd": "YANDEX_VECTOR_STORE_ID_RPD",
    "books": "YANDEX_VECTOR_STORE_ID_BOOKS",
    "hh": "YANDEX_VECTOR_STORE_ID_HH",
}
YANDEX_BASE_URL = "https://ai.api.cloud.yandex.net/v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", choices=INDEX_SETTINGS)
    parser.add_argument("--query", help="Search text for the selected index")
    parser.add_argument("--check-model", action="store_true", help="Also send a short Responses API request")
    args = parser.parse_args()
    if args.query is not None and (not args.index or not args.query.strip()):
        parser.error("--query requires --index and a nonempty search text")

    try:
        from dotenv import load_dotenv
        from openai import APIConnectionError, APIStatusError, OpenAI
    except ImportError:
        parser.exit(2, "Install dependencies: python -m pip install openai python-dotenv\n")

    load_dotenv(Path(__file__).with_name(".env"), encoding="utf-8-sig", interpolate=False)
    names = (args.index,) if args.index else tuple(INDEX_SETTINGS)
    required = ["YANDEX_API_KEY", "YANDEX_FOLDER_ID", *(INDEX_SETTINGS[name] for name in names)]
    if args.check_model:
        required.append("YANDEX_MODEL_URI")
    settings = {key: os.environ.get(key, "").strip() for key in required}
    missing = [key for key, value in settings.items() if not value]
    if missing:
        parser.exit(2, "Fill in .env: " + ", ".join(missing) + "\n")

    base_url = os.environ.get("YANDEX_BASE_URL", YANDEX_BASE_URL).strip().rstrip("/")
    if base_url != YANDEX_BASE_URL:
        parser.exit(2, f"Set YANDEX_BASE_URL to {YANDEX_BASE_URL}\n")

    all_ready = True
    with OpenAI(api_key=settings["YANDEX_API_KEY"], project=settings["YANDEX_FOLDER_ID"],
                base_url=base_url, timeout=20, max_retries=0) as client:
        for name in names:
            store_id = settings[INDEX_SETTINGS[name]]
            try:
                store = client.vector_stores.retrieve(store_id)
                counts = store.file_counts
                print(f"{name.upper()}: {store.status}, files ready {counts.completed}/{counts.total}, "
                      f"failed {counts.failed}, cancelled {counts.cancelled}, in progress {counts.in_progress}")
                ready = store.status == "completed" and counts.total > 0 and counts.completed == counts.total
                all_ready = all_ready and ready
                if args.query and ready:
                    results = client.vector_stores.search(store_id, query=args.query, max_num_results=3)
                    if not results.data:
                        print("Search returned no matches.")
                        all_ready = False
                    for hit in results.data:
                        print(f"  {hit.filename} ({hit.file_id}), score={hit.score:.3f}")
            except APIStatusError as error:
                print(f"{name.upper()}: Yandex AI Studio returned HTTP {error.status_code}.", file=sys.stderr)
                all_ready = False
            except APIConnectionError:
                print(f"{name.upper()}: cannot reach Yandex AI Studio.", file=sys.stderr)
                all_ready = False
        if args.check_model:
            try:
                response = client.responses.create(
                    model=settings["YANDEX_MODEL_URI"],
                    input="Ответь одним словом: готово.",
                    max_output_tokens=128,
                    store=False,
                )
                model_ok = response.status == "completed" and bool(response.output_text.strip())
                print("Model: OK" if model_ok else "Model: no completed text response.")
                all_ready = all_ready and model_ok
            except APIStatusError as error:
                print(f"Model: Yandex AI Studio returned HTTP {error.status_code}.", file=sys.stderr)
                all_ready = False
            except APIConnectionError:
                print("Model: cannot reach Yandex AI Studio.", file=sys.stderr)
                all_ready = False
    return 0 if all_ready else 1


if __name__ == "__main__":
    sys.exit(main())
