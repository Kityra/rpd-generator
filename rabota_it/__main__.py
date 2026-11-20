"""CLI импорта, периодического обновления и поиска."""

import argparse
import json
import logging
import os
from pathlib import Path
import sqlite3
from threading import Event

from .service import BusinessAPI, VacancyStore, import_export, sync


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("data/rabota_it.sqlite"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("import").add_argument("path", type=Path)
    for command in ("sync", "watch"):
        source = commands.add_parser(command)
        source.add_argument("endpoint")
        source.add_argument("--authorised", action="store_true", required=True)
        source.add_argument("--interval", type=int, default=900, help="Секунды между запросами")
    for command in ("search", "summary"):
        search = commands.add_parser(command)
        search.add_argument("query", nargs="?", default="")
        search.add_argument("--region", default="")
        search.add_argument("--technology", action="append", default=[])
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    try:
        store = VacancyStore(args.database)
        if args.command == "import":
            result = import_export(store, args.path)
        elif args.command in {"sync", "watch"}:
            stop = Event()
            while True:
                result = sync(store, args.endpoint, authorised=args.authorised,
                              token=os.environ.get("RABOTA_IT_TOKEN"), interval=args.interval)
                if args.command == "sync":
                    break
                logging.info("Обновление Работа.ру — %s", result["status"])
                stop.wait(args.interval)
        else:
            result = BusinessAPI(store).handle(operation=args.command, query=args.query,
                                               region=args.region, technologies=args.technology)
        print(json.dumps(result, ensure_ascii=False))
        return int(result.get("status") == "failed")
    except (ValueError, OSError, sqlite3.Error, RecursionError) as error:
        logging.error("Операция не выполнена (%s). Проверьте аргументы, формат и доступ к файлам.",
                      type(error).__name__)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
