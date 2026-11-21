"""Постановка утверждённых курсов в очередь повторного аудита."""
import argparse
import asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker

from auth.settings import Settings
from storage.database import DatabaseSetupError, ensure_schema, make_engine, run_async
from storage.portal import schedule_due


async def run(once):
    engine = make_engine(Settings().app_database_url)
    try:
        factory = async_sessionmaker(engine, expire_on_commit=False)
        while True:
            count = await schedule_due(factory)
            if once:
                print(f"Курсов поставлено в очередь: {count}")
                return
            await asyncio.sleep(300)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    arguments = parser.parse_args()
    try:
        ensure_schema(Settings().app_database_url)
    except DatabaseSetupError as error:
        raise SystemExit(str(error))
    run_async(run(arguments.once))
