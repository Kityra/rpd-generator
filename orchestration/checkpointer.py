from contextlib import asynccontextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


CHECKPOINT_SETUP_LOCK = 625149384


@asynccontextmanager
async def open_checkpointer(url: str):
    if url.startswith(("postgresql://", "postgres://")):
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        async with AsyncPostgresSaver.from_conn_string(url) as saver:
            # LangGraph записывает версии миграций отдельными командами autocommit.
            await saver.conn.execute("SELECT pg_advisory_lock(%s)", (CHECKPOINT_SETUP_LOCK,))
            try:
                await saver.setup()
            finally:
                await saver.conn.execute("SELECT pg_advisory_unlock(%s)", (CHECKPOINT_SETUP_LOCK,))
            yield saver
    else:
        Path(url).parent.mkdir(parents=True, exist_ok=True)
        async with AsyncSqliteSaver.from_conn_string(url) as saver:
            await saver.setup()
            yield saver
