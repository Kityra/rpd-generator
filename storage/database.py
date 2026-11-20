from contextlib import asynccontextmanager
from pathlib import Path
from collections.abc import AsyncIterator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine


def make_engine(url: str) -> AsyncEngine:
    if url.startswith("sqlite"):
        Path("data").mkdir(exist_ok=True)
    return create_async_engine(url, pool_pre_ping=True)


@asynccontextmanager
async def write_transaction(factory: async_sessionmaker[AsyncSession]) -> AsyncIterator[AsyncSession]:
    async with factory() as db:
        try:
            if db.bind.dialect.name == "sqlite":
                await db.execute(text("BEGIN IMMEDIATE"))
            yield db
            await db.commit()
        except BaseException:
            await db.rollback()
            raise
