"""Short Script to check the status of Postgres"""
import asyncio
import os
from pathlib import Path

import nest_asyncio
from dotenv import load_dotenv
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.ext.asyncio.engine import AsyncEngine
from sqlalchemy.future import select

from pg_manager import Image

_ = load_dotenv()

POSTGRES_DB = os.getenv("POSTGRES_DB") or ""
POSTGRES_USER = os.getenv("POSTGRES_USER") or ""
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD") or ""
POSTGRES_ENDPOINT = os.getenv("POSTGRES_ENDPOINT") or ""
DB_URL = os.getenv("DB_URI") or ""
DATA_CSV = Path(os.getenv("DATA_CSV") or "")
DATA_DIR = Path(os.getenv("DATA_DIR") or "")
CHUNK = 100

async_engine: AsyncEngine = create_async_engine(
    DB_URL,
    pool_size=5,
    max_overflow=10,
    future=True,
)
AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    async_engine, class_=AsyncSession
)


async def main() -> None:
    async with AsyncSessionLocal() as db:
        print(await db.scalar(select(func.count(Image.id))))


if __name__ == "__main__":
    nest_asyncio.apply()
    asyncio.run(main())
