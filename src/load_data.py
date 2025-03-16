"""Loads image data into the database"""

import asyncio
import csv
import os
from pathlib import Path
from types import CoroutineType

import cv2
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
THREADS = 20
CHUNK = 10

# Create an async engine and sessionmaker
async_engine: AsyncEngine = create_async_engine(
    DB_URL,
    pool_size=20,
    max_overflow=10,
    future=True,
)
AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    async_engine, class_=AsyncSession
)


def insert_image(db: AsyncSession, i: Path, labels: dict[str, bool]) -> bool:
    """Insert images into the database asynchronously

    Args:
        db (AsyncSession): SQLAlchemy session
        i (Path): Image file path
        labels (dict[str, bool]): Label for the image

    Returns:
        bool: True if the image was inserted successfully, False otherwise

    """
    print(f"Processing {i.name}")
    image = cv2.imread(str(i), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (500, 500), interpolation=cv2.INTER_NEAREST)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    b, g, r = cv2.split(image)
    h, s, v = cv2.split(hsv)
    edges = cv2.Canny(v, 100, 200)
    erode = cv2.erode(
        edges, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1
    )
    dilate = cv2.dilate(
        edges, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1
    )

    db.add(
        Image(
            red=r.tolist(),
            gre=g.tolist(),
            blu=b.tolist(),
            hue=h.tolist(),
            sat=s.tolist(),
            val=v.tolist(),
            edge=edges.tolist(),
            dilate=dilate.tolist(),
            erode=erode.tolist(),
            label=labels[i.name],
        )
    )
    return True


async def image_loop(files: list[Path], labels: dict[str, bool]) -> None:
    """Async loop for inserting images into the database"""
    f = iter(files)
    n = next(f, None)
    async with AsyncSessionLocal() as db:
        while n:
            for _ in range(CHUNK):
                if not n:
                    break
                _ = insert_image(db, n, labels)
                n = next(f, None)
            print(
                f"""Currently inserted: {
                    await db.scalar(select(func.count(Image.id)))
                } images"""
            )
            await db.commit()
            await db.flush()


async def main() -> None:
    """Main function for loading images into the database"""
    labels: dict[str, bool] = {}
    with Path.open(DATA_CSV, "r") as f:
        csvreader = csv.DictReader(f)
        for row in csvreader:
            labels[Path(row["file_name"]).name] = row["label"] == "1"

    files = list(DATA_DIR.glob("*"))
    chunk_size = len(files) // THREADS
    print(f"Loading {len(files)} images into the database")
    print(f"Using {THREADS} threads")
    print(f"Loading {chunk_size} images per thread")
    tasks: list[CoroutineType[None, None, None]] = []

    for i in range(THREADS):
        start = i * chunk_size
        end = None if i == THREADS - 1 else start + chunk_size
        chunk = files[start:end]
        tasks.append(image_loop(chunk, labels))

    _ = await asyncio.gather(*tasks)


if __name__ == "__main__":
    _ = asyncio.run(main())
