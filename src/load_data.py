"""Loads image data into the database"""

import asyncio
import csv
import os
from pathlib import Path
from types import CoroutineType

import cv2
from cv2.typing import MatLike
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from numpy import dtype, generic, ndarray
from numpy._typing._shape import _Shape
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
    pool_size=THREADS,
    max_overflow=10,
    future=True,
)
AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    async_engine, class_=AsyncSession
)


def resize_and_crop(
    image: MatLike, target_size: tuple[int, int] = (500, 500)
) -> ndarray[_Shape, dtype[generic]]:
    """Resize and crop an image to the target size without distortion.

    Args:
        image (numpy.ndarray): The input image.
        target_size (tuple): The target size (width, height).

    Returns:
        numpy.ndarray: The resized and cropped image.

    """
    # Get the dimensions of the original image
    dims: tuple[int, int] = image.shape[:2]

    res = (
        cv2.resize(
            image,
            (target_size[0], dims[0] * target_size[0] // dims[1]),
            interpolation=cv2.INTER_AREA,
        )
        if dims[0] > dims[1]
        else cv2.resize(
            image,
            (dims[1] * target_size[1] // dims[0], target_size[1]),
            interpolation=cv2.INTER_AREA,
        )
    )

    center: tuple[int, int] = res.shape[:2]
    y = center[0] // 2 - target_size[0] // 2
    x = center[1] // 2 - target_size[1] // 2

    return res[y : y + target_size[0], x : x + target_size[1]]


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
    image = resize_and_crop(image, (500, 500))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    b, g, r = cv2.split(image)
    h, s, v = cv2.split(hsv)
    edges = cv2.Canny(v, 100, 200)
    erode = cv2.erode(
        v, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1
    )
    dilate = cv2.dilate(
        v, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1
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
            label=labels[i.name] or False,
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
            print(f"    Inserting {CHUNK} images")
            await db.commit()
            print(f"        Inserted {CHUNK} images")


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