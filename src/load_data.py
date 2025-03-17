"""Loads image data into the database"""

import asyncio
import csv
import os
from pathlib import Path
from types import CoroutineType

import cv2
import numpy as np
from cv2.typing import MatLike
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from numpy import dtype, generic, ndarray

from file_manager import write_file

_ = load_dotenv()

DATA_CSV = Path(os.getenv("DATA_CSV") or "")
DATA_DIR = Path(os.getenv("DATA_DIR") or "")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR") or "")
THREADS = 20
CHUNK = 10
SIZE = (500, 500)


def resize_and_crop(
    image: MatLike, target_size: tuple[int, int] = SIZE
) -> ndarray[tuple[int, ...], dtype[generic]]:
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


def insert_image(i: Path, labels: dict[str, bool]) -> bool:
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
    image = resize_and_crop(image, SIZE)
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

    s = np.array([
        r.tolist(),
        g.tolist(),
        b.tolist(),
        h.tolist(),
        s.tolist(),
        v.tolist(),
        edges.tolist(),
        dilate.tolist(),
        erode.tolist(),
    ], dtype=np.uint8)

    subdir = "ai" if labels[i.name] else "no_ai"

    Path.mkdir(OUTPUT_DIR / subdir, exist_ok=True)
    _ = write_file(s, OUTPUT_DIR / subdir / f"{i.stem}.npzstd")
    return True


async def image_loop(files: list[Path], labels: dict[str, bool]) -> None:
    """Async loop for inserting images into the database"""
    f = iter(files)
    n = next(f, None)
    while n:
        for _ in range(CHUNK):
            if not n:
                break
            _ = insert_image(n, labels)
            n = next(f, None)


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
    # _ = insert_image(AsyncSessionLocal(), DATA_DIR / "1160958731514f6ca15bc9fe62570f01.jpg", {})