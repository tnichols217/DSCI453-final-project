"""Imports image data to the PG database"""

import os
from collections.abc import Iterable
from pathlib import Path

import cv2
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.session import Session

from pg_manager import Image

_ = load_dotenv()

POSTGRES_DB = os.getenv("POSTGRES_DB") or ""
POSTGRES_USER = os.getenv("POSTGRES_USER") or ""
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD") or ""
POSTGRES_ENDPOINT = os.getenv("POSTGRES_ENDPOINT") or ""
DB_URL = os.getenv("DB_URI") or ""
DATA_DIR = Path(os.getenv("DATA_DIR") or "")


def activate_con() -> Session:
    """Create an sqlalchemy session

    Returns:
        Session: An active sqlalchemy session

    """
    engine = create_engine(DB_URL)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)()


files = DATA_DIR.glob("*")


def insert_image(db: Session, images: Iterable[Path]) -> bool:
    """Insert images into the database

    Args:
        db (Session): A PostgreSQL connection
        images (list[Path]): A list of image paths

    Returns:
        bool: True if all images were inserted successfully, False otherwise

    """
    for i in images:
        image = cv2.imread(str(i), cv2.IMREAD_COLOR)
        image = cv2.resize(image, (500, 500), interpolation=cv2.INTER_NEAREST)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
        b, g, r = cv2.split(image)
        h, s, v = cv2.split(hsv)
        print(r.tolist())
        db.add(Image(
            R=r.tolist(),
            G=g.tolist(),
            B=b.tolist(),
            H=h.tolist(),
            S=s.tolist(),
            V=v.tolist(),
        ))
        db.commit()

    return True


_ = insert_image(activate_con(), [list(files)[0]])
