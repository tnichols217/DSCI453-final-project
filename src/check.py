import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, session, sessionmaker

_ = load_dotenv()

POSTGRES_DB = os.getenv("POSTGRES_DB") or ""
POSTGRES_USER = os.getenv("POSTGRES_USER") or ""
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD") or ""
POSTGRES_ENDPOINT = os.getenv("POSTGRES_ENDPOINT") or ""
DB_URL = os.getenv("DB_URI") or ""
DATA_CSV = Path(os.getenv("DATA_CSV") or "")
DATA_DIR = Path(os.getenv("DATA_DIR") or "")
CHUNK = 100

engine: Engine = create_engine(
    DB_URL,
    pool_size=5,
    max_overflow=10,
    future=True,
)
SessionLocal: sessionmaker[Session] = sessionmaker(engine, class_=Session)

s = SessionLocal()

print(list(s.execute(text("SELECT * FROM pg_stat_activity"))))