"""
Database connection and session management.

SQLite is used by default for local development.
PostgreSQL can be configured through DATABASE_URL.
"""

import os
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker


DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "autostream.db"

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{DEFAULT_DB_PATH}",
)

connect_args = {}

if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)


def get_db() -> Generator[Session, None, None]:
    """Provide a database session and close it safely."""
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create database tables."""
    from database.models import Base

    DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)