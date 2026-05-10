from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.models import Base
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./mega_ai.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # needed for SQLite with FastAPI
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency — gives a DB session to each request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()