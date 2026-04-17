"""
RetailFlow AI — Database Layer
SQLite persistence via SQLAlchemy for footfall snapshots.
"""

from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, String, DateTime, desc
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "sqlite:///./retail_data.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class FootfallRecord(Base):
    __tablename__ = "footfall_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    total_count = Column(Integer, nullable=False)
    top_zone = Column(String, nullable=False)


Base.metadata.create_all(bind=engine)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def save_log(count: int, top_zone: str) -> FootfallRecord:
    """Persist a snapshot record and return it."""
    db = SessionLocal()
    try:
        record = FootfallRecord(total_count=count, top_zone=top_zone)
        db.add(record)
        db.commit()
        db.refresh(record)
        return record
    finally:
        db.close()


def get_recent_logs(limit: int = 60) -> list[FootfallRecord]:
    """Return the most recent *limit* records, newest first."""
    db = SessionLocal()
    try:
        return (
            db.query(FootfallRecord)
            .order_by(desc(FootfallRecord.timestamp))
            .limit(limit)
            .all()
        )
    finally:
        db.close()
