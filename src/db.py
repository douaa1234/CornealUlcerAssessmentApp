# db.py
from __future__ import annotations

import json
import logging
import os
import pickle
import zlib
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from sqlalchemy import DateTime, Float, ForeignKey, LargeBinary, String, Text, UniqueConstraint, create_engine, event, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", str(BASE_DIR / "data_store"))).expanduser().resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "app.db"
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{DB_PATH}")

logging.basicConfig(
    filename=str(DATA_DIR / "audit.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def _engine_kwargs(url: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "future": True,
        "pool_pre_ping": True,
    }
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
    return kwargs


engine = create_engine(DATABASE_URL, **_engine_kwargs(DATABASE_URL))


@event.listens_for(engine, "connect")
def _configure_sqlite(dbapi_connection, _connection_record):
    if not DATABASE_URL.startswith("sqlite"):
        return
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=30000")
    cursor.close()


SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    future=True,
)


class Base(DeclarativeBase):
    pass


class Case(Base):
    __tablename__ = "cases"

    case_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    visits = relationship("Visit", back_populates="case", cascade="all, delete-orphan")


class Visit(Base):
    __tablename__ = "visits"

    visit_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.case_id", ondelete="CASCADE"), index=True)
    visit_date: Mapped[str] = mapped_column(String(32), index=True)
    eye: Mapped[str] = mapped_column(String(16))
    mode: Mapped[str] = mapped_column(String(16))
    image_hash: Mapped[str] = mapped_column(String(32), index=True)

    mm_per_px: Mapped[float] = mapped_column(Float)
    area_mm2: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    eq_diameter_mm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    opacity_zscore: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    blur: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    qc_flags: Mapped[str] = mapped_column(Text, default="")

    metrics_json: Mapped[str] = mapped_column(Text, default="{}")
    report_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    case = relationship("Case", back_populates="visits")

    __table_args__ = (
        UniqueConstraint("case_id", "visit_date", "eye", "mode", "image_hash", name="uix_visit_dedupe"),
    )


class AppSession(Base):
    __tablename__ = "app_sessions"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    state_blob: Mapped[bytes] = mapped_column(LargeBinary)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DatabaseError(RuntimeError):
    pass


def init_db() -> None:
    Base.metadata.create_all(engine)


@contextmanager
def db_session() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _clean_case_id(case_id: str) -> str:
    value = (case_id or "").strip()
    if not value:
        raise ValueError("Case ID is required.")
    if len(value) > 64:
        raise ValueError("Case ID must be 64 characters or fewer.")
    return value


def upsert_case(db: Session, case_id: str) -> Case:
    clean_id = _clean_case_id(case_id)
    case = db.get(Case, clean_id)
    if case is None:
        case = Case(case_id=clean_id)
        db.add(case)
    return case


def _encode_session_state(state: dict[str, Any]) -> bytes:
    return zlib.compress(pickle.dumps(dict(state), protocol=pickle.HIGHEST_PROTOCOL))


def _decode_session_state(blob: bytes) -> dict[str, Any]:
    return pickle.loads(zlib.decompress(blob))


def load_app_session(session_id: str) -> dict[str, Any] | None:
    try:
        with db_session() as db:
            row = db.get(AppSession, session_id)
            if row is None:
                return None
            return _decode_session_state(row.state_blob)
    except SQLAlchemyError as exc:
        logger.exception("Session load failed | session_id=%s", session_id)
        raise DatabaseError("Could not load session.") from exc


def save_app_session(session_id: str, state: dict[str, Any]) -> None:
    try:
        with db_session() as db:
            blob = _encode_session_state(state)
            row = db.get(AppSession, session_id)
            if row is None:
                db.add(AppSession(session_id=session_id, state_blob=blob))
            else:
                row.state_blob = blob
                row.updated_at = datetime.utcnow()
    except SQLAlchemyError as exc:
        logger.exception("Session save failed | session_id=%s", session_id)
        raise DatabaseError("Could not save session.") from exc


def delete_app_session(session_id: str) -> None:
    try:
        with db_session() as db:
            row = db.get(AppSession, session_id)
            if row is not None:
                db.delete(row)
    except SQLAlchemyError as exc:
        logger.exception("Session delete failed | session_id=%s", session_id)
        raise DatabaseError("Could not delete session.") from exc


def save_visit(
    *,
    visit_id: str,
    case_id: str,
    visit_date: str,
    eye: str,
    mode: str,
    image_hash: str,
    mm_per_px: float,
    analysis_result: Dict[str, Any],
    report_text: str,
) -> None:
    clean_case_id = _clean_case_id(case_id)
    try:
        with db_session() as db:
            upsert_case(db, clean_case_id)
            visit = Visit(
                visit_id=visit_id,
                case_id=clean_case_id,
                visit_date=visit_date,
                eye=eye,
                mode=mode,
                image_hash=image_hash,
                mm_per_px=float(mm_per_px),
                area_mm2=analysis_result.get("area_mm2"),
                eq_diameter_mm=analysis_result.get("eq_diameter_mm"),
                opacity_zscore=analysis_result.get("opacity_zscore"),
                blur=analysis_result.get("blur_laplacian_var"),
                qc_flags=";".join(analysis_result.get("analysis_flags") or []),
                metrics_json=json.dumps(analysis_result, ensure_ascii=False),
                report_text=report_text,
            )
            db.add(visit)
        logger.info(
            "Visit saved | case_id=%s | date=%s | eye=%s | mode=%s | visit_id=%s",
            clean_case_id,
            visit_date,
            eye,
            mode,
            visit_id,
        )
    except IntegrityError as exc:
        raise ValueError(
            f"This visit already exists for Case {clean_case_id} on {visit_date} ({eye}, {mode})."
        ) from exc
    except SQLAlchemyError as exc:
        logger.exception("Database save failed | case_id=%s | visit_id=%s", clean_case_id, visit_id)
        raise DatabaseError("Could not save visit.") from exc


def load_case_visits(case_id: str) -> list[Visit]:
    clean_case_id = (case_id or "").strip()
    if not clean_case_id:
        return []
    try:
        with db_session() as db:
            query = select(Visit).where(Visit.case_id == clean_case_id).order_by(Visit.visit_date.asc())
            return list(db.execute(query).scalars().all())
    except SQLAlchemyError as exc:
        logger.exception("Database load failed | case_id=%s", clean_case_id)
        raise DatabaseError("Could not load case visits.") from exc


def delete_case(case_id: str) -> None:
    clean_case_id = _clean_case_id(case_id)
    try:
        with db_session() as db:
            case = db.get(Case, clean_case_id)
            if case is not None:
                db.delete(case)
        logger.info("Case deleted | case_id=%s", clean_case_id)
    except SQLAlchemyError as exc:
        logger.exception("Database delete failed | case_id=%s", clean_case_id)
        raise DatabaseError("Could not delete case.") from exc
