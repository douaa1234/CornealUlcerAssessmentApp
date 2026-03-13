# db.py
from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from sqlalchemy import (
    create_engine, String, DateTime, Float, Text, ForeignKey, UniqueConstraint
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker, relationship
import json

BASE_DIR= Path(__file__).resolve().parent
DB_PATH= BASE_DIR / "data_store" / "app.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

engine= create_engine(f"sqlite:///{DB_PATH}", future=True)
SessionLocal= sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

class Base(DeclarativeBase):
    pass

class Case(Base):
    __tablename__ = "cases"
    case_id: Mapped[str]= mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime]= mapped_column(DateTime, default=datetime.utcnow)
    visits = relationship("Visit", back_populates="case")

class Visit(Base):
    __tablename__= "visits"
    visit_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.case_id"), index=True)
    visit_date: Mapped[str] = mapped_column(String(32), index=True)  # keep as YYYY-MM-DD string
    eye: Mapped[str] = mapped_column(String(16))
    mode: Mapped[str] = mapped_column(String(16))  # white / fluor
    image_hash: Mapped[str] = mapped_column(String(32), index=True)

    mm_per_px: Mapped[float] = mapped_column(Float)
    area_mm2: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    eq_diameter_mm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    opacity_zscore: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    blur: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    qc_flags: Mapped[str] = mapped_column(Text, default="")

    metrics_json: Mapped[str] = mapped_column(Text, default="{}")  # store full analysis result
    report_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    case = relationship("Case", back_populates="visits")

    __table_args__ = (
        UniqueConstraint("case_id", "visit_date", "eye", "mode", "image_hash", name="uix_visit_dedupe"),
    )

def init_db():
    Base.metadata.create_all(engine)

def upsert_case(db, case_id: str):
    c= db.get(Case, case_id)
    if c is None:
        c = Case(case_id=case_id)
        db.add(c)
    return c

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
):
    with SessionLocal() as db:
        upsert_case(db, case_id)

        v = Visit(
            visit_id=visit_id,
            case_id=case_id,
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
        db.add(v)
        db.commit()

def load_case_visits(case_id: str):
    from sqlalchemy import select
    with SessionLocal() as db:
        q = select(Visit).where(Visit.case_id == case_id).order_by(Visit.visit_date.asc())
        return db.execute(q).scalars().all()