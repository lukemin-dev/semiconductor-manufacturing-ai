import os
from datetime import datetime, timezone
from sqlalchemy import (
    create_engine,
    String,
    Float,
    Integer,
    JSON,
    ForeignKey,
    DateTime,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session


class Base(DeclarativeBase):
    pass


class ModelVersion(Base):
    __tablename__ = "model_version"
    version: Mapped[str] = mapped_column(String(80), primary_key=True)
    metadata_json: Mapped[dict] = mapped_column(JSON)


class ProcessData(Base):
    __tablename__ = "process_data"
    id: Mapped[int] = mapped_column(primary_key=True)
    sample_id: Mapped[str] = mapped_column(String(100), index=True)
    lot_id: Mapped[str] = mapped_column(String(100), index=True)
    features: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class PredictionResult(Base):
    __tablename__ = "prediction_result"
    id: Mapped[int] = mapped_column(primary_key=True)
    process_id: Mapped[int] = mapped_column(ForeignKey("process_data.id"))
    model_version: Mapped[str] = mapped_column(ForeignKey("model_version.version"))
    risk_score: Mapped[float] = mapped_column(Float)
    predicted_fail: Mapped[int] = mapped_column(Integer)
    result: Mapped[dict] = mapped_column(JSON)


class AlertHistory(Base):
    __tablename__ = "alert_history"
    id: Mapped[int] = mapped_column(primary_key=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("prediction_result.id"))
    status: Mapped[str] = mapped_column(String(30), default="open")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


def engine():
    return create_engine(
        os.environ.get(
            "DATABASE_URL",
            "postgresql+psycopg2://secom:secom_local@localhost:55432/secom",
        ),
        pool_pre_ping=True,
    )


def initialize(db, metadata):
    Base.metadata.create_all(db)
    with Session(db) as s, s.begin():
        if not s.get(ModelVersion, metadata["model_version"]):
            s.add(
                ModelVersion(version=metadata["model_version"], metadata_json=metadata)
            )


def save_prediction(db, sample_id, lot_id, features, result):
    with Session(db) as s, s.begin():
        data = ProcessData(sample_id=sample_id, lot_id=lot_id, features=features)
        s.add(data)
        s.flush()
        pred = PredictionResult(
            process_id=data.id,
            model_version=result["model_version"],
            risk_score=result["risk_score"],
            predicted_fail=result["predicted_fail"],
            result=result,
        )
        s.add(pred)
        s.flush()
        if result["predicted_fail"]:
            s.add(AlertHistory(prediction_id=pred.id))
        return pred.id


def history(db, lot_id=None):
    with Session(db) as s:
        q = (
            select(ProcessData, PredictionResult, AlertHistory)
            .join(PredictionResult, PredictionResult.process_id == ProcessData.id)
            .outerjoin(AlertHistory, AlertHistory.prediction_id == PredictionResult.id)
        )
        if lot_id:
            q = q.where(ProcessData.lot_id == lot_id)
        rows = s.execute(q.order_by(PredictionResult.id.desc()).limit(500)).all()
        return [
            {
                "prediction_id": p.id,
                "sample_id": d.sample_id,
                "lot_id": d.lot_id,
                "created_at": d.created_at.isoformat(),
                "alert_status": a.status if a else None,
                **p.result,
            }
            for d, p, a in rows
        ]


def acknowledge_alert(db, prediction_id):
    """Idempotently acknowledge an existing alert; this never changes a prediction."""
    with Session(db) as session, session.begin():
        alert = session.scalar(
            select(AlertHistory)
            .where(AlertHistory.prediction_id == prediction_id)
            .with_for_update()
        )
        if alert is None:
            return None
        alert.status = "acknowledged"
        return {"prediction_id": prediction_id, "alert_status": alert.status}
