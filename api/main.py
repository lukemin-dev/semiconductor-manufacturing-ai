from contextlib import asynccontextmanager
from pathlib import Path
import os
from uuid import uuid4
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, FiniteFloat
from sqlalchemy import text
from src.inference import Predictor
from src.database import engine, initialize, save_prediction, history


@asynccontextmanager
async def lifespan(app):
    app.state.predictor = Predictor(Path(os.environ.get("ARTIFACT_DIR", "artifacts")))
    app.state.db = engine()
    initialize(app.state.db, app.state.predictor.metadata)
    yield
    app.state.db.dispose()


app = FastAPI(
    title="SECOM Manufacturing AI · Portfolio", version="1.0.0", lifespan=lifespan
)


class PredictionRequest(BaseModel):
    sample_id: str = Field(
        default_factory=lambda: str(uuid4()), min_length=1, max_length=100
    )
    lot_id: str = Field(default="DEMO-API", min_length=1, max_length=100)
    features: dict[str, FiniteFloat | None] = Field(min_length=1)


@app.get("/health")
def health():
    with app.state.db.connect() as conn:
        conn.execute(text("SELECT 1"))
    return {
        "status": "ok",
        "database": "connected",
        "model_version": app.state.predictor.metadata["model_version"],
    }


@app.get("/model")
def model():
    return app.state.predictor.metadata


@app.get("/history")
def get_history(lot_id: str | None = None):
    return history(app.state.db, lot_id)


@app.post("/predict")
def predict(req: PredictionRequest):
    try:
        result = app.state.predictor.predict_one(req.features)
    except ValueError as exc:
        raise HTTPException(422, detail=str(exc)) from exc
    prediction_id = save_prediction(
        app.state.db, req.sample_id, req.lot_id, req.features, result
    )
    return {
        "prediction_id": prediction_id,
        "sample_id": req.sample_id,
        "lot_id": req.lot_id,
        **result,
    }


@app.post("/alerts/{prediction_id}/acknowledge")
def acknowledge(prediction_id: int):
    from src.database import acknowledge_alert

    result = acknowledge_alert(app.state.db, prediction_id)
    if result is None:
        raise HTTPException(404, detail="Alert not found for this prediction")
    return result
