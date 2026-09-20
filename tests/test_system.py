import os
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, func
from sqlalchemy.orm import Session
from src.inference import Predictor
from src.data import load_secom
from src.preprocess import fit_feature_filter
from src.database import ProcessData, PredictionResult, AlertHistory, ModelVersion
from api.main import app


def test_train_only_filter():
    train = pd.DataFrame(
        {"constant": [1, 1, 1], "missing": [None, None, 2], "signal": [1, 2, 3]}
    )
    ff = fit_feature_filter(train)
    assert ff.kept_columns == ["signal"]


def test_attribution_and_input():
    p = Predictor()
    row = load_secom().X.iloc[0].to_dict()
    result = p.predict_one(row)
    frame = pd.DataFrame([row])[p.feature_filter.kept_columns]
    effect = p.effects(frame)[0]
    for item in result["top_features"]:
        assert item["contribution"] == pytest.approx(
            effect[list(frame.columns).index(item["feature"])]
        )
    assert result["predicted_fail"] == int(
        result["risk_score"] >= p.metadata["threshold"]
    )
    for bad in ({}, {"nonsense": 3}, {"f_000": float("inf")}, {"f_000": None}):
        with pytest.raises(ValueError):
            p.predict_one(bad)


def test_api_transactions(tmp_path, monkeypatch):
    url = os.environ.get("TEST_DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("DATABASE_URL", url)
    ds = load_secom()
    predictions = pd.read_csv("artifacts/test_predictions.csv")
    indices = [
        int(predictions.query("predicted_fail == 1").iloc[0].row_index),
        int(predictions.query("predicted_fail == 0").iloc[0].row_index),
    ]
    with TestClient(app) as client:
        assert client.get("/health").json()["database"] == "connected"
        for idx in indices:
            row = {
                k: None if pd.isna(v) else float(v) for k, v in ds.X.loc[idx].items()
            }
            r = client.post(
                "/predict",
                json={
                    "sample_id": f"TEST-{idx}",
                    "lot_id": "TEST-LOT",
                    "features": row,
                },
            )
            assert r.status_code == 200
            assert len(r.json()["top_features"]) == 5
        assert (
            client.post("/predict", json={"features": {"unknown": 1}}).status_code
            == 422
        )
        assert client.post("/predict", json={"features": {}}).status_code == 422
        history = client.get("/history", params={"lot_id": "TEST-LOT"}).json()
        assert len(history) >= 2
        assert any(h["alert_status"] is None for h in history)
        alert = next(h for h in history if h["alert_status"] == "open")
        route = f"/alerts/{alert['prediction_id']}/acknowledge"
        first = client.post(route)
        assert first.status_code == 200
        assert first.json()["alert_status"] == "acknowledged"
        assert client.post(route).json() == first.json()
        assert client.post("/alerts/99999999/acknowledge").status_code == 404
        with Session(app.state.db) as s:
            assert s.scalar(select(func.count()).select_from(ModelVersion)) >= 1
            assert s.scalar(select(func.count()).select_from(ProcessData)) >= 2
            assert s.scalar(select(func.count()).select_from(PredictionResult)) >= 2
            assert s.scalar(select(func.count()).select_from(AlertHistory)) >= 1
