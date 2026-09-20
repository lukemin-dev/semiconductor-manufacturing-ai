from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np

from .config import SECOM_DATA, SECOM_LABELS


@dataclass
class SecomDataset:
    X: pd.DataFrame
    y: pd.Series
    timestamps: pd.Series


def load_secom(
    data_path: Path = SECOM_DATA, labels_path: Path = SECOM_LABELS
) -> SecomDataset:
    if not data_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            "SECOM raw files not found. Run `python scripts/download_data.py` "
            "or place secom.data and secom_labels.data in data/raw/."
        )

    X = pd.read_csv(data_path, sep=r"\s+", header=None, na_values="NaN")
    X.columns = [f"f_{i:03d}" for i in range(X.shape[1])]

    labels = pd.read_csv(labels_path, sep=r"\s+", header=None)
    raw_y = labels.iloc[:, 0].astype(int)
    if not raw_y.isin([-1, 1]).all():
        raise ValueError("Unexpected SECOM labels")
    if np.isinf(X.to_numpy()).any():
        raise ValueError("Infinite raw feature values")
    # UCI: -1 = pass, 1 = fail -> ML: 0 = pass, 1 = fail
    y = raw_y.map({-1: 0, 1: 1}).rename("is_fail")
    timestamps = (
        labels.iloc[:, 1:].astype(str).agg(" ".join, axis=1).rename("timestamp")
    )

    if len(X) != len(y):
        raise ValueError(f"Feature/label row mismatch: X={len(X)}, y={len(y)}")

    return SecomDataset(X=X, y=y, timestamps=timestamps)


def make_demo_dataset(
    n_samples: int = 500, n_features: int = 60, random_state: int = 42
) -> SecomDataset:
    """Offline smoke-test data. Not used for portfolio metrics."""
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, n_features))
    i1, i2, i3 = 3, min(11, n_features - 1), min(25, n_features - 1)
    logits = 1.6 * X[:, i1] - 1.2 * X[:, i2] + 0.8 * X[:, i3] - 3.2
    p = 1 / (1 + np.exp(-logits))
    y = (rng.random(n_samples) < p).astype(int)
    # Add missingness and a constant column to exercise preprocessing.
    miss = rng.random(X.shape) < 0.05
    X[miss] = np.nan
    X[:, 0] = 1.0
    frame = pd.DataFrame(X, columns=[f"f_{i:03d}" for i in range(n_features)])
    ts = pd.Series(
        pd.date_range("2026-01-01", periods=n_samples, freq="min").astype(str),
        name="timestamp",
    )
    return SecomDataset(frame, pd.Series(y, name="is_fail"), ts)
