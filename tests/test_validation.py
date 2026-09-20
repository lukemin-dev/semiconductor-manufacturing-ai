import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from src.preprocess import FoldFeatureFilter
from src.thresholds import select_threshold


def test_fold_filter_does_not_see_validation():
    train = pd.DataFrame(
        {"constant": [1, 1, 1, 1], "a": [1, 2, 3, 4], "missing": [None, None, None, 1]}
    )
    validation = pd.DataFrame({"constant": [2], "a": [1000], "missing": [8]})
    fitted = clone(FoldFeatureFilter()).fit(train)
    result = fitted.transform(validation)
    assert list(result.columns) == ["a"]
    assert result.iloc[0, 0] == 1000
    assert fitted.feature_filter_.dropped_constant == ["constant"]


def test_threshold_limits_queue_with_tied_scores():
    y = np.array([0, 1, 0, 1, 0, 0, 0, 1, 0, 0])
    scores = np.array([0.1, 0.8, 0.8, 0.9, 0.2, 0.3, 0.2, 0.7, 0.4, 0.1])
    threshold = select_threshold(y, scores, 0.3)
    assert (scores >= threshold).mean() <= 0.3
    assert np.isfinite(select_threshold(y, np.ones(10), 0.3))
    assert (np.ones(10) >= select_threshold(y, np.ones(10), 0.3)).sum() == 0


def test_cv_and_temporal_manifests():
    root = Path("artifacts")
    if not (root / "cv_splits.json").exists():
        return
    split = json.loads((root / "splits.json").read_text())
    test = set(split["test"])
    dev = set(split["development"])
    seen = {i: 0 for i in dev}
    folds = json.loads((root / "cv_splits.json").read_text())
    for fold in folds:
        train, valid = set(fold["train"]), set(fold["validation"])
        assert not train & valid and not (train | valid) & test
        assert train | valid == dev
        for i in valid:
            seen[i] += 1
    assert set(seen.values()) == {len(folds) // 5}
    temporal = json.loads((root / "temporal_metrics.json").read_text())["splits"]
    assert temporal["train"]["end"] < temporal["validation"]["start"]
    assert temporal["validation"]["end"] < temporal["test"]["start"]
