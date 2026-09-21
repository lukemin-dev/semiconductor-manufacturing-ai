"""Behavioral checks for v3 feature ablations and audit boundaries."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from src.research_features import ResearchFeatures
from src.research_v3 import review_metrics


def test_missing_indicator_preserves_dropped_raw_column():
    train = pd.DataFrame(
        {"signal": [1.0, 2.0, 3.0, 4.0], "sparse": [np.nan, np.nan, np.nan, 8.0]}
    )
    prep = clone(ResearchFeatures(missing_indicators=True)).fit(train)
    assert list(prep.get_feature_names_out()) == ["signal", "missing__sparse"]
    # Validation-only missingness must not change the learned schema or median.
    future = pd.DataFrame({"signal": [np.nan, 99.0], "sparse": [7.0, np.nan]})
    np.testing.assert_allclose(prep.transform(future), [[2.5, 0.0], [99.0, 1.0]])


def test_correlation_filter_is_frozen_at_fit():
    train = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 8], "c": [0, 1, 0, 1]})
    prep = ResearchFeatures(correlation_threshold=0.98).fit(train)
    assert list(prep.get_feature_names_out()) == ["a", "c"]
    # b may become different in validation, but it cannot re-enter the schema.
    future = pd.DataFrame({"a": [4], "b": [-1000], "c": [1]})
    np.testing.assert_allclose(prep.transform(future), [[4, 1]])


def test_review_queue_respects_budget_and_tie_order():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    result = review_metrics(y, np.ones(10))
    assert result["review_count_10"] == 1
    assert result["recall_at_10"] == 0.5
    assert result["review_count_30"] == 3
    assert result["recall_at_30"] == 1.0
    for invalid in (0, -1, 1.1):
        with pytest.raises(ValueError):
            review_metrics(y, np.ones(10), budgets=(invalid,))


def test_v3_audit_excludes_historical_test():
    path = Path("experiments/v3/audit.json")
    if not path.exists():
        pytest.skip("V3 experiment has not been run")
    audit = json.loads(path.read_text())
    split = json.loads(Path("artifacts/splits.json").read_text())
    development, historical = set(split["development"]), set(split["test"])
    outer_seen = []
    for fold in audit["outer_folds"]:
        fit, assess = set(fold["train"]), set(fold["validation"])
        assert fit | assess == development
        assert not fit & assess
        assert not (fit | assess) & historical
        outer_seen.extend(assess)
        for inner in fold["inner"]:
            inner_fit, inner_valid = set(inner["train"]), set(inner["validation"])
            assert inner_fit | inner_valid == fit
            assert not inner_fit & inner_valid
            assert not (inner_fit | inner_valid) & assess
    assert len(outer_seen) == len(set(outer_seen)) == len(development)
    for window in audit["temporal_windows"]:
        groups = [set(window[key]) for key in ("train", "calibration", "assessment")]
        assert not groups[0] & groups[1] and not groups[0] & groups[2]
        assert not groups[1] & groups[2]
        assert not set.union(*groups) & historical
    assert audit["historical_test_rows_evaluated"] == 0
    assert audit["v2_artifacts_unchanged"]
    for name, expected in audit["protected_v2_artifact_sha256"].items():
        actual = hashlib.sha256((Path("artifacts") / name).read_bytes()).hexdigest()
        assert actual == expected
