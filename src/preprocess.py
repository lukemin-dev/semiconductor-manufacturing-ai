from __future__ import annotations

from dataclasses import dataclass
from typing import List
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class FeatureFilter:
    kept_columns: List[str]
    dropped_high_missing: List[str]
    dropped_constant: List[str]


def fit_feature_filter(
    X_train: pd.DataFrame, max_missing_ratio: float = 0.40
) -> FeatureFilter:
    missing_ratio = X_train.isna().mean()
    high_missing = missing_ratio[missing_ratio > max_missing_ratio].index.tolist()
    candidate = X_train.drop(columns=high_missing)

    nunique = candidate.nunique(dropna=True)
    constant = nunique[nunique <= 1].index.tolist()
    kept = [c for c in candidate.columns if c not in constant]

    return FeatureFilter(kept, high_missing, constant)


def apply_feature_filter(
    X: pd.DataFrame, feature_filter: FeatureFilter
) -> pd.DataFrame:
    missing = [c for c in feature_filter.kept_columns if c not in X.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing[:10]}")
    return X[feature_filter.kept_columns].copy()


def build_scaled_preprocessor() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def build_tree_preprocessor() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )


from sklearn.base import BaseEstimator, TransformerMixin


class FoldFeatureFilter(TransformerMixin, BaseEstimator):
    """Learn missing/constant filters independently inside each CV training fold."""

    def __init__(self, max_missing_ratio=0.4):
        self.max_missing_ratio = max_missing_ratio

    def fit(self, X, y=None):
        self.feature_filter_ = fit_feature_filter(X, self.max_missing_ratio)
        self.feature_names_in_ = X.columns.to_numpy()
        self.n_features_in_ = X.shape[1]
        if not self.feature_filter_.kept_columns:
            raise ValueError("No usable features in training fold")
        return self

    def transform(self, X):
        return apply_feature_filter(X, self.feature_filter_)

    def get_feature_names_out(self, input_features=None):
        import numpy as np

        return np.asarray(self.feature_filter_.kept_columns)
