"""Fold-local feature ablations for the v3 research experiment."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.utils.validation import check_is_fitted

from .preprocess import fit_feature_filter


class ResearchFeatures(TransformerMixin, BaseEstimator):
    """Median-imputed values plus optional raw missingness indicators.

    The column filter, imputer, indicator schema and correlation pruning are all
    learned from fit rows. Indicators can preserve missingness from raw columns
    whose numeric values are removed by the high-missingness filter.
    """

    def __init__(self, missing_indicators=False, correlation_threshold=None):
        self.missing_indicators = missing_indicators
        self.correlation_threshold = correlation_threshold

    def fit(self, X, y=None):
        self.feature_filter_ = fit_feature_filter(X, max_missing_ratio=0.4)
        self.value_columns_ = self.feature_filter_.kept_columns
        if not self.value_columns_:
            raise ValueError("No usable numeric columns in training data")
        self.imputer_ = SimpleImputer(strategy="median")
        values = self.imputer_.fit_transform(X[self.value_columns_])
        self.value_indices_ = np.arange(values.shape[1])
        if self.correlation_threshold is not None:
            if not 0 < self.correlation_threshold <= 1:
                raise ValueError("correlation_threshold must be in (0, 1]")
            correlation = np.abs(np.corrcoef(values, rowvar=False))
            if values.shape[1] > 1:
                kept = []
                for index in range(values.shape[1]):
                    if not kept or not np.any(
                        correlation[index, kept] >= self.correlation_threshold
                    ):
                        kept.append(index)
                self.value_indices_ = np.asarray(kept)
        missing_rate = X.isna().mean()
        self.indicator_columns_ = (
            list(missing_rate[(missing_rate > 0) & (missing_rate < 1)].index)
            if self.missing_indicators
            else []
        )
        self.feature_names_out_ = np.asarray(
            [self.value_columns_[i] for i in self.value_indices_]
            + [f"missing__{column}" for column in self.indicator_columns_]
        )
        return self

    def transform(self, X):
        check_is_fitted(self, "imputer_")
        values = self.imputer_.transform(X[self.value_columns_])
        values = values[:, self.value_indices_]
        if self.indicator_columns_:
            flags = X[self.indicator_columns_].isna().to_numpy(dtype=float)
            values = np.column_stack([values, flags])
        return values

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()
