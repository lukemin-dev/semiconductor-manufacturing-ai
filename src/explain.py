from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def permutation_feature_importance(
    model, X, y, feature_names, n_repeats=5, random_state=42
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        X,
        y,
        scoring="average_precision",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    return df.reset_index(drop=True)


def top_contributors_linear(model, transformed_row, feature_names, top_k=5):
    coef = np.ravel(model.coef_)
    row = np.ravel(transformed_row)
    contributions = coef * row
    idx = np.argsort(np.abs(contributions))[::-1][:top_k]
    return [
        {"feature": str(feature_names[i]), "contribution": float(contributions[i])}
        for i in idx
    ]


def try_shap_top_features(model, transformed_row, feature_names, top_k=5):
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(transformed_row)
        if isinstance(values, list):
            values = values[-1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, -1]
        vals = values[0]
        idx = np.argsort(np.abs(vals))[::-1][:top_k]
        return [
            {"feature": str(feature_names[i]), "contribution": float(vals[i])}
            for i in idx
        ]
    except Exception:
        return []
