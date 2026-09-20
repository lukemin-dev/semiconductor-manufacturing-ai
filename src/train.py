"""Frozen train/validation/test experiment; test is never used for selection."""

import json
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    IsolationForest,
    HistGradientBoostingClassifier,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import precision_recall_curve
from .config import ARTIFACTS
from .preprocess import (
    fit_feature_filter,
    apply_feature_filter,
    build_scaled_preprocessor,
)
from .metrics import classification_metrics


def choose_threshold(y, scores):
    p, r, t = precision_recall_curve(y, scores)
    f2 = 5 * p[:-1] * r[:-1] / (4 * p[:-1] + r[:-1] + 1e-12)
    return float(t[np.argmax(f2)])


def train_all(dataset, output_dir=ARTIFACTS):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    X, y = dataset.X, dataset.y
    trainval, test = train_test_split(
        X.index, test_size=0.25, stratify=y, random_state=42
    )
    train, val = train_test_split(
        trainval, test_size=1 / 3, stratify=y.loc[trainval], random_state=42
    )
    splits = {
        "train": list(map(int, train)),
        "validation": list(map(int, val)),
        "test": list(map(int, test)),
    }
    ff = fit_feature_filter(X.loc[train], 0.4)
    frames = {k: apply_feature_filter(X.loc[v], ff) for k, v in splits.items()}
    models = {}
    for weighted in (False, True):
        suffix = "balanced" if weighted else "unweighted"
        models["logistic_" + suffix] = LogisticRegression(
            max_iter=3000,
            class_weight="balanced" if weighted else None,
            random_state=42,
        )
        models["random_forest_" + suffix] = RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=2,
            class_weight="balanced_subsample" if weighted else None,
            random_state=42,
            n_jobs=4,
        )
    try:
        from xgboost import XGBClassifier

        for weighted in (False, True):
            models["xgboost_" + ("balanced" if weighted else "unweighted")] = (
                XGBClassifier(
                    n_estimators=180,
                    max_depth=3,
                    learning_rate=0.04,
                    subsample=0.85,
                    colsample_bytree=0.75,
                    n_jobs=4,
                    random_state=42,
                    eval_metric="logloss",
                )
            )
        fallback = None
    except (ImportError, OSError) as exc:
        models["hist_gradient_boosting_balanced"] = HistGradientBoostingClassifier(
            max_iter=180, max_leaf_nodes=15, class_weight="balanced", random_state=42
        )
        fallback = str(exc)
    fitted, validation, thresholds = {}, {}, {}
    for name, model in models.items():
        pipe = Pipeline([("prep", build_scaled_preprocessor()), ("model", model)])
        kw = (
            {"model__sample_weight": compute_sample_weight("balanced", y.loc[train])}
            if name == "xgboost_balanced"
            else {}
        )
        pipe.fit(frames["train"], y.loc[train], **kw)
        scores = pipe.predict_proba(frames["validation"])[:, 1]
        threshold = choose_threshold(y.loc[val], scores)
        validation[name] = classification_metrics(y.loc[val], scores, threshold)
        thresholds[name] = threshold
        fitted[name] = pipe
    # Choose supervised model by validation AP, then F1. Freeze before test access.
    selected = max(fitted, key=lambda n: (validation[n]["pr_auc"], validation[n]["f1"]))
    iso = Pipeline(
        [
            ("prep", build_scaled_preprocessor()),
            ("model", IsolationForest(n_estimators=250, random_state=42, n_jobs=4)),
        ]
    )
    iso.fit(frames["train"].loc[y.loc[train] == 0])
    val_scores = -iso.decision_function(frames["validation"])
    thresholds["isolation_forest"] = choose_threshold(y.loc[val], val_scores)
    validation["isolation_forest"] = classification_metrics(
        y.loc[val], val_scores, thresholds["isolation_forest"]
    )
    fitted["isolation_forest"] = iso
    metrics, prediction_rows = {}, []
    for name, pipe in fitted.items():
        scores = (
            -pipe.decision_function(frames["test"])
            if name == "isolation_forest"
            else pipe.predict_proba(frames["test"])[:, 1]
        )
        metrics[name] = classification_metrics(y.loc[test], scores, thresholds[name])
        for idx, score in zip(test, scores):
            prediction_rows.append(
                {
                    "row_index": int(idx),
                    "sample_id": f"SECOM-{idx:04d}",
                    "lot_id": f"DEMO-{idx // 25:03d}",
                    "timestamp": str(dataset.timestamps.loc[idx]),
                    "actual_fail": int(y.loc[idx]),
                    "risk_score": float(score),
                    "predicted_fail": int(score >= thresholds[name]),
                    "model_name": name,
                }
            )
    metrics["always_pass"] = classification_metrics(
        y.loc[test], np.zeros(len(test)), 0.5
    )
    best = fitted[selected]
    version = datetime.now(timezone.utc).strftime("secom-%Y%m%d-%H%M%S")
    metadata = {
        "model_version": version,
        "selected_model": selected,
        "threshold": thresholds[selected],
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_rows": len(X),
        "n_features_raw": X.shape[1],
        "n_features_kept": len(ff.kept_columns),
        "fail_count": int(y.sum()),
        "fail_rate": float(y.mean()),
        "raw_columns": list(X.columns),
        "kept_columns": ff.kept_columns,
        "dropped_high_missing": ff.dropped_high_missing,
        "dropped_constant": ff.dropped_constant,
        "selection": "validation average precision; threshold maximizes validation F2",
        "score_note": "Uncalibrated classifier score, not validated defect probability",
        "lot_note": "Synthetic DEMO grouping of 25 row IDs; no real lot metadata",
        "fallback": fallback,
    }
    bundle = {"pipeline": best, "feature_filter": ff}
    joblib.dump(bundle, output_dir / "model.joblib")
    joblib.dump(iso, output_dir / "isolation_forest.joblib")
    pd.DataFrame(prediction_rows).to_csv(
        output_dir / "all_model_predictions.csv", index=False
    )
    pd.DataFrame(prediction_rows).query("model_name == @selected").to_csv(
        output_dir / "test_predictions.csv", index=False
    )
    audit = {
        "splits": {
            k: {"n": len(v), "fails": int(y.loc[v].sum())} for k, v in splits.items()
        },
        "overlap": len(
            set(train) & set(val) | set(train) & set(test) | set(val) & set(test)
        ),
        "overall_missing_fraction": float(X.isna().mean().mean()),
        "duplicate_rows": int(X.duplicated().sum()),
        "feature_filter_fit": "train only",
        "imputer_scaler_fit": "train only; IsolationForest normal train only",
        "test_used_for_selection": False,
    }
    for name, value in [
        ("metadata", metadata),
        ("metrics", metrics),
        ("validation_metrics", validation),
        ("splits", splits),
        ("data_audit", audit),
    ]:
        (output_dir / f"{name}.json").write_text(
            json.dumps(value, ensure_ascii=False, indent=2)
        )
    pd.DataFrame(
        {
            "feature": X.columns,
            "train_missing_fraction": X.loc[train].isna().mean().values,
            "train_unique_values": X.loc[train].nunique().values,
        }
    ).to_csv(output_dir / "feature_audit.csv", index=False)
    from .inference import Predictor

    predictor = Predictor(output_dir)
    explanations = []
    for idx in test:
        result = predictor.predict_one(X.loc[idx].to_dict())
        for rank, item in enumerate(result["top_features"], 1):
            explanations.append({"row_index": int(idx), "rank": rank, **item})
    pd.DataFrame(explanations).to_csv(
        output_dir / "local_attributions.csv", index=False
    )
    # Global average absolute local replacement effect on held-out test (descriptive only).
    effects = predictor.effects(frames["test"])
    pd.DataFrame(
        {"feature": ff.kept_columns, "importance": np.abs(effects).mean(axis=0)}
    ).sort_values("importance", ascending=False).to_csv(
        output_dir / "feature_importance.csv", index=False
    )
    return {"metadata": metadata, "metrics": metrics, "best_model": selected}
