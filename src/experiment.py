"""Reproducible v2: fold-local preprocessing, repeated CV and bounded alerts.

The historical test is a reused holdout, never an independent new validation.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    RandomForestClassifier,
    IsolationForest,
    HistGradientBoostingClassifier,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from .preprocess import FoldFeatureFilter, build_scaled_preprocessor
from .thresholds import select_threshold
from .metrics import classification_metrics


def candidates():
    models = {
        "logistic_full_C01": (
            LogisticRegression(
                C=0.1, max_iter=3000, class_weight="balanced", random_state=42
            ),
            False,
        ),
        "logistic_top40_C01": (
            LogisticRegression(
                C=0.1, max_iter=3000, class_weight="balanced", random_state=42
            ),
            True,
        ),
        "logistic_top40_C001": (
            LogisticRegression(
                C=0.01, max_iter=3000, class_weight="balanced", random_state=42
            ),
            True,
        ),
        "random_forest_unweighted": (
            RandomForestClassifier(
                n_estimators=250, min_samples_leaf=2, n_jobs=4, random_state=42
            ),
            False,
        ),
        "random_forest_balanced": (
            RandomForestClassifier(
                n_estimators=250,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=4,
                random_state=42,
            ),
            False,
        ),
    }
    try:
        from xgboost import XGBClassifier

        for suffix in ["unweighted", "balanced"]:
            models["xgboost_" + suffix] = (
                XGBClassifier(
                    n_estimators=180,
                    max_depth=3,
                    learning_rate=0.04,
                    subsample=0.85,
                    colsample_bytree=0.75,
                    n_jobs=4,
                    random_state=42,
                    eval_metric="logloss",
                ),
                False,
            )
    except (ImportError, OSError):
        models["hist_gradient_boosting_balanced"] = (
            HistGradientBoostingClassifier(
                max_iter=180,
                max_leaf_nodes=15,
                class_weight="balanced",
                random_state=42,
            ),
            False,
        )
    models["isolation_forest"] = (
        IsolationForest(n_estimators=250, n_jobs=4, random_state=42),
        False,
    )
    return {
        name: Pipeline(
            [
                ("filter", FoldFeatureFilter()),
                ("prep", build_scaled_preprocessor()),
                ("select", SelectKBest(f_classif, k=40) if top40 else "passthrough"),
                ("model", model),
            ]
        )
        for name, (model, top40) in models.items()
    }


def fit_candidate(pipe, name, X, y):
    if name == "isolation_forest":
        return pipe.fit(X.loc[y == 0])
    kwargs = (
        {"model__sample_weight": compute_sample_weight("balanced", y)}
        if name == "xgboost_balanced"
        else {}
    )
    return pipe.fit(X, y, **kwargs)


def score_candidate(pipe, name, X):
    return (
        -pipe.decision_function(X)
        if name == "isolation_forest"
        else pipe.predict_proba(X)[:, 1]
    )


def bootstrap_metrics(y, scores, threshold, repetitions=1000):
    """Stratified bootstrap conditional on this fitted model and class counts."""
    y, scores = np.asarray(y), np.asarray(scores)
    groups = [np.flatnonzero(y == label) for label in [0, 1]]
    rng = np.random.default_rng(20260920)
    samples = {k: [] for k in ["fail_recall", "fail_precision", "f1", "pr_auc"]}
    for _ in range(repetitions):
        indices = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        m = classification_metrics(y[indices], scores[indices], threshold)
        for k in samples:
            samples[k].append(m[k])
    return {
        k: {
            "lower": float(np.quantile(v, 0.025)),
            "upper": float(np.quantile(v, 0.975)),
        }
        for k, v in samples.items()
    }


def run_experiment(dataset, output_dir, split_file, repeats=3, max_alert_rate=0.30):
    if repeats < 1:
        raise ValueError("repeats must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    historical = json.loads(Path(split_file).read_text())
    development = sorted(historical["train"] + historical["validation"])
    test = historical["test"]
    assert not set(development) & set(test)
    X, y = dataset.X, dataset.y
    cv = list(
        RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=42).split(
            X.loc[development], y.loc[development]
        )
    )
    fold_manifest = []
    for fold, (fit_idx, valid_idx) in enumerate(cv):
        fold_manifest.append(
            {
                "fold": fold,
                "train": [development[i] for i in fit_idx],
                "validation": [development[i] for i in valid_idx],
            }
        )
    (output_dir / "cv_splits.json").write_text(json.dumps(fold_manifest))
    folds, oof, feature_choices = [], [], []
    candidates_by_name = candidates()
    for name, template in candidates_by_name.items():
        print("Cross-validation:", name, flush=True)
        for fold_info in fold_manifest:
            train, valid = fold_info["train"], fold_info["validation"]
            pipe = fit_candidate(clone(template), name, X.loc[train], y.loc[train])
            scores = score_candidate(pipe, name, X.loc[valid])
            folds.append(
                {
                    "model": name,
                    "fold": fold_info["fold"],
                    "pr_auc": float(average_precision_score(y.loc[valid], scores)),
                    "n_train": len(train),
                    "n_validation": len(valid),
                    "n_features_kept": len(
                        pipe.named_steps["filter"].feature_filter_.kept_columns
                    ),
                }
            )
            oof.extend(
                {
                    "model": name,
                    "fold": fold_info["fold"],
                    "row_index": int(idx),
                    "actual_fail": int(y.loc[idx]),
                    "risk_score": float(score),
                }
                for idx, score in zip(valid, scores)
            )
            names = np.array(pipe.named_steps["filter"].feature_filter_.kept_columns)
            selector = pipe.named_steps["select"]
            if selector != "passthrough":
                names = names[selector.get_support()]
            feature_choices.extend(
                {"model": name, "fold": fold_info["fold"], "feature": str(f)}
                for f in names
            )
    fold_frame = pd.DataFrame(folds)
    oof_frame = pd.DataFrame(oof)
    summary = (
        fold_frame.groupby("model")
        .pr_auc.agg(["mean", "std", "min", "max"])
        .sort_values("mean", ascending=False)
    )
    supervised = summary.drop(index="isolation_forest")
    selected = supervised.index[0]
    thresholds = {}
    oof_metrics = {}
    for name, group in oof_frame.groupby("model"):
        thresholds[name] = select_threshold(
            group.actual_fail, group.risk_score, max_alert_rate
        )
        oof_metrics[name] = classification_metrics(
            group.actual_fail, group.risk_score, thresholds[name]
        )
        oof_metrics[name]["alert_rate"] = float(
            (group.risk_score >= thresholds[name]).mean()
        )
    # Freeze model/threshold selection before generating any test scores.
    decision = {
        "selected_model": selected,
        "thresholds": thresholds,
        "selection": "maximum mean fold AP on development, excluding IsolationForest",
        "max_oof_alert_rate": max_alert_rate,
        "n_folds": len(cv),
        "test_status": "Previously viewed v1 test; reused exploratory holdout, not new independent validation",
    }
    (output_dir / "selection_decision.json").write_text(json.dumps(decision, indent=2))
    print("Frozen selection:", selected, flush=True)
    fitted = {}
    metrics = {}
    predictions = []
    for name, template in candidates_by_name.items():
        pipe = fit_candidate(
            clone(template), name, X.loc[development], y.loc[development]
        )
        fitted[name] = pipe
        scores = score_candidate(pipe, name, X.loc[test])
        metrics[name] = classification_metrics(y.loc[test], scores, thresholds[name])
        metrics[name]["alert_rate"] = float((scores >= thresholds[name]).mean())
        predictions.extend(
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
            for idx, score in zip(test, scores)
        )
    metrics["always_pass"] = classification_metrics(
        y.loc[test], np.zeros(len(test)), 0.5
    )
    best = fitted[selected]
    ff = best.named_steps["filter"].feature_filter_
    selected_features = np.array(ff.kept_columns)
    if best.named_steps["select"] != "passthrough":
        selected_features = selected_features[best.named_steps["select"].get_support()]
    metadata = {
        "model_version": datetime.now(timezone.utc).strftime("secom-v2-%Y%m%d-%H%M%S"),
        "selected_model": selected,
        "threshold": thresholds[selected],
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_rows": len(X),
        "n_features_raw": X.shape[1],
        "n_features_kept": len(ff.kept_columns),
        "n_features_selected": len(selected_features),
        "selected_features": list(selected_features),
        "fail_count": int(y.sum()),
        "fail_rate": float(y.mean()),
        "raw_columns": list(X.columns),
        "kept_columns": ff.kept_columns,
        "dropped_high_missing": ff.dropped_high_missing,
        "dropped_constant": ff.dropped_constant,
        "selection": decision,
        "score_note": "Uncalibrated score, not validated defect probability",
        "lot_note": "Synthetic grouping, not actual manufacturing lots",
    }
    joblib.dump({"pipeline": best, "feature_filter": ff}, output_dir / "model.joblib")
    joblib.dump(fitted["isolation_forest"], output_dir / "isolation_forest.joblib")
    results = pd.DataFrame(predictions)
    results.to_csv(output_dir / "all_model_predictions.csv", index=False)
    selected_predictions = results.query("model_name == @selected")
    selected_predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    fold_frame.to_csv(output_dir / "cv_fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "cv_summary.csv")
    oof_frame.to_csv(output_dir / "oof_predictions.csv", index=False)
    choices = (
        pd.DataFrame(feature_choices)
        .groupby(["model", "feature"])
        .size()
        .rename("selected_folds")
        .reset_index()
    )
    choices["fraction"] = choices.selected_folds / len(cv)
    choices.to_csv(output_dir / "feature_selection_stability.csv", index=False)
    audit = {
        "splits": {
            "development": {
                "n": len(development),
                "fails": int(y.loc[development].sum()),
            },
            "test": {"n": len(test), "fails": int(y.loc[test].sum())},
        },
        "cv_folds": len(cv),
        "overlap": len(set(development) & set(test)),
        "overall_missing_fraction": float(X.isna().mean().mean()),
        "duplicate_rows": int(X.duplicated().sum()),
        "preprocessing_fit": "Within each fold; final fit development only",
        "test_used_for_selection": False,
        "test_previously_viewed": True,
    }
    intervals = bootstrap_metrics(
        selected_predictions.actual_fail,
        selected_predictions.risk_score,
        thresholds[selected],
    )
    for name, value in [
        ("metadata", metadata),
        ("metrics", metrics),
        ("validation_metrics", oof_metrics),
        ("splits", {"development": development, "test": test}),
        ("data_audit", audit),
        ("confidence_intervals", intervals),
    ]:
        (output_dir / f"{name}.json").write_text(
            json.dumps(value, ensure_ascii=False, indent=2)
        )
    pd.DataFrame(
        {
            "feature": X.columns,
            "development_missing_fraction": X.loc[development].isna().mean().values,
            "development_unique_values": X.loc[development].nunique().values,
        }
    ).to_csv(output_dir / "feature_audit.csv", index=False)
    from .inference import Predictor

    predictor = Predictor(output_dir)
    local = []
    for idx in test:
        for rank, item in enumerate(
            predictor.predict_one(X.loc[idx].to_dict())["top_features"], 1
        ):
            local.append({"row_index": int(idx), "rank": rank, **item})
    pd.DataFrame(local).to_csv(output_dir / "local_attributions.csv", index=False)
    effects = predictor.effects(X.loc[test, ff.kept_columns])
    pd.DataFrame(
        {"feature": ff.kept_columns, "importance": np.abs(effects).mean(axis=0)}
    ).sort_values("importance", ascending=False).to_csv(
        output_dir / "feature_importance.csv", index=False
    )
    print(
        json.dumps(
            {
                "selected": selected,
                "metrics": metrics[selected],
                "confidence_intervals": intervals,
            },
            indent=2,
        ),
        flush=True,
    )
    return metadata
