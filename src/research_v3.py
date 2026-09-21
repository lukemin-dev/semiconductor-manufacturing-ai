"""Development-only v3 experiment. Historical test labels are never evaluated."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from .metrics import classification_metrics
from .research_features import ResearchFeatures
from .thresholds import select_threshold

BASELINE = "rf_weighted_baseline"


def candidate_pipelines():
    configs = {
        BASELINE: (False, None, False, 2),
        "rf_missing": (True, None, False, 2),
        "rf_missing_pruned": (True, 0.98, False, 2),
        "balanced_rf": (False, None, True, 2),
        "balanced_rf_missing": (True, None, True, 2),
        "balanced_rf_missing_leaf5": (True, None, True, 5),
    }
    pipelines = {}
    for name, (missing, correlation, sampled, leaf) in configs.items():
        common = dict(
            n_estimators=250,
            min_samples_leaf=leaf,
            random_state=42,
            n_jobs=4,
            max_features="sqrt",
        )
        model = (
            BalancedRandomForestClassifier(
                sampling_strategy="all", replacement=True, bootstrap=False, **common
            )
            if sampled
            else RandomForestClassifier(class_weight="balanced_subsample", **common)
        )
        pipelines[name] = Pipeline(
            [
                ("features", ResearchFeatures(missing, correlation)),
                ("model", model),
            ]
        )
    return pipelines


def review_metrics(y, scores, budgets=(0.1, 0.2, 0.3)):
    """Ranking diagnostics: inspect exactly floor(budget*n) rows.

    Ties use stable input order, never labels. This is a finite-batch queue,
    not a globally calibrated score threshold or a future capacity guarantee.
    """
    y, scores = np.asarray(y, dtype=int), np.asarray(scores, dtype=float)
    if len(y) != len(scores) or not len(y) or not np.isfinite(scores).all():
        raise ValueError("Expected aligned nonempty finite scores")
    order = np.argsort(-scores, kind="stable")
    result = {}
    for budget in budgets:
        if not 0 < budget <= 1:
            raise ValueError("budgets must be in (0, 1]")
        count = int(np.floor(len(y) * budget))
        found = int(y[order[:count]].sum())
        suffix = str(int(budget * 100))
        result[f"review_count_{suffix}"] = count
        result[f"recall_at_{suffix}"] = found / int(y.sum()) if y.sum() else None
        result[f"precision_at_{suffix}"] = found / count if count else 0.0
    return result


def diagnose_v2(dataset, artifacts, output):
    oof = pd.read_csv(artifacts / "oof_predictions.csv")
    metadata = json.loads((artifacts / "metadata.json").read_text())
    selected = oof[oof.model == metadata["selected_model"]]
    row_scores = selected.groupby("row_index").risk_score.mean().sort_index()
    ids = row_scores.index
    dates = pd.to_datetime(
        dataset.timestamps.loc[ids].str.replace(r"\s+:", ":", regex=True),
        format="%d/%m/%Y %H:%M:%S",
    )
    rows = pd.DataFrame(
        {
            "row_index": ids,
            "actual_fail": dataset.y.loc[ids].to_numpy(),
            "mean_repeated_oof_score": row_scores.to_numpy(),
            "missing_fraction": dataset.X.loc[ids].isna().mean(axis=1).to_numpy(),
            "month": dates.dt.strftime("%Y-%m").to_numpy(),
        }
    )
    rows["alert"] = rows.mean_repeated_oof_score >= metadata["threshold"]
    rows["outcome"] = np.select(
        [
            (rows.actual_fail == 1) & rows.alert,
            (rows.actual_fail == 1) & ~rows.alert,
            (rows.actual_fail == 0) & rows.alert,
        ],
        ["true_positive", "false_negative", "false_positive"],
        default="true_negative",
    )
    rows.to_csv(output / "v2_failure_rows.csv", index=False)
    rows.groupby("outcome").agg(
        n=("row_index", "size"),
        mean_missing=("missing_fraction", "mean"),
        median_missing=("missing_fraction", "median"),
        mean_score=("mean_repeated_oof_score", "mean"),
    ).to_csv(output / "v2_failure_groups.csv")
    monthly = []
    for month, group in rows.groupby("month"):
        fails = group[group.actual_fail == 1]
        monthly.append(
            {
                "month": month,
                "n": len(group),
                "fails": len(fails),
                "missed": int((~fails.alert).sum()),
                "fail_miss_rate": float((~fails.alert).mean()) if len(fails) else None,
            }
        )
    pd.DataFrame(monthly).to_csv(output / "v2_failure_by_month.csv", index=False)
    return rows


def temporal_partitions(timestamps, development):
    """Three expanding train/calibration/assessment windows; keep time ties together."""
    dates = pd.to_datetime(
        timestamps.loc[development].str.replace(r"\s+:", ":", regex=True),
        format="%d/%m/%Y %H:%M:%S",
    )
    unique = np.sort(dates.unique())
    windows = []
    for first, second, end in [(0.4, 0.55, 0.7), (0.55, 0.7, 0.85), (0.7, 0.85, 1.0)]:
        low, high = unique[int(len(unique) * first)], unique[int(len(unique) * second)]
        final_mask = dates >= high
        if end < 1:
            final_mask &= dates < unique[int(len(unique) * end)]
        windows.append(
            {
                "train": dates.index[dates < low].tolist(),
                "calibration": dates.index[(dates >= low) & (dates < high)].tolist(),
                "assessment": dates.index[final_mask].tolist(),
            }
        )
    return windows, dates


def run_v3(dataset, artifacts, output):
    artifacts, output = Path(artifacts), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    split = json.loads((artifacts / "splits.json").read_text())
    development = sorted(split["development"])
    historical_test = set(split["test"])
    assert not historical_test.intersection(development)
    protected_files = [
        "model.joblib",
        "metadata.json",
        "metrics.json",
        "test_predictions.csv",
    ]
    fingerprints = {
        name: hashlib.sha256((artifacts / name).read_bytes()).hexdigest()
        for name in protected_files
    }
    diagnose_v2(dataset, artifacts, output)
    X, y = dataset.X.loc[development], dataset.y.loc[development]
    templates = candidate_pipelines()
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260921)
    manifest, inner_results, decisions, outer_results, predictions = [], [], [], [], []
    nested_predictions = []
    for fold, (train_pos, valid_pos) in enumerate(outer.split(X, y)):
        train_ids, valid_ids = X.index[train_pos], X.index[valid_pos]
        inner = list(
            StratifiedKFold(n_splits=3, shuffle=True, random_state=42 + fold).split(
                X.loc[train_ids], y.loc[train_ids]
            )
        )
        fold_info = {
            "outer_fold": fold,
            "train": train_ids.tolist(),
            "validation": valid_ids.tolist(),
            "inner": [
                {"train": train_ids[a].tolist(), "validation": train_ids[b].tolist()}
                for a, b in inner
            ],
        }
        manifest.append(fold_info)
        choices, thresholds = {}, {}
        fitted_scores = {}
        for name, template in templates.items():
            print(f"Outer fold {fold + 1}/5: {name}", flush=True)
            oof = np.zeros(len(train_ids))
            inner_ap = []
            for inner_fold, (fit_pos, calibration_pos) in enumerate(inner):
                fit_ids, calibration_ids = (
                    train_ids[fit_pos],
                    train_ids[calibration_pos],
                )
                pipe = clone(template).fit(X.loc[fit_ids], y.loc[fit_ids])
                score = pipe.predict_proba(X.loc[calibration_ids])[:, 1]
                oof[calibration_pos] = score
                ap = float(average_precision_score(y.loc[calibration_ids], score))
                inner_ap.append(ap)
                inner_results.append(
                    {
                        "outer_fold": fold,
                        "inner_fold": inner_fold,
                        "model": name,
                        "ap": ap,
                    }
                )
            choices[name] = float(np.mean(inner_ap))
            thresholds[name] = select_threshold(y.loc[train_ids], oof, 0.30)
            pipe = clone(template).fit(X.loc[train_ids], y.loc[train_ids])
            score = pipe.predict_proba(X.loc[valid_ids])[:, 1]
            fitted_scores[name] = score
            measured = classification_metrics(y.loc[valid_ids], score, thresholds[name])
            measured.update(review_metrics(y.loc[valid_ids], score))
            outer_results.append(
                {
                    "outer_fold": fold,
                    "model": name,
                    "n_features": len(
                        pipe.named_steps["features"].get_feature_names_out()
                    ),
                    **measured,
                }
            )
            predictions.extend(
                {
                    "outer_fold": fold,
                    "model": name,
                    "row_index": int(idx),
                    "actual_fail": int(y.loc[idx]),
                    "risk_score": float(value),
                    "threshold": thresholds[name],
                }
                for idx, value in zip(valid_ids, score)
            )
        chosen = max(choices, key=choices.get)
        decisions.append(
            {
                "outer_fold": fold,
                "selected_model": chosen,
                "inner_mean_ap": choices[chosen],
                "threshold": thresholds[chosen],
            }
        )
        nested_predictions.extend(
            {
                "outer_fold": fold,
                "model": chosen,
                "row_index": int(idx),
                "actual_fail": int(y.loc[idx]),
                "risk_score": float(value),
                "threshold": thresholds[chosen],
            }
            for idx, value in zip(valid_ids, fitted_scores[chosen])
        )
    windows, dates = temporal_partitions(dataset.timestamps, development)
    temporal_results, temporal_predictions = [], []
    for window, group in enumerate(windows):
        train, calibration, assess = (
            group["train"],
            group["calibration"],
            group["assessment"],
        )
        assert dates.loc[train].max() < dates.loc[calibration].min()
        assert dates.loc[calibration].max() < dates.loc[assess].min()
        for name, template in templates.items():
            print(f"Temporal window {window + 1}/3: {name}", flush=True)
            pipe = clone(template).fit(X.loc[train], y.loc[train])
            calibration_scores = pipe.predict_proba(X.loc[calibration])[:, 1]
            threshold = select_threshold(y.loc[calibration], calibration_scores, 0.30)
            scores = pipe.predict_proba(X.loc[assess])[:, 1]
            measured = classification_metrics(y.loc[assess], scores, threshold)
            measured.update(review_metrics(y.loc[assess], scores))
            temporal_results.append(
                {
                    "window": window,
                    "model": name,
                    "n_train": len(train),
                    "n_calibration": len(calibration),
                    "n_assessment": len(assess),
                    "assessment_fails": int(y.loc[assess].sum()),
                    "train_end": dates.loc[train].max().isoformat(),
                    "assessment_start": dates.loc[assess].min().isoformat(),
                    "assessment_end": dates.loc[assess].max().isoformat(),
                    **measured,
                }
            )
            temporal_predictions.extend(
                {
                    "window": window,
                    "model": name,
                    "row_index": int(idx),
                    "actual_fail": int(y.loc[idx]),
                    "risk_score": float(value),
                    "threshold": threshold,
                }
                for idx, value in zip(assess, scores)
            )
    outer_frame, temporal_frame = (
        pd.DataFrame(outer_results),
        pd.DataFrame(temporal_results),
    )
    summary = (
        outer_frame.groupby("model")
        .agg(
            outer_ap_mean=("pr_auc", "mean"),
            outer_ap_std=("pr_auc", "std"),
            outer_recall_at_10=("recall_at_10", "mean"),
            outer_recall_at_20=("recall_at_20", "mean"),
            outer_recall_at_30=("recall_at_30", "mean"),
        )
        .join(
            temporal_frame.groupby("model").agg(
                temporal_ap_mean=("pr_auc", "mean"),
                temporal_recall_at_30=("recall_at_30", "mean"),
                temporal_threshold_recall=("fail_recall", "mean"),
                temporal_threshold_precision=("fail_precision", "mean"),
            )
        )
    )
    base = summary.loc[BASELINE]
    summary["passes_research_gate"] = (
        (summary.outer_ap_mean >= base.outer_ap_mean + 0.02)
        & (summary.temporal_ap_mean >= base.temporal_ap_mean + 0.02)
        & (summary.temporal_recall_at_30 >= base.temporal_recall_at_30)
    )
    summary.to_csv(output / "comparison.csv")
    for name, value in [
        ("outer_fold_metrics", outer_frame),
        ("inner_fold_metrics", pd.DataFrame(inner_results)),
        ("selection_decisions", pd.DataFrame(decisions)),
        ("outer_predictions", pd.DataFrame(predictions)),
        ("nested_selected_predictions", pd.DataFrame(nested_predictions)),
        ("temporal_metrics", temporal_frame),
        ("temporal_predictions", pd.DataFrame(temporal_predictions)),
    ]:
        value.to_csv(output / f"{name}.csv", index=False)
    audit = {
        "development_rows": len(development),
        "historical_test_rows_evaluated": 0,
        "outer_folds": manifest,
        "temporal_windows": windows,
        "protected_v2_artifact_sha256": fingerprints,
        "v2_artifacts_unchanged": all(
            hashlib.sha256((artifacts / name).read_bytes()).hexdigest() == digest
            for name, digest in fingerprints.items()
        ),
        "candidate_settings": {
            name: {key: str(value) for key, value in pipe.get_params(deep=True).items()}
            for name, pipe in templates.items()
        },
    }
    assert audit["v2_artifacts_unchanged"]
    (output / "audit.json").write_text(json.dumps(audit, indent=2))
    print(summary.to_string(), flush=True)
    return summary
