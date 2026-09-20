from __future__ import annotations

from typing import Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def classification_metrics(y_true, y_prob, threshold: float = 0.5) -> Dict:
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "fail_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "fail_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob))
        if len(set(y_true)) > 1
        else None,
        "pr_auc": float(average_precision_score(y_true, y_prob))
        if len(set(y_true)) > 1
        else None,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return out
