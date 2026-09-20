"""Threshold policies learned exclusively from development predictions."""

import numpy as np
from sklearn.metrics import precision_recall_curve


def select_threshold(y, scores, max_alert_rate=0.30):
    if not 0 < max_alert_rate <= 1:
        raise ValueError("max_alert_rate must be in (0, 1]")
    y, scores = np.asarray(y), np.asarray(scores)
    if len(y) != len(scores) or not len(y) or not np.isfinite(scores).all():
        raise ValueError("Expected nonempty aligned finite scores")
    precision, recall, thresholds = precision_recall_curve(y, scores)
    sorted_scores = np.sort(scores)
    rates = (
        len(scores) - np.searchsorted(sorted_scores, thresholds, side="left")
    ) / len(scores)
    f2 = 5 * precision[:-1] * recall[:-1] / (4 * precision[:-1] + recall[:-1] + 1e-12)
    eligible = np.flatnonzero(rates <= max_alert_rate)
    if not len(eligible):
        return float(np.nextafter(scores.max(), np.inf))
    # In an F2 tie, prefer the larger threshold / smaller review queue.
    best = max(eligible, key=lambda i: (f2[i], thresholds[i]))
    return float(thresholds[best])
