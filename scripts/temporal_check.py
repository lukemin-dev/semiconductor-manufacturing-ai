"""Chronological sensitivity check, not a previously unseen external validation."""

import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from src.data import load_secom
from src.experiment import candidates, fit_candidate, score_candidate, bootstrap_metrics
from src.thresholds import select_threshold
from src.metrics import classification_metrics


def main():
    ds = load_secom()
    meta = json.loads((ROOT / "artifacts/selection_decision.json").read_text())
    timestamps = pd.to_datetime(
        ds.timestamps.str.replace(r"\s+:", ":", regex=True),
        format="%d/%m/%Y %H:%M:%S",
        errors="raise",
    )
    unique = sorted(timestamps.unique())
    cut1 = unique[int(len(unique) * 0.6)]
    cut2 = unique[int(len(unique) * 0.8)]
    groups = {
        "train": timestamps.index[timestamps < cut1].tolist(),
        "validation": timestamps.index[
            (timestamps >= cut1) & (timestamps < cut2)
        ].tolist(),
        "test": timestamps.index[timestamps >= cut2].tolist(),
    }
    name = meta["selected_model"]
    pipe = fit_candidate(
        candidates()[name], name, ds.X.loc[groups["train"]], ds.y.loc[groups["train"]]
    )
    val = score_candidate(pipe, name, ds.X.loc[groups["validation"]])
    threshold = select_threshold(
        ds.y.loc[groups["validation"]], val, meta["max_oof_alert_rate"]
    )
    test = groups["test"]
    scores = score_candidate(pipe, name, ds.X.loc[test])
    result = {
        "model": name,
        "threshold": threshold,
        "method": "chronological 60/20/20 by unique timestamp; no timestamp shared across partitions",
        "status": "sensitivity analysis on previously explored SECOM; not independent external validation",
        "metrics": classification_metrics(ds.y.loc[test], scores, threshold),
        "intervals": bootstrap_metrics(ds.y.loc[test], scores, threshold),
        "splits": {
            k: {
                "n": len(ids),
                "fails": int(ds.y.loc[ids].sum()),
                "start": timestamps.loc[ids].min().isoformat(),
                "end": timestamps.loc[ids].max().isoformat(),
            }
            for k, ids in groups.items()
        },
    }
    assert (
        timestamps.loc[groups["train"]].max()
        < timestamps.loc[groups["validation"]].min()
    )
    assert timestamps.loc[groups["validation"]].max() < timestamps.loc[test].min()
    (ROOT / "artifacts/temporal_metrics.json").write_text(json.dumps(result, indent=2))
    (ROOT / "artifacts/temporal_splits.json").write_text(json.dumps(groups))
    pd.DataFrame(
        {
            "row_index": test,
            "actual_fail": ds.y.loc[test].values,
            "risk_score": scores,
            "predicted_fail": (scores >= threshold).astype(int),
        }
    ).to_csv(ROOT / "artifacts/temporal_predictions.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
