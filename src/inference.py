import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .config import ARTIFACTS


class Predictor:
    def __init__(self, artifact_dir=ARTIFACTS):
        artifact_dir = Path(artifact_dir)
        bundle = joblib.load(artifact_dir / "model.joblib")
        self.pipeline = bundle["pipeline"]
        self.feature_filter = bundle["feature_filter"]
        self.metadata = json.loads((artifact_dir / "metadata.json").read_text())

    def effects(self, frame):
        """Local median replacement attribution. Positive = observed value raises score.
        Not additive SHAP; correlated features and interactions limit interpretation.
        """
        baseline = self.pipeline.predict_proba(frame)[:, 1]
        medians = self.pipeline.named_steps["prep"].named_steps["imputer"].statistics_
        effects = np.zeros((len(frame), len(frame.columns)))
        for i, col in enumerate(frame.columns):
            changed = frame.copy()
            changed[col] = medians[i]
            effects[:, i] = baseline - self.pipeline.predict_proba(changed)[:, 1]
        return effects

    def predict_one(self, row, top_k=5):
        unknown = set(row) - set(self.metadata["raw_columns"])
        if unknown:
            raise ValueError("Unknown features: " + str(sorted(unknown)[:5]))
        if not row or not any(
            v is not None and pd.notna(v)
            for k, v in row.items()
            if k in self.feature_filter.kept_columns
        ):
            raise ValueError("At least one observed retained feature is required")
        if any(
            v is not None and not pd.isna(v) and not np.isfinite(v)
            for v in row.values()
        ):
            raise ValueError("Infinite values are not allowed")
        frame = (
            pd.DataFrame([row])
            .reindex(columns=self.feature_filter.kept_columns)
            .astype(float)
        )
        prob = float(self.pipeline.predict_proba(frame)[0, 1])
        # Batch all feature replacements for a single sample, avoiding hundreds of model calls.
        changed = pd.DataFrame(
            np.repeat(frame.to_numpy(), len(frame.columns), axis=0),
            columns=frame.columns,
        )
        med = self.pipeline.named_steps["prep"].named_steps["imputer"].statistics_
        arr = changed.to_numpy(copy=True)
        np.fill_diagonal(arr, med)
        effects = (
            prob
            - self.pipeline.predict_proba(pd.DataFrame(arr, columns=frame.columns))[
                :, 1
            ]
        )
        order = np.argsort(np.abs(effects))[::-1][:top_k]
        return {
            "predicted_fail": int(prob >= self.metadata["threshold"]),
            "risk_score": prob,
            "threshold": self.metadata["threshold"],
            "top_features": [
                {
                    "feature": frame.columns[i],
                    "contribution": float(effects[i]),
                    "observed_value": None
                    if pd.isna(frame.iloc[0, i])
                    else float(frame.iloc[0, i]),
                }
                for i in order
            ],
            "attribution_method": "local median replacement (score difference), root-cause candidate only",
            "missing_fraction": float(frame.isna().mean(axis=1).iloc[0]),
            "model_version": self.metadata["model_version"],
            "model_name": self.metadata["selected_model"],
        }
