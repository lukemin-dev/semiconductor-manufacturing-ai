import json
from src.data import make_demo_dataset
from src.train import train_all


def test_train_smoke(tmp_path):
    result = train_all(make_demo_dataset(n_samples=220, n_features=24), tmp_path)
    val = json.loads((tmp_path / "validation_metrics.json").read_text())
    expected = max(
        (n for n in val if n != "isolation_forest"),
        key=lambda n: (val[n]["pr_auc"], val[n]["f1"]),
    )
    assert result["best_model"] == expected
    splits = json.loads((tmp_path / "splits.json").read_text())
    assert not set(splits["train"]) & set(splits["test"])
    assert not set(splits["validation"]) & set(splits["test"])
    assert (tmp_path / "local_attributions.csv").exists()
