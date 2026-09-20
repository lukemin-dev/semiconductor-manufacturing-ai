from src.metrics import classification_metrics


def test_metrics_contains_fail_recall():
    y = [0, 0, 1, 1]
    p = [0.1, 0.6, 0.7, 0.9]
    m = classification_metrics(y, p, 0.5)
    assert 0 <= m["fail_recall"] <= 1
    assert m["tp"] == 2
