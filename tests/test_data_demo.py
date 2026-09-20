from src.data import make_demo_dataset
from src.preprocess import fit_feature_filter


def test_demo_dataset_has_missing_and_label():
    ds = make_demo_dataset(n_samples=120, n_features=20)
    assert ds.X.shape == (120, 20)
    assert set(ds.y.unique()).issubset({0, 1})
    ff = fit_feature_filter(ds.X)
    assert "f_000" in ff.dropped_constant
