import numpy as np
import pandas as pd

from koi_system.utils import data_utils
from koi_system.config.koi_config import KOIConfig


def test_load_and_prepare_data_filters_and_encodes(monkeypatch, koi_sample_df):
    def fake_read_csv(path, *args, **kwargs):
        return koi_sample_df.copy()

    monkeypatch.setattr(data_utils.pd, "read_csv", fake_read_csv)

    X_train, X_test, y_train, y_test = data_utils.load_and_prepare_data("dummy.csv")

    total_rows = len(koi_sample_df)
    assert X_train.shape[1] == len(KOIConfig.FEATURES)
    assert set(np.unique(np.concatenate([y_train, y_test]))) == {0, 1}
    assert X_train.shape[0] + X_test.shape[0] == total_rows
    assert y_train.shape[0] + y_test.shape[0] == total_rows


def test_get_feature_names_matches_config():
    assert data_utils.get_feature_names() == KOIConfig.FEATURES
