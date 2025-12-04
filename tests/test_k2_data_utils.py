import numpy as np
import pandas as pd
import pytest

from k2_system.utils.data_utils import (
    K2DataLoader,
    build_physical_target,
)


def test_build_physical_target_recognizes_bounds(k2_sample_df):
    labels = build_physical_target(k2_sample_df)
    assert labels.tolist() == [1, 1, 0, 0]


def test_prepare_target_derives_when_missing_column(k2_sample_df):
    loader = K2DataLoader()
    target = loader.prepare_target(k2_sample_df)
    assert loader.target_column == 'k2_physical_target'
    assert set(np.unique(target)) <= {0, 1}


def test_get_processed_data_returns_aligned_frames(k2_sample_df):
    loader = K2DataLoader()
    X, y = loader.get_processed_data(k2_sample_df)
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == len(loader.feature_columns)
    assert X.shape[0] == len(k2_sample_df)
