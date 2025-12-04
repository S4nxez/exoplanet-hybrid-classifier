import numpy as np
import pandas as pd

from toi_system.utils.config import TargetConfig
from toi_system.utils import data_utils


def test_prepare_dataset_returns_binary_targets():
    X, y, features, meta = data_utils.prepare_dataset()
    assert X.shape[0] == y.shape[0] > 0
    assert X.shape[1] == len(features)
    assert set(np.unique(y)).issubset({0, 1})
    assert meta["positives"] + meta["negatives"] == meta["samples"]


def test_build_target_vector_maps_dispositions():
    df = pd.DataFrame({
        TargetConfig.COLUMN: ['CP', 'FP', 'UNKNOWN', None]
    })
    y = data_utils.build_target_vector(df)
    assert (y[:1] == 1).all()
    assert (y[1:2] == 0).all()
    assert y.shape[0] == 4
