import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def k2_sample_df():
    data = {
        'pl_orbper': [10.0, 150.0, 620.0, 40.0],
        'pl_rade': [2.3, 5.1, 2.8, 30.0],
        'pl_eqt': [900, 700, 400, 1500],
        'pl_trandep': [500, 800, 1200, 600],
        'pl_trandur': [2.1, 3.5, 5.0, 1.2],
        'pl_imppar': [0.2, 0.3, 0.5, 0.1],
        'st_teff': [5500, 6200, 9000, 4500],
        'st_rad': [1.0, 1.2, 0.05, 11.0],
        'st_mass': [1.0, 0.9, 4.5, 6.0],
        'st_logg': [4.4, 4.3, 3.8, 3.5],
        'sy_dist': [100, 350, 2500, 1200],
        'sy_gaiamag': [11.3, 12.1, 13.4, 14.0],
        'sy_kepmag': [12.0, 13.2, 15.5, 16.1],
        'sy_tmag': [11.5, 12.6, 14.8, 15.2],
    }
    return pd.DataFrame(data)


@pytest.fixture
def koi_sample_df():
    values = {
        'koi_period': [10, 20, 30, 15, 40, 50],
        'koi_depth': [0.02, 0.01, 0.03, 0.025, 0.04, 0.05],
        'koi_duration': [2.5, 3.0, 4.0, 3.5, 2.0, 5.0],
        'koi_prad': [1.2, 2.5, 1.8, 2.2, 1.5, 2.8],
        'koi_teq': [800, 900, 750, 820, 960, 700],
        'koi_insol': [150, 200, 180, 160, 140, 130],
        'koi_steff': [5500, 5600, 5700, 5400, 5300, 5200],
        'koi_slogg': [4.3, 4.2, 4.4, 4.1, 4.0, 4.5],
        'koi_srad': [1.0, 1.1, 0.9, 1.2, 1.3, 0.8],
        'koi_pdisposition': ['CANDIDATE', 'FALSE POSITIVE', 'CANDIDATE', 'FALSE POSITIVE', 'CANDIDATE', 'FALSE POSITIVE'],
    }
    return pd.DataFrame(values)
