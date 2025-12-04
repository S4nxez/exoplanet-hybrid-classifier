import numpy as np

from koi_system.core.director import KOIDirector
from koi_system.config.koi_config import KOIConfig
from k2_system.models.k2_director import K2Director, K2_FEATURE_COLUMNS
from toi_system.core.director import TOIDirector
from toi_system.utils.config import TOIFeatures


class IdentityScaler:
    def transform(self, X):
        return np.asarray(X)


class MockRFModel:
    def __init__(self, prob_fn=None, constant=0.6):
        self.prob_fn = prob_fn
        self.constant = constant

    def _prob(self, X):
        if self.prob_fn is not None:
            return np.clip(self.prob_fn(np.asarray(X)), 0.001, 0.999)
        return np.full(np.asarray(X).shape[0], self.constant)

    def predict_proba(self, X):
        probs = self._prob(X)
        return np.vstack([1 - probs, probs]).T

    def predict(self, X):
        probs = self._prob(X)
        return (probs > 0.5).astype(int)


class MockTFModel:
    def __init__(self, prob_fn=None, constant=0.4):
        self.prob_fn = prob_fn
        self.constant = constant

    def _prob(self, X):
        if self.prob_fn is not None:
            return np.clip(self.prob_fn(np.asarray(X)), 0.001, 0.999)
        return np.full(np.asarray(X).shape[0], self.constant)

    def predict(self, X, verbose=0):
        probs = self._prob(X)
        return probs.reshape(-1, 1)


def build_vector(feature_names, values):
    vector = np.zeros(len(feature_names), dtype=float)
    for idx, name in enumerate(feature_names):
        if name in values:
            vector[idx] = values[name]
    return vector


def test_koi_director_prefers_random_forest_with_high_confidence():
    director = KOIDirector()
    scaler = IdentityScaler()
    director.configure(
        rf_model=MockRFModel(constant=0.9),
        rf_scaler=scaler,
        tf_model=MockTFModel(constant=0.55),
        tf_scaler=scaler
    )

    sample = build_vector(
        KOIConfig.FEATURES,
        {
            'koi_period': 10,
            'koi_duration': 3,
            'koi_depth': 0.002,
            'koi_prad': 1.2,
            'koi_teq': 800
        }
    )

    preds, choice = director.predict(sample, return_model_choice=True)
    assert choice == 'RandomForest'
    assert preds.shape == (1,)


def test_koi_director_routes_complex_cases_to_tensorflow():
    director = KOIDirector()
    scaler = IdentityScaler()
    director.configure(
        rf_model=MockRFModel(constant=0.55),
        rf_scaler=scaler,
        tf_model=MockTFModel(constant=0.52),
        tf_scaler=scaler
    )

    sample = build_vector(
        KOIConfig.FEATURES,
        {
            'koi_period': 160,
            'koi_duration': 12,
            'koi_depth': 0.0002,
            'koi_teq': 1900
        }
    )

    _, choice = director.predict(sample, return_model_choice=True)
    assert choice == 'TensorFlow'
    assert director.decision_stats['TensorFlow'] == 1


def test_k2_director_vectorized_selection_returns_per_sample_choices():
    director = K2Director()
    scaler = IdentityScaler()
    director.configure(
        rf_model=MockRFModel(constant=0.55),
        rf_scaler=scaler,
        tf_model=MockTFModel(constant=0.52),
        tf_scaler=scaler
    )

    samples = np.vstack([
        build_vector(
            K2_FEATURE_COLUMNS,
            {
                'pl_orbper': 5,
                'pl_rade': 2,
                'st_teff': 5000,
                'st_rad': 0.9,
                'st_mass': 0.8,
                'sy_dist': 200,
                'pl_eqt': 1200,
                'pl_orbsmax': 0.1
            }
        ),
        build_vector(
            K2_FEATURE_COLUMNS,
            {
                'pl_orbper': 280,
                'pl_rade': 4,
                'st_teff': 6500,
                'st_rad': 1.4,
                'st_mass': 1.3,
                'sy_dist': 900,
                'pl_eqt': 500,
                'pl_orbsmax': 2.5
            }
        )
    ])

    preds, choices = director.predict(samples, return_model_choice=True)
    assert preds.shape == (2,)
    assert choices == ['RandomForest', 'TensorFlow']
    assert director.decision_stats['RandomForest'] == 1
    assert director.decision_stats['TensorFlow'] == 1


def test_toi_director_gates_based_on_complexity():
    director = TOIDirector()
    scaler = IdentityScaler()
    director.configure(
        rf_model=MockRFModel(constant=0.65),
        rf_scaler=scaler,
        tf_model=MockTFModel(constant=0.55),
        tf_scaler=scaler,
    )

    easy_case = build_vector(
        TOIFeatures.CORE,
        {
            'pl_orbper': 5,
            'pl_trandurh': 1.5,
            'pl_trandep': 800,
            'pl_rade': 1.2,
            'pl_insol': 150,
            'pl_eqt': 500,
            'st_tmag': 8,
            'st_dist': 40,
            'st_teff': 5200,
            'st_logg': 4.5,
            'st_rad': 0.9,
        },
    )

    complex_case = build_vector(
        TOIFeatures.CORE,
        {
            'pl_orbper': 400,
            'pl_trandurh': 15,
            'pl_trandep': 50,
            'pl_rade': 4,
            'pl_insol': 20,
            'pl_eqt': 2000,
            'st_tmag': 15,
            'st_dist': 400,
            'st_teff': 7800,
            'st_logg': 3.8,
            'st_rad': 2.5,
        },
    )

    samples = np.vstack([easy_case, complex_case])
    _, choices = director.predict(samples, return_model_choice=True)
    assert choices == ['RandomForest', 'TensorFlow']