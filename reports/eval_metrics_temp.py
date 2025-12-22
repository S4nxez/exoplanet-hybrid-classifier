import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from koi_system.config.koi_config import KOIConfig
from koi_system.core.director import KOIDirector
from k2_system.models.k2_ensemble import K2EnsembleSystem
from k2_system.models.k2_director import K2_FEATURE_COLUMNS
from toi_system.core.director import TOIDirector
from toi_system.utils.data_utils import build_feature_matrix, build_target_vector, load_raw_dataset

np.set_printoptions(suppress=True)


def eval_metrics(y_true, proba, pred):
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) > 1 else None,
        "confusion": confusion_matrix(y_true, pred).tolist(),
    }


def evaluate_koi():
    df = pd.read_csv("data/clean/koi_clean.csv")
    X = df[KOIConfig.FEATURES].to_numpy()
    y = df["is_planet"].astype(int).to_numpy()

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = joblib.load("koi_system/saved_models/koi_randomforest_model.pkl")
    rf_scaler = joblib.load("koi_system/saved_models/koi_randomforest_scaler.pkl")
    tf_model = tf.keras.models.load_model("koi_system/saved_models/koi_tensorflow_model.h5")
    tf_scaler = joblib.load("koi_system/saved_models/koi_tensorflow_scaler.pkl")

    rf_proba = rf.predict_proba(rf_scaler.transform(X_test))[:, 1]
    rf_pred = (rf_proba > 0.5).astype(int)

    tf_proba = tf_model.predict(tf_scaler.transform(X_test), verbose=0).flatten()
    tf_pred = (tf_proba > 0.5).astype(int)

    director = KOIDirector()
    director.configure(rf, rf_scaler, tf_model, tf_scaler)
    selections = director.select_expert(X_test)
    if isinstance(selections, str):
        selections = np.array([selections])

    dir_proba = np.zeros_like(rf_proba)
    dir_pred = np.zeros_like(rf_pred)

    rf_mask = selections == "RandomForest"
    tf_mask = selections == "TensorFlow"

    if np.any(rf_mask):
        pr = rf.predict_proba(rf_scaler.transform(X_test[rf_mask]))[:, 1]
        dir_proba[rf_mask] = pr
        dir_pred[rf_mask] = (pr > 0.5).astype(int)
    if np.any(tf_mask):
        pr = tf_model.predict(tf_scaler.transform(X_test[tf_mask]), verbose=0).flatten()
        dir_proba[tf_mask] = pr
        dir_pred[tf_mask] = (pr > 0.5).astype(int)

    return {
        "rf": eval_metrics(y_test, rf_proba, rf_pred),
        "tf": eval_metrics(y_test, tf_proba, tf_pred),
        "director": eval_metrics(y_test, dir_proba, dir_pred),
        "sample_count": int(len(y_test)),
        "positive_rate": float(y_test.mean()),
        "director_usage": {
            "rf": int(rf_mask.sum()),
            "tf": int(tf_mask.sum()),
        },
    }


def evaluate_toi(thresholds=None):
    df = load_raw_dataset("data/clean/toi_full_clean.csv")
    X, _ = build_feature_matrix(df)
    y = build_target_vector(df)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = joblib.load("toi_system/saved_models/rf_model.pkl")
    rf_scaler = joblib.load("toi_system/saved_models/rf_scaler.pkl")
    tf_model = tf.keras.models.load_model("toi_system/saved_models/tf_model.h5")
    tf_scaler = joblib.load("toi_system/saved_models/tf_scaler.pkl")

    rf_proba = rf.predict_proba(rf_scaler.transform(X_test))[:, 1]
    rf_pred = (rf_proba > 0.5).astype(int)

    tf_proba = tf_model.predict(tf_scaler.transform(X_test), verbose=0).flatten()
    tf_pred = (tf_proba > 0.5).astype(int)

    director = TOIDirector()
    if thresholds:
        director.thresholds = thresholds
    director.configure(rf, rf_scaler, tf_model, tf_scaler)
    selections = director.select_expert(X_test)
    if isinstance(selections, str):
        selections = np.array([selections])

    dir_proba = np.zeros_like(rf_proba)
    dir_pred = np.zeros_like(rf_pred)

    rf_mask = selections == "RandomForest"
    tf_mask = selections == "TensorFlow"

    if np.any(rf_mask):
        pr = rf.predict_proba(rf_scaler.transform(X_test[rf_mask]))[:, 1]
        dir_proba[rf_mask] = pr
        dir_pred[rf_mask] = (pr > 0.5).astype(int)
    if np.any(tf_mask):
        pr = tf_model.predict(tf_scaler.transform(X_test[tf_mask]), verbose=0).flatten()
        dir_proba[tf_mask] = pr
        dir_pred[tf_mask] = (pr > 0.5).astype(int)

    return {
        "rf": eval_metrics(y_test, rf_proba, rf_pred),
        "tf": eval_metrics(y_test, tf_proba, tf_pred),
        "director": eval_metrics(y_test, dir_proba, dir_pred),
        "sample_count": int(len(y_test)),
        "positive_rate": float(y_test.mean()),
        "director_usage": {
            "rf": int(rf_mask.sum()),
            "tf": int(tf_mask.sum()),
        },
    }


def evaluate_k2():
    df = pd.read_csv("data/clean/k2_full.csv")

    # Ensure required columns exist and keep negatives by imputing missing features.
    for col in K2_FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    X = df[K2_FEATURE_COLUMNS].to_numpy(dtype=float, copy=True)
    y = df["disposition"].isin(["CONFIRMED", "CANDIDATE"]).astype(int).to_numpy()

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    system = K2EnsembleSystem()
    system.load_system()

    rf = system.rf_model
    rf_scaler = system.rf_scaler
    tf_model = system.tf_model
    tf_scaler = system.tf_scaler

    rf_proba = rf.predict_proba(rf_scaler.transform(X_test))[:, 1]
    rf_pred = (rf_proba > 0.5).astype(int)

    tf_proba = tf_model.predict(tf_scaler.transform(X_test), verbose=0).flatten()
    tf_pred = (tf_proba > 0.5).astype(int)

    selections = system.director.select_expert(X_test)
    if isinstance(selections, str):
        selections = np.array([selections])

    dir_proba = np.zeros_like(rf_proba)
    dir_pred = np.zeros_like(rf_pred)

    rf_mask = selections == "RandomForest"
    tf_mask = selections == "TensorFlow"

    if np.any(rf_mask):
        pr = rf.predict_proba(rf_scaler.transform(X_test[rf_mask]))[:, 1]
        dir_proba[rf_mask] = pr
        dir_pred[rf_mask] = (pr > 0.5).astype(int)
    if np.any(tf_mask):
        pr = tf_model.predict(tf_scaler.transform(X_test[tf_mask]), verbose=0).flatten()
        dir_proba[tf_mask] = pr
        dir_pred[tf_mask] = (pr > 0.5).astype(int)

    return {
        "rf": eval_metrics(y_test, rf_proba, rf_pred),
        "tf": eval_metrics(y_test, tf_proba, tf_pred),
        "director": eval_metrics(y_test, dir_proba, dir_pred),
        "sample_count": int(len(y_test)),
        "positive_rate": float(y_test.mean()),
        "director_usage": {
            "rf": int(rf_mask.sum()),
            "tf": int(tf_mask.sum()),
        },
    }


def sweep_toi_gating():
    gaps = [0.05, 0.08, 0.12, 0.16]
    mins = [0.04, 0.08, 0.12]
    base_thresh = TOIDirector().thresholds
    results = []
    for g in gaps:
        for m in mins:
            th = type(base_thresh)(confidence_gap=g, minimum_confidence=m, complexity_threshold=base_thresh.complexity_threshold)
            metrics = evaluate_toi(th)
            results.append({
                "confidence_gap": g,
                "minimum_confidence": m,
                "accuracy": metrics["director"]["accuracy"],
                "precision": metrics["director"]["precision"],
                "recall": metrics["director"]["recall"],
                "auc": metrics["director"]["auc"],
                "rf_usage": metrics["director_usage"]["rf"],
                "tf_usage": metrics["director_usage"]["tf"],
            })
    return sorted(results, key=lambda r: r["accuracy"], reverse=True)


if __name__ == "__main__":
    koi = evaluate_koi()
    toi_current = evaluate_toi()
    toi_sweep = sweep_toi_gating()[:5]
    k2 = evaluate_k2()

    print(json.dumps({
        "koi": koi,
        "toi_current": toi_current,
        "k2": k2,
        "toi_sweep_top": toi_sweep,
    }, indent=2))
