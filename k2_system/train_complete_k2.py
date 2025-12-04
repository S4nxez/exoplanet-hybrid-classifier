"""End-to-end training pipeline for the K2 mission experts."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from k2_system.utils.data_utils import K2DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / 'data' / 'clean' / 'k2_clean.csv'
MODEL_DIR = Path(__file__).resolve().parent / 'saved_models'


def prepare_dataset():
    loader = K2DataLoader()
    X, y = loader.get_processed_data(DATA_PATH)
    logger.info("K2 dataset ready: %s samples, %s features", X.shape[0], X.shape[1])
    return X.values, y.values


def train_random_forest(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=3,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    logger.info("RandomForest accuracy: %.2f%%", acc * 100)
    logger.info("RandomForest report:\n%s", classification_report(y_test, preds))

    return model, scaler, acc


def build_tf_model(input_dim: int) -> tf.keras.Model:
    tf.random.set_seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    return model


def train_tensorflow(X_train, X_val, y_train, y_val):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = build_tf_model(X_train_scaled.shape[1])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    ]
    history = model.fit(
        X_train_scaled,
        y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=50,
        batch_size=64,
        callbacks=callbacks,
        verbose=0,
    )

    val_pred = (model.predict(X_val_scaled, verbose=0).flatten() > 0.5).astype(int)
    acc = accuracy_score(y_val, val_pred)
    logger.info("TensorFlow accuracy: %.2f%%", acc * 100)

    return model, scaler, acc, history.history


def save_artifacts(rf_model, rf_scaler, tf_model, tf_scaler, summary):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(rf_model, MODEL_DIR / 'k2_rf_model.pkl')
    joblib.dump(rf_scaler, MODEL_DIR / 'k2_rf_scaler.pkl')
    tf_model.save(MODEL_DIR / 'k2_tf_model.h5')
    joblib.dump(tf_scaler, MODEL_DIR / 'k2_tf_scaler.pkl')
    joblib.dump(summary, MODEL_DIR / 'training_summary.pkl')

    logger.info("Saved artifacts to %s", MODEL_DIR)


def main():
    X, y = prepare_dataset()
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    # Split holdout for TF validation so RF and TF see disjoint data slices
    X_val_tf, X_test_rf, y_val_tf, y_test_rf = train_test_split(
        X_holdout, y_holdout, test_size=0.5, random_state=42, stratify=y_holdout,
    )

    rf_model, rf_scaler, rf_acc = train_random_forest(X_train, X_test_rf, y_train, y_test_rf)
    tf_model, tf_scaler, tf_acc, history = train_tensorflow(X_train, X_val_tf, y_train, y_val_tf)

    summary = {
        'rf_accuracy': rf_acc,
        'tf_accuracy': tf_acc,
        'samples': len(X),
        'features': X.shape[1],
        'tf_history': history,
    }
    save_artifacts(rf_model, rf_scaler, tf_model, tf_scaler, summary)

    logger.info("Training summary: %s", summary)


if __name__ == '__main__':
    main()
